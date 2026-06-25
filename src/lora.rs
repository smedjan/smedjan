//! LoRA (Low-Rank Adaptation) for the Qwen3.5 hybrid transformer in smedjan.
//!
//! The base model stays frozen (quantized int4 weights, no gradients). LoRA adds low-rank
//! A/B matrix pairs to selected linear layers: `output = base_qmatmul(x) + scale * (x @ A) @ B`.
//! Only A/B are trainable (with gradients + optimizer state); the 9B base is never touched.
//!
//! Training: `qwen35_lora_train` loads the Q4 artifact, creates LoRA adapters, runs forward
//! with the LoRA delta, computes cross-entropy loss, backprops through A/B only, and steps
//! the optimizer. The quantized GEMM kernel handles the frozen base; the LoRA delta is a
//! regular f32 matmul (small: rank × in_dim + rank × out_dim per layer).

use crate::autograd;
use crate::gated_deltanet::{Mixer, Qwen35Layer, Qwen35Model};
use crate::gpu::MetalContext;
use crate::tensor::Tensor;
use std::sync::Arc;

/// One LoRA adapter: A `[in_dim, rank]`, B `[rank, out_dim]`, scale `alpha / rank`.
/// Initialized: A = small random, B = zeros (so the initial delta is zero — the model starts
/// identical to the base, and the adapter gradually learns the task-specific delta).
pub struct LoraAdapter {
    pub a: Tensor,  // [in_dim, rank], trainable
    pub b: Tensor,  // [rank, out_dim], trainable
    pub scale: f32, // alpha / rank
}

impl LoraAdapter {
    pub fn new(
        ctx: &Arc<MetalContext>,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> Self {
        // A: small random (std = 0.02), B: zeros (standard LoRA init).
        let a = Tensor::randn(ctx, vec![in_dim, rank], 0.02).with_grad();
        let b = Tensor::zeros(ctx, vec![rank, out_dim]).with_grad();
        LoraAdapter {
            a,
            b,
            scale: alpha / rank as f32,
        }
    }

    /// Compute the LoRA delta: `scale * (x @ A) @ B` where x is `[M, in_dim]`.
    /// This is added to the base (frozen) qmatmul output.
    pub fn delta(&self, x: &Tensor) -> Tensor {
        let xa = x.matmul(&self.a); // [M, rank]
        xa.matmul(&self.b).scale(self.scale) // [M, out_dim]
    }

    /// All trainable parameters (for the optimizer).
    pub fn params(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// LoRA adapters attached to one layer. Each field is `Some` when that linear has a LoRA adapter.
pub struct LoraLayer {
    pub qkv: Option<LoraAdapter>,      // DeltaNet in_proj_qkv
    pub w_a: Option<LoraAdapter>,      // DeltaNet gate pre-activation
    pub w_b: Option<LoraAdapter>,      // DeltaNet beta pre-activation
    pub z_gate: Option<LoraAdapter>,   // DeltaNet output-gate projection
    pub w_o: Option<LoraAdapter>,      // DeltaNet out_proj
    pub q_proj: Option<LoraAdapter>,   // Full-attn q_proj (doubled)
    pub k_proj: Option<LoraAdapter>,   // Full-attn k_proj
    pub v_proj: Option<LoraAdapter>,   // Full-attn v_proj
    pub o_proj: Option<LoraAdapter>,   // Full-attn o_proj
    pub ffn_gate: Option<LoraAdapter>, // MLP gate_proj
    pub ffn_up: Option<LoraAdapter>,   // MLP up_proj
    pub ffn_down: Option<LoraAdapter>, // MLP down_proj
}

/// The full LoRA-wrapped model: frozen base + per-layer adapters.
pub struct Qwen35LoraModel {
    pub base: Qwen35Model,
    pub lora_layers: Vec<LoraLayer>,
    pub rank: usize,
    pub alpha: f32,
}

impl Qwen35LoraModel {
    /// Create LoRA adapters for every linear layer in the model.
    /// `target_modules` controls which layers get adapters (empty = all).
    pub fn new(base: Qwen35Model, rank: usize, alpha: f32) -> Self {
        let ctx = &base.layers[0].ln1.ctx;
        let d = base.cfg.hidden_size;
        let inter = base.cfg.intermediate_size;
        let n_h = base.cfg.num_attention_heads;
        let n_kv = base.cfg.num_key_value_heads;
        let hd = base.cfg.head_dim;
        let n_k = base.cfg.linear_num_key_heads;
        let n_v = base.cfg.linear_num_value_heads;
        let ldh = base.cfg.linear_key_head_dim;
        let lvh = base.cfg.linear_value_head_dim;
        let qkv_out = 2 * n_k * ldh + n_v * lvh;

        let lora_layers = base
            .layers
            .iter()
            .map(|layer| {
                let is_full = matches!(layer.mixer, Mixer::Full(_));
                if is_full {
                    LoraLayer {
                        qkv: None,
                        w_a: None,
                        w_b: None,
                        z_gate: None,
                        w_o: None,
                        q_proj: Some(LoraAdapter::new(ctx, d, n_h * hd * 2, rank, alpha)),
                        k_proj: Some(LoraAdapter::new(ctx, d, n_kv * hd, rank, alpha)),
                        v_proj: Some(LoraAdapter::new(ctx, d, n_kv * hd, rank, alpha)),
                        o_proj: Some(LoraAdapter::new(ctx, n_h * hd, d, rank, alpha)),
                        ffn_gate: Some(LoraAdapter::new(ctx, d, inter, rank, alpha)),
                        ffn_up: Some(LoraAdapter::new(ctx, d, inter, rank, alpha)),
                        ffn_down: Some(LoraAdapter::new(ctx, inter, d, rank, alpha)),
                    }
                } else {
                    LoraLayer {
                        qkv: Some(LoraAdapter::new(ctx, d, qkv_out, rank, alpha)),
                        w_a: Some(LoraAdapter::new(ctx, d, n_v, rank, alpha)),
                        w_b: Some(LoraAdapter::new(ctx, d, n_v, rank, alpha)),
                        z_gate: Some(LoraAdapter::new(ctx, d, n_v * lvh, rank, alpha)),
                        w_o: Some(LoraAdapter::new(ctx, n_v * lvh, d, rank, alpha)),
                        q_proj: None,
                        k_proj: None,
                        v_proj: None,
                        o_proj: None,
                        ffn_gate: Some(LoraAdapter::new(ctx, d, inter, rank, alpha)),
                        ffn_up: Some(LoraAdapter::new(ctx, d, inter, rank, alpha)),
                        ffn_down: Some(LoraAdapter::new(ctx, inter, d, rank, alpha)),
                    }
                }
            })
            .collect();

        Qwen35LoraModel {
            base,
            lora_layers,
            rank,
            alpha,
        }
    }

    /// Collect all trainable LoRA parameters (for the optimizer).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.lora_layers {
            for adapter in [
                &layer.qkv,
                &layer.w_a,
                &layer.w_b,
                &layer.z_gate,
                &layer.w_o,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
            ] {
                if let Some(a) = adapter {
                    params.extend(a.params());
                }
            }
        }
        params
    }

    /// Forward with LoRA deltas applied. The base model runs with `no_grad` (frozen int4 weights);
    /// only the LoRA A/B matrices get gradients.
    ///
    /// This is a simplified version that adds LoRA deltas to the FFN layers only (the highest-impact
    /// target for fine-tuning). A full version would also intercept the attention projections — that
    /// requires modifying the strict forward functions to accept optional LoRA deltas. For now, the
    /// FFN-only path is functional and demonstrates the training loop.
    pub fn forward_lora(&self, x: &Tensor) -> Tensor {
        let c = &self.base.cfg;
        let eps = c.rms_norm_eps;
        let rot = (c.head_dim as f32 * c.partial_rotary_factor) as usize;
        let mut h = x.clone();

        for (i, layer) in self.base.layers.iter().enumerate() {
            let lora = &self.lora_layers[i];
            // Mixer: run base (frozen, no grad through int4 weights).
            let normed = h.rms_norm(&layer.ln1, eps);
            let mixed = autograd::no_grad(|| {
                if c.strict_qwen35 {
                    match &layer.mixer {
                        Mixer::Delta(d) => crate::gated_deltanet::qwen3_deltanet_mixer_strict(
                            &normed,
                            d,
                            c.linear_num_key_heads,
                            c.linear_num_value_heads,
                            c.linear_key_head_dim,
                            c.linear_conv_kernel_dim,
                            eps,
                        ),
                        Mixer::Full(f) => crate::gated_deltanet::qwen3_full_attention_mixer_strict(
                            &normed,
                            f,
                            c.num_attention_heads,
                            c.num_key_value_heads,
                            c.head_dim,
                            rot,
                            c.rope_theta,
                        ),
                    }
                } else {
                    // Placeholder path (synthetic tests).
                    match &layer.mixer {
                        Mixer::Delta(d) => crate::gated_deltanet::qwen3_deltanet_mixer(
                            &normed,
                            &crate::gated_deltanet::DeltaNetWeights {
                                w_q: &d.w_q,
                                w_k: &d.w_k,
                                w_v: &d.w_v,
                                conv_q: &d.conv_q,
                                conv_k: &d.conv_k,
                                conv_v: &d.conv_v,
                                w_a: &d.w_a,
                                w_b: &d.w_b,
                                w_gate: &d.w_gate,
                                out_norm: &d.out_norm,
                                w_o: &d.w_o,
                            },
                            c.linear_num_key_heads,
                            c.linear_num_value_heads,
                            c.linear_key_head_dim,
                            c.linear_conv_kernel_dim,
                        ),
                        Mixer::Full(f) => crate::gated_deltanet::qwen3_full_attention_mixer(
                            &normed,
                            &crate::gated_deltanet::FullAttnWeights {
                                w_q: &f.w_q,
                                w_k: &f.w_k,
                                w_v: &f.w_v,
                                qk_norm: &f.qk_norm,
                                w_gate: &f.w_gate,
                                w_o: &f.w_o,
                            },
                            c.num_attention_heads,
                            c.num_key_value_heads,
                            c.head_dim,
                            rot,
                            c.rope_theta,
                        ),
                    }
                }
            });
            h = h.add(&mixed);

            // FFN with LoRA: base (frozen) + LoRA delta (trainable).
            let normed2 = h.rms_norm(&layer.ln2, eps);
            let (b, s, d) = (normed2.shape[0], normed2.shape[1], normed2.shape[2]);
            let xf = normed2.reshape(vec![b * s, d]);

            // Base FFN (frozen).
            let base_g = autograd::no_grad(|| {
                crate::gated_deltanet::qmul(&xf, &layer.q_ffn_gate, &layer.ffn_gate)
            });
            let base_u = autograd::no_grad(|| {
                crate::gated_deltanet::qmul(&xf, &layer.q_ffn_up, &layer.ffn_up)
            });
            let base_silu = base_g.silu_gate(&base_u);

            // LoRA delta for gate/up/down (trainable, with gradients).
            let lora_g = lora.ffn_gate.as_ref().map(|a| a.delta(&xf));
            let lora_u = lora.ffn_up.as_ref().map(|a| a.delta(&xf));
            let total_g = match lora_g {
                Some(lg) => base_g.add(&lg),
                None => base_g,
            };
            let total_u = match lora_u {
                Some(lu) => base_u.add(&lu),
                None => base_u,
            };
            let silu_out = total_g.silu_gate(&total_u);

            // Base down (frozen) + LoRA down (trainable).
            let base_down = autograd::no_grad(|| {
                crate::gated_deltanet::qmul(&silu_out, &layer.q_ffn_down, &layer.ffn_down)
            });
            let lora_down = lora.ffn_down.as_ref().map(|a| a.delta(&silu_out));
            let total_down = match lora_down {
                Some(ld) => base_down.add(&ld),
                None => base_down,
            };
            h = h.add(&total_down.reshape(vec![b, s, d]));
        }

        let h = h.rms_norm(&self.base.final_norm, eps);
        let (b, s, d) = (h.shape[0], h.shape[1], h.shape[2]);
        let hf = h.reshape(vec![b * s, d]);
        // Base lm_head (frozen) + LoRA lm_head delta would go here if we add it.
        let logits = autograd::no_grad(|| {
            crate::gated_deltanet::qmul(&hf, &self.base.q_lm_head, &self.base.lm_head)
        });
        logits.reshape(vec![b, s, c.vocab_size as usize])
    }

    /// Save LoRA adapters to a safetensors file.
    pub fn save_lora(&self, path: &str) -> std::io::Result<()> {
        let mut blob = Vec::new();
        let mut entries = Vec::new();
        for (i, layer) in self.lora_layers.iter().enumerate() {
            for (name, adapter) in [
                ("qkv", &layer.qkv),
                ("w_a", &layer.w_a),
                ("w_b", &layer.w_b),
                ("z_gate", &layer.z_gate),
                ("w_o", &layer.w_o),
                ("q_proj", &layer.q_proj),
                ("k_proj", &layer.k_proj),
                ("v_proj", &layer.v_proj),
                ("o_proj", &layer.o_proj),
                ("ffn_gate", &layer.ffn_gate),
                ("ffn_up", &layer.ffn_up),
                ("ffn_down", &layer.ffn_down),
            ] {
                if let Some(a) = adapter {
                    for (suffix, t) in [("a", &a.a), ("b", &a.b)] {
                        let data = t.to_vec();
                        let start = blob.len();
                        blob.extend(data.iter().flat_map(|f| f.to_le_bytes()));
                        let end = blob.len();
                        let shape_s = t
                            .shape
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        entries.push(format!(
                            "\"lora.layers.{i}.{name}.{suffix}\":{{\"dtype\":\"F32\",\"shape\":[{shape_s}],\"data_offsets\":[{start},{end}]}}"
                        ));
                    }
                }
            }
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        file.write_all(&blob)?;
        eprintln!("LoRA adapters saved to {path} ({} tensors)", entries.len());
        Ok(())
    }
}

/// LoRA training configuration.
pub struct LoraTrainConfig {
    pub model_path: String,
    pub config_path: String,
    pub data_path: String,
    pub output_dir: String,
    pub rank: usize,
    pub alpha: f32,
    pub lr: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub iters: usize,
    pub save_every: usize,
    pub report_every: usize,
}

/// Run LoRA fine-tuning on the Qwen3.5 model.
pub fn qwen35_lora_train(ctx: &Arc<MetalContext>, config: &LoraTrainConfig) -> std::io::Result<()> {
    use crate::autograd;
    use crate::loss::cross_entropy_loss;
    use crate::optim::AdamW;
    use crate::safetensors::{config_from_hf_qwen35, import_qwen35_safetensors};

    eprintln!("=== Smedjan Qwen3.5 LoRA Fine-Tuning ===");
    eprintln!(
        "LoRA: rank={}, alpha={}, lr={:.1e}, batch={}, seq={}, iters={}",
        config.rank, config.alpha, config.lr, config.batch_size, config.seq_len, config.iters
    );

    // Load the base model (frozen, quantized int4).
    let cfg = config_from_hf_qwen35(&config.config_path)?;
    let mut base = import_qwen35_safetensors(ctx, &config.model_path, cfg.clone(), 64)?;
    base.cfg.strict_qwen35 = true;
    eprintln!(
        "Base model loaded: {} layers, d={}",
        cfg.num_hidden_layers, cfg.hidden_size
    );

    // Create LoRA wrappers.
    let model = Qwen35LoraModel::new(base, config.rank, config.alpha);
    let lora_params = model.lora_params();
    eprintln!(
        "LoRA parameters: {} tensors, {:.1}M params",
        lora_params.len(),
        lora_params.iter().map(|t| t.numel() as f64).sum::<f64>() / 1e6
    );

    // Optimizer on LoRA params only.
    let param_refs: Vec<&Tensor> = lora_params.iter().collect();
    let mut optimizer = AdamW::new(ctx, &param_refs, 0.0); // no weight decay for LoRA

    // Load training data (JSONL with "text" field → tokenized).
    let dataset = load_lora_dataset(&config.data_path)?;
    eprintln!("Dataset: {} examples", dataset.len());

    std::fs::create_dir_all(&config.output_dir)?;

    // Training loop.
    for step in 0..config.iters {
        // Get a batch: random sequences from the dataset.
        let batch = get_lora_batch(&dataset, config.batch_size, config.seq_len, &cfg);
        let x = batch.input; // [batch, seq, d]
        let targets = batch.targets; // [batch * seq]

        // Forward with LoRA (gradients flow through A/B only).
        autograd::clear_tape();
        let logits = model.forward_lora(&x); // [batch, seq, vocab]
        let (b, s, v) = (logits.shape[0], logits.shape[1], logits.shape[2]);
        let logits_flat = logits.reshape(vec![b * s, v]);

        // Cross-entropy loss.
        let (loss, grad_logits) = cross_entropy_loss(ctx, &logits_flat, &targets);

        // Backward through LoRA params only.
        autograd::backward(ctx, loss.id);

        // Optimizer step.
        optimizer.step(config.lr);

        if step % config.report_every == 0 {
            let loss_val = loss.to_vec()[0];
            eprintln!("step {step}/{}: loss={loss_val:.4}", config.iters);
        }

        if step > 0 && step % config.save_every == 0 {
            let path = format!("{}/lora_step_{step}.safetensors", config.output_dir);
            model.save_lora(&path)?;
        }
    }

    // Final save.
    let path = format!("{}/lora_final.safetensors", config.output_dir);
    model.save_lora(&path)?;
    eprintln!("LoRA training complete. Final adapter: {path}");

    Ok(())
}

/// Simple training dataset: list of token sequences.
struct LoraBatch {
    input: Tensor,     // [batch, seq, d_model] — embedded tokens
    targets: Vec<u32>, // [batch * seq] — next-token targets
}

/// Load a JSONL dataset where each line has a "text" field. Tokenizes using the model's tokenizer
/// (simplified: uses character-level tokenization for now — real BPE integration is a follow-up).
fn load_lora_dataset(path: &str) -> std::io::Result<Vec<Vec<u32>>> {
    let content = std::fs::read_to_string(path)?;
    let mut dataset = Vec::new();
    for line in content.lines() {
        // Simple JSONL "text" field extraction (avoids serde_json dependency).
        // Looks for "text": "..." pattern.
        if let Some(text) = extract_json_field(line, "text") {
            // Simple character-level tokenization (placeholder for BPE).
            let tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            if tokens.len() > 10 {
                dataset.push(tokens);
            }
        }
    }
    if dataset.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no valid training data",
        ));
    }
    Ok(dataset)
}

/// Extract a string field from a JSON line: finds `"field": "value"` and returns `value`.
/// Simple parser — handles escaped quotes minimally. For production, use a real JSON parser.
fn extract_json_field(line: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\":");
    let key_pos = line.find(&key)?;
    let after_key = &line[key_pos + key.len()..];
    // Skip whitespace.
    let after_key = after_key.trim_start();
    if !after_key.starts_with('"') {
        return None;
    }
    // Find closing quote (handle escaped quotes).
    let mut chars = after_key[1..].chars().peekable();
    let mut result = String::new();
    let mut escaped = false;
    for c in chars.by_ref() {
        if escaped {
            match c {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                _ => result.push(c),
            }
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(result);
        } else {
            result.push(c);
        }
    }
    None
}

fn get_lora_batch(
    dataset: &[Vec<u32>],
    batch_size: usize,
    seq_len: usize,
    cfg: &crate::gated_deltanet::Qwen35Config,
) -> LoraBatch {
    let ctx = Arc::new(MetalContext::new()); // TODO: share ctx
    let d = cfg.hidden_size;
    let vocab = cfg.vocab_size as usize;
    let mut input_data = vec![0.0f32; batch_size * seq_len * d];
    let mut targets = vec![0u32; batch_size * seq_len];

    for b in 0..batch_size {
        // Pick a random example, truncate/pad to seq_len.
        let example = &dataset[b % dataset.len()];
        let tokens: Vec<u32> = example
            .iter()
            .take(seq_len)
            .map(|&t| t % vocab as u32)
            .collect();
        // Embed: one-hot * random projection (placeholder — real embedding via q_embed).
        for (t, &tid) in tokens.iter().enumerate() {
            // Simple embedding: hash token into d-dim space (placeholder).
            for j in 0..d {
                let hash = (tid as usize).wrapping_mul(31).wrapping_add(j);
                input_data[b * seq_len * d + t * d + j] = ((hash % 100) as f32 - 50.0) * 0.01;
            }
            // Target = next token (shifted by 1).
            targets[b * seq_len + t] = if t + 1 < tokens.len() {
                tokens[t + 1]
            } else {
                0
            };
        }
        // Pad remaining positions.
        for t in tokens.len()..seq_len {
            targets[b * seq_len + t] = 0;
        }
    }

    let input = Tensor::from_slice(&ctx, &input_data, vec![batch_size, seq_len, d]);
    LoraBatch { input, targets }
}
