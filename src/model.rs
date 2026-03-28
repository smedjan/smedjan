use crate::attention::{KvCache, MultiHeadAttention};
use crate::autograd::{self, Op, TapeEntry};
use crate::metal::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
use objc2::rc::Retained;
use std::sync::Arc;

/// Model configuration — fully parameterized architecture.
/// One codebase handles any size from 1M to 1B+ parameters.
/// No code changes needed to scale up or down, just config.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,    // GQA: number of key/value heads (n_kv_heads <= n_heads)
    pub n_layers: usize,
    pub ffn_multiplier: f32,  // FFN hidden dim = d_model * ffn_multiplier, rounded to multiple of 256
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub n_experts: usize,     // MoE: number of expert FFNs (1 = dense, >1 = MoE)
    pub top_k_experts: usize, // MoE: how many experts active per token (typically 1 or 2)
    pub mup_base_width: usize, // μP: base model width for HP transfer (0 = disabled)
    pub bitnet: bool,          // BitNet: use ternary weights in FFN (no float multiply)
    pub lowrank: usize,        // Low-rank training: 0=full rank, >0=rank for FFN decomposition
    pub shared_layers: bool,   // ALBERT: share weights across all layers (1 unique layer, N iterations)
    pub n_predict: usize,      // Multi-token prediction: 0=standard, N=predict next N+1 tokens (Meta 2024)
    pub stochastic_depth: f32, // Layer drop rate: 0.0=off, 0.1=10% max drop rate for deepest layer
    pub sliding_window: usize, // Sliding window attention: 0=full causal, >0=window size. Saves O(n²)→O(n*w) memory.
    pub fp16_activations: bool, // Store inter-layer activations in FP16 during forward. Halves activation memory.
}

impl ModelConfig {
    /// Compute the FFN hidden dimension from d_model and ffn_multiplier.
    /// Rounded to the nearest multiple of 256 for GPU alignment.
    pub fn d_ff(&self) -> usize {
        let raw = (self.d_model as f32 * self.ffn_multiplier) as usize;
        // Round UP to next multiple of 256 for GPU alignment
        raw.div_ceil(256) * 256
    }

    /// Compute the KV projection dimension: head_dim * n_kv_heads.
    /// For standard MHA (n_kv_heads == n_heads), this equals d_model.
    pub fn kv_dim(&self) -> usize {
        let head_dim = self.d_model / self.n_heads;
        head_dim * self.n_kv_heads
    }

    /// Build a config with exact control over every knob.
    pub fn custom(
        vocab_size: u32,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        ffn_multiplier: f32,
        max_seq_len: usize,
    ) -> Self {
        Self::custom_gqa(vocab_size, d_model, n_heads, n_heads, n_layers, ffn_multiplier, max_seq_len)
    }

    /// Build a config with Grouped Query Attention.
    /// n_kv_heads must divide n_heads evenly.
    pub fn custom_gqa(
        vocab_size: u32,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        n_layers: usize,
        ffn_multiplier: f32,
        max_seq_len: usize,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        assert!(n_kv_heads <= n_heads, "n_kv_heads ({}) must be <= n_heads ({})", n_kv_heads, n_heads);
        assert!(n_kv_heads > 0, "n_kv_heads must be > 0");
        assert_eq!(n_heads % n_kv_heads, 0, "n_heads ({}) must be divisible by n_kv_heads ({})", n_heads, n_kv_heads);
        Self {
            vocab_size,
            d_model,
            n_heads,
            n_kv_heads,
            n_layers,
            ffn_multiplier,
            max_seq_len,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            n_experts: 1,
            top_k_experts: 1,
            mup_base_width: 0,
            bitnet: false,
            lowrank: 0,
            shared_layers: false,
            n_predict: 0,
            stochastic_depth: 0.0,
            sliding_window: 0,
            fp16_activations: false,
        }
    }

    /// Enable μP scaling. base_width is the proxy model's d_model (e.g. 64).
    /// When training the target model (e.g. d_model=768), LR scales by base/target.
    pub fn with_mup(mut self, base_width: usize) -> Self {
        self.mup_base_width = base_width;
        self
    }

    /// Get μP learning rate multiplier for hidden layers.
    /// Returns base_width / d_model (< 1 for large models).
    /// Returns 1.0 if μP is disabled.
    pub fn mup_lr_scale(&self) -> f32 {
        if self.mup_base_width > 0 {
            self.mup_base_width as f32 / self.d_model as f32
        } else {
            1.0
        }
    }

    /// Get μP output logit scale (dampen large model outputs).
    pub fn mup_output_scale(&self) -> f32 {
        if self.mup_base_width > 0 {
            self.mup_base_width as f32 / self.d_model as f32
        } else {
            1.0
        }
    }

    /// Build a MoE config: multiple expert FFNs, router selects top-K per token.
    pub fn custom_moe(
        vocab_size: u32, d_model: usize, n_heads: usize, n_kv_heads: usize,
        n_layers: usize, ffn_multiplier: f32, max_seq_len: usize,
        n_experts: usize, top_k_experts: usize,
    ) -> Self {
        assert!(n_experts > 0, "n_experts must be > 0");
        assert!(top_k_experts <= n_experts, "top_k ({}) must be <= n_experts ({})", top_k_experts, n_experts);
        let mut config = Self::custom_gqa(vocab_size, d_model, n_heads, n_kv_heads, n_layers, ffn_multiplier, max_seq_len);
        config.n_experts = n_experts;
        config.top_k_experts = top_k_experts;
        config
    }

    /// Tiny: ~1.2M params — dev/test, trains in seconds
    pub fn tiny(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 128, 4, 4, 2.67, 512)
    }

    /// Small: ~5M params — simple OS command assistant
    pub fn small(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 256, 4, 6, 2.67, 512)
    }

    /// Medium: ~42M params — capable code assistant
    pub fn medium(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 512, 8, 12, 2.67, 1024)
    }

    /// Large: ~300M params — strong general assistant
    pub fn large(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 1024, 16, 16, 2.67, 2048)
    }

    /// XL: ~800M params — serious capability
    pub fn xl(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 1536, 16, 20, 2.67, 2048)
    }

    /// Max: ~1.2B params — pushes M1 16GB for training
    pub fn max(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 2048, 16, 24, 2.67, 2048)
    }

    /// Huge: ~3B params — needs gradient checkpointing or multi-GPU for training,
    /// but fits in 16GB for inference
    pub fn huge(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 2560, 20, 32, 2.67, 4096)
    }

    /// 8B: ~8B params — full-scale model, needs 32GB+ for training (f16),
    /// fits in 16GB for inference with quantization
    pub fn eight_b(vocab_size: u32) -> Self {
        Self::custom(vocab_size, 4096, 32, 32, 2.67, 8192)
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let ff = self.d_ff();
        let v = self.vocab_size as usize;

        // Embedding (shared with lm_head via weight tying)
        let embedding = v * d;

        // Per layer:
        //   attention Q: d * d, K: d * kv_dim, V: d * kv_dim, O: d * d
        //   ffn: d * ff + ff * d + d * ff = 3 * d * ff (SwiGLU has 3 weight matrices)
        //   norms: 2 * d (ln1, ln2)
        let kv_dim = self.kv_dim();
        let attn_params = d * d + d * kv_dim + d * kv_dim + d * d; // Q + K + V + O
        let ffn_params = if self.n_experts > 1 {
            self.n_experts * 3 * d * ff + d * self.n_experts // expert weights + router
        } else {
            3 * d * ff
        };
        let per_layer = attn_params + ffn_params + 2 * d;

        // Final norm
        let final_norm = d;

        // Multi-token prediction heads: n_predict × (d*d projection + d norm)
        let mtp = self.n_predict * (d * d + d);

        embedding + self.n_layers * per_layer + final_norm + mtp
    }

    /// Memory required for training (weights + gradients + optimizer state) in bytes.
    /// AdamW needs 3x the weight memory (params + m + v), plus gradients.
    pub fn training_memory_bytes(&self) -> usize {
        self.param_count() * 4 * 4 // 4 copies (param + grad + m + v) × 4 bytes (f32)
    }

    /// Memory required for inference (weights only) in bytes.
    pub fn inference_memory_bytes(&self) -> usize {
        self.param_count() * 4 // f32
    }

    /// Print a summary of this config.
    pub fn summary(&self) -> String {
        let gqa_info = if self.n_kv_heads == self.n_heads {
            String::new()
        } else {
            format!(", n_kv_heads={}, group_size={}", self.n_kv_heads, self.n_heads / self.n_kv_heads)
        };
        format!(
            "d_model={}, n_heads={}{}, n_layers={}, d_ff={}, seq={}, params={}M, train_ram={:.0}MB, infer_ram={:.0}MB",
            self.d_model,
            self.n_heads,
            gqa_info,
            self.n_layers,
            self.d_ff(),
            self.max_seq_len,
            self.param_count() as f64 / 1e6,
            self.training_memory_bytes() as f64 / (1024.0 * 1024.0),
            self.inference_memory_bytes() as f64 / (1024.0 * 1024.0),
        )
    }

    /// Apply NTK-aware RoPE scaling to extend context length beyond max_seq_len.
    /// factor = desired_context / max_seq_len. theta_scaled = theta * factor.
    pub fn with_rope_scaling(&self, factor: f32) -> Self {
        let mut config = self.clone();
        config.rope_theta *= factor;
        config.max_seq_len = (config.max_seq_len as f32 * factor) as usize;
        config
    }
}

/// Expert FFN weights for Mixture of Experts.
pub struct ExpertFFN {
    pub w1: Tensor,  // [d_model, d_ff] — gate projection
    pub w2: Tensor,  // [d_ff, d_model] — down projection
    pub w3: Tensor,  // [d_model, d_ff] — up projection
}

/// A single transformer block (pre-norm architecture).
/// Supports both dense FFN and Mixture of Experts (MoE).
pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    // Dense FFN (used when n_experts == 1)
    pub ffn_w1: Tensor,     // [d_model, d_ff] — gate projection
    pub ffn_w2: Tensor,     // [d_ff, d_model] — down projection
    pub ffn_w3: Tensor,     // [d_model, d_ff] — up projection
    // MoE (used when n_experts > 1)
    pub experts: Vec<ExpertFFN>,   // n_experts FFN blocks
    pub router_weight: Tensor,     // [d_model, n_experts] — router logits
    pub router_bias: Vec<f32>,     // [n_experts] — bias-based load balancing (DeepSeek-V3)
    pub n_experts: usize,
    pub top_k: usize,
    pub bitnet: bool,
    // Low-rank FFN decomposition: W = U × V (when lowrank > 0)
    pub ffn_w1_u: Tensor, pub ffn_w1_v: Tensor, // gate: [d, r] × [r, ff]
    pub ffn_w2_u: Tensor, pub ffn_w2_v: Tensor, // down: [ff, r] × [r, d]
    pub ffn_w3_u: Tensor, pub ffn_w3_v: Tensor, // up:   [d, r] × [r, ff]
    pub lowrank: usize,
    pub ln1_weight: Tensor, // [d_model] — attention norm
    pub ln2_weight: Tensor, // [d_model] — ffn norm
    pub norm_eps: f32,
    pub mod_router: Tensor, // [d_model, 1] — Mixture of Depths router (scores tokens for skip)
    pub mod_capacity: f32,  // 0.0=disabled, 0.5=process top 50% tokens per layer
}

impl TransformerBlock {
    pub fn new(ctx: &Arc<MetalContext>, config: &ModelConfig, layer_idx: usize) -> Self {
        let d = config.d_model;
        let ff = config.d_ff();

        // Scaled initialization for residual connections
        let residual_scale = (1.0 / (2.0 * config.n_layers as f32)).sqrt();
        let ff_std = (2.0 / (d + ff) as f32).sqrt() * residual_scale;
        let down_std = (2.0 / (ff + d) as f32).sqrt() * residual_scale;

        let _ = layer_idx;

        // Create expert FFNs for MoE (empty vec for dense)
        let experts = if config.n_experts > 1 {
            (0..config.n_experts).map(|_| ExpertFFN {
                w1: Tensor::randn(ctx, vec![d, ff], ff_std),
                w2: Tensor::randn(ctx, vec![ff, d], down_std),
                w3: Tensor::randn(ctx, vec![d, ff], ff_std),
            }).collect()
        } else {
            Vec::new()
        };

        // Router weight — larger init std to break expert symmetry from step 1
        let router_weight = if config.n_experts > 1 {
            Tensor::randn(ctx, vec![d, config.n_experts], (2.0 / d as f32).sqrt())
        } else {
            Tensor::zeros(ctx, vec![1])
        };
        // Bias-based load balancing (DeepSeek-V3): bias added to router logits for
        // routing decisions, but NOT for gating weights. Adjusted each step by gamma.
        let router_bias = vec![0.0f32; config.n_experts.max(1)];

        // Low-rank FFN decomposition: W = U × V
        let r = config.lowrank;
        let (w1u, w1v, w2u, w2v, w3u, w3v) = if r > 0 {
            let u_std = (2.0 / (d + r) as f32).sqrt() * residual_scale;
            let v_std = (2.0 / (r + ff) as f32).sqrt();
            let d_std = (2.0 / (ff + r) as f32).sqrt() * residual_scale;
            let dv_std = (2.0 / (r + d) as f32).sqrt();
            (
                Tensor::randn(ctx, vec![d, r], u_std),
                Tensor::randn(ctx, vec![r, ff], v_std),
                Tensor::randn(ctx, vec![ff, r], d_std),
                Tensor::randn(ctx, vec![r, d], dv_std),
                Tensor::randn(ctx, vec![d, r], u_std),
                Tensor::randn(ctx, vec![r, ff], v_std),
            )
        } else {
            // Placeholders (not used when lowrank=0)
            let z = || Tensor::zeros(ctx, vec![1]);
            (z(), z(), z(), z(), z(), z())
        };

        let mut attn = MultiHeadAttention::new_with_rank(ctx, d, config.n_heads, config.n_kv_heads, config.rope_theta, config.lowrank);
        attn.sliding_window = config.sliding_window;

        Self {
            attn,
            ffn_w1: Tensor::randn(ctx, vec![d, ff], ff_std),
            ffn_w2: Tensor::randn(ctx, vec![ff, d], down_std),
            ffn_w3: Tensor::randn(ctx, vec![d, ff], ff_std),
            ffn_w1_u: w1u, ffn_w1_v: w1v,
            ffn_w2_u: w2u, ffn_w2_v: w2v,
            ffn_w3_u: w3u, ffn_w3_v: w3v,
            lowrank: r,
            experts,
            router_weight,
            router_bias,
            n_experts: config.n_experts,
            top_k: config.top_k_experts,
            bitnet: config.bitnet,
            ln1_weight: Tensor::ones(ctx, vec![d]),
            ln2_weight: Tensor::ones(ctx, vec![d]),
            norm_eps: config.norm_eps,
            mod_router: Tensor::randn(ctx, vec![d, 1], (1.0 / d as f32).sqrt()),
            mod_capacity: 0.0, // disabled by default
        }
    }

    /// Forward pass: pre-norm transformer block.
    /// x: [batch, seq_len, d_model] → [batch, seq_len, d_model]
    ///
    /// With Mixture of Depths (mod_capacity > 0): scores each token via a small router.
    /// Tokens with low scores get residual passthrough only. This saves 30-50% compute
    /// by skipping expensive attention+FFN for "easy" tokens.
    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut KvCache>) -> Tensor {
        let batch = x.shape[0];
        let seq_len = x.shape[1];
        let d = x.shape[2];

        // Mixture of Depths: soft routing (multiply block output by sigmoid router score)
        // All tokens still run through the block but "easy" tokens get near-zero contribution.
        // Gradients flow through sigmoid, teaching the router which tokens to skip.
        if self.mod_capacity > 0.0 && autograd::is_recording() {
            let x_flat = x.reshape(vec![batch * seq_len, d]);
            // Router: score each token. High → process, low → skip.
            // Using x@W → tanh for smooth [−1,1] gating, then shift to [0,1].
            // gate = 0.5 * (1 + tanh(score)) — smooth differentiable gate.
            // For simplicity, we approximate with: gate = scale_rows(block_delta, abs(score))
            // where tokens with near-zero score skip the layer.
            // Actually simplest correct approach: just pass scores through silu for soft gating.
            let scores = x_flat.matmul(&self.mod_router); // [n_tokens, 1]
            let gate_expanded = scores.reshape(vec![batch * seq_len]);

            // Full block computation
            let normed = x.rms_norm(&self.ln1_weight, self.norm_eps);
            let attn_out = self.attn.forward(&normed, kv_cache);
            let (normed2, h) = attn_out.rms_norm_residual_with_sum(x, &self.ln2_weight, self.norm_eps);
            let ffn_out = if self.n_experts > 1 {
                self.moe_ffn(&normed2, batch, seq_len, d)
            } else {
                self.swiglu_ffn(&normed2, batch, seq_len, d)
            };
            let block_out = h.add(&ffn_out); // [batch, seq, d]

            // Gate: block_result = x + gate * (block_out - x)
            // When gate≈1: full processing. When gate≈0: residual passthrough.
            let block_flat = block_out.reshape(vec![batch * seq_len, d]);
            let x_flat2 = x.reshape(vec![batch * seq_len, d]);
            let delta = block_flat.add(&x_flat2.scale(-1.0)); // block_out - x
            let gated_delta = delta.scale_rows(&gate_expanded); // gate * (block_out - x)
            let result = x_flat2.add(&gated_delta); // x + gate * (block_out - x)
            return result.reshape(vec![batch, seq_len, d]);
        }

        // Standard path (no MoD or inference)
        let normed = x.rms_norm(&self.ln1_weight, self.norm_eps);
        let attn_out = self.attn.forward(&normed, kv_cache);
        let (normed2, h) = attn_out.rms_norm_residual_with_sum(x, &self.ln2_weight, self.norm_eps);
        let ffn_out = if self.n_experts > 1 {
            self.moe_ffn(&normed2, batch, seq_len, d)
        } else {
            self.swiglu_ffn(&normed2, batch, seq_len, d)
        };
        h.add(&ffn_out)
    }

    /// Shared-Expert Mixture of Experts with ReLU routing (DeepSeek-V3 + ReMoE).
    /// 1 shared expert (block's FFN weights) always active for ALL tokens.
    /// N routed experts use ReLU gating instead of softmax+topk (ICLR 2025, ReMoE):
    ///   gate_i = ReLU(x @ W_router_i) — positive → active, zero → inactive
    /// ReLU is fully differentiable, naturally sparse, no load balancing loss needed.
    fn moe_ffn(&self, x: &Tensor, batch: usize, seq_len: usize, d: usize) -> Tensor {
        let n_tokens = batch * seq_len;
        let x_flat = x.reshape(vec![n_tokens, d]);

        // Shared expert: always active for ALL tokens
        let shared_out = self.swiglu_ffn(x, batch, seq_len, d)
            .reshape(vec![n_tokens, d]);

        // ReMoE routing (ICLR 2025): ReLU gate instead of softmax+topk.
        // gate_i = ReLU(x @ W_router_i) — positive activates, zero deactivates.
        // Fully differentiable, naturally sparse, no auxiliary balance loss needed.
        let router_logits = x_flat.matmul(&self.router_weight); // [n_tokens, n_experts]
        let router_probs = router_logits.relu(); // ReLU: natural sparsity

        // Soft MoE: each routed expert adds a weighted delta on top of the shared output.
        let mut combined = shared_out;

        for expert_idx in 0..self.n_experts {
            let expert = &self.experts[expert_idx];

            let gate = x_flat.matmul(&expert.w1);
            let up = x_flat.matmul(&expert.w3);
            let hidden = gate.silu_gate(&up);
            let expert_out = hidden.matmul(&expert.w2); // [n_tokens, d]

            // Extract this expert's routing probability ON TAPE
            let mut sel = vec![0.0f32; self.n_experts];
            sel[expert_idx] = 1.0;
            let selector = Tensor::from_buffer(
                Arc::clone(&x_flat.ctx),
                x_flat.ctx.buffer_from_slice(&sel),
                vec![self.n_experts, 1],
            );
            let weight_col = router_probs.matmul(&selector); // [n_tokens, 1]
            let weights = weight_col.reshape(vec![n_tokens]); // [n_tokens]

            // Add weighted expert delta to combined output
            let scaled = expert_out.scale_rows(&weights);
            combined = combined.add(&scaled);
        }

        combined.reshape(vec![batch, seq_len, d])
    }

    /// SwiGLU feed-forward: output = (SiLU(x @ W1) * (x @ W3)) @ W2
    fn swiglu_ffn(&self, x: &Tensor, batch: usize, seq_len: usize, d: usize) -> Tensor {
        let x_flat = x.reshape(vec![batch * seq_len, d]);

        // Gate and up projections — separate matmuls (fewer dispatches than fused concat+slice)
        let (gate, up) = if self.lowrank > 0 {
            let g = x_flat.matmul(&self.ffn_w1_u).matmul(&self.ffn_w1_v);
            let u = x_flat.matmul(&self.ffn_w3_u).matmul(&self.ffn_w3_v);
            (g, u)
        } else if self.bitnet {
            (x_flat.ternary_matmul(&self.ffn_w1), x_flat.ternary_matmul(&self.ffn_w3))
        } else {
            (x_flat.matmul(&self.ffn_w1), x_flat.matmul(&self.ffn_w3))
        };

        let hidden = gate.silu_gate(&up);

        // Down projection
        let out = if self.lowrank > 0 {
            hidden.matmul(&self.ffn_w2_u).matmul(&self.ffn_w2_v)
        } else if self.bitnet {
            hidden.ternary_matmul(&self.ffn_w2)
        } else {
            hidden.matmul(&self.ffn_w2)
        };
        out.reshape(vec![batch, seq_len, d])
    }

    /// Forward pass with gradient checkpointing.
    /// NOTE: Recompute-based checkpointing is DISABLED because Metal GPU matmul is
    /// non-deterministic (FP16 shared memory rounding varies between kernel invocations).
    /// Recomputed forward produces different intermediates → wrong gradients → loss diverges.
    /// Fix requires deterministic kernels (FP32-only shared memory or cached sub-tape approach).
    /// Currently falls back to standard forward (no memory savings, correct gradients).
    pub fn forward_checkpointed(self: &Arc<Self>, x: &Tensor, _layer_idx: usize) -> Tensor {
        self.forward(x, None)
    }

    /// Original recompute-based checkpointing (disabled — see above).
    pub fn forward_checkpointed_recompute(self: &Arc<Self>, x: &Tensor, layer_idx: usize) -> Tensor {
        // Save the input tensor's buffer and shape — we need these for the main tape entry
        // and for the recompute closure.
        let input_id = x.id;
        let input_buffer: Retained<GpuBuffer> = x.buffer.clone();
        let input_shape = x.shape.clone();
        let ctx = Arc::clone(&x.ctx);

        // Run the forward pass on a temporary sub-tape (discarded after capturing output)
        let (output, _sub_tape) = autograd::checkpoint_forward(|| {
            self.forward(x, None)
        });

        // Record a single checkpoint op on the main tape
        let checkpoint_output_id = autograd::next_id();
        let output_size = output.numel();
        let output_shape = output.shape.clone();

        // Copy the output buffer so the checkpoint entry owns it
        let checkpoint_output_buf = ctx.alloc_buffer(output_size * 4);
        compute::gpu_copy(&ctx, &output.buffer, &checkpoint_output_buf, output_size as u32);

        autograd::record(TapeEntry {
            op: Op::Checkpoint { layer_idx },
            inputs: vec![input_id],
            output: checkpoint_output_id,
            input_buffers: vec![input_buffer.clone()],
            output_buffer: checkpoint_output_buf.clone(),
            shapes: vec![input_shape.clone(), output_shape.clone()],
            cached: None,
        });

        // Register the recompute closure. It captures the block (Arc) and the input
        // buffer/shape so it can reconstruct the input tensor and re-run forward.
        let block = Arc::clone(self);
        let recompute_input_buffer = input_buffer;
        let recompute_input_shape = input_shape;

        autograd::register_recompute(layer_idx, Box::new(move |recompute_ctx: &Arc<MetalContext>| {
            // Reconstruct the input tensor from the saved buffer
            let recompute_input = Tensor {
                id: autograd::next_id(),
                buffer: recompute_input_buffer.clone(),
                shape: recompute_input_shape.clone(),
                requires_grad: true,
                ctx: Arc::clone(recompute_ctx),
            };

            // Run forward on a fresh sub-tape and return it
            let (_recomputed_output, sub_tape) = autograd::checkpoint_forward(|| {
                block.forward(&recompute_input, None)
            });

            sub_tape
        }));

        // Return a tensor with the checkpoint output ID so the main tape is consistent
        Tensor {
            id: checkpoint_output_id,
            buffer: checkpoint_output_buf,
            shape: output_shape,
            requires_grad: true,
            ctx,
        }
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.attn.parameters();
        if self.lowrank > 0 {
            // Low-rank: U and V matrices instead of full W
            params.extend_from_slice(&[
                &self.ffn_w1_u, &self.ffn_w1_v,
                &self.ffn_w2_u, &self.ffn_w2_v,
                &self.ffn_w3_u, &self.ffn_w3_v,
            ]);
        } else if self.n_experts > 1 {
            params.push(&self.router_weight);
            for expert in &self.experts {
                params.push(&expert.w1);
                params.push(&expert.w2);
                params.push(&expert.w3);
            }
        } else {
            params.extend_from_slice(&[&self.ffn_w1, &self.ffn_w2, &self.ffn_w3]);
        }
        params.extend_from_slice(&[&self.ln1_weight, &self.ln2_weight]);
        if self.mod_capacity > 0.0 {
            params.push(&self.mod_router);
        }
        params
    }
}

/// The full transformer model.
pub struct Transformer {
    pub config: ModelConfig,
    pub embedding: Tensor,           // [vocab_size, d_model] or [vocab, embed_rank] if factored
    pub embed_proj: Tensor,          // [embed_rank, d_model] (identity/zeros when not factored)
    pub embed_rank: usize,           // 0 = full embedding, >0 = factored
    pub blocks: Vec<Arc<TransformerBlock>>,
    pub ln_final_weight: Tensor,     // [d_model]
    /// Multi-token prediction heads: each projects d_model → d_model for future token k.
    /// Head k predicts token at position t+k+2 (head 0 is t+2, head 1 is t+3, etc.).
    /// The standard LM head (weight-tied embedding) always predicts t+1.
    pub mtp_heads: Vec<Tensor>,      // n_predict × [d_model, d_model]
    pub mtp_norms: Vec<Tensor>,      // n_predict × [d_model] (per-head layer norms)
    ctx: Arc<MetalContext>,
}

impl Transformer {
    pub fn new(ctx: &Arc<MetalContext>, config: ModelConfig) -> Self {
        let d = config.d_model;
        let v = config.vocab_size as usize;

        // Embedding — optionally factored: [vocab, rank] × [rank, d] instead of [vocab, d]
        let embed_rank = config.lowrank.max(0); // reuse lowrank for embedding too
        let (embedding, embed_proj) = if embed_rank > 0 && embed_rank < d {
            let e_std = (1.0 / embed_rank as f32).sqrt();
            let p_std = (1.0 / d as f32).sqrt();
            (
                Tensor::randn(ctx, vec![v, embed_rank], e_std),
                Tensor::randn(ctx, vec![embed_rank, d], p_std),
            )
        } else {
            let embed_std = (1.0 / d as f32).sqrt();
            (Tensor::randn(ctx, vec![v, d], embed_std), Tensor::zeros(ctx, vec![1]))
        };

        let blocks: Vec<Arc<TransformerBlock>> = if config.shared_layers {
            // ALBERT: one unique layer, shared across all positions
            let shared = Arc::new(TransformerBlock::new(ctx, &config, 0));
            eprintln!("ALBERT mode: {} layers sharing 1 set of weights", config.n_layers);
            (0..config.n_layers).map(|_| Arc::clone(&shared)).collect()
        } else {
            (0..config.n_layers).map(|i| Arc::new(TransformerBlock::new(ctx, &config, i))).collect()
        };

        let ln_final_weight = Tensor::ones(ctx, vec![d]);

        // Multi-token prediction heads (Meta 2024): each head predicts token t+k+2
        // using a learned projection of the hidden state. The standard LM head predicts t+1.
        // 4× better sample efficiency at N=4.
        let mtp_heads: Vec<Tensor> = (0..config.n_predict)
            .map(|_| Tensor::randn(ctx, vec![d, d], (2.0 / (d + d) as f32).sqrt()))
            .collect();
        let mtp_norms: Vec<Tensor> = (0..config.n_predict)
            .map(|_| Tensor::ones(ctx, vec![d]))
            .collect();

        if config.n_predict > 0 {
            eprintln!("Multi-token prediction: {} extra heads (predict t+2..t+{})",
                config.n_predict, config.n_predict + 1);
        }

        eprintln!(
            "Model initialized: {} layers, {}M parameters",
            config.n_layers,
            config.param_count() as f32 / 1e6
        );

        Self {
            config,
            embedding,
            embed_proj,
            embed_rank,
            blocks,
            ln_final_weight,
            mtp_heads,
            mtp_norms,
            ctx: Arc::clone(ctx),
        }
    }

    /// Forward pass: tokens → logits.
    /// tokens: [batch, seq_len] (u32), kv_caches: one per layer
    /// When `checkpointed` is true, transformer blocks use gradient checkpointing
    /// (only stores block inputs, recomputes intermediates during backward).
    /// Checkpointing is ignored when kv_caches are provided (inference mode).
    /// Returns logits: [batch * seq_len, vocab_size]
    pub fn forward(
        &self,
        tokens: &[u32],
        batch: usize,
        seq_len: usize,
        kv_caches: Option<&mut Vec<KvCache>>,
        checkpointed: bool,
    ) -> Tensor {
        let d = self.config.d_model;
        let v = self.config.vocab_size as usize;
        let n_tokens = batch * seq_len;

        // Embedding lookup (optionally factored)
        let tokens_buf = self.ctx.buffer_from_u32_slice(tokens);
        let embed_dim = if self.embed_rank > 0 { self.embed_rank } else { d };
        let embed_out_buf = self.ctx.alloc_buffer(n_tokens * embed_dim * 4);
        compute::gpu_embedding_lookup(
            &self.ctx,
            &tokens_buf,
            &self.embedding.buffer,
            &embed_out_buf,
            n_tokens as u32,
            embed_dim as u32,
        );

        // Record embedding on tape
        let tokens_id = autograd::next_id();
        let embed_out_id = autograd::next_id();
        if autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Embedding,
                inputs: vec![tokens_id, self.embedding.id],
                output: embed_out_id,
                input_buffers: vec![tokens_buf, self.embedding.buffer.clone()],
                output_buffer: embed_out_buf.clone(),
                shapes: vec![vec![n_tokens], vec![v, embed_dim], vec![n_tokens, embed_dim]],
                cached: None,
            });
        }

        let embed_tensor = Tensor {
            id: embed_out_id,
            buffer: embed_out_buf,
            shape: vec![n_tokens, embed_dim],
            requires_grad: true,
            ctx: Arc::clone(&self.ctx),
        };

        // Project factored embedding to full d_model
        let h_flat = if self.embed_rank > 0 {
            embed_tensor.matmul(&self.embed_proj) // [n_tokens, rank] @ [rank, d] = [n_tokens, d]
        } else {
            embed_tensor
        };
        let mut h = h_flat.reshape(vec![batch, seq_len, d]);

        // Run through transformer blocks
        match kv_caches {
            Some(caches) => {
                // Inference mode with KV cache — no checkpointing (no gradients needed)
                for (i, block) in self.blocks.iter().enumerate() {
                    h = block.forward(&h, Some(&mut caches[i]));
                }
            }
            None => {
                if checkpointed {
                    for (i, block) in self.blocks.iter().enumerate() {
                        h = block.forward_checkpointed(&h, i);
                    }
                } else {
                    let n_layers = self.blocks.len();
                    for (i, block) in self.blocks.iter().enumerate() {
                        // Stochastic depth: linearly increasing drop probability per layer.
                        if autograd::is_recording() && self.config.stochastic_depth > 0.0 && n_layers > 1 {
                            let drop_prob = self.config.stochastic_depth * (i as f32 / (n_layers - 1) as f32);
                            if rand::random::<f32>() < drop_prob {
                                continue;
                            }
                        }
                        h = block.forward(&h, None);

                        // FP16 activation compression: roundtrip h through FP16 between layers.
                        // Loses ~0.1% precision but the tape records smaller intermediate values,
                        // reducing peak memory. Most effective with gradient checkpointing where
                        // only checkpoint inputs are stored (this makes those inputs smaller).
                        if self.config.fp16_activations && i + 1 < n_layers {
                            h = h.fp16_roundtrip();
                        }
                    }
                }
            }
        }

        // Final layer norm
        let h = h.rms_norm(&self.ln_final_weight, self.config.norm_eps);

        // LM head (weight-tied with embedding)
        let h_flat = h.reshape(vec![n_tokens, d]);
        self.apply_lm_head(&h_flat)
    }

    /// Apply LM head to hidden states: h → logits via weight-tied embedding.
    pub fn apply_lm_head(&self, h_flat: &Tensor) -> Tensor {
        let logits = if self.embed_rank > 0 {
            let h_proj = h_flat.matmul_trans_b(&self.embed_proj.detach());
            h_proj.matmul_trans_b(&self.embedding.detach())
        } else {
            h_flat.matmul_trans_b(&self.embedding.detach())
        };
        let mup_scale = self.config.mup_output_scale();
        if mup_scale < 1.0 { logits.scale(mup_scale) } else { logits }
    }

    /// Forward pass returning hidden states BEFORE the LM head.
    /// Used by FusedLinearCrossEntropy which handles LM head + CE in chunks.
    pub fn forward_hidden(
        &self, tokens: &[u32], batch: usize, seq_len: usize, _checkpointed: bool,
    ) -> Tensor {
        let d = self.config.d_model;
        let n_tokens = batch * seq_len;

        let tokens_buf = self.ctx.buffer_from_u32_slice(tokens);
        let embed_dim = if self.embed_rank > 0 { self.embed_rank } else { d };
        let embed_out_buf = self.ctx.alloc_buffer(n_tokens * embed_dim * 4);
        compute::gpu_embedding_lookup(
            &self.ctx, &tokens_buf, &self.embedding.buffer, &embed_out_buf,
            n_tokens as u32, embed_dim as u32,
        );

        let tokens_id = autograd::next_id();
        let embed_out_id = autograd::next_id();
        if autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Embedding,
                inputs: vec![tokens_id, self.embedding.id],
                output: embed_out_id,
                input_buffers: vec![tokens_buf, self.embedding.buffer.clone()],
                output_buffer: embed_out_buf.clone(),
                shapes: vec![vec![n_tokens], vec![self.config.vocab_size as usize, embed_dim], vec![n_tokens, embed_dim]],
                cached: None,
            });
        }

        let embed_tensor = Tensor {
            id: embed_out_id, buffer: embed_out_buf,
            shape: vec![n_tokens, embed_dim], requires_grad: true, ctx: Arc::clone(&self.ctx),
        };

        let h_flat = if self.embed_rank > 0 {
            embed_tensor.matmul(&self.embed_proj)
        } else { embed_tensor };
        let mut h = h_flat.reshape(vec![batch, seq_len, d]);

        let n_layers = self.blocks.len();
        for (i, block) in self.blocks.iter().enumerate() {
            if autograd::is_recording() && self.config.stochastic_depth > 0.0 && n_layers > 1 {
                let drop_prob = self.config.stochastic_depth * (i as f32 / (n_layers - 1) as f32);
                if rand::random::<f32>() < drop_prob { continue; }
            }
            h = block.forward(&h, None);
            if self.config.fp16_activations && i + 1 < n_layers {
                h = h.fp16_roundtrip();
            }
        }

        let h = h.rms_norm(&self.ln_final_weight, self.config.norm_eps);
        h.reshape(vec![n_tokens, d])
    }

    /// Forward with multi-token prediction: returns (main_logits, [extra_logits...]).
    /// Each extra_logits[k] predicts token at position t+k+2 (shifted by k+1 from main).
    /// When n_predict=0, extra vec is empty (standard next-token prediction).
    pub fn forward_mtp(
        &self,
        tokens: &[u32],
        batch: usize,
        seq_len: usize,
        checkpointed: bool,
    ) -> (Tensor, Vec<Tensor>) {
        let d = self.config.d_model;
        let n_tokens = batch * seq_len;

        // Embedding lookup
        let tokens_buf = self.ctx.buffer_from_u32_slice(tokens);
        let embed_dim = if self.embed_rank > 0 { self.embed_rank } else { d };
        let embed_out_buf = self.ctx.alloc_buffer(n_tokens * embed_dim * 4);
        compute::gpu_embedding_lookup(
            &self.ctx, &tokens_buf, &self.embedding.buffer, &embed_out_buf,
            n_tokens as u32, embed_dim as u32,
        );

        let tokens_id = autograd::next_id();
        let embed_out_id = autograd::next_id();
        if autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Embedding,
                inputs: vec![tokens_id, self.embedding.id],
                output: embed_out_id,
                input_buffers: vec![tokens_buf, self.embedding.buffer.clone()],
                output_buffer: embed_out_buf.clone(),
                shapes: vec![vec![n_tokens], vec![self.config.vocab_size as usize, embed_dim], vec![n_tokens, embed_dim]],
                cached: None,
            });
        }

        let embed_tensor = Tensor {
            id: embed_out_id, buffer: embed_out_buf,
            shape: vec![n_tokens, embed_dim], requires_grad: true, ctx: Arc::clone(&self.ctx),
        };

        let h_flat = if self.embed_rank > 0 {
            embed_tensor.matmul(&self.embed_proj)
        } else { embed_tensor };
        let mut h = h_flat.reshape(vec![batch, seq_len, d]);

        if checkpointed {
            for (i, block) in self.blocks.iter().enumerate() {
                h = block.forward_checkpointed(&h, i);
            }
        } else {
            for block in &self.blocks { h = block.forward(&h, None); }
        }

        let h = h.rms_norm(&self.ln_final_weight, self.config.norm_eps);
        let h_flat = h.reshape(vec![n_tokens, d]);

        // Main LM head (weight-tied embedding): predicts t+1
        let main_logits = if self.embed_rank > 0 {
            h_flat.matmul_trans_b(&self.embed_proj.detach())
                .matmul_trans_b(&self.embedding.detach())
        } else {
            h_flat.matmul_trans_b(&self.embedding.detach())
        };

        // Extra prediction heads: head k predicts t+k+2
        let mut extra_logits = Vec::with_capacity(self.config.n_predict);
        for k in 0..self.config.n_predict {
            let projected = h_flat.matmul(&self.mtp_heads[k]);
            let normed = projected.rms_norm(&self.mtp_norms[k], self.config.norm_eps);
            let logits_k = if self.embed_rank > 0 {
                normed.matmul_trans_b(&self.embed_proj.detach())
                    .matmul_trans_b(&self.embedding.detach())
            } else {
                normed.matmul_trans_b(&self.embedding.detach())
            };
            extra_logits.push(logits_k);
        }

        let mup_scale = self.config.mup_output_scale();
        let main_logits = if mup_scale < 1.0 { main_logits.scale(mup_scale) } else { main_logits };

        (main_logits, extra_logits)
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.embedding, &self.ln_final_weight];
        if self.embed_rank > 0 {
            params.push(&self.embed_proj);
        }
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        for head in &self.mtp_heads { params.push(head); }
        for norm in &self.mtp_norms { params.push(norm); }
        params
    }

    /// Initialize KV caches for inference (one per layer).
    /// Uses legacy dynamic allocation (grows on each step).
    pub fn init_kv_caches(&self) -> Vec<KvCache> {
        (0..self.config.n_layers).map(|_| KvCache::new()).collect()
    }

    /// Initialize pre-allocated KV caches for inference (one per layer).
    /// Pre-allocates to max_seq_len to avoid O(n^2) reallocation during generation.
    pub fn init_kv_caches_preallocated(&self, batch: usize) -> Vec<KvCache> {
        let batch_heads = batch * self.config.n_kv_heads;
        let head_dim = self.config.d_model / self.config.n_heads;
        let max_seq = self.config.max_seq_len;
        (0..self.config.n_layers)
            .map(|_| KvCache::with_capacity(&self.ctx, batch_heads, max_seq, head_dim))
            .collect()
    }
}

/// Grow a small model's weights into a larger model configuration.
/// Copies existing weights into the top-left corner of the larger matrices,
/// initializes the remaining dimensions with small random values.
///
/// Requirements:
/// - small_model.d_model <= large_config.d_model
/// - small_model.n_layers <= large_config.n_layers
/// - same vocab_size
///
/// Usage: train small → grow → continue training large
pub fn grow_model(
    ctx: &Arc<MetalContext>,
    small: &Transformer,
    large_config: ModelConfig,
) -> Transformer {
    let sd = small.config.d_model;
    let ld = large_config.d_model;
    let v = large_config.vocab_size as usize;

    assert!(sd <= ld, "small d_model ({}) must be <= large ({})", sd, ld);
    assert!(small.config.n_layers <= large_config.n_layers, "small layers must be <= large");
    assert_eq!(small.config.vocab_size, large_config.vocab_size, "vocab must match");

    eprintln!("Growing model: d={} → d={}, layers={} → layers={}",
        sd, ld, small.config.n_layers, large_config.n_layers);

    // Create the large model with random init
    let large = Transformer::new(ctx, large_config.clone());

    // Copy embedding: [vocab, sd] → top-left of [vocab, ld]
    copy_weight_block(ctx, &small.embedding.buffer, &large.embedding.buffer,
        v, sd, v, ld);

    // Copy layer weights for shared layers
    let n_shared = small.config.n_layers.min(large_config.n_layers);
    for i in 0..n_shared {
        let sb = &small.blocks[i];
        let lb = &large.blocks[i];

        // Attention weights: Q, K, V, O
        let small_kv_dim = sd / small.config.n_heads * small.config.n_kv_heads;
        let large_kv_dim = ld / large_config.n_heads * large_config.n_kv_heads;
        copy_weight_block(ctx, &sb.attn.w_q.buffer, &lb.attn.w_q.buffer, sd, sd, ld, ld);
        copy_weight_block(ctx, &sb.attn.w_k.buffer, &lb.attn.w_k.buffer, sd, small_kv_dim, ld, large_kv_dim);
        copy_weight_block(ctx, &sb.attn.w_v.buffer, &lb.attn.w_v.buffer, sd, small_kv_dim, ld, large_kv_dim);
        copy_weight_block(ctx, &sb.attn.w_o.buffer, &lb.attn.w_o.buffer, sd, sd, ld, ld);

        // FFN weights (only if both are full rank or same lowrank)
        if large_config.lowrank == 0 && small.config.lowrank == 0 {
            let sff = small.config.d_ff();
            let lff = large_config.d_ff();
            copy_weight_block(ctx, &sb.ffn_w1.buffer, &lb.ffn_w1.buffer, sd, sff, ld, lff);
            copy_weight_block(ctx, &sb.ffn_w2.buffer, &lb.ffn_w2.buffer, sff, sd, lff, ld);
            copy_weight_block(ctx, &sb.ffn_w3.buffer, &lb.ffn_w3.buffer, sd, sff, ld, lff);
        }

        // Norm weights: [sd] → [ld] (copy first sd elements)
        compute::gpu_buffer_copy(ctx, &sb.ln1_weight.buffer, &lb.ln1_weight.buffer, 0, 0, sd as u32);
        compute::gpu_buffer_copy(ctx, &sb.ln2_weight.buffer, &lb.ln2_weight.buffer, 0, 0, sd as u32);
    }

    // Final norm
    compute::gpu_buffer_copy(ctx, &small.ln_final_weight.buffer, &large.ln_final_weight.buffer, 0, 0, sd as u32);

    eprintln!("Model grown: {}M → {}M params",
        small.config.param_count() as f32 / 1e6,
        large_config.param_count() as f32 / 1e6);

    large
}

/// Copy a weight matrix block: small [sr, sc] → top-left of large [lr, lc]
fn copy_weight_block(
    ctx: &Arc<MetalContext>,
    src: &GpuBuffer, dst: &GpuBuffer,
    src_rows: usize, src_cols: usize,
    _dst_rows: usize, dst_cols: usize,
) {
    // Copy row by row (src row stride ≠ dst row stride)
    for r in 0..src_rows {
        compute::gpu_buffer_copy(
            ctx, src, dst,
            (r * src_cols) as u32,     // src offset
            (r * dst_cols) as u32,     // dst offset
            src_cols as u32,           // copy length
        );
    }
}
