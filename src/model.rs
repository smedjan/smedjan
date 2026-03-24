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
    pub n_layers: usize,
    pub ffn_multiplier: f32,  // FFN hidden dim = d_model * ffn_multiplier, rounded to multiple of 256
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl ModelConfig {
    /// Compute the FFN hidden dimension from d_model and ffn_multiplier.
    /// Rounded to the nearest multiple of 256 for GPU alignment.
    pub fn d_ff(&self) -> usize {
        let raw = (self.d_model as f32 * self.ffn_multiplier) as usize;
        // Round UP to next multiple of 256 for GPU alignment
        raw.div_ceil(256) * 256
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
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        Self {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            ffn_multiplier,
            max_seq_len,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        }
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
        //   attention: 4 * d * d (Q, K, V, O)
        //   ffn: d * ff + ff * d + d * ff = 3 * d * ff (SwiGLU has 3 weight matrices)
        //   norms: 2 * d (ln1, ln2)
        let per_layer = 4 * d * d + 3 * d * ff + 2 * d;

        // Final norm
        let final_norm = d;

        embedding + self.n_layers * per_layer + final_norm
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
        format!(
            "d_model={}, n_heads={}, n_layers={}, d_ff={}, seq={}, params={}M, train_ram={:.0}MB, infer_ram={:.0}MB",
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff(),
            self.max_seq_len,
            self.param_count() as f64 / 1e6,
            self.training_memory_bytes() as f64 / (1024.0 * 1024.0),
            self.inference_memory_bytes() as f64 / (1024.0 * 1024.0),
        )
    }
}

/// A single transformer block (pre-norm architecture).
pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub ffn_w1: Tensor,     // [d_model, d_ff] — gate projection
    pub ffn_w2: Tensor,     // [d_ff, d_model] — down projection
    pub ffn_w3: Tensor,     // [d_model, d_ff] — up projection
    pub ln1_weight: Tensor, // [d_model] — attention norm
    pub ln2_weight: Tensor, // [d_model] — ffn norm
    pub norm_eps: f32,
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

        Self {
            attn: MultiHeadAttention::new(ctx, d, config.n_heads),
            ffn_w1: Tensor::randn(ctx, vec![d, ff], ff_std),
            ffn_w2: Tensor::randn(ctx, vec![ff, d], down_std),
            ffn_w3: Tensor::randn(ctx, vec![d, ff], ff_std),
            ln1_weight: Tensor::ones(ctx, vec![d]),
            ln2_weight: Tensor::ones(ctx, vec![d]),
            norm_eps: config.norm_eps,
        }
    }

    /// Forward pass: pre-norm transformer block.
    /// x: [batch, seq_len, d_model] → [batch, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, kv_cache: Option<&mut KvCache>) -> Tensor {
        let batch = x.shape[0];
        let seq_len = x.shape[1];
        let d = x.shape[2];

        // Attention sub-layer with residual
        let normed = x.rms_norm(&self.ln1_weight, self.norm_eps);
        let attn_out = self.attn.forward(&normed, kv_cache);
        let h = x.add(&attn_out);

        // SwiGLU FFN with residual
        let normed2 = h.rms_norm(&self.ln2_weight, self.norm_eps);
        let ffn_out = self.swiglu_ffn(&normed2, batch, seq_len, d);
        h.add(&ffn_out)
    }

    /// SwiGLU feed-forward: output = (SiLU(x @ W1) * (x @ W3)) @ W2
    fn swiglu_ffn(&self, x: &Tensor, batch: usize, seq_len: usize, d: usize) -> Tensor {
        let x_flat = x.reshape(vec![batch * seq_len, d]);

        // Gate and up projections
        let gate = x_flat.matmul(&self.ffn_w1); // [bs, d_ff]
        let up = x_flat.matmul(&self.ffn_w3);   // [bs, d_ff]

        // SwiGLU activation (fused: silu(gate) * up in one kernel)
        let hidden = gate.silu_gate(&up);

        // Down projection
        let out = hidden.matmul(&self.ffn_w2); // [bs, d]
        out.reshape(vec![batch, seq_len, d])
    }

    /// Forward pass with gradient checkpointing.
    /// Runs the normal forward inside `checkpoint_forward()`, records a single
    /// `Op::Checkpoint` entry on the main tape, and registers a recompute closure
    /// that can re-run this block's forward during backward.
    pub fn forward_checkpointed(self: &Arc<Self>, x: &Tensor, layer_idx: usize) -> Tensor {
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
        params.extend_from_slice(&[
            &self.ffn_w1,
            &self.ffn_w2,
            &self.ffn_w3,
            &self.ln1_weight,
            &self.ln2_weight,
        ]);
        params
    }
}

/// The full transformer model.
pub struct Transformer {
    pub config: ModelConfig,
    pub embedding: Tensor,           // [vocab_size, d_model]
    pub blocks: Vec<Arc<TransformerBlock>>,
    pub ln_final_weight: Tensor,     // [d_model]
    // lm_head shares weights with embedding (weight tying)
    ctx: Arc<MetalContext>,
}

impl Transformer {
    pub fn new(ctx: &Arc<MetalContext>, config: ModelConfig) -> Self {
        let d = config.d_model;
        let v = config.vocab_size as usize;

        // Embedding with small init
        let embed_std = (1.0 / d as f32).sqrt();
        let embedding = Tensor::randn(ctx, vec![v, d], embed_std);

        let blocks: Vec<Arc<TransformerBlock>> = (0..config.n_layers)
            .map(|i| Arc::new(TransformerBlock::new(ctx, &config, i)))
            .collect();

        let ln_final_weight = Tensor::ones(ctx, vec![d]);

        eprintln!(
            "Model initialized: {} layers, {}M parameters",
            config.n_layers,
            config.param_count() as f32 / 1e6
        );

        Self {
            config,
            embedding,
            blocks,
            ln_final_weight,
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

        // Embedding lookup
        let tokens_buf = self.ctx.buffer_from_u32_slice(tokens);
        let embed_out_buf = self.ctx.alloc_buffer(n_tokens * d * 4);
        compute::gpu_embedding_lookup(
            &self.ctx,
            &tokens_buf,
            &self.embedding.buffer,
            &embed_out_buf,
            n_tokens as u32,
            d as u32,
        );

        // Record embedding on tape
        let tokens_id = autograd::next_id(); // separate ID for the non-differentiable tokens tensor
        let embed_out_id = autograd::next_id();
        if autograd::is_recording() {
            autograd::record(TapeEntry {
                op: Op::Embedding,
                inputs: vec![tokens_id, self.embedding.id],
                output: embed_out_id,
                input_buffers: vec![tokens_buf, self.embedding.buffer.clone()],
                output_buffer: embed_out_buf.clone(),
                shapes: vec![vec![n_tokens], vec![v, d], vec![n_tokens, d]],
                cached: None,
            });
        }

        let mut h = Tensor {
            id: embed_out_id,
            buffer: embed_out_buf,
            shape: vec![batch, seq_len, d],
            requires_grad: true,
            ctx: Arc::clone(&self.ctx),
        };

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
                    for block in &self.blocks {
                        h = block.forward(&h, None);
                    }
                }
            }
        }

        // Final layer norm
        let h = h.rms_norm(&self.ln_final_weight, self.config.norm_eps);

        // LM head (weight-tied with embedding): logits = h @ embedding^T
        // h: [batch*seq, d_model], embedding: [vocab, d_model]
        // logits: [batch*seq, vocab]
        let h_flat = h.reshape(vec![n_tokens, d]);
        h_flat.matmul_trans_b(&self.embedding.detach())
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.embedding, &self.ln_final_weight];
        for block in &self.blocks {
            params.extend(block.parameters());
        }
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
        let batch_heads = batch * self.config.n_heads;
        let head_dim = self.config.d_model / self.config.n_heads;
        let max_seq = self.config.max_seq_len;
        (0..self.config.n_layers)
            .map(|_| KvCache::with_capacity(&self.ctx, batch_heads, max_seq, head_dim))
            .collect()
    }
}
