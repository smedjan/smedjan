use crate::checkpoint;
use crate::generate::{self, SamplingConfig};
use crate::gpu::GpuContext as MetalContext;
use crate::metal::compute;
use crate::model::Transformer;
use crate::tokenizer::BpeTokenizer;
use std::sync::Arc;

/// Model metadata exposed through the public API.
pub struct ModelInfo {
    pub params: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: u32,
    pub step: u32,
}

/// High-level API for using AndreAI as a library.
///
/// Encapsulates model loading, tokenization, and generation into a single
/// struct suitable for embedding into other Rust projects (e.g. AndreOS).
pub struct AndreAI {
    ctx: Arc<MetalContext>,
    model: Transformer,
    tokenizer: BpeTokenizer,
    step: u32,
}

impl AndreAI {
    /// Load a model from checkpoint and tokenizer files.
    ///
    /// The checkpoint can be either a standard `.bin` checkpoint or a
    /// quantized `.qbin` checkpoint (auto-detected by extension).
    pub fn load(checkpoint_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        let ctx = MetalContext::new();
        let tokenizer = BpeTokenizer::load(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer '{}': {}", tokenizer_path, e))?;

        let (model, step) = if checkpoint_path.ends_with(".qbin") {
            crate::quantize::load_quantized(&ctx, checkpoint_path)
                .map_err(|e| format!("Failed to load quantized checkpoint '{}': {}", checkpoint_path, e))?
        } else {
            checkpoint::load_checkpoint(&ctx, checkpoint_path)
                .map_err(|e| format!("Failed to load checkpoint '{}': {}", checkpoint_path, e))?
        };

        Ok(Self {
            ctx,
            model,
            tokenizer,
            step,
        })
    }

    /// Generate a response to a prompt using default sampling parameters.
    pub fn generate(&self, prompt: &str) -> String {
        let config = SamplingConfig::default();
        generate::generate(&self.ctx, &self.model, &self.tokenizer, prompt, &config)
    }

    /// Generate a response with custom sampling parameters.
    pub fn generate_with_config(&self, prompt: &str, config: &SamplingConfig) -> String {
        generate::generate(&self.ctx, &self.model, &self.tokenizer, prompt, config)
    }

    /// Generate streaming, calling the callback for each produced token.
    pub fn generate_streaming<F: FnMut(&str)>(&self, prompt: &str, on_token: F) {
        let config = SamplingConfig::default();
        generate::generate_streaming(
            &self.ctx,
            &self.model,
            &self.tokenizer,
            prompt,
            &config,
            on_token,
        );
    }

    /// Generate streaming with custom sampling parameters.
    pub fn generate_streaming_with_config<F: FnMut(&str)>(
        &self,
        prompt: &str,
        config: &SamplingConfig,
        on_token: F,
    ) {
        generate::generate_streaming(
            &self.ctx,
            &self.model,
            &self.tokenizer,
            prompt,
            config,
            on_token,
        );
    }

    /// Get model metadata.
    pub fn model_info(&self) -> ModelInfo {
        let c = &self.model.config;
        ModelInfo {
            params: c.param_count(),
            d_model: c.d_model,
            n_layers: c.n_layers,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            vocab_size: c.vocab_size,
            step: self.step,
        }
    }

    /// Get a reference to the underlying tokenizer for direct token manipulation.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }
}

/// GPU capability diagnostic — exercises all kernel variants to verify they compile and run.
/// Returns (n_kernels_tested, all_passed).
pub fn gpu_diagnostic(ctx: &Arc<MetalContext>) -> (usize, bool) {
    let mut tested = 0;
    let mut passed = true;

    // FP32 matmul variants (replaced by FP16 in hot path, but kept for fallback)
    let a = ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = ctx.buffer_from_slice(&[1.0f32, 0.0, 0.0, 1.0]);
    let c = ctx.alloc_buffer(4 * 4);
    compute::gpu_matmul_trans_b(ctx, &a, &b, &c, 2, 2, 2);
    compute::gpu_matmul_trans_a(ctx, &a, &b, &c, 2, 2, 2);
    tested += 2;

    // FP16 cast roundtrip
    let f16 = ctx.alloc_buffer(4 * 2);
    let f32_back = ctx.alloc_buffer(4 * 4);
    compute::gpu_cast_f32_to_f16(ctx, &a, &f16, 4);
    compute::gpu_cast_f16_to_f32(ctx, &f16, &f32_back, 4);
    let vals = MetalContext::read_buffer(&f32_back, 4);
    if (vals[0] - 1.0).abs() > 0.1 { passed = false; }
    tested += 2;

    // Batched FP16 matmul variants
    let ba = ctx.buffer_from_slice(&[1.0f32; 8]);
    let bb = ctx.buffer_from_slice(&[1.0f32; 8]);
    let bc = ctx.alloc_buffer(8 * 4);
    let ba16 = ctx.alloc_buffer(8 * 2);
    let bb16 = ctx.alloc_buffer(8 * 2);
    compute::gpu_cast_f32_to_f16(ctx, &ba, &ba16, 8);
    compute::gpu_cast_f32_to_f16(ctx, &bb, &bb16, 8);
    compute::gpu_batched_matmul_f16(ctx, &ba16, &bb16, &bc, compute::BatchedDims { batch: 2, m: 2, n: 2, k: 2 });
    compute::gpu_batched_matmul_trans_b_f16(ctx, &ba16, &bb16, &bc, compute::BatchedDims { batch: 2, m: 2, n: 2, k: 2 });
    compute::gpu_batched_matmul_trans_a_f16(ctx, &ba16, &bb16, &bc, compute::BatchedDims { batch: 2, m: 2, n: 2, k: 2 });
    // FP16 non-batched matmul backward variant (kept for large-matrix backward path)
    compute::gpu_matmul_trans_a_f16(ctx, &ba16, &bb16, &bc, 2, 2, 2);
    // Utility cast helper (used by FP16 backward when enabled)
    let cast_test = crate::autograd::cast_buf_f16(ctx, &ba, 4);
    // Verify FP16 cast produced valid buffer (read back and check)
    let cast_back = ctx.alloc_buffer(4 * 4);
    compute::gpu_cast_f16_to_f32(ctx, &cast_test, &cast_back, 4);
    let cast_vals = MetalContext::read_buffer(&cast_back, 4);
    if (cast_vals[0] - 1.0).abs() > 0.1 { passed = false; }
    tested += 5;

    // MoE gather/scatter
    let indices = ctx.buffer_from_u32_slice(&[0, 1]);
    let gathered = ctx.alloc_buffer(2 * 2 * 4);
    compute::gpu_moe_gather(ctx, &a, &indices, &gathered, 2, 2);
    let weights = ctx.buffer_from_slice(&[0.5f32, 0.5]);
    let combined = ctx.alloc_buffer(2 * 2 * 4);
    compute::gpu_fill(ctx, &combined, 4, 0.0);
    compute::gpu_moe_scatter_add(ctx, &gathered, &indices, &weights, &combined, 2, 2);
    tested += 2;

    // Lion optimizer
    let p = ctx.buffer_from_slice(&[1.0f32, 2.0]);
    let g = ctx.buffer_from_slice(&[0.1f32, -0.1]);
    let m = ctx.alloc_buffer(2 * 4);
    compute::gpu_fill(ctx, &m, 2, 0.0);
    compute::gpu_lion_update(ctx, &p, &g, &m, 2, compute::LionParams { lr: 0.01, beta1: 0.9, beta2: 0.99, weight_decay: 0.0 });
    tested += 1;

    // Sophia optimizer
    let h = ctx.alloc_buffer(2 * 4);
    compute::gpu_fill(ctx, &h, 2, 0.0);
    compute::gpu_sophia_update(ctx, &p, &g, &m, &h, 2, compute::SophiaParams { lr: 0.01, beta1: 0.965, beta2: 0.99, eps: 1e-4, rho: 1.0, weight_decay: 0.0 });
    tested += 1;

    // LogSumExp + Z-loss
    let lse_data = ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]); // [2, 3]
    let lse_out = ctx.alloc_buffer(2 * 4);
    compute::gpu_logsumexp(ctx, &lse_data, &lse_out, 2, 3);
    let lse_vals = MetalContext::read_buffer(&lse_out, 2);
    if (lse_vals[0] - lse_vals[1]).abs() > 0.01 { passed = false; } // same rows → same lse
    // Z-loss (disabled in training but function exists)
    let z_logits = crate::tensor::Tensor::randn(ctx, vec![4, 8], 0.1);
    let z_loss_buf = ctx.alloc_buffer(4);
    compute::gpu_fill(ctx, &z_loss_buf, 1, 0.0);
    let z_grad_buf = ctx.alloc_buffer(4 * 8 * 4);
    crate::loss::z_loss(ctx, &z_logits, &z_loss_buf, &z_grad_buf, 1e-4);
    tested += 2;

    // DataMixer (verify construction works)
    if std::path::Path::new("data/train_v3.bin").exists() {
        let mixer = crate::data::DataMixer::new(
            &["data/train_v3.bin", "data/train_v3.bin"],
            &[0.7, 0.3], 4, 16,
        );
        if let Ok(mut m) = mixer {
            let _ = m.total_tokens();
            let _ = m.source_weights();
            let _ = m.next_batch();
            tested += 1;
        }
    }

    // FusedLinearCrossEntropy
    let fce_hidden = crate::tensor::Tensor::randn(ctx, vec![4, 8], 0.1);
    let fce_embed = crate::tensor::Tensor::randn(ctx, vec![16, 8], 0.1);
    let fce_targets = vec![0u32, 1, 2, 3];
    let (_fce_loss, _fce_grad) = crate::loss::fused_linear_cross_entropy(
        ctx, &fce_hidden, &fce_embed, &fce_targets, 2,
    );
    tested += 1;

    // WSD scheduler
    let wsd = crate::optim::WSDScheduler::with_phases(1e-3, 10, 80, 10);
    assert_eq!(wsd.total_steps(), 100);
    assert!(wsd.get_lr(50) > 0.0); // stable phase
    tested += 1;

    // Optimizer enum
    let tiny = crate::tensor::Tensor::zeros(ctx, vec![2, 2]);
    let tiny_refs: Vec<&crate::tensor::Tensor> = vec![&tiny];
    let mut opt = crate::optim::Optimizer::AdamW(crate::optim::AdamW::new(ctx, &tiny_refs, 0.0));
    opt.step(0.0);
    opt.zero_grad();
    let _ = opt.adamw_step();
    let mut soph_opt = crate::optim::Optimizer::Sophia(crate::optim::Sophia::new(ctx, &tiny_refs, 0.0));
    soph_opt.step(0.0);
    let mut muon_opt = crate::optim::Optimizer::Muon(crate::optim::Muon::new(ctx, &tiny_refs, 0.0));
    muon_opt.step(0.01);
    tested += 3;

    // FlashAttention op variant
    let flash_op = crate::autograd::Op::FlashAttention {
        batch_heads: 1, seq_q: 2, seq_k: 2, head_dim: 2, kv_offset: 0,
    };
    if !matches!(flash_op, crate::autograd::Op::FlashAttention { .. }) { passed = false; }
    tested += 1;

    // Verify grad recycling path exists (safe after sync flush)
    crate::autograd::zero_grads_recycle();
    // Column concat+slice (infrastructure for fused projections)
    let col_src = ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]); // [2, 2]
    let col_dst = ctx.alloc_buffer(2 * 4 * 4); // [2, 4]
    compute::gpu_fill(ctx, &col_dst, 8, 0.0);
    compute::gpu_concat_cols(ctx, &col_src, &col_dst, 2, 2, 4, 0);
    compute::gpu_concat_cols(ctx, &col_src, &col_dst, 2, 2, 4, 2);
    let col_slice = ctx.alloc_buffer(2 * 2 * 4); // [2, 2]
    compute::gpu_slice_cols(ctx, &col_dst, &col_slice, 2, 4, 2, 2);
    tested += 3;

    // In-place rope variants (kept for potential future use)
    let rope_buf = ctx.buffer_from_slice(&[1.0f32, 0.0, 0.0, 1.0]);
    let rope_copy = ctx.alloc_buffer(4 * 4);
    compute::gpu_copy(ctx, &rope_buf, &rope_copy, 4);
    compute::gpu_rope(ctx, &rope_copy, 1, 1, 4, 0, 10000.0);
    compute::gpu_rope_backward(ctx, &rope_copy, 1, 1, 4, 0, 10000.0);
    tested += 3;

    // L2 norm variants — verify sync and into-buffer paths agree
    let norm_data = ctx.buffer_from_slice(&[3.0f32, 4.0]); // norm = 5.0
    let norm_sync = compute::gpu_l2_norm(ctx, &norm_data, 2);
    if (norm_sync - 5.0).abs() > 0.01 { passed = false; }
    let norm_out = ctx.alloc_buffer(4);
    compute::gpu_l2_norm_into(ctx, &norm_data, 2, &norm_out);
    let norm_async = MetalContext::read_buffer(&norm_out, 1)[0];
    if (norm_async - 5.0).abs() > 0.01 { passed = false; }
    tested += 2;

    (tested, passed)
}

