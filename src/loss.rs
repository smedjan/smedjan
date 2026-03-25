use crate::autograd::{self, Op, TapeEntry};
use crate::metal::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
use objc2::rc::Retained;
use std::sync::Arc;

/// Cross-entropy loss for next-token prediction.
/// logits: [batch * seq_len, vocab_size], targets: [batch * seq_len] (u32 token IDs)
/// Returns a scalar loss tensor and the gradient buffer.
pub fn cross_entropy_loss(
    ctx: &Arc<MetalContext>,
    logits: &Tensor,
    targets: &[u32],
) -> (Tensor, Retained<GpuBuffer>) {
    let logits_shape = &logits.shape;
    assert_eq!(logits_shape.len(), 2, "logits must be [batch, vocab]");
    let batch = logits_shape[0];
    let vocab = logits_shape[1];
    assert_eq!(targets.len(), batch, "targets length must match batch size");

    let targets_buf = ctx.buffer_from_u32_slice(targets);
    let losses_buf = ctx.alloc_buffer(batch * 4);
    let grad_logits_buf = ctx.alloc_buffer(batch * vocab * 4);

    compute::gpu_cross_entropy(
        ctx,
        &logits.buffer,
        &targets_buf,
        &losses_buf,
        &grad_logits_buf,
        batch as u32,
        vocab as u32,
    );

    // Reduce per-sample losses to scalar mean
    let scalar_buf = ctx.alloc_buffer(4);
    compute::gpu_reduce_sum(ctx, &losses_buf, &scalar_buf, batch as u32);
    compute::gpu_scale(ctx, &scalar_buf, 1, 1.0 / batch as f32);

    let loss_id = autograd::next_id();
    let loss = Tensor {
        id: loss_id,
        buffer: scalar_buf,
        shape: vec![1],
        requires_grad: true,
        ctx: Arc::clone(ctx),
    };

    if autograd::is_recording() {
        autograd::record(TapeEntry {
            op: Op::CrossEntropy,
            inputs: vec![logits.id],
            output: loss_id,
            input_buffers: vec![logits.buffer.clone()],
            output_buffer: loss.buffer.clone(),
            shapes: vec![logits_shape.clone(), vec![1]],
            cached: Some(grad_logits_buf.clone()),
        });
    }

    (loss, grad_logits_buf)
}

/// Knowledge distillation loss combining KL divergence and cross-entropy.
///
/// final_loss = alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
///            + (1 - alpha) * cross_entropy(student, targets)
///
/// Returns combined scalar loss + combined gradient buffer for student_logits.
/// Teacher logits are treated as detached (no gradient flows through them).
pub fn distillation_loss(
    ctx: &Arc<MetalContext>,
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f32,
    alpha: f32,
    targets: &[u32],
) -> (Tensor, Retained<GpuBuffer>) {
    let shape = &student_logits.shape;
    assert_eq!(shape.len(), 2, "student_logits must be [batch, vocab]");
    let batch = shape[0];
    let vocab = shape[1];
    assert_eq!(teacher_logits.shape, *shape, "teacher and student logit shapes must match");
    assert_eq!(targets.len(), batch, "targets length must match batch size");
    assert!(temperature > 0.0, "temperature must be positive");
    assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1]");

    // --- KL divergence component ---
    let kl_losses_buf = ctx.alloc_buffer(batch * 4);
    let kl_grad_buf = ctx.alloc_buffer(batch * vocab * 4);

    compute::gpu_kl_divergence(
        ctx,
        &teacher_logits.buffer,
        &student_logits.buffer,
        &kl_losses_buf,
        &kl_grad_buf,
        batch as u32,
        vocab as u32,
        temperature,
    );

    // KL scalar mean
    let kl_scalar_buf = ctx.alloc_buffer(4);
    compute::gpu_reduce_sum(ctx, &kl_losses_buf, &kl_scalar_buf, batch as u32);
    compute::gpu_scale(ctx, &kl_scalar_buf, 1, 1.0 / batch as f32);

    // Scale KL gradient by alpha * T^2.
    // The shader outputs raw d_KL/d_z = (1/T)(q - p) / batch. Multiplying by alpha * T^2
    // produces alpha * T * (q - p) / batch = d(alpha * T^2 * KL) / d_z / batch.
    let t_sq = temperature * temperature;
    compute::gpu_scale(ctx, &kl_grad_buf, (batch * vocab) as u32, alpha * t_sq);

    // --- Cross-entropy component ---
    let targets_buf = ctx.buffer_from_u32_slice(targets);
    let ce_losses_buf = ctx.alloc_buffer(batch * 4);
    let ce_grad_buf = ctx.alloc_buffer(batch * vocab * 4);

    compute::gpu_cross_entropy(
        ctx,
        &student_logits.buffer,
        &targets_buf,
        &ce_losses_buf,
        &ce_grad_buf,
        batch as u32,
        vocab as u32,
    );

    // CE scalar mean
    let ce_scalar_buf = ctx.alloc_buffer(4);
    compute::gpu_reduce_sum(ctx, &ce_losses_buf, &ce_scalar_buf, batch as u32);
    compute::gpu_scale(ctx, &ce_scalar_buf, 1, 1.0 / batch as f32);

    // Scale CE gradient by (1 - alpha)
    compute::gpu_scale(ctx, &ce_grad_buf, (batch * vocab) as u32, 1.0 - alpha);

    // --- Combine: combined_grad = kl_grad + ce_grad ---
    // gpu_add_inplace: a += b, so kl_grad_buf += ce_grad_buf
    compute::gpu_add_inplace(ctx, &kl_grad_buf, &ce_grad_buf, (batch * vocab) as u32);
    let combined_grad = kl_grad_buf;

    // Combined scalar loss = alpha * T^2 * kl_mean + (1 - alpha) * ce_mean
    // Scale each scalar then add
    compute::gpu_scale(ctx, &kl_scalar_buf, 1, alpha * t_sq);
    compute::gpu_scale(ctx, &ce_scalar_buf, 1, 1.0 - alpha);
    compute::gpu_add_inplace(ctx, &kl_scalar_buf, &ce_scalar_buf, 1);
    let combined_loss_buf = kl_scalar_buf;

    let loss_id = autograd::next_id();
    let loss = Tensor {
        id: loss_id,
        buffer: combined_loss_buf,
        shape: vec![1],
        requires_grad: true,
        ctx: Arc::clone(ctx),
    };

    if autograd::is_recording() {
        autograd::record(TapeEntry {
            op: Op::CrossEntropy, // Reuse CrossEntropy op — gradient is precomputed
            inputs: vec![student_logits.id],
            output: loss_id,
            input_buffers: vec![student_logits.buffer.clone()],
            output_buffer: loss.buffer.clone(),
            shapes: vec![shape.clone(), vec![1]],
            cached: Some(combined_grad.clone()),
        });
    }

    (loss, combined_grad)
}
