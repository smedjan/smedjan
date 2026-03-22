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
        requires_grad: false,
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
