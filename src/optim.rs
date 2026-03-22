use crate::autograd;
use crate::metal::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
use objc2::rc::Retained;
use std::sync::Arc;

/// AdamW optimizer with decoupled weight decay.
pub struct AdamW {
    pub params: Vec<ParamState>,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
    ctx: Arc<MetalContext>,
}

pub struct ParamState {
    pub tensor_id: usize,
    pub buffer: Retained<GpuBuffer>,
    pub size: usize,
    pub m: Retained<GpuBuffer>, // first moment
    pub v: Retained<GpuBuffer>, // second moment
}

impl AdamW {
    pub fn new(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32) -> Self {
        let param_states: Vec<ParamState> = params
            .iter()
            .map(|t| {
                let size = t.numel();
                let m = ctx.alloc_buffer(size * 4);
                let v = ctx.alloc_buffer(size * 4);
                compute::gpu_fill(ctx, &m, size as u32, 0.0);
                compute::gpu_fill(ctx, &v, size as u32, 0.0);
                ParamState {
                    tensor_id: t.id,
                    buffer: t.buffer.clone(),
                    size,
                    m,
                    v,
                }
            })
            .collect();

        Self {
            params: param_states,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
            step: 0,
            ctx: Arc::clone(ctx),
        }
    }

    /// Perform one optimizer step with the given learning rate.
    pub fn step(&mut self, lr: f32) {
        self.step += 1;

        for ps in &self.params {
            let grad = autograd::get_grad(ps.tensor_id);
            let grad = match grad {
                Some(g) => g,
                None => continue,
            };

            compute::gpu_adamw_update(
                &self.ctx,
                &ps.buffer,
                &grad,
                &ps.m,
                &ps.v,
                ps.size as u32,
                &compute::AdamWHyperparams {
                    lr,
                    beta1: self.beta1,
                    beta2: self.beta2,
                    eps: self.eps,
                    weight_decay: self.weight_decay,
                    step: self.step,
                },
            );
        }
    }

    /// Zero all gradients and clear recompute closures.
    pub fn zero_grad(&self) {
        autograd::clear_tape();
        autograd::clear_recompute_registry();
    }
}

/// Cosine warmup learning rate scheduler.
pub struct CosineWarmupScheduler {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: u32,
    pub total_steps: u32,
}

impl CosineWarmupScheduler {
    pub fn new(max_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        Self {
            max_lr,
            min_lr: max_lr * 0.1,
            warmup_steps,
            total_steps,
        }
    }

    pub fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            let progress = progress.min(1.0);
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}
