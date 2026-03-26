use crate::autograd;
use crate::metal::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
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

    /// Load optimizer state from checkpoint data. Sets m, v buffers and step counter.
    pub fn load_state(&mut self, states: &[(Vec<f32>, Vec<f32>)], opt_step: u32) {
        assert_eq!(states.len(), self.params.len(), "Optimizer state count mismatch");
        self.step = opt_step;
        for (ps, (m_data, v_data)) in self.params.iter().zip(states.iter()) {
            assert_eq!(m_data.len(), ps.size, "m state size mismatch");
            assert_eq!(v_data.len(), ps.size, "v state size mismatch");
            unsafe {
                let m_ptr = ps.m.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(m_data.as_ptr(), m_ptr, ps.size);
                let v_ptr = ps.v.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, ps.size);
            }
        }
    }

    /// Zero all gradients and clear recompute closures.
    pub fn zero_grad(&self) {
        autograd::clear_tape();
        autograd::clear_recompute_registry();
    }
}

/// Cosine warmup learning rate scheduler with optional warm restarts.
/// When restart_period > 0, the cosine cycle repeats every restart_period steps
/// (after warmup), resetting LR to max_lr. This is SGDR (Loshchilov & Hutter, 2017).
pub struct CosineWarmupScheduler {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: u32,
    pub total_steps: u32,
    pub restart_period: u32, // 0 = no restarts (standard cosine decay)
}

impl CosineWarmupScheduler {
    pub fn new(max_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        Self {
            max_lr,
            min_lr: max_lr * 0.1,
            warmup_steps,
            total_steps,
            restart_period: 0,
        }
    }

    pub fn with_restarts(max_lr: f32, warmup_steps: u32, total_steps: u32, restart_period: u32) -> Self {
        Self {
            max_lr,
            min_lr: max_lr * 0.1,
            warmup_steps,
            total_steps,
            restart_period,
        }
    }

    pub fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            if self.warmup_steps == 0 {
                return self.max_lr;
            }
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else if self.total_steps <= self.warmup_steps {
            self.max_lr
        } else {
            let decay_step = step - self.warmup_steps;
            let decay_total = self.total_steps - self.warmup_steps;

            let progress = if self.restart_period > 0 {
                // Warm restarts: progress resets every restart_period steps
                (decay_step % self.restart_period) as f32 / self.restart_period as f32
            } else {
                (decay_step as f32 / decay_total as f32).min(1.0)
            };

            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}
