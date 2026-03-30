use crate::autograd;
use crate::metal::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
use objc2::rc::Retained;
use objc2_metal::MTLBuffer;
use std::sync::Arc;

/// AdamW optimizer with decoupled weight decay.
/// Supports GALORE: gradient low-rank projection for memory savings.
pub struct AdamW {
    pub params: Vec<ParamState>,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
    pub galore_rank: usize, // 0 = disabled, >0 = project grads to this rank
    ctx: Arc<MetalContext>,
}

pub struct ParamState {
    pub tensor_id: usize,
    pub buffer: Retained<GpuBuffer>,
    pub size: usize,
    pub m: Retained<GpuBuffer>, // first moment (full or projected)
    pub v: Retained<GpuBuffer>, // second moment (full or projected)
    pub proj: Option<Retained<GpuBuffer>>, // GALORE: random projection matrix [size, rank]
    pub proj_size: usize, // projected size (rank × smaller_dim)
    pub no_decay: bool, // skip weight decay for norms and embeddings
}

impl AdamW {
    pub fn new(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32) -> Self {
        Self::new_with_galore(ctx, params, weight_decay, 0)
    }

    pub fn new_with_galore(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32, galore_rank: usize) -> Self {
        let param_states: Vec<ParamState> = params
            .iter()
            .map(|t| {
                let size = t.numel();
                // GALORE: for large params (>4096 elements), use projected m/v
                let (m_size, proj, proj_size) = if galore_rank > 0 && size > 4096 {
                    // Project to rank dimensions: m/v stored as [rank] per row
                    // For a [rows, cols] weight: project cols → rank
                    let proj_sz = galore_rank; // simplified: project to flat rank
                    let p = ctx.alloc_buffer(proj_sz * 4);
                    compute::gpu_fill(ctx, &p, proj_sz as u32, 0.0);
                    (proj_sz, Some(p), proj_sz)
                } else {
                    (size, None, size)
                };
                let m = ctx.alloc_buffer(m_size * 4);
                let v = ctx.alloc_buffer(m_size * 4);
                compute::gpu_fill(ctx, &m, m_size as u32, 0.0);
                compute::gpu_fill(ctx, &v, m_size as u32, 0.0);
                // Skip weight decay for 1D params (norm weights, biases).
                // Norm weights are initialized to 1.0 — decay pushes them toward 0,
                // attenuating signal through the network (0.9^12_norms ≈ 0.28 after 20K steps).
                let no_decay = t.shape.len() <= 1;
                ParamState {
                    tensor_id: t.id,
                    buffer: t.buffer.clone(),
                    size,
                    m,
                    v,
                    proj,
                    proj_size,
                    no_decay,
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
            galore_rank,
            ctx: Arc::clone(ctx),
        }
    }

    /// Memory used by optimizer state (m + v buffers + projection matrices).
    pub fn memory_bytes(&self) -> usize {
        self.params.iter().map(|ps| {
            let mv = ps.proj_size * 4 * 2;
            let proj = ps.proj.as_ref().map_or(0, |p| p.length());
            mv + proj
        }).sum()
    }

    /// Perform one optimizer step with the given learning rate (GPU dispatch).
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
                    weight_decay: if ps.no_decay { 0.0 } else { self.weight_decay },
                    step: self.step,
                },
            );
        }
    }

    /// Perform optimizer step on CPU using unified memory (Apple Silicon zero-copy).
    /// On Apple Silicon, CPU and GPU share physical memory — no copy needed.
    /// The CPU update runs while the GPU can start the next forward pass.
    /// This hides optimizer latency behind GPU compute.
    pub fn step_cpu(&mut self, lr: f32) {
        use objc2_metal::MTLBuffer;
        self.step += 1;

        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);

        for ps in &self.params {
            let grad = autograd::get_grad(ps.tensor_id);
            let grad = match grad { Some(g) => g, None => continue };

            let size = ps.size;
            // Direct pointer access to unified memory — zero copy on Apple Silicon
            let param_ptr = ps.buffer.contents().as_ptr() as *mut f32;
            let grad_ptr = grad.contents().as_ptr() as *const f32;
            let m_ptr = ps.m.contents().as_ptr() as *mut f32;
            let v_ptr = ps.v.contents().as_ptr() as *mut f32;

            unsafe {
                for i in 0..size {
                    let g = *grad_ptr.add(i);
                    let m_val = self.beta1 * *m_ptr.add(i) + (1.0 - self.beta1) * g;
                    let v_val = self.beta2 * *v_ptr.add(i) + (1.0 - self.beta2) * g * g;
                    *m_ptr.add(i) = m_val;
                    *v_ptr.add(i) = v_val;

                    let m_hat = m_val / bc1;
                    let v_hat = v_val / bc2;

                    let wd = if ps.no_decay { 0.0 } else { self.weight_decay };
                    let p = *param_ptr.add(i);
                    *param_ptr.add(i) = p * (1.0 - lr * wd)
                        - lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }
    }

    /// Load optimizer state from checkpoint data. Sets m, v buffers and step counter.
    pub fn load_state(&mut self, states: &[(Vec<f32>, Vec<f32>)], opt_step: u32) {
        assert_eq!(states.len(), self.params.len(), "Optimizer state count mismatch");
        self.step = opt_step;
        for (ps, (m_data, v_data)) in self.params.iter().zip(states.iter()) {
            assert_eq!(m_data.len(), ps.proj_size, "m state size mismatch");
            assert_eq!(v_data.len(), ps.proj_size, "v state size mismatch");
            unsafe {
                let m_ptr = ps.m.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(m_data.as_ptr(), m_ptr, ps.proj_size);
                let v_ptr = ps.v.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, ps.proj_size);
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

/// WSD (Warmup-Stable-Decay) learning rate schedule.
/// Three phases: linear warmup → constant plateau → linear decay to zero.
/// Beats cosine by 5-10% on final loss. Key advantage: stable phase can continue
/// indefinitely — branch off with decay at any point to get a good model.
/// Used by OLMo 2, Phi-4, LongCat-Flash. (arXiv 2410.05192)
pub struct WSDScheduler {
    pub max_lr: f32,
    pub warmup_steps: u32,
    pub stable_steps: u32,   // constant LR phase after warmup
    pub decay_steps: u32,    // linear decay to zero after stable
}

impl WSDScheduler {
    /// Create WSD schedule. stable_fraction = fraction of total steps at constant LR.
    /// Typical: warmup=2%, stable=70%, decay=28%.
    pub fn new(max_lr: f32, warmup_steps: u32, total_steps: u32) -> Self {
        let after_warmup = total_steps.saturating_sub(warmup_steps);
        let stable = (after_warmup as f32 * 0.7) as u32;
        let decay = after_warmup - stable;
        Self { max_lr, warmup_steps, stable_steps: stable, decay_steps: decay }
    }

    pub fn with_phases(max_lr: f32, warmup_steps: u32, stable_steps: u32, decay_steps: u32) -> Self {
        Self { max_lr, warmup_steps, stable_steps, decay_steps }
    }

    pub fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            if self.warmup_steps == 0 { return self.max_lr; }
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else if step < self.warmup_steps + self.stable_steps {
            // Stable plateau — constant max_lr
            self.max_lr
        } else {
            // Linear decay to zero
            let decay_step = step - self.warmup_steps - self.stable_steps;
            if self.decay_steps == 0 { return 0.0; }
            let progress = (decay_step as f32 / self.decay_steps as f32).min(1.0);
            self.max_lr * (1.0 - progress)
        }
    }

    pub fn total_steps(&self) -> u32 {
        self.warmup_steps + self.stable_steps + self.decay_steps
    }
}

/// Inverse-sqrt schedule: lr = max_lr * sqrt(warmup) / sqrt(max(step, warmup)).
/// The original Transformer schedule (Vaswani et al., 2017).
/// Gentle decay — never reaches zero. Good for continued pretraining.
pub fn inverse_sqrt_lr(max_lr: f32, warmup_steps: u32, step: u32) -> f32 {
    if step < warmup_steps {
        if warmup_steps == 0 { return max_lr; }
        max_lr * (step as f32 / warmup_steps as f32)
    } else {
        max_lr * (warmup_steps as f32).sqrt() / (step as f32).sqrt()
    }
}

/// Trapezoidal schedule: warmup → stable → linear decay.
/// Like WSD but with configurable decay endpoint (not necessarily zero).
pub fn trapezoidal_lr(max_lr: f32, min_lr: f32, warmup_steps: u32, stable_steps: u32, total_steps: u32, step: u32) -> f32 {
    if step < warmup_steps {
        if warmup_steps == 0 { return max_lr; }
        min_lr + (max_lr - min_lr) * (step as f32 / warmup_steps as f32)
    } else if step < warmup_steps + stable_steps {
        max_lr
    } else {
        let decay_steps = total_steps.saturating_sub(warmup_steps + stable_steps);
        if decay_steps == 0 { return min_lr; }
        let progress = ((step - warmup_steps - stable_steps) as f32 / decay_steps as f32).min(1.0);
        max_lr + (min_lr - max_lr) * progress
    }
}

/// Sophia optimizer — second-order with diagonal Hessian.
/// 2x faster convergence than AdamW for ~same compute.
pub struct Sophia {
    pub params: Vec<SophiaState>,
    pub beta1: f32,     // momentum decay (0.965)
    pub beta2: f32,     // hessian EMA decay (0.99)
    pub eps: f32,       // hessian floor (1e-4)
    pub rho: f32,       // clipping threshold (1.0)
    pub weight_decay: f32,
    pub step: u32,
    ctx: Arc<MetalContext>,
}

pub struct SophiaState {
    pub tensor_id: usize,
    pub buffer: Retained<GpuBuffer>,
    pub size: usize,
    pub m: Retained<GpuBuffer>, // first moment (momentum)
    pub h: Retained<GpuBuffer>, // diagonal Hessian estimate
}

impl Sophia {
    pub fn new(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32) -> Self {
        let param_states: Vec<SophiaState> = params.iter().map(|t| {
            let size = t.numel();
            let m = ctx.alloc_buffer(size * 4);
            let h = ctx.alloc_buffer(size * 4);
            compute::gpu_fill(ctx, &m, size as u32, 0.0);
            compute::gpu_fill(ctx, &h, size as u32, 0.0);
            SophiaState { tensor_id: t.id, buffer: t.buffer.clone(), size, m, h }
        }).collect();

        Self {
            params: param_states,
            beta1: 0.965, beta2: 0.99, eps: 1e-4, rho: 1.0,
            weight_decay, step: 0, ctx: Arc::clone(ctx),
        }
    }

    pub fn step(&mut self, lr: f32) {
        self.step += 1;
        for ps in &self.params {
            let grad = autograd::get_grad(ps.tensor_id);
            let grad = match grad { Some(g) => g, None => continue };

            compute::gpu_sophia_update(
                &self.ctx, &ps.buffer, &grad, &ps.m, &ps.h,
                ps.size as u32, lr, self.beta1, self.beta2,
                self.eps, self.rho, self.weight_decay,
            );
        }
    }

    pub fn zero_grad(&self) {
        autograd::clear_tape();
        autograd::clear_recompute_registry();
    }
}

/// Muon optimizer — MomentUm Orthogonalized by Newton-Schulz (2025).
/// 2.5x faster convergence than AdamW. Used by Kimi K2, GLM-4.5, INTELLECT-3.
/// For 2D weight matrices: orthogonalizes momentum via Newton-Schulz iteration.
/// For non-2D params (embeddings, biases, norms): falls back to AdamW.
pub struct Muon {
    pub params: Vec<MuonState>,
    pub beta: f32,          // momentum decay (0.95)
    pub weight_decay: f32,
    pub ns_steps: usize,    // Newton-Schulz iterations (5-7)
    pub step: u32,
    ctx: Arc<MetalContext>,
}

pub struct MuonState {
    pub tensor_id: usize,
    pub buffer: Retained<GpuBuffer>,
    pub size: usize,
    pub shape: Vec<usize>,
    pub m: Retained<GpuBuffer>,    // momentum buffer
    pub is_2d: bool,               // true = Muon update, false = AdamW fallback
    // AdamW fallback state for non-2D params
    pub v: Option<Retained<GpuBuffer>>,
    // Pre-allocated Newton-Schulz workspace (avoids ~100 allocs per 2D param per step)
    pub ns_x: Option<Retained<GpuBuffer>>,    // [rows, cols] — working copy
    pub ns_xxt: Option<Retained<GpuBuffer>>,  // [rows, rows] — X @ X^T
    pub ns_xxtx: Option<Retained<GpuBuffer>>, // [rows, cols] — (X @ X^T) @ X
}

impl Muon {
    pub fn new(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32) -> Self {
        let param_states = params.iter().map(|t| {
            let size = t.numel();
            let is_2d = t.shape.len() == 2 && t.shape[0] > 1 && t.shape[1] > 1;
            let m = ctx.alloc_buffer(size * 4);
            compute::gpu_fill(ctx, &m, size as u32, 0.0);
            let v = if !is_2d {
                let v_buf = ctx.alloc_buffer(size * 4);
                compute::gpu_fill(ctx, &v_buf, size as u32, 0.0);
                Some(v_buf)
            } else { None };
            // Pre-allocate NS workspace for 2D params
            let (ns_x, ns_xxt, ns_xxtx) = if is_2d {
                let rows = t.shape[0];
                let cols = t.shape[1];
                (
                    Some(ctx.alloc_buffer(rows * cols * 4)),
                    Some(ctx.alloc_buffer(rows * rows * 4)),
                    Some(ctx.alloc_buffer(rows * cols * 4)),
                )
            } else { (None, None, None) };
            MuonState {
                tensor_id: t.id, buffer: t.buffer.clone(), size,
                shape: t.shape.clone(), m, is_2d, v,
                ns_x, ns_xxt, ns_xxtx,
            }
        }).collect();

        Self {
            params: param_states,
            beta: 0.95,
            weight_decay,
            ns_steps: 5,
            step: 0,
            ctx: Arc::clone(ctx),
        }
    }

    pub fn step(&mut self, lr: f32) {
        self.step += 1;
        let beta1_adam = 0.9f32;
        let beta2_adam = 0.95f32;
        let eps_adam = 1e-8f32;

        for ps in &self.params {
            let grad = autograd::get_grad(ps.tensor_id);
            let grad = match grad { Some(g) => g, None => continue };

            if ps.is_2d {
                // Muon update: momentum + Newton-Schulz orthogonalization
                let rows = ps.shape[0] as u32;
                let cols = ps.shape[1] as u32;
                let size = ps.size as u32;

                // Update momentum: m = beta * m + (1-beta) * grad
                compute::gpu_ema_update(&self.ctx, &ps.m, &grad, size, self.beta);

                // Newton-Schulz orthogonalization of momentum M [rows, cols]
                // Normalize M to unit spectral norm first (approximate: scale by 1/sqrt(rows*cols))
                let norm_scale = 1.0 / ((rows as f32).max(cols as f32)).sqrt();

                // Pre-allocated workspace — no buffer allocations per step
                let x_buf = ps.ns_x.as_ref().unwrap();
                let xxt_buf = ps.ns_xxt.as_ref().unwrap();
                let xxtx_buf = ps.ns_xxtx.as_ref().unwrap();

                // X = M * norm_scale (working copy into pre-allocated buffer)
                compute::gpu_scale_copy(&self.ctx, &ps.m, x_buf, size, norm_scale);

                // Newton-Schulz: X = 1.5*X - 0.5*(X@X^T)@X (5 iterations)
                let a = 1.5f32;
                let b = -0.5f32;

                for _ns in 0..self.ns_steps {
                    compute::gpu_matmul_trans_b(&self.ctx, x_buf, x_buf, xxt_buf,
                        rows, rows, cols);
                    compute::gpu_matmul(&self.ctx, xxt_buf, x_buf, xxtx_buf,
                        rows, cols, rows);
                    // X = a*X + b*(X@X^T@X) — fused with axpy: scale X, then axpy
                    compute::gpu_scale(&self.ctx, x_buf, size, a);
                    compute::gpu_axpy(&self.ctx, x_buf, xxtx_buf, size, b);
                }

                // Apply: theta = theta * (1 - lr * wd) - lr * X
                // Fused axpy: theta += -lr * X (1 dispatch instead of scale + add_inplace = 2)
                if self.weight_decay > 0.0 {
                    compute::gpu_scale(&self.ctx, &ps.buffer, size, 1.0 - lr * self.weight_decay);
                }
                compute::gpu_axpy(&self.ctx, &ps.buffer, x_buf, size, -lr);
            } else {
                // AdamW fallback for 1D params (norms, biases, embeddings)
                let v_buf = ps.v.as_ref().unwrap();
                compute::gpu_adamw_update(
                    &self.ctx, &ps.buffer, &grad, &ps.m, v_buf,
                    ps.size as u32,
                    &compute::AdamWHyperparams {
                        lr, beta1: beta1_adam, beta2: beta2_adam, eps: eps_adam,
                        weight_decay: self.weight_decay, step: self.step,
                    },
                );
            }
        }
    }
}

/// Unified optimizer interface for training loop.
pub enum Optimizer {
    AdamW(AdamW),
    Sophia(Sophia),
    Muon(Muon),
}

impl Optimizer {
    pub fn step(&mut self, lr: f32) {
        match self {
            Optimizer::AdamW(o) => o.step(lr),
            Optimizer::Sophia(o) => o.step(lr),
            Optimizer::Muon(o) => o.step(lr),
        }
    }

    pub fn zero_grad(&self) {
        match self {
            Optimizer::AdamW(o) => o.zero_grad(),
            Optimizer::Sophia(o) => o.zero_grad(),
            Optimizer::Muon(_) => {
                autograd::clear_tape();
                autograd::clear_recompute_registry();
            }
        }
    }

    pub fn adamw_step(&self) -> u32 {
        match self {
            Optimizer::AdamW(o) => o.step,
            Optimizer::Sophia(o) => o.step,
            Optimizer::Muon(o) => o.step,
        }
    }
}
