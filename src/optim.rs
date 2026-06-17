use crate::autograd;
use crate::gpu::{compute, GpuBuffer, MetalContext};
use crate::tensor::Tensor;
#[cfg(feature = "metal")]
use objc2_metal::MTLBuffer;
use std::collections::HashSet;
use std::sync::Arc;

/// Read a GPU buffer's first `byte_len` bytes (unified memory — direct, no DMA). Used to serialize
/// optimizer state (f32 or int8) to a resume sidecar.
#[cfg(feature = "metal")]
fn buf_bytes(buf: &GpuBuffer, byte_len: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, byte_len).to_vec() }
}
// CUDA: device memory has no host pointer — dtoh the f32 slice and reinterpret as bytes.
#[cfg(not(feature = "metal"))]
fn buf_bytes(buf: &GpuBuffer, byte_len: usize) -> Vec<u8> {
    let v = crate::gpu::MetalContext::read_buffer(buf, byte_len / 4);
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Write bytes into a GPU buffer (unified memory). Caller guarantees `bytes.len()` ≤ buffer size.
#[cfg(not(feature = "metal"))]
fn buf_set_bytes(buf: &GpuBuffer, bytes: &[u8]) {
    let n = bytes.len() / 4;
    let mut f = vec![0f32; n];
    for i in 0..n {
        f[i] = f32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
    }
    use cudarc::driver::DevicePtr;
    unsafe { cudarc::driver::result::memcpy_htod_sync(*buf.device_ptr(), &f).expect("htod buf_set_bytes"); }
}
#[cfg(feature = "metal")]
fn buf_set_bytes(buf: &GpuBuffer, bytes: &[u8]) {
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents().as_ptr() as *mut u8, bytes.len());
    }
}

/// Tunable AdamW hyperparameters. Defaults are the hardened values established by the instability
/// fix: beta2=0.95 (short second-moment memory) with eps=1e-5 (Llama's value) so the update
/// denominator can't collapse when gradients momentarily shrink. `update_clip` bounds the
/// normalized per-element update at the source (0 = disabled).
#[derive(Clone, Copy, Debug)]
pub struct AdamWHyper {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub update_clip: f32,
}

impl Default for AdamWHyper {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.95, eps: 1e-5, update_clip: 0.0 }
    }
}

/// AdamW optimizer with decoupled weight decay.
pub struct AdamW {
    pub params: Vec<ParamState>,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Per-element ceiling on the normalized update m_hat/(sqrt(v_hat)+eps); 0 = disabled.
    pub update_clip: f32,
    pub weight_decay: f32,
    pub step: u32,
    ctx: Arc<MetalContext>,
}

pub struct ParamState {
    pub tensor_id: usize,
    pub buffer: crate::gpu::Buf,
    pub size: usize,
    pub m: crate::gpu::Buf, // first moment
    pub v: crate::gpu::Buf, // second moment
    pub no_decay: bool, // skip weight decay for norms and embeddings
}

impl AdamW {
    pub fn new(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32) -> Self {
        Self::new_with_config(ctx, params, weight_decay, AdamWHyper::default())
    }

    /// Construct AdamW with explicit hyperparameters (betas/eps/update_clip).
    pub fn new_with_config(
        ctx: &Arc<MetalContext>,
        params: &[&Tensor],
        weight_decay: f32,
        hyper: AdamWHyper,
    ) -> Self {
        let param_states: Vec<ParamState> = params
            .iter()
            .map(|t| {
                let size = t.numel();
                let m = ctx.alloc_buffer(size * 4);
                let v = ctx.alloc_buffer(size * 4);
                compute::gpu_fill(ctx, &m, size as u32, 0.0);
                compute::gpu_fill(ctx, &v, size as u32, 0.0);
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
                    no_decay,
                }
            })
            .collect();

        Self {
            params: param_states,
            // eps floors the update denominator sqrt(v_hat)+eps. With beta2=0.95 (short second-moment
            // memory) the running variance v collapses fast when gradients momentarily shrink; at the
            // old eps=1e-8 the denominator then collapsed too, producing huge/oscillating updates that
            // wandered and (with a degenerate RMSNorm row) diverged. 1e-5 (Llama's value, the default)
            // floors it — negligible when gradients are healthy (sqrt(v_hat) >> eps), stabilising when
            // they aren't. Configurable via AdamWHyper.
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            update_clip: hyper.update_clip,
            weight_decay,
            step: 0,
            ctx: Arc::clone(ctx),
        }
    }

    /// Memory used by optimizer state (m + v buffers). Test-only since the GaLore memory log
    /// was removed; exercised by `adamw_8bit_memory_is_4x_smaller`.
    #[cfg(test)]
    pub fn memory_bytes(&self) -> usize {
        self.params.iter().map(|ps| ps.size * 4 * 2).sum()
    }

    /// Reset m/v states for specific parameters (by tensor ID).
    /// Used after ReLoRA merge: stale momentum from pre-merge would push fresh adapters
    /// in the wrong direction.
    pub fn reset_states_for_params(&self, ctx: &Arc<MetalContext>, params: &[&Tensor]) {
        let ids: std::collections::HashSet<usize> = params.iter().map(|p| p.id).collect();
        for ps in &self.params {
            if ids.contains(&ps.tensor_id) {
                compute::gpu_fill(ctx, &ps.m, ps.size as u32, 0.0);
                compute::gpu_fill(ctx, &ps.v, ps.size as u32, 0.0);
            }
        }
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
                    update_clip: self.update_clip,
                },
            );
        }
    }

    /// Perform optimizer step on CPU using unified memory (Apple Silicon zero-copy).
    /// On Apple Silicon, CPU and GPU share physical memory — no copy needed.
    /// The CPU update runs while the GPU can start the next forward pass.
    /// This hides optimizer latency behind GPU compute.
    #[cfg(feature = "metal")]
    pub fn step_cpu(&mut self, lr: f32) {
        #[cfg(feature = "metal")]
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

                    let mut update = m_hat / (v_hat.sqrt() + self.eps);
                    if self.update_clip > 0.0 {
                        update = update.clamp(-self.update_clip, self.update_clip);
                    }
                    let wd = if ps.no_decay { 0.0 } else { self.weight_decay };
                    let p = *param_ptr.add(i);
                    *param_ptr.add(i) = p * (1.0 - lr * wd) - lr * update;
                }
            }
        }
    }

    /// Load optimizer state from checkpoint data. Sets m, v buffers and step counter.
    #[cfg(feature = "metal")]
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

    /// Serialize state for resume: per param, m then v (fp32 bytes). (Sidecar form, parallel to the
    /// AMDT format's m/v save; used when AdamW is a sub-optimizer of the hybrid.)
    pub fn save_state_blobs(&self) -> Vec<Vec<u8>> {
        let mut blobs = Vec::with_capacity(self.params.len() * 2);
        for ps in &self.params {
            blobs.push(buf_bytes(&ps.m, ps.size * 4));
            blobs.push(buf_bytes(&ps.v, ps.size * 4));
        }
        blobs
    }

    /// Restore state from `save_state_blobs` (same order) + the step counter.
    pub fn load_state_blobs(&mut self, step: u32, blobs: &[Vec<u8>]) {
        assert_eq!(blobs.len(), self.params.len() * 2, "AdamW state blob count mismatch");
        self.step = step;
        for (i, ps) in self.params.iter().enumerate() {
            buf_set_bytes(&ps.m, &blobs[i * 2]);
            buf_set_bytes(&ps.v, &blobs[i * 2 + 1]);
        }
    }
}

/// 8-bit AdamW: the first/second moments m,v are stored as block-wise int8 (one fp32 absmax scale
/// per 256-element block) instead of fp32 — ~4× less optimizer memory, the direct lever for fitting
/// a bigger model on the same RAM ("10× capacity"). The update dequantizes, runs the standard AdamW
/// math (same hardened eps/no_decay/update_clip as `AdamW`), applies the param step, then requantizes
/// with fresh per-block scales. Pairs with Muon (no v) for further savings.
pub struct AdamW8bit {
    pub params: Vec<Param8State>,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub update_clip: f32,
    pub weight_decay: f32,
    pub step: u32,
    ctx: Arc<MetalContext>,
}

pub struct Param8State {
    pub tensor_id: usize,
    pub buffer: crate::gpu::Buf,
    pub size: usize,
    pub n_blocks: usize,
    pub no_decay: bool,
    pub m_q: crate::gpu::Buf,    // int8 of m, `size` bytes
    pub v_q: crate::gpu::Buf,    // int8 of √v (range-compressed), `size` bytes
    pub m_scale: crate::gpu::Buf, // fp32 absmax(|m|)/127 per block, n_blocks entries
    pub v_scale: crate::gpu::Buf, // fp32 absmax(√v)/127 per block, n_blocks entries
}

impl AdamW8bit {
    pub fn new(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32) -> Self {
        Self::new_with_config(ctx, params, weight_decay, AdamWHyper::default())
    }

    pub fn new_with_config(ctx: &Arc<MetalContext>, params: &[&Tensor], weight_decay: f32, hyper: AdamWHyper) -> Self {
        let block = compute::ADAM8_BLOCK;
        let param_states = params.iter().map(|t| {
            let size = t.numel();
            let n_blocks = size.div_ceil(block);
            let m_q = ctx.alloc_buffer(size.max(1));        // 1 byte per int8 element
            let v_q = ctx.alloc_buffer(size.max(1));
            let m_scale = ctx.alloc_buffer(n_blocks * 4);   // fp32 per block
            let v_scale = ctx.alloc_buffer(n_blocks * 4);
            // Zero all state: int8 0 and fp32 0.0 are both all-zero bytes. Buffers may be pooled
            // (stale), so this is required.
            // Metal: pooled buffers may be stale → must zero. CUDA: alloc_zeros already zeroes.
            #[cfg(feature = "metal")]
            unsafe {
                std::ptr::write_bytes(m_q.contents().as_ptr() as *mut u8, 0, size.max(1));
                std::ptr::write_bytes(v_q.contents().as_ptr() as *mut u8, 0, size.max(1));
                std::ptr::write_bytes(m_scale.contents().as_ptr() as *mut u8, 0, n_blocks * 4);
                std::ptr::write_bytes(v_scale.contents().as_ptr() as *mut u8, 0, n_blocks * 4);
            }
            Param8State {
                tensor_id: t.id,
                buffer: t.buffer.clone(),
                size,
                n_blocks,
                no_decay: t.shape.len() <= 1,
                m_q,
                v_q,
                m_scale,
                v_scale,
            }
        }).collect();

        Self {
            params: param_states,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            update_clip: hyper.update_clip,
            weight_decay,
            step: 0,
            ctx: Arc::clone(ctx),
        }
    }

    /// Optimizer-state memory: int8 m+v + fp32 block scales. ~4× smaller than fp32 AdamW.
    pub fn memory_bytes(&self) -> usize {
        self.params.iter().map(|ps| ps.size * 2 + ps.n_blocks * 4 * 2).sum()
    }

    pub fn step(&mut self, lr: f32) {
        self.step += 1;
        for ps in &self.params {
            let grad = match autograd::get_grad(ps.tensor_id) {
                Some(g) => g,
                None => continue,
            };
            compute::gpu_adamw_8bit_update(
                &self.ctx,
                &ps.buffer,
                &grad,
                &compute::Adam8Buffers {
                    m_q: &ps.m_q,
                    v_q: &ps.v_q,
                    m_scale: &ps.m_scale,
                    v_scale: &ps.v_scale,
                },
                ps.size as u32,
                &compute::AdamWHyperparams {
                    lr,
                    beta1: self.beta1,
                    beta2: self.beta2,
                    eps: self.eps,
                    weight_decay: if ps.no_decay { 0.0 } else { self.weight_decay },
                    step: self.step,
                    update_clip: self.update_clip,
                },
            );
        }
    }

    pub fn zero_grad(&self) {
        autograd::clear_tape();
        autograd::clear_recompute_registry();
    }

    /// Serialize state for resume: per param, m_q + v_q (int8) then m_scale + v_scale (fp32 bytes).
    pub fn save_state_blobs(&self) -> Vec<Vec<u8>> {
        let mut blobs = Vec::with_capacity(self.params.len() * 4);
        for ps in &self.params {
            blobs.push(buf_bytes(&ps.m_q, ps.size));
            blobs.push(buf_bytes(&ps.v_q, ps.size));
            blobs.push(buf_bytes(&ps.m_scale, ps.n_blocks * 4));
            blobs.push(buf_bytes(&ps.v_scale, ps.n_blocks * 4));
        }
        blobs
    }

    /// Restore state from `save_state_blobs` (same order) + the step counter.
    pub fn load_state_blobs(&mut self, step: u32, blobs: &[Vec<u8>]) {
        assert_eq!(blobs.len(), self.params.len() * 4, "8-bit AdamW state blob count mismatch");
        self.step = step;
        for (i, ps) in self.params.iter().enumerate() {
            buf_set_bytes(&ps.m_q, &blobs[i * 4]);
            buf_set_bytes(&ps.v_q, &blobs[i * 4 + 1]);
            buf_set_bytes(&ps.m_scale, &blobs[i * 4 + 2]);
            buf_set_bytes(&ps.v_scale, &blobs[i * 4 + 3]);
        }
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
    if warmup_steps == 0 {
        // No warmup: apply inverse-sqrt decay from step 1 onward, max_lr at step 0
        if step == 0 { return max_lr; }
        max_lr / (step as f32).sqrt()
    } else if step < warmup_steps {
        // Linear warmup
        max_lr * (step as f32 / warmup_steps as f32)
    } else {
        // Inverse-sqrt decay: lr = max_lr * sqrt(warmup) / sqrt(step)
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
    pub buffer: crate::gpu::Buf,
    pub size: usize,
    pub m: crate::gpu::Buf, // first moment (momentum)
    pub h: crate::gpu::Buf, // diagonal Hessian estimate
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
                ps.size as u32, compute::SophiaParams { lr, beta1: self.beta1, beta2: self.beta2, eps: self.eps, rho: self.rho, weight_decay: self.weight_decay },
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
    /// AdamW-fallback hyperparameters for the non-2-D params (norms/biases). eps defaults to the
    /// hardened 1e-5 — the old hardcoded 1e-8 here was the same denominator-collapse bug that was
    /// fixed in the standalone AdamW but had been left latent on this fallback path.
    pub adamw_hyper: AdamWHyper,
    /// NorMuon: when true, normalize the orthogonalized update per-neuron (per output row) by a
    /// running second moment — adds Adam-like per-neuron adaptivity on top of orthogonalization
    /// (arXiv NorMuon, ~+11% over Muon). Off by default = plain Muon. The per-row moment is held in
    /// `MuonState::ns_vrow` (in-memory only; re-warms after a resume — it's a fast beta2=0.95 EMA, so
    /// not persisting it costs only a few steps and keeps the optimizer state-blob format unchanged).
    pub normalized: bool,
    pub norm_beta2: f32,
    pub norm_eps: f32,
    /// Cautious Muon (Liang et al. 2024, arXiv 2411.16085): each step, zero the orthogonalized-update
    /// components whose sign disagrees with the current gradient, then renormalize to preserve the
    /// update magnitude. A near-free convergence improvement. Off by default; composes with NorMuon
    /// (applied after it, to the final update).
    pub cautious: bool,
    ctx: Arc<MetalContext>,
}

pub struct MuonState {
    pub tensor_id: usize,
    pub buffer: crate::gpu::Buf,
    pub size: usize,
    pub shape: Vec<usize>,
    pub m: crate::gpu::Buf,    // momentum buffer
    pub is_2d: bool,               // true = Muon update, false = AdamW fallback
    pub no_decay: bool,            // AdamW fallback: skip weight decay for 1-D norms/biases
    // AdamW fallback state for non-2D params
    pub v: Option<crate::gpu::Buf>,
    // Pre-allocated Newton-Schulz workspace (avoids ~100 allocs per 2D param per step)
    pub ns_x: Option<crate::gpu::Buf>,    // [rows, cols] — working copy
    pub ns_xxt: Option<crate::gpu::Buf>,  // [rows, rows] — X @ X^T
    pub ns_xxtx: Option<crate::gpu::Buf>, // [rows, cols] — (X @ X^T) @ X
    // NorMuon per-row state (only Some for 2-D params): the running second moment and a scratch buffer.
    pub ns_vrow: Option<crate::gpu::Buf>,  // [rows] — EMA of per-row mean-square of the orthogonal update
    pub ns_rowss: Option<crate::gpu::Buf>, // [rows] — scratch (per-row sum-of-squares, then the scale)
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
            // Pre-allocate NS workspace for 2D params (+ NorMuon per-row buffers, used only when enabled).
            let (ns_x, ns_xxt, ns_xxtx, ns_vrow, ns_rowss) = if is_2d {
                let rows = t.shape[0];
                let cols = t.shape[1];
                let vrow = ctx.alloc_buffer(rows * 4);
                compute::gpu_fill(ctx, &vrow, rows as u32, 0.0); // second moment starts at 0
                (
                    Some(ctx.alloc_buffer(rows * cols * 4)),
                    Some(ctx.alloc_buffer(rows * rows * 4)),
                    Some(ctx.alloc_buffer(rows * cols * 4)),
                    Some(vrow),
                    Some(ctx.alloc_buffer(rows * 4)),
                )
            } else { (None, None, None, None, None) };
            MuonState {
                tensor_id: t.id, buffer: t.buffer.clone(), size,
                shape: t.shape.clone(), m, is_2d, no_decay: t.shape.len() <= 1, v,
                ns_x, ns_xxt, ns_xxtx, ns_vrow, ns_rowss,
            }
        }).collect();

        Self {
            params: param_states,
            beta: 0.95,
            weight_decay,
            ns_steps: 5,
            step: 0,
            adamw_hyper: AdamWHyper::default(),
            normalized: false,
            norm_beta2: 0.95,
            norm_eps: 1e-8,
            cautious: false,
            ctx: Arc::clone(ctx),
        }
    }

    /// Enable NorMuon: neuron-wise (per-row) second-moment normalization of the orthogonalized update
    /// (~+11% over Muon). Off by default. `beta2` is the per-row second-moment EMA decay (try 0.95),
    /// `eps` floors the denominator (try 1e-8). Applies to every 2-D param; the per-row state lives in
    /// `ns_vrow` (already allocated in `new`).
    pub fn set_normalization(&mut self, beta2: f32, eps: f32) {
        self.normalized = true;
        self.norm_beta2 = beta2;
        self.norm_eps = eps;
    }

    /// Enable Cautious Muon (Liang et al. 2024): each step, mask out the orthogonalized-update
    /// components whose sign disagrees with the current gradient, then renormalize to preserve the
    /// update magnitude (so it is LR-neutral vs plain Muon). A near-free convergence gain. Composes
    /// with NorMuon. Off by default.
    pub fn set_cautious(&mut self, on: bool) {
        self.cautious = on;
    }

    pub fn step(&mut self, lr: f32) {
        self.step += 1;

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

                // Newton-Schulz orthogonalization of momentum M [rows, cols].
                // Normalize M by its FROBENIUS norm first: X = M / (‖M‖_F + eps). This guarantees
                // σ_max(X) ≤ ‖X‖_2 ≤ ‖X‖_F = 1 < √3, the convergence radius of the cubic NS map
                // g(σ)=1.5σ−0.5σ³. The previous 1/√max(rows,cols) scale was a dimension heuristic that
                // did NOT bound the spectral norm, so at larger momentum magnitude (bigger batch /
                // higher effective LR) σ_max could exceed √3 and the iteration diverged cubically —
                // the suspected cause of the batch-32 Muon/hybrid divergence (#6). Orthogonalization
                // is scale-free, so when NS already converged (small batch) the update is unchanged;
                // only the divergent regime is fixed. NOTE: a partially-converged NS (small ns_steps)
                // is mildly scale-sensitive, so re-check --muon-lr-scale after this change.
                // Done on-GPU in one dispatch (no CPU readback → no command-batch flush/hazard).

                // Pre-allocated workspace — no buffer allocations per step
                let x_buf = ps.ns_x.as_ref().unwrap();
                let xxt_buf = ps.ns_xxt.as_ref().unwrap();
                let xxtx_buf = ps.ns_xxtx.as_ref().unwrap();

                // X = M / (‖M‖_F + eps) into the pre-allocated working buffer.
                compute::gpu_muon_frob_normalize(&self.ctx, &ps.m, x_buf, size);

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

                // NorMuon: per-neuron (per output row) normalization of the orthogonalized update X.
                // Adds Adam-like per-neuron adaptivity on top of Muon's orthogonalization (~+11% over
                // Muon). Off by default (plain Muon). Reuses existing kernels — no new optimizer state
                // on disk (ns_vrow is in-memory, re-warms after resume).
                if self.normalized {
                    let vrow = ps.ns_vrow.as_ref().unwrap();   // [rows] running second moment
                    let rowss = ps.ns_rowss.as_ref().unwrap(); // [rows] scratch
                    // r[i] = mean_j X[i,j]^2  (per-row mean-square of the orthogonal update)
                    compute::gpu_row_dot_reduce(&self.ctx, x_buf, x_buf, rowss, rows, cols);
                    compute::gpu_scale(&self.ctx, rowss, rows, 1.0 / cols as f32);
                    // v[i] = beta2*v[i] + (1-beta2)*r[i]
                    compute::gpu_ema_update(&self.ctx, vrow, rowss, rows, self.norm_beta2);
                    // scale[i] = 1 / (sqrt(v[i] / (1 - beta2^t)) + eps)  — bias-corrected; written into rowss
                    let bias_correction = 1.0 / (1.0 - self.norm_beta2.powi(self.step as i32));
                    compute::gpu_inv_sqrt_bc(&self.ctx, vrow, rowss, rows, bias_correction, self.norm_eps);
                    // X[i,j] *= scale[i]  (in-place: each thread reads+writes its own element)
                    compute::gpu_scale_rows(&self.ctx, x_buf, rowss, x_buf, rows, cols);
                }

                // Cautious Muon (Liang et al. 2024): zero update components whose sign disagrees with
                // the current gradient (u·g ≤ 0), then renormalize by size/(kept+1) so the update
                // magnitude — and the effective LR — is preserved. Applied to the FINAL update (after
                // NorMuon). Reuses the post-NS scratch (ns_xxtx [size] as the keep-mask, ns_xxt for the
                // 1-element kept-count) — no new optimizer state, no CPU readback.
                if self.cautious {
                    let keep = ps.ns_xxtx.as_ref().unwrap();
                    let kept_sum = ps.ns_xxt.as_ref().unwrap();
                    compute::gpu_cautious_mask(&self.ctx, x_buf, &grad, keep, size);
                    compute::gpu_reduce_sum(&self.ctx, keep, kept_sum, size);
                    compute::gpu_cautious_scale(&self.ctx, x_buf, kept_sum, size);
                }

                // Apply: theta = theta * (1 - lr * wd) - lr * X
                // Fused axpy: theta += -lr * X (1 dispatch instead of scale + add_inplace = 2)
                if self.weight_decay > 0.0 {
                    compute::gpu_scale(&self.ctx, &ps.buffer, size, 1.0 - lr * self.weight_decay);
                }
                compute::gpu_axpy(&self.ctx, &ps.buffer, x_buf, size, -lr);
            } else {
                // AdamW fallback for non-2-D params (norms, biases). Uses the hardened eps (1e-5)
                // and respects no_decay so norm weights aren't decayed toward zero — matching the
                // standalone AdamW exactly (the old path hardcoded eps=1e-8 + decayed norms).
                let v_buf = ps.v.as_ref().unwrap();
                compute::gpu_adamw_update(
                    &self.ctx, &ps.buffer, &grad, &ps.m, v_buf,
                    ps.size as u32,
                    &compute::AdamWHyperparams {
                        lr,
                        beta1: self.adamw_hyper.beta1,
                        beta2: self.adamw_hyper.beta2,
                        eps: self.adamw_hyper.eps,
                        weight_decay: if ps.no_decay { 0.0 } else { self.weight_decay },
                        step: self.step,
                        update_clip: self.adamw_hyper.update_clip,
                    },
                );
            }
        }
    }

    /// Serialize state for resume: per param, the momentum `m` (fp32 bytes), plus the AdamW-fallback
    /// `v` (fp32 bytes) for the non-2-D params that use it.
    pub fn save_state_blobs(&self) -> Vec<Vec<u8>> {
        let mut blobs = Vec::new();
        for ps in &self.params {
            blobs.push(buf_bytes(&ps.m, ps.size * 4));
            if let Some(v) = &ps.v {
                blobs.push(buf_bytes(v, ps.size * 4));
            }
        }
        blobs
    }

    /// Restore state from `save_state_blobs` (same order) + the step counter.
    pub fn load_state_blobs(&mut self, step: u32, blobs: &[Vec<u8>]) {
        self.step = step;
        let mut i = 0;
        for ps in &self.params {
            buf_set_bytes(&ps.m, &blobs[i]);
            i += 1;
            if let Some(v) = &ps.v {
                buf_set_bytes(v, &blobs[i]);
                i += 1;
            }
        }
        assert_eq!(i, blobs.len(), "Muon state blob count mismatch");
    }
}

/// Muon+AdamW hybrid — the canonical Muon recipe (Keller Jordan et al., Kimi K2, GLM-4.5).
///
/// Muon's full-batch speed comes from orthogonalizing the *hidden* 2-D weight matrices
/// (attention/FFN projections) via Newton-Schulz; but applying it to embeddings, the tied LM head,
/// MoE routers, and 1-D norms/biases is its well-known pathology. The hybrid routes each parameter
/// by role: true hidden 2-D matrices → Muon, everything else → the hardened AdamW (eps=1e-5,
/// no_decay for norms). This gets Muon's speed without the embedding/norm degradation, and it is
/// strictly better than the half-baked AdamW fallback that lived inside `Muon` (which still
/// orthogonalized embeddings because they are 2-D in shape).
pub struct HybridOptimizer {
    pub muon: Muon,   // hidden 2-D weight matrices
    pub adamw: AdamW, // embeddings, tied head, routers, all 1-D norms/biases
    /// Per-group LR multipliers on the base lr passed to `step`. The canonical recipe drives the
    /// orthogonalized Muon group harder than the AdamW group (e.g. Muon ~2e-2, AdamW ~few e-3): the
    /// AdamW group includes embeddings/the tied head, which oscillate if driven at Muon's LR.
    pub muon_lr_scale: f32,
    pub adamw_lr_scale: f32,
}

impl HybridOptimizer {
    /// Partition `params` by role. A parameter goes to Muon iff it is a 2-D matrix (both dims > 1)
    /// AND is not in `force_adamw_ids` (embeddings/head/routers — see
    /// `Transformer::force_adamw_param_ids`). Everything else (1-D and the forced set) goes to AdamW.
    pub fn new(
        ctx: &Arc<MetalContext>,
        params: &[&Tensor],
        weight_decay: f32,
        force_adamw_ids: &HashSet<usize>,
        hyper: AdamWHyper,
    ) -> Self {
        let mut muon_params: Vec<&Tensor> = Vec::new();
        let mut adamw_params: Vec<&Tensor> = Vec::new();
        for &t in params {
            let is_matrix = t.shape.len() == 2 && t.shape[0] > 1 && t.shape[1] > 1;
            if is_matrix && !force_adamw_ids.contains(&t.id) {
                muon_params.push(t);
            } else {
                adamw_params.push(t);
            }
        }
        let mut muon = Muon::new(ctx, &muon_params, weight_decay);
        muon.adamw_hyper = hyper; // keep the (unused-here) fallback consistent if a degenerate sneaks in
        let adamw = AdamW::new_with_config(ctx, &adamw_params, weight_decay, hyper);
        Self { muon, adamw, muon_lr_scale: 1.0, adamw_lr_scale: 1.0 }
    }

    /// Set the per-group LR multipliers (Muon group, AdamW group) on the base lr.
    pub fn set_lr_scales(&mut self, muon: f32, adamw: f32) {
        self.muon_lr_scale = muon;
        self.adamw_lr_scale = adamw;
    }

    pub fn step(&mut self, lr: f32) {
        self.muon.step(lr * self.muon_lr_scale);
        self.adamw.step(lr * self.adamw_lr_scale);
    }

    pub fn zero_grad(&self) {
        autograd::clear_tape();
        autograd::clear_recompute_registry();
    }

    /// Number of parameters routed to each sub-optimizer (for logging / tests).
    pub fn split_counts(&self) -> (usize, usize) {
        (self.muon.params.len(), self.adamw.params.len())
    }

    /// Serialize state for resume: Muon blobs (one m per 2-D matrix) followed by AdamW blobs (m,v per
    /// embedding/head/norm param). The Muon-portion length is exactly `muon.params.len()` (all hybrid
    /// Muon params are 2-D, so none carry a fallback `v`), which makes the split unambiguous on load.
    pub fn save_state_blobs(&self) -> Vec<Vec<u8>> {
        let mut blobs = self.muon.save_state_blobs();
        blobs.extend(self.adamw.save_state_blobs());
        blobs
    }

    /// Restore state from `save_state_blobs` (same order) + the step counter.
    pub fn load_state_blobs(&mut self, step: u32, blobs: &[Vec<u8>]) {
        let muon_n = self.muon.params.len(); // 1 blob (m) per 2-D matrix
        assert!(blobs.len() >= muon_n, "hybrid state blob count too small");
        self.muon.load_state_blobs(step, &blobs[..muon_n]);
        self.adamw.load_state_blobs(step, &blobs[muon_n..]);
    }
}

/// Unified optimizer interface for training loop.
pub enum Optimizer {
    AdamW(AdamW),
    Sophia(Sophia),
    Muon(Muon),
    Hybrid(HybridOptimizer),
    AdamW8bit(AdamW8bit),
}

impl Optimizer {
    pub fn step(&mut self, lr: f32) {
        match self {
            Optimizer::AdamW(o) => o.step(lr),
            Optimizer::Sophia(o) => o.step(lr),
            Optimizer::Muon(o) => o.step(lr),
            Optimizer::Hybrid(o) => o.step(lr),
            Optimizer::AdamW8bit(o) => o.step(lr),
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
            Optimizer::Hybrid(o) => o.zero_grad(),
            Optimizer::AdamW8bit(o) => o.zero_grad(),
        }
    }

    pub fn adamw_step(&self) -> u32 {
        match self {
            Optimizer::AdamW(o) => o.step,
            Optimizer::Sophia(o) => o.step,
            Optimizer::Muon(o) => o.step,
            Optimizer::Hybrid(o) => o.adamw.step,
            Optimizer::AdamW8bit(o) => o.step,
        }
    }
}
