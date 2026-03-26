use crate::autograd;
use crate::checkpoint;
use crate::data::DataLoader;
use crate::loss;
use crate::metal::compute;
use crate::metal::MetalContext;
use crate::model::{ModelConfig, Transformer};
use crate::optim::{AdamW, CosineWarmupScheduler};
use std::sync::Arc;
use std::time::Instant;

/// Training configuration.
pub struct TrainConfig {
    pub model_config: ModelConfig,
    pub dataset_path: String,
    pub tokenizer_path: String,
    pub checkpoint_dir: String,
    pub batch_size: usize,
    pub seq_len: usize,
    pub total_steps: u32,
    pub max_lr: f32,
    pub warmup_steps: u32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub log_interval: u32,
    pub checkpoint_interval: u32,
    pub gradient_checkpointing: bool,
    /// Knowledge distillation: path to teacher model checkpoint.
    /// When set, training uses distillation loss instead of plain cross-entropy.
    pub teacher_checkpoint: Option<String>,
    /// Distillation temperature (softens teacher/student distributions). Default: 4.0.
    pub distill_temperature: f32,
    /// Distillation mixing weight: loss = alpha * T^2 * KL + (1-alpha) * CE. Default: 0.5.
    pub distill_alpha: f32,
    /// Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps. Default: 1.
    pub grad_accum_steps: u32,
    /// Resume from a training state file (saves optimizer + model + step).
    pub resume_from: Option<String>,
    /// Path to validation dataset (optional). Eval every checkpoint_interval steps.
    pub val_dataset: Option<String>,
    /// Dropout rate for regularization. Default: 0.0 (no dropout).
    pub dropout: f32,
}

impl TrainConfig {
    pub fn default_small(dataset_path: &str, tokenizer_path: &str) -> Self {
        Self {
            model_config: ModelConfig::small(8192),
            dataset_path: dataset_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            checkpoint_dir: "checkpoints".to_string(),
            batch_size: 32,
            seq_len: 256,
            total_steps: 50000,
            max_lr: 3e-4,
            warmup_steps: 2000,
            weight_decay: 0.1,
            max_grad_norm: 1.0,
            log_interval: 10,
            checkpoint_interval: 5000,
            gradient_checkpointing: false,
            teacher_checkpoint: None,
            distill_temperature: 4.0,
            distill_alpha: 0.5,
            grad_accum_steps: 1,
            resume_from: None,
            val_dataset: None,
            dropout: 0.0,
        }
    }
}

/// Run the training loop.
pub fn train(ctx: &Arc<MetalContext>, config: &TrainConfig) -> std::io::Result<()> {
    eprintln!("=== AndreAI Training ===");
    eprintln!(
        "Model: {}M params, {} layers, d_model={}, {} heads",
        config.model_config.param_count() as f32 / 1e6,
        config.model_config.n_layers,
        config.model_config.d_model,
        config.model_config.n_heads
    );
    let effective_batch = config.batch_size * config.grad_accum_steps as usize;
    eprintln!(
        "Training: batch_size={}, seq_len={}, total_steps={}, gradient_checkpointing={}, grad_accum_steps={}, effective_batch={}",
        config.batch_size, config.seq_len, config.total_steps, config.gradient_checkpointing,
        config.grad_accum_steps, effective_batch,
    );
    eprintln!("Tokenizer: {}", config.tokenizer_path);

    // Create checkpoint directory
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    // Load teacher model for distillation (frozen, no grad)
    let teacher_model = match &config.teacher_checkpoint {
        Some(teacher_path) => {
            eprintln!("Distillation mode: loading teacher from {}", teacher_path);
            eprintln!("  temperature={}, alpha={}", config.distill_temperature, config.distill_alpha);
            let (teacher, _step) = checkpoint::load_checkpoint(ctx, teacher_path)?;
            Some(teacher)
        }
        None => None,
    };

    // Initialize model + optimizer (fresh or from resume checkpoint)
    let (model, mut optimizer, start_step, mut total_tokens) = if let Some(ref resume_path) = config.resume_from {
        eprintln!("Resuming from: {}", resume_path);
        let (model, opt_states, step, opt_step, tokens) = checkpoint::load_training_state(ctx, resume_path)?;
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let mut optimizer = AdamW::new(ctx, &param_refs, config.weight_decay);
        optimizer.load_state(&opt_states, opt_step);
        eprintln!("Resumed at step {}, {} tokens, optimizer step {}", step, tokens, opt_step);
        (model, optimizer, step, tokens)
    } else {
        let model = Transformer::new(ctx, config.model_config.clone());
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let optimizer = AdamW::new(ctx, &param_refs, config.weight_decay);
        (model, optimizer, 0, 0u64)
    };

    // Learning rate scheduler
    let scheduler = CosineWarmupScheduler::new(
        config.max_lr,
        config.warmup_steps,
        config.total_steps,
    );

    // Data loader
    let mut data_loader = DataLoader::new(&config.dataset_path, config.batch_size, config.seq_len)?;
    let batches_per_epoch = data_loader.batches_per_epoch();
    eprintln!("Dataset: {} tokens, ~{} batches/epoch", data_loader.total_tokens(), batches_per_epoch);

    // total_tokens initialized from resume state or 0
    let start_time = Instant::now();

    let grad_accum_steps = config.grad_accum_steps.max(1);
    let loss_scale = 1.0 / grad_accum_steps as f32;

    for step in start_step..config.total_steps {
        let step_start = Instant::now();
        let lr = scheduler.get_lr(step);

        // Track the last micro-step's loss for logging
        let mut last_loss_tensor: Option<crate::tensor::Tensor> = None;

        // === Gradient accumulation loop ===
        for _micro_step in 0..grad_accum_steps {
            // Get a micro-batch (DataLoader uses config.batch_size as micro-batch size)
            let (inputs, targets) = data_loader.next_batch();

            // Forward pass (batched GPU dispatch — all kernels encode into one command buffer)
            ctx.begin_batch();
            let logits = model.forward(&inputs, config.batch_size, config.seq_len, None, config.gradient_checkpointing);

            // Compute loss — distillation or plain cross-entropy
            let (loss_tensor, grad_logits) = if let Some(ref teacher) = teacher_model {
                // Teacher forward pass (no gradient recording)
                let teacher_logits = autograd::no_grad(|| {
                    teacher.forward(&inputs, config.batch_size, config.seq_len, None, false)
                });
                loss::distillation_loss(
                    ctx,
                    &logits,
                    &teacher_logits,
                    config.distill_temperature,
                    config.distill_alpha,
                    &targets,
                )
            } else {
                loss::cross_entropy_loss(ctx, &logits, &targets)
            };

            // Scale both loss AND gradient by 1/grad_accum_steps.
            // CrossEntropy backward uses the pre-computed gradient (cached buffer),
            // NOT the loss value, so scaling only the loss leaves gradients 8x too large.
            if grad_accum_steps > 1 {
                let grad_size = (config.batch_size * config.seq_len * config.model_config.vocab_size as usize) as u32;
                compute::gpu_scale(ctx, &grad_logits, grad_size, loss_scale);
                compute::gpu_scale(ctx, &loss_tensor.buffer, 1, loss_scale);
            }

            // Backward pass in the SAME command batch as forward — one fewer GPU sync
            autograd::backward(ctx, loss_tensor.id);
            ctx.flush_batch();

            // Free the tape (activations) but keep accumulated gradients
            autograd::clear_tape_keep_grads();
            autograd::clear_recompute_registry();

            last_loss_tensor = Some(loss_tensor);
        }

        // === After all micro-steps: clip, step, zero ===
        // Gradient clipping every step (required for stability).
        // clip_gradients does its own GPU batching internally.
        clip_gradients(ctx, &model, config.max_grad_norm);

        // Optimizer step
        ctx.begin_batch();
        if lr > 1e-10 {
            optimizer.step(lr);
        }
        ctx.flush_batch();

        // Zero gradients for next accumulation cycle
        autograd::zero_grads();

        let tokens_this_step = (config.batch_size * config.seq_len * grad_accum_steps as usize) as u64;
        total_tokens += tokens_this_step;

        // Logging + NaN detection (only at log intervals to avoid GPU→CPU sync every step)
        if step % config.log_interval == 0 {
            // Read back the last micro-step's loss (scaled). Undo the scale for display.
            let raw_loss = last_loss_tensor.as_ref().map(|t| t.to_vec()[0]).unwrap_or(0.0);
            let loss_val = if grad_accum_steps > 1 { raw_loss / loss_scale } else { raw_loss };
            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!("FATAL: loss is {} at step {}. Training diverged.", loss_val, step);
                eprintln!("Try: lower --lr, increase --warmup, or check data quality.");
                break;
            }
            let step_time = step_start.elapsed().as_secs_f32();
            let tokens_per_sec = tokens_this_step as f32 / step_time;
            let elapsed = start_time.elapsed().as_secs();
            let (tape_ops, tape_bytes) = autograd::tape_stats();
            if step == 0 {
                eprintln!("Tape: {} ops, {:.1} MB activation memory", tape_ops, tape_bytes as f64 / (1024.0 * 1024.0));
            }

            let (pool_hits, pool_misses) = MetalContext::pool_stats();

            // Periodic weight health check: every 100 steps to avoid GPU→CPU sync overhead.
            // gpu_l2_norm_check forces a batch flush + readback (8 bytes).
            let weight_norm = if step % 100 == 0 {
                let param0 = &model.parameters()[0];
                let (weight_norm_sq, has_nan) = compute::gpu_l2_norm_check(ctx, &param0.buffer, param0.numel() as u32);
                if has_nan {
                    eprintln!("[WARN] NaN detected in model weights at step {}", step);
                }
                weight_norm_sq.sqrt()
            } else {
                0.0
            };

            eprintln!(
                "step {:>6} | loss {:>8.4} | lr {:.2e} | {:.0} tok/s | {:.1}s/step | {}s elapsed | {}M tokens | epoch {} ({}/{}) | pool {}/{} | w_norm {:.4}",
                step,
                loss_val,
                lr,
                tokens_per_sec,
                step_time,
                elapsed,
                total_tokens / 1_000_000,
                data_loader.epoch(),
                step as usize % batches_per_epoch,
                batches_per_epoch,
                pool_hits,
                pool_hits + pool_misses,
                weight_norm,
            );
        }

        // Checkpointing — save both model-only and full training state for resume
        if step > 0 && step % config.checkpoint_interval == 0 {
            let path = format!("{}/step_{}.bin", config.checkpoint_dir, step);
            checkpoint::save_checkpoint(&path, &model, step)?;
            let state_path = format!("{}/state_{}.bin", config.checkpoint_dir, step);
            checkpoint::save_training_state(&state_path, &model, &optimizer, step, total_tokens)?;
        }

        // Validation loss (if validation dataset provided)
        if let Some(ref val_path) = config.val_dataset {
            if step > 0 && step % config.checkpoint_interval == 0 {
                let val_loss = compute_validation_loss(ctx, &model, val_path, config.batch_size, config.seq_len)?;
                eprintln!("  val_loss: {:.4}", val_loss);
            }
        }
    }

    // Final checkpoint
    let path = format!("{}/final.bin", config.checkpoint_dir);
    checkpoint::save_checkpoint(&path, &model, config.total_steps)?;
    let state_path = format!("{}/state_final.bin", config.checkpoint_dir);
    checkpoint::save_training_state(&state_path, &model, &optimizer, config.total_steps, total_tokens)?;
    eprintln!("Training complete. Final checkpoint: {}", path);

    Ok(())
}

/// Compute average cross-entropy loss on a validation dataset (no gradients).
fn compute_validation_loss(
    ctx: &Arc<MetalContext>,
    model: &Transformer,
    val_path: &str,
    batch_size: usize,
    seq_len: usize,
) -> std::io::Result<f32> {
    let mut val_loader = DataLoader::new(val_path, batch_size, seq_len)?;
    let n_batches = val_loader.batches_per_epoch().min(50); // cap at 50 batches for speed
    let mut total_loss = 0.0f32;
    let mut count = 0u32;

    autograd::no_grad(|| {
        for _ in 0..n_batches {
            let (inputs, targets) = val_loader.next_batch();
            ctx.begin_batch();
            let logits = model.forward(&inputs, batch_size, seq_len, None, false);
            let (loss_tensor, _) = loss::cross_entropy_loss(ctx, &logits, &targets);
            ctx.flush_batch();
            let val = loss_tensor.to_vec()[0];
            if val.is_finite() {
                total_loss += val;
                count += 1;
            }
        }
    });
    autograd::clear_tape();

    Ok(if count > 0 { total_loss / count as f32 } else { f32::NAN })
}

/// Clip gradients by global L2 norm. Also zeroes NaN/Inf gradients.
fn clip_gradients(ctx: &Arc<MetalContext>, model: &Transformer, max_norm: f32) {
    let params = model.parameters();

    // Phase 1: Compute all per-parameter L2 norms + NaN checks on GPU (batched).
    let mut norm_bufs: Vec<Option<(objc2::rc::Retained<crate::metal::GpuBuffer>, usize)>> = Vec::with_capacity(params.len());

    ctx.begin_batch();
    for param in &params {
        if let Some(grad) = autograd::get_grad(param.id) {
            let norm_out = ctx.alloc_buffer(std::mem::size_of::<f32>() * 2);
            compute::gpu_l2_norm_check_into(ctx, &grad, param.numel() as u32, &norm_out);
            norm_bufs.push(Some((norm_out, param.numel())));
        } else {
            norm_bufs.push(None);
        }
    }
    ctx.flush_batch();

    // Phase 2: Read all norms back (shared memory = direct pointer, no DMA).
    let mut total_norm_sq = 0.0f32;
    let mut nan_indices = Vec::new();
    for (i, entry) in norm_bufs.iter().enumerate() {
        if let Some((norm_buf, _size)) = entry {
            let vals = MetalContext::read_buffer(norm_buf, 2);
            let sum_sq = vals[0];
            let has_nan = vals[1] > 0.5;
            if has_nan || sum_sq.is_nan() || sum_sq.is_infinite() {
                nan_indices.push(i);
            } else {
                total_norm_sq += sum_sq;
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // Phase 3: Zero NaN grads and scale if needed (batched).
    if !nan_indices.is_empty() {
        eprintln!(
            "[WARN] NaN/Inf detected in {} gradient(s) (param indices: {:?}) — zeroing affected gradients",
            nan_indices.len(),
            &nan_indices[..nan_indices.len().min(10)],
        );
    }
    let needs_scale = total_norm > max_norm && total_norm.is_finite();
    let scale = if needs_scale { max_norm / (total_norm + 1e-6) } else { 1.0 };

    if !nan_indices.is_empty() || needs_scale {
        ctx.begin_batch();

        // Zero NaN gradients
        for &i in &nan_indices {
            if let Some(grad) = autograd::get_grad(params[i].id) {
                compute::gpu_fill(ctx, &grad, params[i].numel() as u32, 0.0);
            }
        }

        // Scale all gradients if norm exceeds max_norm
        if needs_scale {
            for param in &params {
                if let Some(grad) = autograd::get_grad(param.id) {
                    compute::gpu_scale(ctx, &grad, param.numel() as u32, scale);
                }
            }
        }

        ctx.flush_batch();
    }
}
