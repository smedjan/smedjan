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
    eprintln!(
        "Training: batch_size={}, seq_len={}, total_steps={}, gradient_checkpointing={}",
        config.batch_size, config.seq_len, config.total_steps, config.gradient_checkpointing
    );
    eprintln!("Tokenizer: {}", config.tokenizer_path);

    // Create checkpoint directory
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    // Initialize model
    let model = Transformer::new(ctx, config.model_config.clone());

    // Initialize optimizer
    let param_refs: Vec<&_> = model.parameters().into_iter().collect();
    let mut optimizer = AdamW::new(ctx, &param_refs, config.weight_decay);

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

    let mut total_tokens: u64 = 0;
    let start_time = Instant::now();

    for step in 0..config.total_steps {
        let step_start = Instant::now();
        let lr = scheduler.get_lr(step);

        // Get batch (wraps around automatically)
        let (inputs, targets) = data_loader.next_batch();

        // Forward pass (batched GPU dispatch — all kernels encode into one command buffer)
        ctx.begin_batch();
        let logits = model.forward(&inputs, config.batch_size, config.seq_len, None, config.gradient_checkpointing);

        // Compute loss
        let (loss_tensor, _grad_logits) = loss::cross_entropy_loss(ctx, &logits, &targets);
        ctx.flush_batch();

        // Backward pass (batched)
        ctx.begin_batch();
        autograd::backward(ctx, loss_tensor.id);
        ctx.flush_batch();

        // Gradient clipping (also filters NaN gradients)
        ctx.begin_batch();
        clip_gradients(ctx, &model, config.max_grad_norm);

        // Optimizer step (skip if lr is effectively zero to avoid NaN momentum from 0*NaN)
        if lr > 1e-10 {
            optimizer.step(lr);
        }
        optimizer.zero_grad();
        ctx.flush_batch();

        let tokens_this_step = (config.batch_size * config.seq_len) as u64;
        total_tokens += tokens_this_step;

        // NaN detection — abort early to save time
        let loss_val = loss_tensor.to_vec()[0];
        if loss_val.is_nan() || loss_val.is_infinite() {
            eprintln!("FATAL: loss is {} at step {}. Training diverged.", loss_val, step);
            eprintln!("Try: lower --lr, increase --warmup, or check data quality.");
            break;
        }

        // Logging
        if step % config.log_interval == 0 {
            let step_time = step_start.elapsed().as_secs_f32();
            let tokens_per_sec = tokens_this_step as f32 / step_time;
            let elapsed = start_time.elapsed().as_secs();
            let (tape_ops, tape_bytes) = autograd::tape_stats();
            if step == 0 {
                eprintln!("Tape: {} ops, {:.1} MB activation memory", tape_ops, tape_bytes as f64 / (1024.0 * 1024.0));
            }

            let (pool_hits, pool_misses) = MetalContext::pool_stats();

            eprintln!(
                "step {:>6} | loss {:>8.4} | lr {:.2e} | {:.0} tok/s | {:.1}s/step | {}s elapsed | {}M tokens | epoch {} ({}/{}) | pool {}/{}",
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
            );
        }

        // Checkpointing
        if step > 0 && step % config.checkpoint_interval == 0 {
            let path = format!("{}/step_{}.bin", config.checkpoint_dir, step);
            checkpoint::save_checkpoint(&path, &model, step)?;
            eprintln!("Checkpoint saved: {}", path);
        }
    }

    // Final checkpoint
    let path = format!("{}/final.bin", config.checkpoint_dir);
    checkpoint::save_checkpoint(&path, &model, config.total_steps)?;
    eprintln!("Training complete. Final checkpoint: {}", path);

    Ok(())
}

/// Clip gradients by global L2 norm. Also zeroes NaN/Inf gradients.
fn clip_gradients(ctx: &Arc<MetalContext>, model: &Transformer, max_norm: f32) {
    let params = model.parameters();

    // Compute global gradient norm, zeroing any NaN/Inf gradients
    let mut total_norm_sq = 0.0f32;
    for param in &params {
        if let Some(grad) = autograd::get_grad(param.id) {
            let norm = compute::gpu_l2_norm(ctx, &grad, param.numel() as u32);
            if norm.is_nan() || norm.is_infinite() {
                // NaN gradient — zero it out to prevent corruption
                compute::gpu_fill(ctx, &grad, param.numel() as u32, 0.0);
            } else {
                total_norm_sq += norm * norm;
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // Scale gradients if norm exceeds max_norm
    if total_norm > max_norm && total_norm.is_finite() {
        let scale = max_norm / (total_norm + 1e-6);
        for param in &params {
            if let Some(grad) = autograd::get_grad(param.id) {
                compute::gpu_scale(ctx, &grad, param.numel() as u32, scale);
            }
        }
    }
}
