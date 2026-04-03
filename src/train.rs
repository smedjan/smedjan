use crate::autograd;
use crate::checkpoint;
use crate::data::DataLoader;
use crate::loss;
use crate::metal::compute;
use crate::metal::MetalContext;
use crate::model::{ModelConfig, Transformer};
use crate::optim::{AdamW, CosineWarmupScheduler};
use std::io::Write as IoWrite;
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
    /// LR restart period (steps). 0 = standard cosine decay, >0 = warm restarts.
    pub lr_restart_period: u32,
    /// Data pruning: skip batches where loss < threshold. 0.0 = disabled.
    pub prune_threshold: f32,
    /// GALORE: gradient low-rank projection rank. 0 = disabled.
    pub galore_rank: usize,
    /// Optimizer: "adamw" or "sophia". Default: adamw.
    pub optimizer_type: String,
    /// Speculative pretraining: path to a tiny reference model.
    /// Skip batches where reference model already has low loss (easy data).
    pub reference_model: Option<String>,
    /// Speculative threshold: skip if reference loss < this value.
    pub speculative_threshold: f32,
    /// Curriculum learning: ramp sequence length from seq_len/4 → seq_len over first 25% of training.
    /// Faster early training (short seqs = bigger effective batch), smooth transition to full context.
    pub curriculum: bool,
    /// Z-loss coefficient: penalize large logit magnitudes. 0.0=disabled, 1e-4=recommended for MoE.
    /// Prevents router/logit explosion that causes expert collapse (PaLM, ST-MoE).
    pub z_loss_coefficient: f32,
    /// LR schedule: "cosine" (default) or "wsd" (warmup-stable-decay, 5-10% better)
    pub lr_schedule: String,
    /// Self-distillation via EMA: decay rate for exponential moving average teacher.
    /// 0.0=disabled, 0.999=recommended. EMA model teaches the student with KL divergence.
    /// 20-30% better sample efficiency at ~10% compute overhead. (BYOL-style)
    pub ema_decay: f32,
    /// Anti-PGD noise scale for gradient perturbation. 0.0=off, 0.01=recommended.
    /// Anticorrelated noise between steps navigates to flatter minima. (Orvieto et al.)
    pub noise_scale: f32,
    /// ReLoRA merge interval: periodically merge lowrank U×V into base weights, reinitialize.
    /// 0=disabled, 5000=recommended. After K merges, effective rank = K × lowrank.
    /// Enables full-rank learning through sequential low-rank updates. (arXiv 2307.05695)
    pub relora_interval: u32,
    /// Use FusedLinearCrossEntropy: compute logits+loss in chunks, never materialize full logit tensor.
    /// Saves ~2GB peak memory for vocab=8192. Enable for large vocab or tight memory.
    pub fused_ce: bool,
    /// Progressive layer freezing: freeze bottom N% of layers after this fraction of training.
    /// 0.0=disabled, 0.5=freeze bottom 50% after halfway. Saves 10-30% compute in late training.
    pub freeze_fraction: f32,
    /// Load a pretrained model checkpoint (weights only, fresh optimizer).
    /// Used for progressive training: grow a small model, then continue training the larger one.
    pub pretrained: Option<String>,
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
            lr_restart_period: 0,
            prune_threshold: 0.0,
            galore_rank: 0,
            optimizer_type: "adamw".to_string(),
            reference_model: None,
            speculative_threshold: 7.0,
            curriculum: false,
            z_loss_coefficient: 0.0,
            lr_schedule: "cosine".to_string(),
            ema_decay: 0.0,
            noise_scale: 0.0,
            relora_interval: 0,
            fused_ce: false,
            freeze_fraction: 0.0,
            pretrained: None,
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

    // Training log file (CSV: step, loss, lr, tok/s, elapsed, tokens)
    let log_path = format!("{}/train.csv", config.checkpoint_dir);
    let log_exists = std::path::Path::new(&log_path).exists();
    let mut log_file = std::fs::OpenOptions::new().create(true).append(true).open(&log_path)?;
    if !log_exists {
        writeln!(log_file, "step,loss,lr,tok_per_sec,elapsed_sec,total_tokens")?;
    }

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

    // Load reference model for speculative pretraining (frozen, scores batches)
    let ref_model = match &config.reference_model {
        Some(ref_path) => {
            eprintln!("Speculative pretraining: loading reference from {}", ref_path);
            eprintln!("  threshold={:.1} (skip batches where ref loss < threshold)", config.speculative_threshold);
            let (ref_m, _) = checkpoint::load_checkpoint(ctx, ref_path)?;
            Some(ref_m)
        }
        None => None,
    };

    // Initialize model + optimizer (fresh or from resume checkpoint)
    let (model, mut optimizer, start_step, mut total_tokens) = if let Some(ref resume_path) = config.resume_from {
        eprintln!("=== RESUMING from {} ===", resume_path);
        let (model, opt_states, step, opt_step, tokens) = checkpoint::load_training_state(ctx, resume_path)?;
        eprintln!(
            "Resumed model: {}M params, {} layers, d_model={}, {} heads",
            model.config.param_count() as f32 / 1e6,
            model.config.n_layers, model.config.d_model, model.config.n_heads
        );
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let mut optimizer = AdamW::new_with_galore(ctx, &param_refs, config.weight_decay, config.galore_rank);
        optimizer.load_state(&opt_states, opt_step);
        // Resume from step+1 — the checkpoint was saved AFTER step completed
        let resume_step = step + 1;
        eprintln!("Resuming at step {}/{}, {} tokens, optimizer step {}", resume_step, config.total_steps, tokens, opt_step);
        (model, optimizer, resume_step, tokens)
    } else if let Some(ref pretrained_path) = config.pretrained {
        // Load pretrained model weights (fresh optimizer, step 0).
        // Used for progressive training: grow small → large, then continue.
        eprintln!("=== Loading pretrained model from {} ===", pretrained_path);
        let (model, step) = checkpoint::load_checkpoint(ctx, pretrained_path)?;
        eprintln!(
            "Pretrained model: {}M params, {} layers, d_model={}, {} heads (trained to step {})",
            model.config.param_count() as f32 / 1e6,
            model.config.n_layers, model.config.d_model, model.config.n_heads, step
        );
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let optimizer = AdamW::new_with_galore(ctx, &param_refs, config.weight_decay, config.galore_rank);
        (model, optimizer, 0, 0u64)
    } else {
        let model = Transformer::new(ctx, config.model_config.clone());
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let optimizer = AdamW::new_with_galore(ctx, &param_refs, config.weight_decay, config.galore_rank);
        (model, optimizer, 0, 0u64)
    };

    // Create Sophia optimizer if selected (runs ALONGSIDE AdamW for state compatibility)
    let mut sophia_opt = if config.optimizer_type == "sophia" {
        eprintln!("Using Sophia optimizer (2x faster convergence)");
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        Some(crate::optim::Sophia::new(ctx, &param_refs, config.weight_decay))
    } else {
        None
    };

    // Create Muon optimizer if selected (2.5x faster convergence than AdamW)
    let mut muon_opt = if config.optimizer_type == "muon" {
        eprintln!("Using Muon optimizer (2.5x faster convergence — Newton-Schulz orthogonalization)");
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let n_2d = param_refs.iter().filter(|p| p.shape.len() == 2 && p.shape[0] > 1 && p.shape[1] > 1).count();
        eprintln!("  {}/{} params use Muon (2D), rest use AdamW fallback", n_2d, param_refs.len());
        Some(crate::optim::Muon::new(ctx, &param_refs, config.weight_decay))
    } else {
        None
    };

    // Log optimizer info
    if optimizer.galore_rank > 0 {
        eprintln!("GALORE: rank={}, optimizer memory={:.1}MB (vs {:.1}MB full)",
            optimizer.galore_rank,
            optimizer.memory_bytes() as f32 / 1e6,
            model.parameters().iter().map(|p| p.numel() * 8).sum::<usize>() as f32 / 1e6);
    }

    // μP: scale learning rate by base_width / d_model
    let mup_scale = config.model_config.mup_lr_scale();
    let effective_lr = config.max_lr * mup_scale;
    if mup_scale < 1.0 {
        eprintln!("μP enabled: base_width={}, LR scaled {:.4} → {:.4}",
            config.model_config.mup_base_width, config.max_lr, effective_lr);
    }

    // Learning rate scheduler — multiple options from research
    let get_lr: Box<dyn Fn(u32) -> f32> = if config.lr_schedule == "wsd" {
        let wsd = crate::optim::WSDScheduler::new(effective_lr, config.warmup_steps, config.total_steps);
        eprintln!("LR schedule: WSD (warmup={}, stable={}, decay={})",
            wsd.warmup_steps, wsd.stable_steps, wsd.decay_steps);
        Box::new(move |step| wsd.get_lr(step))
    } else if config.lr_schedule == "wso" {
        // Warmup-Stable-Only: no decay. Best before SFT. (arXiv 2602.06797)
        let warmup = config.warmup_steps;
        let lr = effective_lr;
        eprintln!("LR schedule: WSO (warmup={}, then constant)", warmup);
        Box::new(move |step| {
            if step < warmup { lr * (step as f32 / warmup.max(1) as f32) }
            else { lr }
        })
    } else if config.lr_schedule == "invsqrt" {
        // Inverse sqrt: original Transformer schedule (Vaswani 2017)
        let warmup = config.warmup_steps;
        let lr = effective_lr;
        eprintln!("LR schedule: inverse-sqrt (warmup={})", warmup);
        Box::new(move |step| crate::optim::inverse_sqrt_lr(lr, warmup, step))
    } else if config.lr_schedule == "trapezoid" {
        // Trapezoidal: warmup → stable → linear decay to 10% of max
        let warmup = config.warmup_steps;
        let total = config.total_steps;
        let after_warmup = total.saturating_sub(warmup);
        let stable = (after_warmup as f32 * 0.6) as u32;
        let lr = effective_lr;
        eprintln!("LR schedule: trapezoid (warmup={}, stable={}, decay to 10%)", warmup, stable);
        Box::new(move |step| crate::optim::trapezoidal_lr(lr, lr * 0.1, warmup, stable, total, step))
    } else {
        let scheduler = if config.lr_restart_period > 0 {
            CosineWarmupScheduler::with_restarts(effective_lr, config.warmup_steps, config.total_steps, config.lr_restart_period)
        } else {
            CosineWarmupScheduler::new(effective_lr, config.warmup_steps, config.total_steps)
        };
        Box::new(move |step| scheduler.get_lr(step))
    };

    // Pre-allocate loss workspace (avoids 33MB+ allocation every step)
    let batch_seq = config.batch_size * config.seq_len;
    let loss_ws = loss::LossWorkspace::new(ctx, batch_seq, config.model_config.vocab_size as usize);

    // EMA (Exponential Moving Average) model for self-distillation
    // The EMA is a running average of weights that's always a better model than the snapshot.
    // Used as a teacher for KL-divergence self-distillation during training.
    let ema_buffers: Vec<objc2::rc::Retained<crate::metal::GpuBuffer>> = if config.ema_decay > 0.0 {
        eprintln!("Self-distillation: EMA decay={}", config.ema_decay);
        model.parameters().iter().map(|p| {
            let buf = ctx.alloc_buffer(p.numel() * 4);
            compute::gpu_copy(ctx, &p.buffer, &buf, p.numel() as u32);
            buf
        }).collect()
    } else {
        Vec::new()
    };

    // Data loader
    let mut data_loader = DataLoader::new(&config.dataset_path, config.batch_size, config.seq_len)?;
    let batches_per_epoch = data_loader.batches_per_epoch();
    eprintln!("Dataset: {} tokens, ~{} batches/epoch", data_loader.total_tokens(), batches_per_epoch);


    // total_tokens initialized from resume state or 0
    let start_time = Instant::now();

    let grad_accum_steps = config.grad_accum_steps.max(1);

    // Early stopping state
    let mut best_val_loss = f32::INFINITY;
    let mut val_no_improve = 0u32;
    let early_stop_patience = 3;
    let mut ema_loss = 0.0f32;
    let mut peak_tok_s = 0.0f32;
    let mut best_train_loss = f32::INFINITY;
    let mut prev_loss = 0.0f32; // for gradient noise estimation
    let loss_scale = 1.0 / grad_accum_steps as f32;

    for step in start_step..config.total_steps {
        let step_start = Instant::now();
        let lr = get_lr(step);

        // Curriculum learning: ramp seq_len from min(64, seq/4) → seq over first 25% of training.
        // Short sequences = faster steps + bigger effective batch in early training.
        let effective_seq = if config.curriculum {
            let ramp_end = config.total_steps / 4;
            if step < ramp_end {
                let min_seq = (config.seq_len / 4).max(32);
                let progress = step as f32 / ramp_end as f32;
                let seq = min_seq as f32 + progress * (config.seq_len - min_seq) as f32;
                // Round to multiple of 8 for GPU alignment
                ((seq as usize + 7) / 8 * 8).min(config.seq_len)
            } else {
                config.seq_len
            }
        } else {
            config.seq_len
        };

        // Track the last micro-step's loss for logging
        let mut last_loss_tensor: Option<crate::tensor::Tensor> = None;

        // === Gradient accumulation loop ===
        for _micro_step in 0..grad_accum_steps {
            // Get a micro-batch. With curriculum learning, truncate to effective_seq.
            let (full_inputs, full_targets) = data_loader.next_batch();
            let (inputs, targets): (Vec<u32>, Vec<u32>) = if effective_seq < config.seq_len {
                let bs = config.batch_size;
                let mut si = Vec::with_capacity(bs * effective_seq);
                let mut st = Vec::with_capacity(bs * effective_seq);
                for b in 0..bs {
                    let start = b * config.seq_len;
                    si.extend_from_slice(&full_inputs[start..start + effective_seq]);
                    st.extend_from_slice(&full_targets[start..start + effective_seq]);
                }
                (si, st)
            } else {
                (full_inputs.to_vec(), full_targets.to_vec())
            };

            // Speculative pretraining: score batch with reference model, skip if easy
            if let Some(ref ref_m) = ref_model {
                // Run reference forward + loss entirely in no_grad to avoid tape pollution
                let ref_loss_val = autograd::no_grad(|| {
                    ctx.begin_batch();
                    let ref_logits = ref_m.forward(&inputs, config.batch_size, effective_seq, None, false);
                    let (ref_loss, _) = loss::cross_entropy_loss(ctx, &ref_logits, &targets);
                    ctx.flush_batch();
                    let val = ref_loss.to_vec()[0];
                    // Drop all ref-model intermediate buffers before training forward
                    drop(ref_logits);
                    drop(ref_loss);
                    val
                });
                if ref_loss_val < config.speculative_threshold && ref_loss_val.is_finite() {
                    last_loss_tensor = None;
                    continue;
                }
            }

            // Forward pass (batched GPU dispatch — all kernels encode into one command buffer)
            ctx.begin_batch();

            // Two paths: standard (compute logits, then CE) or fused (LM head + CE in chunks)
            let n_predict = config.model_config.n_predict;

            let (loss_tensor, grad_logits) = if config.fused_ce && n_predict == 0 && config.teacher_checkpoint.is_none() {
                // FusedLinearCrossEntropy: compute logits+loss in chunks, never materialize full logit tensor.
                // Saves ~2GB peak memory. Incompatible with MTP and distillation (they need full logits).
                let hidden = model.forward_hidden(&inputs, config.batch_size, effective_seq, config.gradient_checkpointing);
                // For factored embedding, we need the un-factored embedding for the fused CE.
                // Use the full vocab embedding (either direct or projected).
                let embed_for_ce = if model.embed_rank > 0 {
                    // Factored: need embed_proj @ embedding for the effective [d_model, vocab] weight.
                    // FusedLinearCE handles matmul_trans_b internally, so pass [vocab, d_model].
                    // For factored case, fall back to standard CE (fused CE doesn't handle two-step projection yet).
                    let logits = model.apply_lm_head(&hidden);
                    loss::cross_entropy_loss_with_workspace(ctx, &logits, &targets, &loss_ws)
                } else {
                    loss::fused_linear_cross_entropy(ctx, &hidden, &model.embedding, &targets, 1024)
                };
                embed_for_ce
            } else {
                // Standard path: compute hidden → LM head → CE
                let (logits, extra_logits, hidden_for_distill) = if n_predict > 0 {
                    let (l, e) = model.forward_mtp(&inputs, config.batch_size, effective_seq, config.gradient_checkpointing);
                    (l, e, None)
                } else if config.ema_decay > 0.0 {
                    // When EMA active: use forward_hidden to get hidden states for self-distillation
                    let hidden = model.forward_hidden(&inputs, config.batch_size, effective_seq, config.gradient_checkpointing);
                    let logits = model.apply_lm_head(&hidden);
                    (logits, Vec::new(), Some(hidden))
                } else {
                    let l = model.forward(&inputs, config.batch_size, effective_seq, None, config.gradient_checkpointing);
                    (l, Vec::new(), None)
                };

                let (loss_tensor, grad_logits) = if let Some(ref teacher) = teacher_model {
                    let teacher_logits = autograd::no_grad(|| {
                        teacher.forward(&inputs, config.batch_size, effective_seq, None, false)
                    });
                    loss::distillation_loss(
                        ctx, &logits, &teacher_logits,
                        config.distill_temperature, config.distill_alpha, &targets,
                    )
                } else {
                    loss::cross_entropy_loss_with_workspace(ctx, &logits, &targets, &loss_ws)
                };

                // Z-loss: penalize large logit magnitudes (MoE stability)
                if config.z_loss_coefficient > 0.0 {
                    loss::z_loss(ctx, &logits, &loss_tensor.buffer, &grad_logits, config.z_loss_coefficient);
                }

                // EMA self-distillation: teacher logits from hidden @ ema_embedding^T
                // One extra matmul — no full teacher forward needed.
                // KL(teacher || student) as auxiliary loss with weight 0.1.
                // Self-distillation every 10 steps to amortize the cost (~40% overhead per step)
                if let Some(ref h) = hidden_for_distill {
                    if !ema_buffers.is_empty() && step % 10 == 0 {
                        // Self-distillation: compute teacher logits from hidden + EMA LM head.
                        // Uses EMA embedding (ema_buffers[0]) for the weight-tied LM head.
                        // One extra matmul — no full teacher forward needed.
                        let ema_embed = &ema_buffers[0];
                        let teacher_logits = autograd::no_grad(|| {
                            let vocab = config.model_config.vocab_size as usize;
                            let n = h.shape[0];
                            if model.embed_rank > 0 {
                                // Factored: h @ ema_embed_proj^T @ ema_embedding^T
                                let ema_proj = &ema_buffers[2]; // embed_proj is 3rd param
                                let d = config.model_config.d_model;
                                let r = model.embed_rank;
                                let h_proj_buf = ctx.alloc_buffer(n * r * 4);
                                compute::gpu_matmul_trans_b(ctx, &h.buffer, ema_proj, &h_proj_buf,
                                    n as u32, r as u32, d as u32);
                                let teacher_buf = ctx.alloc_buffer(n * vocab * 4);
                                compute::gpu_matmul_trans_b(ctx, &h_proj_buf, ema_embed, &teacher_buf,
                                    n as u32, vocab as u32, r as u32);
                                crate::tensor::Tensor::from_buffer(Arc::clone(&h.ctx), teacher_buf, vec![n, vocab])
                            } else {
                                let d = config.model_config.d_model;
                                let teacher_buf = ctx.alloc_buffer(n * vocab * 4);
                                compute::gpu_matmul_trans_b(ctx, &h.buffer, ema_embed, &teacher_buf,
                                    n as u32, vocab as u32, d as u32);
                                crate::tensor::Tensor::from_buffer(Arc::clone(&h.ctx), teacher_buf, vec![n, vocab])
                            }
                        });
                        // KL distillation: alpha=0.1, temperature=2.0
                        let (distill_loss, distill_grad) = loss::distillation_loss(
                            ctx, &logits, &teacher_logits, 2.0, 0.1, &targets,
                        );
                        // Add distillation loss to main loss for display
                        compute::gpu_axpy(ctx, &loss_tensor.buffer, &distill_loss.buffer, 1, 0.1);
                        // Add distillation gradient to CE gradient
                        compute::gpu_axpy(ctx, &grad_logits, &distill_grad,
                            (config.batch_size * effective_seq * config.model_config.vocab_size as usize) as u32, 0.1);
                    }
                }

                // Multi-token prediction: add loss from extra heads
                if !extra_logits.is_empty() {
                    let mtp_weight = 1.0 / (n_predict + 1) as f32;
                    compute::gpu_scale(ctx, &loss_tensor.buffer, 1, mtp_weight);
                    compute::gpu_scale(ctx, &grad_logits, (config.batch_size * effective_seq * config.model_config.vocab_size as usize) as u32, mtp_weight);

                    for (k, extra_log) in extra_logits.iter().enumerate() {
                        let shift = k + 1;
                        let bs = config.batch_size * effective_seq;
                        let mut shifted_targets = vec![0u32; bs];
                        for b in 0..config.batch_size {
                            for t in 0..effective_seq {
                                let src_idx = b * effective_seq + t + shift;
                                if t + shift < effective_seq {
                                    shifted_targets[b * effective_seq + t] = targets[src_idx];
                                }
                            }
                        }
                        let (extra_loss, _extra_grad) = loss::cross_entropy_loss(ctx, extra_log, &shifted_targets);
                        compute::gpu_scale(ctx, &extra_loss.buffer, 1, mtp_weight);
                        compute::gpu_add_inplace(ctx, &loss_tensor.buffer, &extra_loss.buffer, 1);
                    }
                }

                (loss_tensor, grad_logits)
            };

            // Online data pruning: skip backward if loss is below threshold.
            // The model already knows this data — training on it wastes compute.
            if config.prune_threshold > 0.0 {
                ctx.flush_batch(); // need to read loss value
                let loss_val = loss_tensor.to_vec()[0];
                if loss_val < config.prune_threshold && loss_val.is_finite() {
                    autograd::clear_tape();
                    autograd::clear_recompute_registry();
                    // Count as processed but skip gradient update
                    last_loss_tensor = Some(loss_tensor);
                    continue; // skip backward + optimizer for this micro-step
                }
                ctx.begin_batch(); // resume batch for backward
            }

            // Note: loss spikes from outlier data (loss > 50) are handled by gradient
            // clipping (max_grad_norm=1.0). Checking loss mid-step requires GPU sync
            // that breaks command batching and kills throughput. The gradient clipper
            // already limits per-step damage from any single batch.

            // Scale both loss AND gradient by 1/grad_accum_steps.
            if grad_accum_steps > 1 {
                let grad_size = (config.batch_size * effective_seq * config.model_config.vocab_size as usize) as u32;
                compute::gpu_scale(ctx, &grad_logits, grad_size, loss_scale);
                compute::gpu_scale(ctx, &loss_tensor.buffer, 1, loss_scale);
            }

            // Backward pass in the SAME command batch as forward — one fewer GPU sync
            autograd::backward(ctx, loss_tensor.id);
            // DON'T flush — gradient norm kernels encode into same batch below

            // Free the tape (activations) but keep accumulated gradients
            autograd::clear_tape_keep_grads();
            autograd::clear_recompute_registry();

            last_loss_tensor = Some(loss_tensor);
        }

        // Progressive layer freezing: zero gradients for bottom layers after training progresses.
        // Frozen layers still run forward but don't get weight updates → saves optimizer compute.
        if config.freeze_fraction > 0.0 {
            let progress = step as f32 / config.total_steps as f32;
            if progress > 0.25 { // start freezing after 25% of training
                let n_layers = model.blocks.len();
                let n_freeze = ((n_layers as f32 * config.freeze_fraction * progress.min(1.0)) as usize).min(n_layers - 1);
                if n_freeze > 0 {
                    // Zero gradients for frozen layer parameters
                    for block in model.blocks.iter().take(n_freeze) {
                        for param in block.parameters() {
                            if let Some(grad) = autograd::get_grad(param.id) {
                                compute::gpu_fill(ctx, &grad, param.numel() as u32, 0.0);
                            }
                        }
                    }
                }
            }
        }

        // Gradient clipping: norm computation fused into same batch as backward.
        clip_gradients_fused(ctx, &model, config.max_grad_norm);

        // Optimizer step: GPU (default) or CPU (Apple Silicon zero-copy)
        if lr > 1e-10 {
            if let Some(ref mut muon) = muon_opt {
                ctx.begin_batch();
                muon.step(lr);
            } else if let Some(ref mut soph) = sophia_opt {
                ctx.begin_batch();
                soph.step(lr);
            } else if config.optimizer_type == "adamw-cpu" {
                // CPU optimizer: runs on unified memory while GPU can start next forward.
                // Apple Silicon advantage: zero-copy, ~same speed as GPU for small param counts.
                optimizer.step_cpu(lr);
                ctx.begin_batch(); // start a dummy batch for the flush below
            } else {
                ctx.begin_batch();
                optimizer.step(lr);
            }
        } else {
            ctx.begin_batch();
        }
        // Anti-PGD noise: add perturbation to weights after optimizer step.
        // Alternating sign each step creates anticorrelated noise that navigates to flatter minima.
        // noise_t ≈ scale * ((-1)^step) * randn — simple sign-flip anticorrelation.
        if config.noise_scale > 0.0 && lr > 1e-10 {
            let sign = if step % 2 == 0 { config.noise_scale } else { -config.noise_scale };
            for param in &model.parameters() {
                let noise = crate::tensor::Tensor::randn(ctx, param.shape.clone(), sign * lr);
                compute::gpu_add_inplace(ctx, &param.buffer, &noise.buffer, param.numel() as u32);
            }
        }

        // EMA update: ema = decay * ema + (1-decay) * model_weights
        if config.ema_decay > 0.0 {
            for (ema_buf, param) in ema_buffers.iter().zip(model.parameters().iter()) {
                compute::gpu_ema_update(ctx, ema_buf, &param.buffer, param.numel() as u32, config.ema_decay);
            }
        }
        // Sync flush: wait for GPU to finish forward+backward+optimizer before reading any buffers.
        // Async flush caused race condition — checkpoint saves and loss reads saw stale data.
        ctx.flush_batch();

        // Zero gradients + invalidate FP16 weight cache (weights changed by optimizer)
        autograd::zero_grads();
        crate::tensor::Tensor::clear_f16_cache();

        let tokens_this_step = (config.batch_size * effective_seq * grad_accum_steps as usize) as u64;
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
            // Auto-detect loss spikes: if loss > 2× EMA, warn (may need lower LR)
            if ema_loss > 0.0 && loss_val > ema_loss * 2.0 && step > config.warmup_steps {
                eprintln!("[WARN] Loss spike: {:.4} > 2× EMA {:.4} at step {}. Consider lowering --lr.",
                    loss_val, ema_loss, step);
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

            // Track EMA loss and peak throughput
            if ema_loss == 0.0 { ema_loss = loss_val; }
            else { ema_loss = 0.95 * ema_loss + 0.05 * loss_val; }
            if tokens_per_sec > peak_tok_s { peak_tok_s = tokens_per_sec; }

            // ETA estimation
            let steps_done = (step - start_step + 1) as f32;
            let steps_remaining = config.total_steps.saturating_sub(step + 1) as f32;
            let avg_step_time = start_time.elapsed().as_secs_f32() / steps_done;
            let eta_secs = (steps_remaining * avg_step_time) as u64;
            let eta_str = if eta_secs > 3600 {
                format!("{}h{}m", eta_secs / 3600, (eta_secs % 3600) / 60)
            } else if eta_secs > 60 {
                format!("{}m{}s", eta_secs / 60, eta_secs % 60)
            } else {
                format!("{}s", eta_secs)
            };

            let loss_delta = if prev_loss > 0.0 { loss_val - prev_loss } else { 0.0 };
            prev_loss = loss_val;

            eprintln!(
                "step {:>6} | loss {:>8.4} ({:+.3}) | lr {:.2e} | {:.0} tok/s | {:.1}s/step | {}M tok | ep {} | pool {}/{} | ETA {} | w{:.1}",
                step, loss_val, loss_delta,
                lr, tokens_per_sec, step_time,
                total_tokens / 1_000_000,
                data_loader.epoch(),
                pool_hits, pool_hits + pool_misses,
                eta_str, weight_norm,
            );

            // Write to CSV log file
            let _ = writeln!(log_file, "{},{:.6},{:.6e},{:.1},{},{}", step, loss_val, lr, tokens_per_sec, elapsed, total_tokens);
        }

        // ReLoRA: periodically merge lowrank adapters into base weights, then reinitialize.
        // W_base += U @ V, reinit U/V. After K merges at rank r, effective rank ≈ K × r.
        if config.relora_interval > 0 && config.model_config.lowrank > 0
            && step > 0 && step % config.relora_interval == 0
        {
            let reinit_scale = 0.01;
            let mut n_merged = 0;

            ctx.begin_batch();
            for block in &model.blocks {
                n_merged += block.relora_merge(ctx, reinit_scale);
            }
            ctx.flush_batch();

            // Reset optimizer momentum/variance for the LoRA params (U, V).
            // Stale momentum from pre-merge would push the fresh adapters in the wrong direction.
            optimizer.reset_states_for_params(ctx, &model.parameters());

            eprintln!("[ReLoRA] Step {}: merged {} weight pairs across {} layers (reinit scale={})",
                step, n_merged, model.blocks.len(), reinit_scale);
        }

        // Checkpointing — save both model-only and full training state for resume
        // CRITICAL: Wait for GPU to finish the optimizer update before reading param buffers.
        // flush_batch_async() only commits — doesn't wait. Without this sync, checkpoint
        // reads stale (pre-optimizer) parameter values, producing corrupt checkpoints.
        if step > 0 && step % config.checkpoint_interval == 0 {
            ctx.wait_gpu();
            let path = format!("{}/step_{}.bin", config.checkpoint_dir, step);
            checkpoint::save_checkpoint(&path, &model, step)?;
            let state_path = format!("{}/state_{}.bin", config.checkpoint_dir, step);
            checkpoint::save_training_state(&state_path, &model, &optimizer, step, total_tokens)?;

            // Auto-save best model based on EMA loss
            if ema_loss > 0.0 && ema_loss < best_train_loss {
                best_train_loss = ema_loss;
                let best_path = format!("{}/best.bin", config.checkpoint_dir);
                checkpoint::save_checkpoint(&best_path, &model, step)?;
                eprintln!("  → New best model (EMA loss {:.4})", ema_loss);
            }
        }

        // Validation loss + early stopping (if validation dataset provided)
        if let Some(ref val_path) = config.val_dataset {
            if step > 0 && step % config.checkpoint_interval == 0 {
                let val_loss = compute_validation_loss(ctx, &model, val_path, config.batch_size, config.seq_len)?;
                let _ = writeln!(log_file, "# val_loss={:.6} at step {}", val_loss, step);
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    val_no_improve = 0;
                    eprintln!("  val_loss: {:.4} (new best)", val_loss);
                } else {
                    val_no_improve += 1;
                    eprintln!("  val_loss: {:.4} (no improve {}/{})", val_loss, val_no_improve, early_stop_patience);
                    if early_stop_patience > 0 && val_no_improve >= early_stop_patience {
                        eprintln!("Early stopping: val_loss didn't improve for {} checks", early_stop_patience);
                        break;
                    }
                }
            }
        }
    }

    // Final checkpoint — wait for GPU before reading weights
    ctx.wait_gpu();
    let path = format!("{}/final.bin", config.checkpoint_dir);
    checkpoint::save_checkpoint(&path, &model, config.total_steps)?;
    let state_path = format!("{}/state_final.bin", config.checkpoint_dir);
    checkpoint::save_training_state(&state_path, &model, &optimizer, config.total_steps, total_tokens)?;
    let total_time = start_time.elapsed().as_secs();
    let total_time_str = if total_time > 3600 {
        format!("{}h{}m", total_time / 3600, (total_time % 3600) / 60)
    } else { format!("{}m{}s", total_time / 60, total_time % 60) };
    let avg_tok_s = if total_time > 0 { total_tokens as f64 / total_time as f64 } else { 0.0 };
    let tok_per_day = avg_tok_s * 86400.0;
    eprintln!("Training complete. Final checkpoint: {}", path);
    eprintln!("=== Training Summary ===");
    eprintln!("  Total time: {}", total_time_str);
    eprintln!("  Total tokens: {}M ({:.1}B/day at avg throughput)", total_tokens / 1_000_000, tok_per_day / 1e9);
    eprintln!("  Peak throughput: {:.0} tok/s | Avg: {:.0} tok/s", peak_tok_s, avg_tok_s);
    eprintln!("  Final EMA loss: {:.4} | Best: {:.4}", ema_loss, best_train_loss);
    eprintln!("  Epochs: {} | Steps: {}", data_loader.epoch(), config.total_steps);
    eprintln!("  Model: {}M params, d={}, {}L, {} heads",
        config.model_config.param_count() as f32 / 1e6,
        config.model_config.d_model, config.model_config.n_layers, config.model_config.n_heads);

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
/// Fused variant: expects a command batch already open (from backward pass).
/// Encodes norm kernels into the existing batch — 1 sync for backward+norms.
fn clip_gradients_fused(ctx: &Arc<MetalContext>, model: &Transformer, max_norm: f32) {
    let params = model.parameters();
    let mut norm_bufs: Vec<Option<(objc2::rc::Retained<crate::metal::GpuBuffer>, usize)>> = Vec::with_capacity(params.len());

    // Encode norm kernels into the EXISTING batch (no begin_batch — reuses backward's)
    for param in &params {
        if let Some(grad) = autograd::get_grad(param.id) {
            let norm_out = ctx.alloc_buffer(std::mem::size_of::<f32>() * 2);
            compute::gpu_l2_norm_check_into(ctx, &grad, param.numel() as u32, &norm_out);
            norm_bufs.push(Some((norm_out, param.numel())));
        } else {
            norm_bufs.push(None);
        }
    }
    ctx.flush_batch(); // Single flush: backward + norms

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

    if !nan_indices.is_empty() {
        eprintln!("[WARN] NaN/Inf detected in {} gradient(s) (param indices: {:?}) — zeroing affected gradients",
            nan_indices.len(), &nan_indices[..nan_indices.len().min(10)]);
    }
    let needs_scale = total_norm > max_norm && total_norm.is_finite();
    let scale = if needs_scale { max_norm / (total_norm + 1e-6) } else { 1.0 };

    if !nan_indices.is_empty() || needs_scale {
        ctx.begin_batch();
        for &i in &nan_indices {
            if let Some(grad) = autograd::get_grad(params[i].id) {
                compute::gpu_fill(ctx, &grad, params[i].numel() as u32, 0.0);
            }
        }
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

/// Standalone clip_gradients — public so SFT/DPO can use the batched implementation.
pub fn clip_gradients(ctx: &Arc<MetalContext>, model: &Transformer, max_norm: f32) {
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
