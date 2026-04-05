use crate::autograd;
use crate::checkpoint;
use crate::metal::compute;
use crate::metal::MetalContext;
use crate::model::Transformer;
use crate::optim::{AdamW, CosineWarmupScheduler};
use crate::tokenizer::{BpeTokenizer, BOS_TOKEN, EOS_TOKEN};
use memmap2::Mmap;
use rand::seq::SliceRandom;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Binary dataset format
// ---------------------------------------------------------------------------

/// A single preference pair: prompt + chosen (winner) + rejected (loser).
pub struct PreferencePair {
    pub prompt: Vec<u32>,
    pub chosen: Vec<u32>,
    pub rejected: Vec<u32>,
}

/// Dataset of preference pairs loaded from a binary file via mmap.
///
/// Binary format:
/// ```text
/// [num_pairs: u32]
/// For each pair:
///   [prompt_len: u32] [prompt_tokens: u32 * prompt_len]
///   [chosen_len: u32] [chosen_tokens: u32 * chosen_len]
///   [rejected_len: u32] [rejected_tokens: u32 * rejected_len]
/// ```
pub struct DpoDataset {
    mmap: Mmap,
    /// Byte offsets into the mmap for the start of each pair.
    offsets: Vec<usize>,
    num_pairs: usize,
}

impl DpoDataset {
    /// Load a binary preference dataset from disk.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        assert!(mmap.len() >= 4, "DPO dataset file too small");
        let num_pairs = read_u32(&mmap, 0) as usize;

        // Build offset index by scanning through the file
        let mut offsets = Vec::with_capacity(num_pairs);
        let mut pos = 4; // skip num_pairs header

        for i in 0..num_pairs {
            offsets.push(pos);

            // prompt
            assert!(pos + 4 <= mmap.len(), "Truncated DPO dataset at pair {}", i);
            let prompt_len = read_u32(&mmap, pos) as usize;
            pos += 4 + prompt_len * 4;

            // chosen
            assert!(pos + 4 <= mmap.len(), "Truncated DPO dataset at pair {}", i);
            let chosen_len = read_u32(&mmap, pos) as usize;
            pos += 4 + chosen_len * 4;

            // rejected
            assert!(pos + 4 <= mmap.len(), "Truncated DPO dataset at pair {}", i);
            let rejected_len = read_u32(&mmap, pos) as usize;
            pos += 4 + rejected_len * 4;
        }

        eprintln!("DPO dataset loaded: {} preference pairs from {}", num_pairs, path);

        Ok(Self { mmap, offsets, num_pairs })
    }

    /// Get a preference pair by index.
    ///
    /// Validates all byte offsets against the mmap length before reading,
    /// preventing out-of-bounds access on truncated or malformed files.
    pub fn get_pair(&self, idx: usize) -> PreferencePair {
        assert!(idx < self.num_pairs, "Pair index {} out of bounds ({})", idx, self.num_pairs);
        let mmap_len = self.mmap.len();
        let mut pos = self.offsets[idx];

        // --- prompt ---
        assert!(pos + 4 <= mmap_len, "get_pair({}): prompt_len field at offset {} exceeds mmap ({})", idx, pos, mmap_len);
        let prompt_len = read_u32(&self.mmap, pos) as usize;
        pos += 4;
        assert!(pos + prompt_len * 4 <= mmap_len, "get_pair({}): prompt tokens ({}) at offset {} exceed mmap ({})", idx, prompt_len, pos, mmap_len);
        let prompt = read_u32_slice(&self.mmap, pos, prompt_len);
        pos += prompt_len * 4;

        // --- chosen ---
        assert!(pos + 4 <= mmap_len, "get_pair({}): chosen_len field at offset {} exceeds mmap ({})", idx, pos, mmap_len);
        let chosen_len = read_u32(&self.mmap, pos) as usize;
        pos += 4;
        assert!(pos + chosen_len * 4 <= mmap_len, "get_pair({}): chosen tokens ({}) at offset {} exceed mmap ({})", idx, chosen_len, pos, mmap_len);
        let chosen = read_u32_slice(&self.mmap, pos, chosen_len);
        pos += chosen_len * 4;

        // --- rejected ---
        assert!(pos + 4 <= mmap_len, "get_pair({}): rejected_len field at offset {} exceeds mmap ({})", idx, pos, mmap_len);
        let rejected_len = read_u32(&self.mmap, pos) as usize;
        pos += 4;
        assert!(pos + rejected_len * 4 <= mmap_len, "get_pair({}): rejected tokens ({}) at offset {} exceed mmap ({})", idx, rejected_len, pos, mmap_len);
        let rejected = read_u32_slice(&self.mmap, pos, rejected_len);
        let _ = pos; // suppress unused after last read

        PreferencePair { prompt, chosen, rejected }
    }

    /// Number of preference pairs in the dataset.
    pub fn len(&self) -> usize {
        self.num_pairs
    }
}

/// Read a little-endian u32 from a byte slice at the given offset.
/// Panics if `offset + 4` exceeds the slice length.
fn read_u32(data: &[u8], offset: usize) -> u32 {
    assert!(
        offset + 4 <= data.len(),
        "read_u32: offset {} + 4 exceeds mmap length {}",
        offset,
        data.len(),
    );
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Read a slice of u32 values from a byte slice.
/// Panics if the requested range exceeds the slice length.
fn read_u32_slice(data: &[u8], offset: usize, len: usize) -> Vec<u32> {
    let required_bytes = len * 4;
    assert!(
        offset + required_bytes <= data.len(),
        "read_u32_slice: offset {} + {} bytes exceeds mmap length {}",
        offset,
        required_bytes,
        data.len(),
    );
    (0..len).map(|i| read_u32(data, offset + i * 4)).collect()
}

// ---------------------------------------------------------------------------
// DPO data loader
// ---------------------------------------------------------------------------

/// Iterates over preference pairs in shuffled order.
pub struct DpoDataLoader {
    dataset: DpoDataset,
    order: Vec<usize>,
    position: usize,
    max_seq_len: usize,
    epoch: usize,
}

impl DpoDataLoader {
    pub fn new(dataset: DpoDataset, max_seq_len: usize) -> Self {
        let n = dataset.len();
        assert!(n > 0, "DPO dataset is empty");

        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        order.shuffle(&mut rng);

        Self {
            dataset,
            order,
            position: 0,
            max_seq_len,
            epoch: 0,
        }
    }

    /// Get the next preference pair, reshuffling at epoch boundaries.
    /// Returns (prompt_tokens, chosen_tokens, rejected_tokens) — all truncated
    /// to fit within max_seq_len when concatenated as prompt+response.
    pub fn next_pair(&mut self) -> PreferencePair {
        if self.position >= self.order.len() {
            self.position = 0;
            self.epoch += 1;
            let mut rng = rand::thread_rng();
            self.order.shuffle(&mut rng);
        }

        let idx = self.order[self.position];
        self.position += 1;

        let pair = self.dataset.get_pair(idx);

        // Truncate so that prompt + response fits in max_seq_len.
        // Strategy: keep as much prompt as needed, then truncate response.
        let max_response_len = self.max_seq_len.saturating_sub(pair.prompt.len());
        let chosen_trunc = pair.chosen.len().min(max_response_len);
        let rejected_trunc = pair.rejected.len().min(max_response_len);

        PreferencePair {
            prompt: pair.prompt.clone(),
            chosen: pair.chosen[..chosen_trunc].to_vec(),
            rejected: pair.rejected[..rejected_trunc].to_vec(),
        }
    }

    /// Current epoch.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Approximate batches (pairs) per epoch.
    pub fn pairs_per_epoch(&self) -> usize {
        self.dataset.len()
    }
}

// ---------------------------------------------------------------------------
// Sequence log-probabilities (CPU, operates on GPU logits read back)
// ---------------------------------------------------------------------------

/// Compute the sum of log P(token_i | tokens_0..i-1) for tokens after `prompt_len`.
///
/// logits_data: [seq_len, vocab_size] — the model's output logits (CPU readback).
/// tokens: the full input sequence (prompt + response).
/// prompt_len: index where the response begins.
///
/// For each position i >= prompt_len, we compute log_softmax(logits[i-1])[tokens[i]],
/// because logits[i-1] predicts the token at position i.
///
/// Returns the sum of per-token log probabilities (a negative number).
fn sequence_log_probs(logits_data: &[f32], vocab_size: usize, tokens: &[u32], prompt_len: usize) -> f32 {
    let seq_len = tokens.len();
    assert_eq!(logits_data.len(), seq_len * vocab_size);

    let mut total_logp: f32 = 0.0;

    for i in prompt_len..seq_len {
        // logits[i-1] predicts token at position i
        let logit_row_start = (i - 1) * vocab_size;
        let logit_row = &logits_data[logit_row_start..logit_row_start + vocab_size];

        // Numerically stable log-softmax: log_softmax(x)_j = x_j - log(sum(exp(x - max(x))))
        let max_logit = logit_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = logit_row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln();

        let target_token = tokens[i] as usize;
        let log_prob = logit_row[target_token] - max_logit - log_sum_exp;
        total_logp += log_prob;
    }

    total_logp
}

// ---------------------------------------------------------------------------
// DPO training configuration
// ---------------------------------------------------------------------------

pub struct DpoConfig {
    /// Path to the policy model checkpoint (will be fine-tuned).
    pub checkpoint_path: String,
    /// Path to the reference model checkpoint (frozen).
    pub ref_checkpoint_path: String,
    /// Path to the tokenizer.
    pub tokenizer_path: String,
    /// Path to the binary preference dataset.
    pub data_path: String,
    /// Output directory for DPO checkpoints.
    pub output_dir: String,
    /// DPO temperature parameter (beta). Controls divergence from reference.
    /// Lower = more conservative. Typical: 0.1 - 0.5.
    pub beta: f32,
    /// Learning rate.
    pub learning_rate: f32,
    /// Maximum sequence length (prompt + response).
    pub max_seq_len: usize,
    /// Number of training steps.
    pub total_steps: u32,
    /// Warmup steps for learning rate scheduler.
    pub warmup_steps: u32,
    /// Weight decay for AdamW.
    pub weight_decay: f32,
    /// Max gradient norm for clipping.
    pub max_grad_norm: f32,
    /// Log every N steps.
    pub log_interval: u32,
    /// Save checkpoint every N steps.
    pub checkpoint_interval: u32,
}

impl DpoConfig {
    pub fn default_dpo(
        checkpoint_path: &str,
        ref_checkpoint_path: &str,
        tokenizer_path: &str,
        data_path: &str,
    ) -> Self {
        Self {
            checkpoint_path: checkpoint_path.to_string(),
            ref_checkpoint_path: ref_checkpoint_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            data_path: data_path.to_string(),
            output_dir: "dpo_checkpoints".to_string(),
            beta: 0.1,
            learning_rate: 1e-6,
            max_seq_len: 512,
            total_steps: 1000,
            warmup_steps: 100,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            log_interval: 10,
            checkpoint_interval: 500,
        }
    }
}

// ---------------------------------------------------------------------------
// DPO training loop
// ---------------------------------------------------------------------------

/// Run Direct Preference Optimization training.
///
/// DPO (Rafailov et al., 2023) trains a policy model to prefer chosen responses
/// over rejected ones, using a frozen reference model as an anchor.
///
/// Loss: L_DPO = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
/// where log_ratio = log pi(y|x) - log pi_ref(y|x)
pub fn dpo_train(ctx: &Arc<MetalContext>, config: &DpoConfig) -> std::io::Result<()> {
    eprintln!("=== AndreAI Direct Preference Optimization ===");

    // Load policy model (will be updated)
    let (policy_model, pretrain_step) = checkpoint::load_checkpoint(ctx, &config.checkpoint_path)?;
    eprintln!(
        "Policy model loaded: step {}, {}M params, {} layers, d_model={}, {} heads",
        pretrain_step,
        policy_model.config.param_count() as f32 / 1e6,
        policy_model.config.n_layers,
        policy_model.config.d_model,
        policy_model.config.n_heads
    );

    // Load reference model (frozen, never updated)
    let (ref_model, ref_step) = checkpoint::load_checkpoint(ctx, &config.ref_checkpoint_path)?;
    eprintln!(
        "Reference model loaded: step {}, {}M params",
        ref_step,
        ref_model.config.param_count() as f32 / 1e6
    );

    // Verify models have the same architecture
    assert_eq!(
        policy_model.config.vocab_size, ref_model.config.vocab_size,
        "Policy and reference models must have the same vocab_size"
    );
    assert_eq!(
        policy_model.config.d_model, ref_model.config.d_model,
        "Policy and reference models must have the same d_model"
    );
    assert_eq!(
        policy_model.config.n_layers, ref_model.config.n_layers,
        "Policy and reference models must have the same n_layers"
    );

    // Load dataset
    let dataset = DpoDataset::load(&config.data_path)?;
    let mut data_loader = DpoDataLoader::new(dataset, config.max_seq_len);

    eprintln!("Tokenizer: {}", config.tokenizer_path);
    eprintln!(
        "DPO: beta={}, lr={:.1e}, max_seq_len={}, total_steps={}",
        config.beta, config.learning_rate, config.max_seq_len, config.total_steps
    );
    eprintln!(
        "Dataset: {} preference pairs, ~{} pairs/epoch",
        data_loader.pairs_per_epoch(),
        data_loader.pairs_per_epoch()
    );

    // Create output directory
    std::fs::create_dir_all(&config.output_dir)?;

    // Initialize optimizer on the policy model
    let param_refs: Vec<&_> = policy_model.parameters().into_iter().collect();
    let mut optimizer = AdamW::new(ctx, &param_refs, config.weight_decay);

    let scheduler = CosineWarmupScheduler::new(
        config.learning_rate,
        config.warmup_steps,
        config.total_steps,
    );

    let vocab_size = policy_model.config.vocab_size as usize;
    let start_time = Instant::now();
    let mut total_pairs: u64 = 0;

    for step in 0..config.total_steps {
        let step_start = Instant::now();
        let lr = scheduler.get_lr(step);

        // Get next preference pair
        let pair = data_loader.next_pair();
        let prompt_len = pair.prompt.len();

        // Build full sequences: prompt + response
        let chosen_input: Vec<u32> = pair.prompt.iter()
            .chain(pair.chosen.iter())
            .copied()
            .collect();
        let rejected_input: Vec<u32> = pair.prompt.iter()
            .chain(pair.rejected.iter())
            .copied()
            .collect();

        let chosen_seq_len = chosen_input.len();
        let rejected_seq_len = rejected_input.len();

        // Skip degenerate pairs (empty responses)
        if chosen_seq_len <= prompt_len || rejected_seq_len <= prompt_len {
            continue;
        }

        // --- Forward pass: policy on chosen ---
        ctx.begin_batch();
        let policy_chosen_logits = policy_model.forward(
            &chosen_input, 1, chosen_seq_len, None, false,
        );
        ctx.flush_batch();
        let policy_chosen_logits_data = policy_chosen_logits.to_vec();

        // --- Forward pass: policy on rejected ---
        ctx.begin_batch();
        let policy_rejected_logits = policy_model.forward(
            &rejected_input, 1, rejected_seq_len, None, false,
        );
        ctx.flush_batch();
        let policy_rejected_logits_data = policy_rejected_logits.to_vec();

        // --- Forward pass: reference on chosen (no grad) ---
        let ref_chosen_logits_data = autograd::no_grad(|| {
            ctx.begin_batch();
            let logits = ref_model.forward(
                &chosen_input, 1, chosen_seq_len, None, false,
            );
            ctx.flush_batch();
            logits.to_vec()
        });

        // --- Forward pass: reference on rejected (no grad) ---
        let ref_rejected_logits_data = autograd::no_grad(|| {
            ctx.begin_batch();
            let logits = ref_model.forward(
                &rejected_input, 1, rejected_seq_len, None, false,
            );
            ctx.flush_batch();
            logits.to_vec()
        });

        // --- Compute sequence log-probabilities (response tokens only) ---
        let policy_chosen_logps = sequence_log_probs(
            &policy_chosen_logits_data, vocab_size, &chosen_input, prompt_len,
        );
        let policy_rejected_logps = sequence_log_probs(
            &policy_rejected_logits_data, vocab_size, &rejected_input, prompt_len,
        );
        let ref_chosen_logps = sequence_log_probs(
            &ref_chosen_logits_data, vocab_size, &chosen_input, prompt_len,
        );
        let ref_rejected_logps = sequence_log_probs(
            &ref_rejected_logits_data, vocab_size, &rejected_input, prompt_len,
        );

        // --- Compute DPO loss and gradients ---
        // log-ratio for chosen: log pi(y_w|x) - log pi_ref(y_w|x)
        let chosen_log_ratio = policy_chosen_logps - ref_chosen_logps;
        // log-ratio for rejected: log pi(y_l|x) - log pi_ref(y_l|x)
        let rejected_log_ratio = policy_rejected_logps - ref_rejected_logps;

        // DPO implicit reward difference
        let reward_diff = config.beta * (chosen_log_ratio - rejected_log_ratio);

        // Loss = -log sigmoid(reward_diff) = log(1 + exp(-reward_diff))
        let loss_val = log1p_exp(-reward_diff);

        // Gradient of DPO w.r.t. policy log-probs:
        // sigmoid(-reward_diff) is the "weight" — how much to push
        let sigmoid_neg = sigmoid(-reward_diff);

        // dL/d(policy_chosen_logps) = -beta * sigmoid(-reward_diff)
        let grad_chosen_scale = -config.beta * sigmoid_neg;
        // dL/d(policy_rejected_logps) = beta * sigmoid(-reward_diff)
        let grad_rejected_scale = config.beta * sigmoid_neg;

        // --- Backward: inject DPO gradients into the policy model ---
        // We need to backpropagate through the policy forward passes.
        // The chain rule: dL/d(logits) = dL/d(logps) * d(logps)/d(logits)
        //
        // For each response position i, d(log_softmax(logits[i-1])[t_i])/d(logits[i-1][j]):
        //   = (1{j == t_i} - softmax(logits[i-1])[j])
        //
        // We compute the gradient w.r.t. logits on CPU then upload to GPU for backward.

        // Backward through chosen path
        let chosen_grad_logits = compute_dpo_logit_gradients(
            &policy_chosen_logits_data,
            vocab_size,
            &chosen_input,
            prompt_len,
            grad_chosen_scale,
        );

        // Upload gradient and run backward through policy model (chosen)
        // We replay the forward pass tape, injecting our gradient at the loss node.
        // Strategy: rerun forward for chosen, then compute CE-like loss with the DPO gradient.
        autograd::clear_tape();

        ctx.begin_batch();
        let policy_chosen_logits_2 = policy_model.forward(
            &chosen_input, 1, chosen_seq_len, None, false,
        );
        ctx.flush_batch();

        // Inject the DPO gradient as the loss gradient for backward
        let grad_buf = ctx.buffer_from_slice(&chosen_grad_logits);
        let chosen_loss_id = inject_loss_gradient(ctx, &policy_chosen_logits_2, grad_buf);

        ctx.begin_batch();
        autograd::backward(ctx, chosen_loss_id);
        ctx.flush_batch();

        // Store chosen gradients, then do rejected pass
        // We need to accumulate gradients from both chosen and rejected paths.
        // autograd::backward accumulates into the same GRADS map, so we do
        // the rejected forward+backward without clearing grads.
        autograd::clear_tape_keep_grads();

        // Backward through rejected path
        let rejected_grad_logits = compute_dpo_logit_gradients(
            &policy_rejected_logits_data,
            vocab_size,
            &rejected_input,
            prompt_len,
            grad_rejected_scale,
        );

        ctx.begin_batch();
        let policy_rejected_logits_2 = policy_model.forward(
            &rejected_input, 1, rejected_seq_len, None, false,
        );
        ctx.flush_batch();

        let grad_buf_rej = ctx.buffer_from_slice(&rejected_grad_logits);
        let rejected_loss_id = inject_loss_gradient(ctx, &policy_rejected_logits_2, grad_buf_rej);

        ctx.begin_batch();
        autograd::backward(ctx, rejected_loss_id);
        ctx.flush_batch();

        // --- Average gradients from chosen + rejected backward passes ---
        autograd::scale_grads(ctx, 0.5);

        // --- Gradient clipping + optimizer step ---
        clip_gradients_dpo(ctx, &policy_model, config.max_grad_norm);

        ctx.begin_batch();
        if lr > 1e-10 {
            optimizer.step(lr);
        }
        ctx.flush_batch();

        // Clear tape, recycle gradient + cache buffers to pool
        autograd::zero_grads_recycle();
        crate::tensor::Tensor::clear_f16_cache_recycle();
        autograd::clear_tape();
        autograd::clear_recompute_registry();
        total_pairs += 1;

        // --- Logging ---
        if step % config.log_interval == 0 {
            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!(
                    "FATAL: DPO loss is {} at step {}. Training diverged.",
                    loss_val, step
                );
                eprintln!("Try: lower --lr or --beta.");
                break;
            }

            let step_time = step_start.elapsed().as_secs_f32();
            let elapsed = start_time.elapsed().as_secs();

            // Accuracy: fraction where the policy prefers chosen over rejected
            let accuracy = if reward_diff > 0.0 { 1.0 } else { 0.0 };

            eprintln!(
                "dpo {:>6} | loss {:>8.4} | acc {:>3.0}% | reward_diff {:>+8.4} | lr {:.2e} | {:.1}s/step | {}s elapsed | {} pairs | epoch {}",
                step,
                loss_val,
                accuracy * 100.0,
                reward_diff,
                lr,
                step_time,
                elapsed,
                total_pairs,
                data_loader.epoch(),
            );
        }

        // --- Checkpointing ---
        if step > 0 && step % config.checkpoint_interval == 0 {
            let path = format!("{}/dpo_step_{}.bin", config.output_dir, step);
            checkpoint::save_checkpoint(&path, &policy_model, pretrain_step + step)?;
            eprintln!("DPO checkpoint saved: {}", path);
        }
    }

    // Final checkpoint
    let path = format!("{}/dpo_final.bin", config.output_dir);
    checkpoint::save_checkpoint(&path, &policy_model, pretrain_step + config.total_steps)?;
    eprintln!("DPO training complete. Final checkpoint: {}", path);

    Ok(())
}

/// Compute the gradient of the DPO loss w.r.t. model logits.
///
/// For each response position i (>= prompt_len), the gradient of log P(t_i | context)
/// w.r.t. logits[i-1] is: (1{j == t_i} - softmax(logits[i-1])[j])
///
/// Multiplied by the DPO gradient scale (grad_scale), this gives:
///   d_logits[i-1][j] = grad_scale * (1{j == t_i} - softmax(logits[i-1])[j])
///
/// Prompt positions get zero gradient.
fn compute_dpo_logit_gradients(
    logits_data: &[f32],
    vocab_size: usize,
    tokens: &[u32],
    prompt_len: usize,
    grad_scale: f32,
) -> Vec<f32> {
    let seq_len = tokens.len();
    let mut grad = vec![0.0f32; seq_len * vocab_size];

    for i in prompt_len..seq_len {
        let logit_row_start = (i - 1) * vocab_size;
        let logit_row = &logits_data[logit_row_start..logit_row_start + vocab_size];

        // Compute softmax for this position
        let max_logit = logit_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logit_row.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let inv_sum = 1.0 / sum_exp;

        let target_token = tokens[i] as usize;
        let grad_row_start = (i - 1) * vocab_size;

        for j in 0..vocab_size {
            let softmax_j = exps[j] * inv_sum;
            let indicator = if j == target_token { 1.0 } else { 0.0 };
            grad[grad_row_start + j] = grad_scale * (indicator - softmax_j);
        }
    }

    grad
}

/// Inject a pre-computed gradient buffer as the loss gradient for backward propagation.
/// This records a synthetic CrossEntropy tape entry whose cached gradient is our DPO gradient.
/// Inject a pre-computed gradient as a synthetic loss node on the tape.
/// Returns the loss tensor ID — backward MUST be called with this ID,
/// not the logits ID, otherwise the gradient is never visited.
fn inject_loss_gradient(
    ctx: &Arc<MetalContext>,
    logits: &crate::tensor::Tensor,
    grad_buf: objc2::rc::Retained<crate::metal::GpuBuffer>,
) -> usize {
    use crate::autograd::{Op, TapeEntry};

    // Create a scalar "loss" tensor for backward to target
    let loss_buf = ctx.alloc_buffer(4);
    compute::gpu_fill(ctx, &loss_buf, 1, 0.0); // value doesn't matter, only gradient does

    let loss_id = autograd::next_id();

    autograd::record(TapeEntry {
        op: Op::CrossEntropy,
        inputs: vec![logits.id],
        output: loss_id,
        input_buffers: vec![logits.buffer.clone()],
        output_buffer: loss_buf,
        shapes: vec![logits.shape.clone(), vec![1]],
        cached: Some(grad_buf),
    });

    loss_id
}

/// Clip gradients — delegates to the shared batched implementation.
fn clip_gradients_dpo(ctx: &Arc<MetalContext>, model: &Transformer, max_norm: f32) {
    crate::train::clip_gradients(ctx, model, max_norm);
}

/// Numerically stable log(1 + exp(x)).
fn log1p_exp(x: f32) -> f32 {
    if x > 20.0 {
        x // for large x, log(1 + exp(x)) ≈ x
    } else if x < -20.0 {
        0.0 // for very negative x, log(1 + exp(x)) ≈ 0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Standard sigmoid function.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ---------------------------------------------------------------------------
// Data preparation: JSONL → binary format
// ---------------------------------------------------------------------------

/// Convert a JSONL file of preference pairs to the binary DPO format.
///
/// Input JSONL format:
/// ```json
/// {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
/// ```
///
/// Output: binary file with the format described in DpoDataset.
pub fn prepare_dpo_dataset(
    input_path: &str,
    output_path: &str,
    tokenizer: &BpeTokenizer,
) -> std::io::Result<usize> {
    let content = std::fs::read_to_string(input_path)?;
    let mut pairs: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let prompt_str = extract_json_string(line, "prompt").unwrap_or_else(|| {
            panic!(
                "DPO JSONL line {} missing \"prompt\" field: {}",
                line_num + 1,
                &line[..line.len().min(80)]
            )
        });
        let chosen_str = extract_json_string(line, "chosen").unwrap_or_else(|| {
            panic!(
                "DPO JSONL line {} missing \"chosen\" field: {}",
                line_num + 1,
                &line[..line.len().min(80)]
            )
        });
        let rejected_str = extract_json_string(line, "rejected").unwrap_or_else(|| {
            panic!(
                "DPO JSONL line {} missing \"rejected\" field: {}",
                line_num + 1,
                &line[..line.len().min(80)]
            )
        });

        // Tokenize with BOS prefix on prompt
        let formatted_prompt = format!("User: {}\nAssistant: ", prompt_str);
        let mut prompt_tokens = vec![BOS_TOKEN];
        prompt_tokens.extend(tokenizer.encode(&formatted_prompt));

        let mut chosen_tokens = tokenizer.encode(&chosen_str);
        chosen_tokens.push(EOS_TOKEN);

        let mut rejected_tokens = tokenizer.encode(&rejected_str);
        rejected_tokens.push(EOS_TOKEN);

        pairs.push((prompt_tokens, chosen_tokens, rejected_tokens));
    }

    // Write binary format
    let mut file = std::fs::File::create(output_path)?;
    let num_pairs = pairs.len() as u32;
    file.write_all(&num_pairs.to_le_bytes())?;

    for (prompt, chosen, rejected) in &pairs {
        // prompt
        let prompt_len = prompt.len() as u32;
        file.write_all(&prompt_len.to_le_bytes())?;
        for &t in prompt {
            file.write_all(&t.to_le_bytes())?;
        }

        // chosen
        let chosen_len = chosen.len() as u32;
        file.write_all(&chosen_len.to_le_bytes())?;
        for &t in chosen {
            file.write_all(&t.to_le_bytes())?;
        }

        // rejected
        let rejected_len = rejected.len() as u32;
        file.write_all(&rejected_len.to_le_bytes())?;
        for &t in rejected {
            file.write_all(&t.to_le_bytes())?;
        }
    }

    eprintln!(
        "DPO dataset prepared: {} preference pairs from {} → {}",
        pairs.len(), input_path, output_path
    );

    Ok(pairs.len())
}

/// Extract a string value for a given key from a JSON object string.
/// Handles escaped quotes and basic escape sequences.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    let key_start = json.find(&search)?;
    let after_key = &json[key_start + search.len()..];

    let after_colon = after_key.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();
    let after_colon = after_colon.strip_prefix('"')?;

    let mut result = String::new();
    let mut chars = after_colon.chars();
    loop {
        match chars.next() {
            None => return None,
            Some('"') => break,
            Some('\\') => match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('b') => result.push('\u{0008}'),
                Some('f') => result.push('\u{000C}'),
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('/') => result.push('/'),
                Some('u') => {
                    // Parse \uXXXX unicode escape
                    let mut hex = String::with_capacity(4);
                    for _ in 0..4 {
                        match chars.next() {
                            Some(h) if h.is_ascii_hexdigit() => hex.push(h),
                            _ => return None, // malformed \uXXXX
                        }
                    }
                    let codepoint = u32::from_str_radix(&hex, 16).ok()?;
                    // Handle UTF-16 surrogate pairs: \uD800-\uDBFF followed by \uDC00-\uDFFF
                    if (0xD800..=0xDBFF).contains(&codepoint) {
                        // High surrogate — expect \uDCxx low surrogate
                        if chars.next() != Some('\\') || chars.next() != Some('u') {
                            return None;
                        }
                        let mut hex2 = String::with_capacity(4);
                        for _ in 0..4 {
                            match chars.next() {
                                Some(h) if h.is_ascii_hexdigit() => hex2.push(h),
                                _ => return None,
                            }
                        }
                        let low = u32::from_str_radix(&hex2, 16).ok()?;
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return None; // invalid surrogate pair
                        }
                        let combined = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                        result.push(char::from_u32(combined)?);
                    } else {
                        result.push(char::from_u32(codepoint)?);
                    }
                }
                Some(c) => {
                    // Unknown escape: preserve literally (non-standard but lenient)
                    result.push('\\');
                    result.push(c);
                }
                None => return None,
            },
            Some(c) => result.push(c),
        }
    }

    Some(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd;
    use crate::metal::MetalContext;
    use crate::model::{ModelConfig, Transformer};
    use std::io::Write;

    fn test_ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(10.0) - 1.0).abs() < 1e-4);
        assert!(sigmoid(-10.0) < 1e-4);
        // Symmetry
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log1p_exp() {
        // log(1 + exp(0)) = log(2)
        assert!((log1p_exp(0.0) - 2.0f32.ln()).abs() < 1e-6);
        // Large positive: log(1 + exp(100)) ≈ 100
        assert!((log1p_exp(100.0) - 100.0).abs() < 1e-3);
        // Large negative: log(1 + exp(-100)) ≈ 0
        assert!(log1p_exp(-100.0) < 1e-6);
    }

    #[test]
    fn test_sequence_log_probs_basic() {
        // 3 positions, vocab size 4
        // Position 0 predicts position 1, position 1 predicts position 2
        let vocab_size = 4;
        let seq_len = 3;
        let mut logits = vec![0.0f32; seq_len * vocab_size];

        // logits[0] = [0, 0, 10, 0] — strongly predicts token 2
        logits[2] = 10.0;
        // logits[1] = [0, 0, 0, 10] — strongly predicts token 3
        logits[1 * vocab_size + 3] = 10.0;

        // tokens = [A, B, C] (len=3), logits = [3, vocab_size]
        // prompt_len = 1: compute log-probs for positions 1 and 2.
        // logits[0] predicts tokens[1], logits[1] predicts tokens[2].
        let tokens = vec![99u32, 2, 3]; // prompt_len = 1
        let logps = sequence_log_probs(&logits, vocab_size, &tokens, 1);

        // logits[0] has strong peak at 2, tokens[1] = 2 → high log-prob
        // logits[1] has strong peak at 3, tokens[2] = 3 → high log-prob
        // Both should be close to 0 (log of ~1.0)
        assert!(logps > -1.0, "Expected high log-probs, got {}", logps);
        assert!(logps < 0.0, "Log-probs should be negative, got {}", logps);
    }

    #[test]
    fn test_dpo_binary_dataset_roundtrip() {
        let dir = std::env::temp_dir().join("andreai_dpo_test");
        std::fs::create_dir_all(&dir).unwrap();
        let bin_path = dir.join("test_prefs.bin");

        // Write a binary dataset with 2 pairs
        let mut file = std::fs::File::create(&bin_path).unwrap();
        let num_pairs: u32 = 2;
        file.write_all(&num_pairs.to_le_bytes()).unwrap();

        // Pair 0: prompt=[1,2], chosen=[3,4,5], rejected=[6]
        write_tokens(&mut file, &[1, 2]);
        write_tokens(&mut file, &[3, 4, 5]);
        write_tokens(&mut file, &[6]);

        // Pair 1: prompt=[10], chosen=[20], rejected=[30, 40]
        write_tokens(&mut file, &[10]);
        write_tokens(&mut file, &[20]);
        write_tokens(&mut file, &[30, 40]);

        drop(file);

        // Load and verify
        let dataset = DpoDataset::load(bin_path.to_str().unwrap()).unwrap();
        assert_eq!(dataset.len(), 2);

        let p0 = dataset.get_pair(0);
        assert_eq!(p0.prompt, vec![1, 2]);
        assert_eq!(p0.chosen, vec![3, 4, 5]);
        assert_eq!(p0.rejected, vec![6]);

        let p1 = dataset.get_pair(1);
        assert_eq!(p1.prompt, vec![10]);
        assert_eq!(p1.chosen, vec![20]);
        assert_eq!(p1.rejected, vec![30, 40]);

        std::fs::remove_dir_all(&dir).ok();
    }

    fn write_tokens(file: &mut std::fs::File, tokens: &[u32]) {
        let len = tokens.len() as u32;
        file.write_all(&len.to_le_bytes()).unwrap();
        for &t in tokens {
            file.write_all(&t.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn test_dpo_jsonl_prepare_roundtrip() {
        let dir = std::env::temp_dir().join("andreai_dpo_jsonl_test");
        std::fs::create_dir_all(&dir).unwrap();
        let jsonl_path = dir.join("prefs.jsonl");
        let bin_path = dir.join("prefs.bin");

        // Write JSONL
        let mut f = std::fs::File::create(&jsonl_path).unwrap();
        writeln!(f, r#"{{"prompt": "What is 1+1?", "chosen": "2", "rejected": "3"}}"#).unwrap();
        writeln!(f, r#"{{"prompt": "Hello", "chosen": "Hi there!", "rejected": "Go away"}}"#).unwrap();
        drop(f);

        // We need a tokenizer — create a minimal one
        // Use byte-level encoding for simplicity in tests
        let tok = BpeTokenizer::train(b"What is 1+1? 2 3 Hello Hi there! Go away User: Assistant:", 300);
        let count = prepare_dpo_dataset(
            jsonl_path.to_str().unwrap(),
            bin_path.to_str().unwrap(),
            &tok,
        ).unwrap();
        assert_eq!(count, 2);

        // Verify the binary file is loadable
        let dataset = DpoDataset::load(bin_path.to_str().unwrap()).unwrap();
        assert_eq!(dataset.len(), 2);

        // Prompt should start with BOS
        let p0 = dataset.get_pair(0);
        assert_eq!(p0.prompt[0], BOS_TOKEN);
        // Chosen and rejected should end with EOS
        assert_eq!(*p0.chosen.last().unwrap(), EOS_TOKEN);
        assert_eq!(*p0.rejected.last().unwrap(), EOS_TOKEN);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_extract_json_string_dpo() {
        let json = r#"{"prompt": "test prompt", "chosen": "good answer", "rejected": "bad answer"}"#;
        assert_eq!(extract_json_string(json, "prompt"), Some("test prompt".to_string()));
        assert_eq!(extract_json_string(json, "chosen"), Some("good answer".to_string()));
        assert_eq!(extract_json_string(json, "rejected"), Some("bad answer".to_string()));
        assert_eq!(extract_json_string(json, "missing"), None);
    }

    #[test]
    fn test_dpo_gradient_computation() {
        // Verify that compute_dpo_logit_gradients produces sensible values
        let vocab_size = 4;
        let seq_len = 3;
        let prompt_len = 1;

        // logits: uniform distribution
        let logits = vec![0.0f32; seq_len * vocab_size];
        let tokens = vec![0u32, 1, 2]; // prompt=[0], response=[1, 2]

        let grad_scale = -0.1; // negative = push probs up (chosen)
        let grad = compute_dpo_logit_gradients(&logits, vocab_size, &tokens, prompt_len, grad_scale);

        assert_eq!(grad.len(), seq_len * vocab_size);

        // Row 0 (position 0 predicts position 1 = token 1):
        // softmax is uniform: 1/4 for each
        // grad[0][1] = grad_scale * (1 - 0.25) = -0.1 * 0.75 = -0.075
        // grad[0][j!=1] = grad_scale * (0 - 0.25) = -0.1 * (-0.25) = 0.025
        let g00 = grad[0 * vocab_size + 0]; // j=0, not target
        let g01 = grad[0 * vocab_size + 1]; // j=1, IS target
        assert!((g01 - (-0.075)).abs() < 1e-5, "Expected -0.075, got {}", g01);
        assert!((g00 - 0.025).abs() < 1e-5, "Expected 0.025, got {}", g00);

        // Prompt position gradients should be zero
        // Actually prompt_len=1, so position 0 in tokens is prompt.
        // Response starts at position 1. logits[0] predicts tokens[1].
        // logits[1] predicts tokens[2]. These are the only nonzero rows.
        // Row 2 (logits[2]) is not used (no token at position 3).
        // So row 2 should be zero.
        for j in 0..vocab_size {
            assert_eq!(grad[2 * vocab_size + j], 0.0);
        }
    }

    #[test]
    fn test_dpo_single_step() {
        let ctx = test_ctx();

        // Create a tiny model
        let config = ModelConfig::custom(64, 32, 2, 2, 2.67, 64);
        let model = Transformer::new(&ctx, config.clone());

        // Capture initial weights
        let initial_weights: Vec<f32> = model.parameters()[0].to_vec();

        // Create a preference pair
        let prompt = vec![1u32, 2, 3];
        let chosen = vec![4u32, 5];
        let rejected = vec![6u32, 7];
        let prompt_len = prompt.len();

        let chosen_input: Vec<u32> = prompt.iter().chain(chosen.iter()).copied().collect();
        let rejected_input: Vec<u32> = prompt.iter().chain(rejected.iter()).copied().collect();

        let vocab_size = config.vocab_size as usize;
        let beta = 0.1f32;

        // --- Forward passes ---
        ctx.begin_batch();
        let policy_chosen_logits = model.forward(&chosen_input, 1, chosen_input.len(), None, false);
        ctx.flush_batch();
        let policy_chosen_data = policy_chosen_logits.to_vec();

        ctx.begin_batch();
        let policy_rejected_logits = model.forward(&rejected_input, 1, rejected_input.len(), None, false);
        ctx.flush_batch();
        let policy_rejected_data = policy_rejected_logits.to_vec();

        // Use same model as ref (before any updates)
        let ref_chosen_data = policy_chosen_data.clone();
        let ref_rejected_data = policy_rejected_data.clone();

        // Compute log-probs
        let policy_chosen_logps = sequence_log_probs(&policy_chosen_data, vocab_size, &chosen_input, prompt_len);
        let policy_rejected_logps = sequence_log_probs(&policy_rejected_data, vocab_size, &rejected_input, prompt_len);
        let ref_chosen_logps = sequence_log_probs(&ref_chosen_data, vocab_size, &chosen_input, prompt_len);
        let ref_rejected_logps = sequence_log_probs(&ref_rejected_data, vocab_size, &rejected_input, prompt_len);

        // DPO loss
        let chosen_log_ratio = policy_chosen_logps - ref_chosen_logps;
        let rejected_log_ratio = policy_rejected_logps - ref_rejected_logps;
        let reward_diff = beta * (chosen_log_ratio - rejected_log_ratio);
        let loss_val = log1p_exp(-reward_diff);

        // Verify loss is finite and positive
        assert!(loss_val.is_finite(), "DPO loss should be finite, got {}", loss_val);
        assert!(loss_val > 0.0, "DPO loss should be positive, got {}", loss_val);

        // When policy == ref, log ratios are 0, reward_diff = 0,
        // loss = log(1 + exp(0)) = log(2) ≈ 0.693
        assert!((loss_val - 2.0f32.ln()).abs() < 1e-4,
            "With identical policy/ref, DPO loss should be log(2) ≈ 0.693, got {}", loss_val);

        // --- Backward and optimizer step ---
        let sigmoid_neg = sigmoid(-reward_diff);
        let grad_chosen_scale = -beta * sigmoid_neg;
        let grad_rejected_scale = beta * sigmoid_neg;

        // Chosen backward
        autograd::clear_tape();
        let chosen_grad_logits = compute_dpo_logit_gradients(
            &policy_chosen_data, vocab_size, &chosen_input, prompt_len, grad_chosen_scale,
        );

        ctx.begin_batch();
        let logits_2 = model.forward(&chosen_input, 1, chosen_input.len(), None, false);
        ctx.flush_batch();

        let grad_buf = ctx.buffer_from_slice(&chosen_grad_logits);
        let chosen_loss_id = inject_loss_gradient(&ctx, &logits_2, grad_buf);

        ctx.begin_batch();
        autograd::backward(&ctx, chosen_loss_id);
        ctx.flush_batch();

        autograd::clear_tape_keep_grads();

        // Rejected backward
        let rejected_grad_logits = compute_dpo_logit_gradients(
            &policy_rejected_data, vocab_size, &rejected_input, prompt_len, grad_rejected_scale,
        );

        ctx.begin_batch();
        let logits_3 = model.forward(&rejected_input, 1, rejected_input.len(), None, false);
        ctx.flush_batch();

        let grad_buf_rej = ctx.buffer_from_slice(&rejected_grad_logits);
        let rejected_loss_id = inject_loss_gradient(&ctx, &logits_3, grad_buf_rej);

        ctx.begin_batch();
        autograd::backward(&ctx, rejected_loss_id);
        ctx.flush_batch();

        // Verify gradients exist
        let has_nonzero_grad = model.parameters().iter().any(|p| {
            autograd::get_grad(p.id).is_some_and(|g| {
                let vals = MetalContext::read_buffer(&g, p.numel());
                vals.iter().any(|&v| v != 0.0)
            })
        });
        assert!(has_nonzero_grad, "At least one parameter should have non-zero gradient");

        // Optimizer step
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let mut optimizer = crate::optim::AdamW::new(&ctx, &param_refs, 0.01);

        ctx.begin_batch();
        optimizer.step(1e-4);
        optimizer.zero_grad();
        ctx.flush_batch();

        autograd::clear_tape();

        // Verify weights changed
        let updated_weights: Vec<f32> = model.parameters()[0].to_vec();
        let weights_changed = initial_weights.iter().zip(updated_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(weights_changed, "Policy model weights should have changed after DPO step");
    }
}
