use crate::autograd;
use crate::checkpoint;
use crate::gpu::MetalContext;
use crate::gpu::compute;
use crate::loss;
use crate::model::Transformer;
use crate::optim::{AdamW, CosineWarmupScheduler};
use crate::tensor::Tensor;
use crate::tokenizer::{BOS_TOKEN, BpeTokenizer, EOS_TOKEN, PAD_TOKEN};
use rand::seq::SliceRandom;
use std::io::{Error, ErrorKind};
use std::sync::Arc;
use std::time::Instant;

/// A single supervised fine-tuning example: tokenized sequence with a boundary
/// marking where the assistant's response starts. Loss is only computed on
/// tokens at and after `response_start`.
pub struct SftExample {
    pub tokens: Vec<u32>,
    pub response_start: usize,
}

/// Dataset of instruction-response pairs loaded from JSONL.
/// Each line: `{"prompt": "...", "response": "..."}`
pub struct SftDataset {
    pub examples: Vec<SftExample>,
}

impl SftDataset {
    /// Load an SFT dataset from a JSONL file. Each line must be valid JSON
    /// with `"prompt"` and `"response"` string fields.
    ///
    /// Format per example:
    ///   `<|bos|>User: {prompt}\nAssistant: {response}<|eos|>`
    pub fn load(path: &str, tokenizer: &BpeTokenizer) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut examples = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let (prompt, response) = parse_jsonl_line(line, line_num)?;

            let formatted_prompt = format!("User: {}\nAssistant: ", prompt);
            let prompt_tokens = tokenizer.encode(&formatted_prompt);
            let response_tokens = tokenizer.encode(&response);

            // Build full sequence: BOS + prompt_tokens + response_tokens + EOS
            let mut tokens =
                Vec::with_capacity(1 + prompt_tokens.len() + response_tokens.len() + 1);
            tokens.push(BOS_TOKEN);
            tokens.extend_from_slice(&prompt_tokens);
            let response_start = tokens.len(); // assistant response begins here
            tokens.extend_from_slice(&response_tokens);
            tokens.push(EOS_TOKEN);

            examples.push(SftExample {
                tokens,
                response_start,
            });
        }

        eprintln!(
            "SFT dataset loaded: {} examples from {}",
            examples.len(),
            path
        );

        if let Some(ex) = examples.first() {
            eprintln!(
                "  first example: {} tokens, response starts at position {}",
                ex.tokens.len(),
                ex.response_start
            );
        }

        if examples.is_empty() {
            return Err(invalid_data("SFT dataset is empty"));
        }

        Ok(Self { examples })
    }
}

/// Minimal JSON parser for `{"prompt": "...", "response": "..."}` lines.
/// Avoids adding a serde dependency. Handles escaped quotes within strings.
fn parse_jsonl_line(line: &str, line_num: usize) -> std::io::Result<(String, String)> {
    let prompt = extract_json_string(line, "prompt").ok_or_else(|| {
        invalid_data(format!(
            "SFT JSONL line {} missing \"prompt\" field: {}",
            line_num + 1,
            crate::truncate_on_char_boundary(line, 80)
        ))
    });
    let response = extract_json_string(line, "response").ok_or_else(|| {
        invalid_data(format!(
            "SFT JSONL line {} missing \"response\" field: {}",
            line_num + 1,
            crate::truncate_on_char_boundary(line, 80)
        ))
    });
    Ok((prompt?, response?))
}

/// Extract a string value for a given key from a JSON object string.
/// Handles escaped quotes and basic escape sequences.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    // Look for "key": "value" pattern
    let search = format!("\"{}\"", key);
    let key_start = json.find(&search)?;
    let after_key = &json[key_start + search.len()..];

    // Skip whitespace and colon
    let after_colon = after_key.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    // Must start with a quote
    let after_colon = after_colon.strip_prefix('"')?;

    // Read until unescaped closing quote
    let mut result = String::new();
    let mut chars = after_colon.chars();
    loop {
        match chars.next() {
            None => return None, // unterminated string
            Some('"') => break,
            Some('\\') => {
                match chars.next() {
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
                }
            }
            Some(c) => result.push(c),
        }
    }

    Some(result)
}

/// Data loader for SFT training. Iterates over examples in shuffled order,
/// packing them into fixed-size batches with padding and loss masks.
pub struct SftDataLoader {
    examples: Vec<SftExample>,
    order: Vec<usize>,
    position: usize,
    batch_size: usize,
    max_seq_len: usize,
    epoch: usize,
}

impl SftDataLoader {
    pub fn new(
        dataset: SftDataset,
        batch_size: usize,
        max_seq_len: usize,
    ) -> std::io::Result<Self> {
        let max_tokens = max_seq_len
            .checked_add(1)
            .ok_or_else(|| invalid_input("seq_len + 1 overflows usize"))?;
        let original_examples = dataset.examples.len();
        let examples: Vec<SftExample> = dataset
            .examples
            .into_iter()
            .filter(|example| example.tokens.len().min(max_tokens) > example.response_start)
            .collect();
        let n = examples.len();
        if n < batch_size {
            return Err(invalid_data(format!(
                "SFT dataset has {n} usable examples at seq_len {max_seq_len} but batch_size is {batch_size}"
            )));
        }
        if n < original_examples {
            eprintln!(
                "SFT dataset: skipped {} examples whose responses are fully truncated at seq_len {}",
                original_examples - n,
                max_seq_len
            );
        }

        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        order.shuffle(&mut rng);

        Ok(Self {
            examples,
            order,
            position: 0,
            batch_size,
            max_seq_len,
            epoch: 0,
        })
    }

    /// Get next batch: (input_tokens, target_tokens, loss_mask).
    ///
    /// - `input_tokens`: `[batch_size * max_seq_len]` — token IDs, right-padded with PAD_TOKEN.
    /// - `target_tokens`: `[batch_size * max_seq_len]` — shifted right by 1, padded with PAD_TOKEN.
    /// - `loss_mask`: `[batch_size * max_seq_len]` — `true` only for positions where the target
    ///   is part of the assistant's response (>= response_start in the original sequence).
    pub fn next_batch(&mut self) -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let total = self.batch_size * self.max_seq_len;
        let mut inputs = vec![PAD_TOKEN; total];
        let mut targets = vec![PAD_TOKEN; total];
        let mut mask = vec![false; total];

        for b in 0..self.batch_size {
            // Wrap around and reshuffle if needed
            if self.position >= self.order.len() {
                self.position = 0;
                self.epoch += 1;
                let mut rng = rand::thread_rng();
                self.order.shuffle(&mut rng);
            }

            let example_idx = self.order[self.position];
            self.position += 1;

            let example = &self.examples[example_idx];
            // Truncate to max_seq_len + 1 (need one extra for shifted target)
            let usable_len = example.tokens.len().min(self.max_seq_len + 1);
            let seq_len = usable_len.saturating_sub(1).min(self.max_seq_len);

            let base = b * self.max_seq_len;

            for i in 0..seq_len {
                inputs[base + i] = example.tokens[i];
                targets[base + i] = example.tokens[i + 1];

                // Loss mask: only compute loss where the target position is
                // within the assistant's response region. Position i+1 in the
                // original token array corresponds to the target at position i.
                // The assistant's response starts at example.response_start, so
                // the first target token we want loss on is at position
                // response_start - 1 in the input (predicting token at response_start).
                if i + 1 >= example.response_start {
                    mask[base + i] = true;
                }
            }
        }

        (inputs, targets, mask)
    }

    /// Current epoch (number of full passes through the dataset).
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Approximate batches per epoch.
    pub fn batches_per_epoch(&self) -> usize {
        self.examples.len() / self.batch_size
    }

    /// Total examples in the dataset.
    pub fn total_examples(&self) -> usize {
        self.examples.len()
    }
}

/// SFT training configuration.
pub struct SftConfig {
    pub checkpoint_path: String,
    pub tokenizer_path: String,
    pub data_path: String,
    pub output_dir: String,
    pub batch_size: usize,
    pub seq_len: usize,
    pub total_steps: u32,
    pub max_lr: f32,
    pub warmup_steps: u32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub log_interval: u32,
    pub checkpoint_interval: u32,
    /// Self-distillation strength (0.0 = disabled). When > 0, each step runs a
    /// second (teacher) forward pass with hints inserted at response positions,
    /// and the loss becomes `CE + alpha * KL(teacher, student)`.
    pub self_distill_alpha: f32,
    /// Optional textual-feedback samples for localized credit assignment. When
    /// non-empty, the teacher input is built from these samples (inserting the
    /// hint at the target turn) instead of the generic response-position hint.
    pub self_distill_samples: Vec<crate::self_distill::TextualFeedbackSample>,
}

impl SftConfig {
    pub fn default_sft(checkpoint_path: &str, tokenizer_path: &str, data_path: &str) -> Self {
        Self {
            checkpoint_path: checkpoint_path.to_string(),
            tokenizer_path: tokenizer_path.to_string(),
            data_path: data_path.to_string(),
            output_dir: "sft_checkpoints".to_string(),
            batch_size: 8,
            seq_len: 256,
            total_steps: 1000,
            max_lr: 2e-5,
            warmup_steps: 100,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            log_interval: 10,
            checkpoint_interval: 500,
            self_distill_alpha: 0.0,
            self_distill_samples: Vec::new(),
        }
    }

    pub fn validate(&self) -> std::io::Result<()> {
        validate_non_empty("checkpoint_path", &self.checkpoint_path)?;
        validate_non_empty("tokenizer_path", &self.tokenizer_path)?;
        validate_non_empty("data_path", &self.data_path)?;
        validate_non_empty("output_dir", &self.output_dir)?;
        validate_positive_usize("batch_size", self.batch_size)?;
        validate_positive_usize("seq_len", self.seq_len)?;
        validate_positive_u32("total_steps", self.total_steps)?;
        validate_positive_u32("log_interval", self.log_interval)?;
        validate_positive_u32("checkpoint_interval", self.checkpoint_interval)?;
        validate_finite_non_negative("max_lr", self.max_lr)?;
        validate_finite_non_negative("weight_decay", self.weight_decay)?;
        validate_finite_positive("max_grad_norm", self.max_grad_norm)?;
        self.seq_len
            .checked_add(1)
            .ok_or_else(|| invalid_input("seq_len + 1 overflows usize"))?;
        self.batch_size
            .checked_mul(self.seq_len)
            .ok_or_else(|| invalid_input("batch_size * seq_len overflows usize"))?;
        Ok(())
    }
}

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

fn validate_non_empty(field: &str, value: &str) -> std::io::Result<()> {
    if value.is_empty() {
        Err(invalid_input(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_positive_usize(field: &str, value: usize) -> std::io::Result<()> {
    if value > 0 {
        Ok(())
    } else {
        Err(invalid_input(format!("{field} must be greater than 0")))
    }
}

fn validate_positive_u32(field: &str, value: u32) -> std::io::Result<()> {
    if value > 0 {
        Ok(())
    } else {
        Err(invalid_input(format!("{field} must be greater than 0")))
    }
}

fn validate_finite_positive(field: &str, value: f32) -> std::io::Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(format!("{field} must be finite and > 0")))
    }
}

fn validate_finite_non_negative(field: &str, value: f32) -> std::io::Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(format!("{field} must be finite and >= 0")))
    }
}

/// Apply a loss mask to the cross-entropy gradient buffer.
/// For every position where `mask[i]` is false, zeros out the corresponding
/// row in the gradient buffer (row i of shape [batch*seq, vocab]).
/// This prevents gradient flow from prompt tokens — only assistant response
/// tokens contribute to the loss.
fn apply_loss_mask(
    ctx: &Arc<MetalContext>,
    grad_logits: &crate::gpu::GpuBuffer,
    mask: &[bool],
    vocab_size: usize,
) -> f32 {
    let total_positions = mask.len();
    let masked_count = mask.iter().filter(|&&m| !m).count();
    let unmasked_count = total_positions - masked_count;

    // Build u32 mask on CPU (cheap: one u32 per position, not per vocab element)
    let mask_u32: Vec<u32> = mask.iter().map(|&m| if m { 1u32 } else { 0u32 }).collect();
    let mask_buf = ctx.buffer_from_u32_slice(&mask_u32);

    // Zero out masked gradient rows entirely on GPU. No CPU roundtrip of the
    // large [positions * vocab] gradient buffer.
    compute::gpu_gradient_mask(
        ctx,
        grad_logits,
        &mask_buf,
        total_positions as u32,
        vocab_size as u32,
    );

    // Rescale unmasked rows: original CE divides by total_positions, we want
    // to divide by unmasked_count. Scale by total_positions / unmasked_count.
    if unmasked_count > 0 && unmasked_count != total_positions {
        let rescale = total_positions as f32 / unmasked_count as f32;
        compute::gpu_scale(
            ctx,
            grad_logits,
            (total_positions * vocab_size) as u32,
            rescale,
        );
    }

    // Report fraction of unmasked positions for logging
    unmasked_count as f32 / total_positions as f32
}

/// Build a teacher input by replacing response positions with the correct target
/// tokens. The teacher sees the same context as the student, but at every
/// response position it also sees the *correct next token* — this creates a
/// "what if the model had known the right answer" distribution for KL
/// distillation. Positions where `loss_mask` is false (prompt tokens) are left
/// unchanged.
///
/// This is the generic path used when `self_distill_samples` is empty. When
/// samples are provided, `build_teacher_input_from_sample` is used instead for
/// finer-grained per-turn control.
fn build_teacher_input(inputs: &[u32], targets: &[u32], loss_mask: &[bool]) -> Vec<u32> {
    assert_eq!(inputs.len(), targets.len());
    assert_eq!(inputs.len(), loss_mask.len());
    inputs
        .iter()
        .enumerate()
        .map(|(i, &tok)| if loss_mask[i] { targets[i] } else { tok })
        .collect()
}

/// Run supervised fine-tuning on a pre-trained checkpoint.
pub fn sft_train(ctx: &Arc<MetalContext>, config: &SftConfig) -> std::io::Result<()> {
    config.validate()?;
    eprintln!("=== Smedjan Supervised Fine-Tuning ===");

    // Load pre-trained checkpoint
    let (model, pretrain_step) = checkpoint::load_checkpoint(ctx, &config.checkpoint_path)?;
    if config.seq_len > model.config.max_seq_len {
        return Err(invalid_input(format!(
            "seq_len {} exceeds checkpoint max_seq_len {}",
            config.seq_len, model.config.max_seq_len
        )));
    }
    eprintln!(
        "Loaded pre-trained model: step {}, {}M params, {} layers, d_model={}, {} heads",
        pretrain_step,
        model.config.param_count() as f32 / 1e6,
        model.config.n_layers,
        model.config.d_model,
        model.config.n_heads
    );

    // Load tokenizer and SFT dataset
    let tokenizer = BpeTokenizer::load(&config.tokenizer_path).map_err(|e| {
        Error::new(
            e.kind(),
            format!(
                "Failed to load tokenizer '{}': {}",
                config.tokenizer_path, e
            ),
        )
    })?;
    let dataset = SftDataset::load(&config.data_path, &tokenizer)?;
    let mut data_loader = SftDataLoader::new(dataset, config.batch_size, config.seq_len)?;

    eprintln!(
        "SFT: batch_size={}, seq_len={}, total_steps={}, lr={:.1e}",
        config.batch_size, config.seq_len, config.total_steps, config.max_lr
    );
    eprintln!(
        "Dataset: {} examples, ~{} batches/epoch",
        data_loader.total_examples(),
        data_loader.batches_per_epoch()
    );

    // Create output directory
    std::fs::create_dir_all(&config.output_dir)?;

    // Initialize optimizer on the pre-trained weights
    let param_refs: Vec<&_> = model.parameters().into_iter().collect();
    let mut optimizer = AdamW::new(ctx, &param_refs, config.weight_decay);

    let scheduler =
        CosineWarmupScheduler::new(config.max_lr, config.warmup_steps, config.total_steps);

    let vocab_size = model.config.vocab_size as usize;
    let mut total_tokens: u64 = 0;
    let start_time = Instant::now();

    for step in 0..config.total_steps {
        let step_start = Instant::now();
        let lr = scheduler.get_lr(step);

        // Get SFT batch with loss mask
        let (inputs, targets, loss_mask) = data_loader.next_batch();

        // --- Self-distillation: teacher forward (detached, no grad) ---
        // When self_distill_alpha > 0, run a second forward pass with hints
        // inserted at response positions. The teacher logits form the target
        // distribution for KL distillation, giving per-turn credit assignment.
        let mut targets = targets;
        let teacher_logits: Option<Tensor> = if config.self_distill_alpha > 0.0 {
            let (teacher_inputs, sample_targets) = if config.self_distill_samples.is_empty() {
                (build_teacher_input(&inputs, &targets, &loss_mask), None)
            } else {
                // Use one sample per step (rotated). The sample defines both the
                // teacher input (hint inserted at the target turn) and the correct
                // target tokens for the CE loss.
                let sample =
                    &config.self_distill_samples[step as usize % config.self_distill_samples.len()];
                let teacher_input = crate::self_distill::build_teacher_input_from_sample(
                    sample,
                    config.batch_size * config.seq_len,
                );
                // Tile the sample's target tokens across the batch.
                let total = config.batch_size * config.seq_len;
                let tiled: Vec<u32> = (0..total)
                    .map(|i| sample.target_token_ids[i % sample.target_token_ids.len()])
                    .collect();
                (teacher_input, Some(tiled))
            };
            if let Some(st) = sample_targets {
                targets = st;
            }
            Some(autograd::no_grad(|| {
                model.forward(
                    &teacher_inputs,
                    config.batch_size,
                    config.seq_len,
                    None,
                    false,
                )
            }))
        } else {
            None
        };

        // --- Student forward (with grad) ---
        ctx.begin_batch();
        let logits = model.forward(
            &inputs,
            config.batch_size,
            config.seq_len,
            None,
            false, // no gradient checkpointing for SFT (small datasets)
        );

        // Loss: CE (+ alpha * KL if teacher is active)
        let (loss_tensor, grad_logits) = match &teacher_logits {
            Some(teacher) => crate::self_distill::textual_feedback_loss(
                ctx,
                &logits,
                teacher,
                &targets,
                &loss_mask,
                config.self_distill_alpha,
            ),
            None => loss::cross_entropy_loss(ctx, &logits, &targets),
        };
        ctx.flush_batch();

        // Apply loss mask: zero out gradients for prompt positions, rescale
        let response_frac = apply_loss_mask(ctx, &grad_logits, &loss_mask, vocab_size);

        // Backward pass (batched — uses the masked gradient buffer via tape)
        ctx.begin_batch();
        autograd::backward(ctx, loss_tensor.id);
        ctx.flush_batch();

        // Gradient clipping (handles its own batching internally)
        clip_gradients_sft(ctx, &model, config.max_grad_norm);

        // Optimizer step (batched — one sync for all params instead of per-param)
        ctx.begin_batch();
        if lr > 1e-10 {
            optimizer.step(lr);
        }
        ctx.flush_batch();

        // Clear tape and gradient buffers, recycle to pool
        autograd::zero_grads_recycle();
        crate::tensor::Tensor::clear_f16_cache_recycle();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let tokens_this_step = (config.batch_size * config.seq_len) as u64;
        total_tokens += tokens_this_step;

        let loss_val = loss_tensor.to_vec()[0];
        if loss_val.is_nan() || loss_val.is_infinite() {
            eprintln!(
                "FATAL: SFT loss is {} at step {}. Training diverged.",
                loss_val, step
            );
            eprintln!("Try: lower --lr or check SFT data quality.");
            return Err(Error::other(format!(
                "SFT loss became non-finite at step {step}: {loss_val}"
            )));
        }

        // Logging
        if step % config.log_interval == 0 {
            let step_time = step_start.elapsed().as_secs_f32();
            let tokens_per_sec = tokens_this_step as f32 / step_time;
            let elapsed = start_time.elapsed().as_secs();
            let (tape_ops, tape_bytes) = autograd::tape_stats();

            if step == 0 {
                eprintln!(
                    "Tape: {} ops, {:.1} MB activation memory",
                    tape_ops,
                    tape_bytes as f64 / (1024.0 * 1024.0)
                );
            }

            eprintln!(
                "sft {:>6} | loss {:>8.4} | lr {:.2e} | resp {:.0}% | {:.0} tok/s | {:.1}s/step | {}s elapsed | {}M tokens | epoch {}",
                step,
                loss_val,
                lr,
                response_frac * 100.0,
                tokens_per_sec,
                step_time,
                elapsed,
                total_tokens / 1_000_000,
                data_loader.epoch(),
            );
        }

        // Checkpointing
        if step > 0 && step % config.checkpoint_interval == 0 {
            let path = format!("{}/sft_step_{}.bin", config.output_dir, step);
            checkpoint::save_checkpoint(&path, &model, pretrain_step + step)?;
            eprintln!("SFT checkpoint saved: {}", path);
        }
    }

    // Final checkpoint
    let path = format!("{}/sft_final.bin", config.output_dir);
    checkpoint::save_checkpoint(&path, &model, pretrain_step + config.total_steps)?;
    eprintln!("SFT complete. Final checkpoint: {}", path);

    Ok(())
}

/// Clip gradients — delegates to the shared batched implementation.
fn clip_gradients_sft(ctx: &Arc<MetalContext>, model: &Transformer, max_norm: f32) {
    crate::train::clip_gradients(ctx, model, max_norm);
}

/// Convert NL2Bash-style data into JSONL SFT format.
///
/// Input format (one pair per two consecutive lines):
/// ```text
/// List all files in current directory
/// ls -la
/// Show disk usage
/// df -h
/// ```
///
/// Or tab-separated:
/// ```text
/// List all files\tls -la
/// ```
///
/// Output: JSONL with `{"prompt": "...", "response": "..."}` per line.
pub fn generate_sft_dataset(input_path: &str, output_path: &str) -> std::io::Result<usize> {
    let content = std::fs::read_to_string(input_path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut out = std::fs::File::create(output_path)?;
    let mut count = 0;

    // Try tab-separated format first: each line has prompt\tresponse
    let is_tsv = lines.first().is_some_and(|l| l.contains('\t'));

    if is_tsv {
        for line in &lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((prompt, response)) = line.split_once('\t') {
                let prompt = prompt.trim();
                let response = response.trim();
                if !prompt.is_empty() && !response.is_empty() {
                    write_jsonl_line(&mut out, prompt, response)?;
                    count += 1;
                }
            }
        }
    } else {
        // Two-line format: prompt on odd lines, response on even lines
        let mut i = 0;
        while i + 1 < lines.len() {
            let prompt = lines[i].trim();
            let response = lines[i + 1].trim();
            if !prompt.is_empty() && !response.is_empty() {
                write_jsonl_line(&mut out, prompt, response)?;
                count += 1;
            }
            i += 2;
        }
    }

    eprintln!(
        "Generated SFT dataset: {} pairs from {} → {}",
        count, input_path, output_path
    );
    Ok(count)
}

/// Write a single JSONL line with proper escaping.
fn write_jsonl_line(out: &mut std::fs::File, prompt: &str, response: &str) -> std::io::Result<()> {
    use std::io::Write;
    writeln!(
        out,
        "{{\"prompt\": \"{}\", \"response\": \"{}\"}}",
        escape_json(prompt),
        escape_json(response)
    )
}

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\x20' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_string_basic() {
        let json = r#"{"prompt": "hello world", "response": "hi there"}"#;
        assert_eq!(
            extract_json_string(json, "prompt"),
            Some("hello world".to_string())
        );
        assert_eq!(
            extract_json_string(json, "response"),
            Some("hi there".to_string())
        );
    }

    #[test]
    fn test_extract_json_string_escaped() {
        let json = r#"{"prompt": "say \"hello\"", "response": "line1\nline2"}"#;
        assert_eq!(
            extract_json_string(json, "prompt"),
            Some("say \"hello\"".to_string())
        );
        assert_eq!(
            extract_json_string(json, "response"),
            Some("line1\nline2".to_string())
        );
    }

    #[test]
    fn test_extract_json_string_missing() {
        let json = r#"{"prompt": "test"}"#;
        assert_eq!(extract_json_string(json, "response"), None);
    }

    #[test]
    fn test_escape_json_roundtrip() {
        let input = "hello \"world\"\nfoo\\bar";
        let escaped = escape_json(input);
        assert_eq!(escaped, r#"hello \"world\"\nfoo\\bar"#);
    }

    #[test]
    fn test_extract_json_string_unicode_escape() {
        // Basic \uXXXX escape
        let json = r#"{"text": "caf\u00e9"}"#;
        assert_eq!(
            extract_json_string(json, "text"),
            Some("caf\u{00e9}".to_string())
        );

        // Surrogate pair: U+1F600 (grinning face) = \uD83D\uDE00
        let json = r#"{"text": "hi \uD83D\uDE00"}"#;
        assert_eq!(
            extract_json_string(json, "text"),
            Some("hi \u{1F600}".to_string())
        );

        // Malformed: incomplete hex digits
        let json = r#"{"text": "bad \u00z9"}"#;
        assert_eq!(extract_json_string(json, "text"), None);
    }

    #[test]
    fn sft_config_rejects_invalid_runtime_values() {
        fn expect_invalid<F>(needle: &str, mutate: F)
        where
            F: FnOnce(&mut SftConfig),
        {
            let mut cfg = SftConfig::default_sft("checkpoint.bin", "tokenizer.bin", "sft.jsonl");
            mutate(&mut cfg);
            let err = cfg.validate().expect_err("invalid SFT config should fail");
            assert_eq!(err.kind(), ErrorKind::InvalidInput);
            assert!(
                err.to_string().contains(needle),
                "expected error containing '{needle}', got '{err}'"
            );
        }

        expect_invalid("checkpoint_path", |c| c.checkpoint_path.clear());
        expect_invalid("batch_size", |c| c.batch_size = 0);
        expect_invalid("seq_len", |c| c.seq_len = 0);
        expect_invalid("total_steps", |c| c.total_steps = 0);
        expect_invalid("log_interval", |c| c.log_interval = 0);
        expect_invalid("checkpoint_interval", |c| c.checkpoint_interval = 0);
        expect_invalid("max_lr", |c| c.max_lr = f32::NAN);
        expect_invalid("max_grad_norm", |c| c.max_grad_norm = 0.0);
        expect_invalid("batch_size * seq_len", |c| {
            c.batch_size = usize::MAX;
            c.seq_len = 2;
        });
    }

    #[test]
    fn sft_dataset_reports_malformed_jsonl_without_panic() {
        let dir = std::env::temp_dir().join("smedjan_sft_bad_jsonl_test");
        std::fs::create_dir_all(&dir).unwrap();
        let jsonl_path = dir.join("bad.jsonl");
        std::fs::write(&jsonl_path, r#"{"prompt":"hello"}"#).unwrap();
        let tok = BpeTokenizer::train(b"hello response User: Assistant:", 300);

        let err = match SftDataset::load(jsonl_path.to_str().unwrap(), &tok) {
            Ok(_) => panic!("missing response should fail"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("missing \"response\""),
            "unexpected error: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sft_dataloader_rejects_fully_truncated_responses() {
        let dataset = SftDataset {
            examples: vec![SftExample {
                tokens: vec![BOS_TOKEN, 42, EOS_TOKEN],
                response_start: 2,
            }],
        };

        let err = match SftDataLoader::new(dataset, 1, 1) {
            Ok(_) => panic!("fully truncated response should fail"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("usable examples"),
            "unexpected error: {err}"
        );
    }
}
