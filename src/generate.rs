use crate::attention::KvCache;
use crate::autograd;
use crate::metal::compute::{gpu_argmax, gpu_temperature_scale};
use crate::metal::MetalContext;
use crate::model::Transformer;
use crate::tokenizer::{BpeTokenizer, BOS_TOKEN, EOS_TOKEN};
use rand::Rng;
use std::sync::Arc;

/// Sampling configuration.
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            max_tokens: 256,
        }
    }
}

/// Generate text autoregressively with KV cache.
pub fn generate(
    ctx: &Arc<MetalContext>,
    model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &SamplingConfig,
) -> String {
    eprintln!("Generating on {} (temp={}, top_p={}, top_k={}, max_tokens={})",
        ctx.device_name(), config.temperature, config.top_p, config.top_k, config.max_tokens);
    autograd::no_grad(|| {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(tokenizer.encode(prompt));

        let mut kv_caches = model.init_kv_caches_preallocated(1);

        // Prefill: process entire prompt at once
        let batch = 1;
        let seq_len = tokens.len();
        ctx.begin_batch();
        let logits = model.forward(&tokens, batch, seq_len, Some(&mut kv_caches), false);
        ctx.flush_batch();

        // Get last token's logits for next prediction
        let vocab_size = model.config.vocab_size as usize;
        let greedy = config.temperature < 0.01;

        let mut next_token = if greedy {
            // PERF-5: For prefill, the logits buffer contains seq_len * vocab_size values.
            // We need argmax of the last token's logits only, so we read back just that slice.
            // This is a one-time cost; the hot loop below uses gpu_argmax on single-token logits.
            let all_logits = logits.to_vec();
            let last_logits = &all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
            sample_token(last_logits, config)
        } else {
            let all_logits = logits.to_vec();
            let last_logits = &all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
            sample_token(last_logits, config)
        };

        let mut generated = Vec::new();
        generated.push(next_token);

        // Autoregressive generation with KV cache
        for _ in 1..config.max_tokens {
            if next_token == EOS_TOKEN {
                break;
            }

            // Forward pass on single token (KV cache handles the context)
            ctx.begin_batch();
            let logits = model.forward(&[next_token], 1, 1, Some(&mut kv_caches), false);
            ctx.flush_batch();

            next_token = if greedy {
                // PERF-5: GPU-side argmax — reads back 4 bytes instead of 128KB (vocab_size * 4)
                gpu_argmax(ctx, &logits.buffer, vocab_size as u32)
            } else {
                // Temperature scaling on GPU, then read back for CPU sampling
                gpu_temperature_scale(ctx, &logits.buffer, 0, vocab_size as u32, config.temperature);
                // Zero-copy: shared memory on Apple Silicon means direct pointer access
                let token_logits = &logits.as_slice()[..vocab_size];
                sample_token_prescaled(token_logits, config)
            };
            generated.push(next_token);
        }

        // Remove EOS if present
        if generated.last() == Some(&EOS_TOKEN) {
            generated.pop();
        }

        tokenizer.decode(&generated)
    })
}

/// Sample a token from pre-scaled logits (temperature already applied on GPU).
/// Uses top-k, top-p filtering and softmax sampling.
fn sample_token_prescaled(logits: &[f32], config: &SamplingConfig) -> u32 {
    let mut rng = rand::thread_rng();

    // Logits are already temperature-scaled, just enumerate
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l))
        .collect();

    // Top-k filtering: keep only top-k logits
    if config.top_k > 0 && config.top_k < scaled.len() {
        scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scaled.truncate(config.top_k);
    }

    // Softmax
    let max_logit = scaled.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = scaled
        .iter()
        .map(|&(i, l)| (i, (l - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    for p in &mut probs {
        p.1 /= sum;
    }

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0;
        let mut cutoff = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        // Re-normalize
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        for p in &mut probs {
            p.1 /= sum;
        }
    }

    // Sample from distribution
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if r <= cumulative {
            return idx as u32;
        }
    }

    probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

/// Sample a token from logits using temperature, top-k, and top-p.
fn sample_token(logits: &[f32], config: &SamplingConfig) -> u32 {
    let mut rng = rand::thread_rng();

    // Apply temperature
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / config.temperature.max(1e-8)))
        .collect();

    // Top-k filtering: keep only top-k logits
    if config.top_k > 0 && config.top_k < scaled.len() {
        scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scaled.truncate(config.top_k);
    }

    // Softmax
    let max_logit = scaled.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = scaled
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    for p in &mut probs {
        p.1 /= sum;
    }

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);

        // Re-normalize
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        for p in &mut probs {
            p.1 /= sum;
        }
    }

    // Sample from the distribution
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if r < cumulative {
            return idx as u32;
        }
    }

    // Fallback: return the highest probability token
    probs[0].0 as u32
}

/// Generate and stream tokens, calling the callback for each new token.
pub fn generate_streaming<F>(
    ctx: &Arc<MetalContext>,
    model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &SamplingConfig,
    mut on_token: F,
) where
    F: FnMut(&str),
{
    eprintln!("Streaming on {} (temp={}, top_p={}, top_k={}, max_tokens={})",
        ctx.device_name(), config.temperature, config.top_p, config.top_k, config.max_tokens);
    autograd::no_grad(|| {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(tokenizer.encode(prompt));

        let mut kv_caches = model.init_kv_caches_preallocated(1);

        // Prefill
        let seq_len = tokens.len();
        ctx.begin_batch();
        let logits = model.forward(&tokens, 1, seq_len, Some(&mut kv_caches), false);
        ctx.flush_batch();
        let vocab_size = model.config.vocab_size as usize;
        let greedy = config.temperature < 0.01;
        // Zero-copy: shared memory on Apple Silicon means direct pointer access
        let all_logits = logits.as_slice();
        let last_logits = &all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let mut next_token = sample_token(last_logits, config);

        for _ in 0..config.max_tokens {
            if next_token == EOS_TOKEN {
                break;
            }

            let text = tokenizer.decode(&[next_token]);
            on_token(&text);

            ctx.begin_batch();
            let logits = model.forward(&[next_token], 1, 1, Some(&mut kv_caches), false);
            ctx.flush_batch();

            next_token = if greedy {
                // PERF-5: GPU-side argmax — reads back 4 bytes instead of 128KB
                gpu_argmax(ctx, &logits.buffer, vocab_size as u32)
            } else {
                // Temperature scaling on GPU, then read back for CPU sampling
                gpu_temperature_scale(ctx, &logits.buffer, 0, vocab_size as u32, config.temperature);
                let logits_data = logits.to_vec();
                let token_logits = &logits_data[..vocab_size];
                sample_token_prescaled(token_logits, config)
            };
        }
    });
}

/// Speculative decoding: use a small draft model to propose tokens,
/// then verify them in a single forward pass through the main model.
/// Achieves up to Nx speedup (where N = draft_tokens) when draft model
/// predictions align well with the main model.
///
/// Algorithm per step:
/// 1. Draft model generates `draft_tokens` candidates autoregressively
/// 2. Main model verifies ALL candidates in one forward pass
/// 3. Accept matching tokens (greedy argmax comparison), reject on first mismatch
/// 4. Use main model's prediction at the rejection point
/// 5. Truncate both KV caches to the accepted prefix length
pub fn generate_speculative(
    ctx: &Arc<MetalContext>,
    main_model: &Transformer,
    draft_model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &SamplingConfig,
    draft_tokens: usize,
) -> String {
    let mut generated = Vec::new();
    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;

    generate_speculative_inner(
        ctx,
        main_model,
        draft_model,
        tokenizer,
        prompt,
        config,
        draft_tokens,
        &mut generated,
        &mut total_accepted,
        &mut total_drafted,
        None::<fn(&str)>,
    );

    let accept_rate = if total_drafted > 0 {
        total_accepted as f64 / total_drafted as f64
    } else {
        0.0
    };
    eprintln!(
        "Speculative decoding: {}/{} drafts accepted ({:.1}%), {} tokens generated",
        total_accepted, total_drafted, accept_rate * 100.0, generated.len()
    );

    // Remove EOS if present
    if generated.last() == Some(&EOS_TOKEN) {
        generated.pop();
    }

    tokenizer.decode(&generated)
}

/// Streaming variant of speculative decoding. Calls `on_token` for each
/// accepted token as soon as it is confirmed by the main model.
pub fn generate_speculative_streaming<F>(
    ctx: &Arc<MetalContext>,
    main_model: &Transformer,
    draft_model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &SamplingConfig,
    draft_tokens: usize,
    on_token: F,
) where
    F: FnMut(&str),
{
    let mut generated = Vec::new();
    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;

    generate_speculative_inner(
        ctx,
        main_model,
        draft_model,
        tokenizer,
        prompt,
        config,
        draft_tokens,
        &mut generated,
        &mut total_accepted,
        &mut total_drafted,
        Some(on_token),
    );

    let accept_rate = if total_drafted > 0 {
        total_accepted as f64 / total_drafted as f64
    } else {
        0.0
    };
    eprintln!(
        "Speculative decoding: {}/{} drafts accepted ({:.1}%), {} tokens generated",
        total_accepted, total_drafted, accept_rate * 100.0, generated.len()
    );
}

/// Core speculative decoding loop, shared by both blocking and streaming variants.
/// When `on_token` is `Some`, streams each accepted token via the callback.
fn generate_speculative_inner<F>(
    ctx: &Arc<MetalContext>,
    main_model: &Transformer,
    draft_model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &SamplingConfig,
    draft_tokens: usize,
    generated: &mut Vec<u32>,
    total_accepted: &mut usize,
    total_drafted: &mut usize,
    mut on_token: Option<F>,
) where
    F: FnMut(&str),
{
    eprintln!(
        "Speculative decoding on {} (draft_tokens={}, temp={}, max_tokens={})",
        ctx.device_name(), draft_tokens, config.temperature, config.max_tokens
    );
    eprintln!(
        "Main model: {} layers, {}M params | Draft model: {} layers, {}M params",
        main_model.config.n_layers,
        main_model.config.param_count() as f32 / 1e6,
        draft_model.config.n_layers,
        draft_model.config.param_count() as f32 / 1e6,
    );

    autograd::no_grad(|| {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(tokenizer.encode(prompt));

        let mut main_kv = main_model.init_kv_caches_preallocated(1);
        let mut draft_kv = draft_model.init_kv_caches_preallocated(1);

        let main_vocab = main_model.config.vocab_size as usize;
        let draft_vocab = draft_model.config.vocab_size as usize;

        // Prefill: process the prompt through both models
        let prompt_len = tokens.len();

        ctx.begin_batch();
        let main_logits = main_model.forward(&tokens, 1, prompt_len, Some(&mut main_kv), false);
        ctx.flush_batch();

        ctx.begin_batch();
        let draft_logits = draft_model.forward(&tokens, 1, prompt_len, Some(&mut draft_kv), false);
        ctx.flush_batch();

        // Get last token prediction from main model after prefill (this is the first generated token)
        let main_all = main_logits.to_vec();
        let main_last = &main_all[(prompt_len - 1) * main_vocab..prompt_len * main_vocab];
        let mut last_main_token = sample_token(main_last, config);

        // Also get the draft model's prediction for the same position
        let draft_all = draft_logits.to_vec();
        let draft_last = &draft_all[(prompt_len - 1) * draft_vocab..prompt_len * draft_vocab];
        let mut last_draft_token = argmax(draft_last);

        // The first token comes from the main model unconditionally
        if last_main_token == EOS_TOKEN {
            return;
        }
        generated.push(last_main_token);
        emit_token(tokenizer, last_main_token, &mut on_token);

        // Main speculative decoding loop
        let mut remaining = config.max_tokens.saturating_sub(1);
        while remaining > 0 {
            let n_draft = draft_tokens.min(remaining);

            // Phase 1: Draft model generates n_draft candidate tokens autoregressively
            let mut draft_candidates = Vec::with_capacity(n_draft);
            let draft_cache_start = draft_kv[0].cached_len();

            // Feed the last accepted token to the draft model to synchronize its state,
            // since the main model may have overridden the draft's prediction.
            let mut draft_input_token = last_main_token;

            for _ in 0..n_draft {
                ctx.begin_batch();
                let d_logits = draft_model.forward(
                    &[draft_input_token], 1, 1, Some(&mut draft_kv), false,
                );
                ctx.flush_batch();

                let d_token = gpu_argmax(ctx, &d_logits.buffer, draft_vocab as u32);
                draft_candidates.push(d_token);

                if d_token == EOS_TOKEN {
                    break;
                }
                draft_input_token = d_token;
            }

            let n_drafted = draft_candidates.len();
            *total_drafted += n_drafted;

            if n_drafted == 0 {
                break;
            }

            // Phase 2: Main model verifies ALL draft candidates in a single forward pass.
            // We feed [last_main_token, draft_0, draft_1, ..., draft_{n-1}] as a sequence.
            // The main model's KV cache already contains the prompt + previous tokens,
            // so we only need to process the new tokens.
            // The output logits at position i give the prediction for position i+1:
            //   logits[0] predicts what comes after last_main_token (should match draft_0)
            //   logits[1] predicts what comes after draft_0 (should match draft_1)
            //   ...
            //   logits[n] predicts what comes after draft_{n-1} (bonus token if all match)
            let mut verify_seq = Vec::with_capacity(1 + n_drafted);
            verify_seq.push(last_main_token);
            verify_seq.extend_from_slice(&draft_candidates);

            let verify_len = verify_seq.len();
            ctx.begin_batch();
            let main_logits = main_model.forward(
                &verify_seq, 1, verify_len, Some(&mut main_kv), false,
            );
            ctx.flush_batch();

            let all_logits = main_logits.to_vec();

            // Phase 3: Compare draft tokens with main model's argmax predictions
            let mut n_accepted = 0usize;
            for i in 0..n_drafted {
                // logits[i] is the main model's prediction for position i+1 in verify_seq
                let logit_offset = i * main_vocab;
                let position_logits = &all_logits[logit_offset..logit_offset + main_vocab];
                let main_argmax = argmax(position_logits);

                if main_argmax == draft_candidates[i] {
                    // Draft token accepted
                    n_accepted += 1;
                    generated.push(draft_candidates[i]);
                    emit_token(tokenizer, draft_candidates[i], &mut on_token);

                    if draft_candidates[i] == EOS_TOKEN {
                        return;
                    }
                } else {
                    // Rejection: use main model's sampled token instead
                    let sampled = sample_token(position_logits, config);
                    generated.push(sampled);
                    emit_token(tokenizer, sampled, &mut on_token);

                    if sampled == EOS_TOKEN {
                        return;
                    }

                    // The main model's token is now the last accepted token
                    last_main_token = sampled;
                    break;
                }
            }

            *total_accepted += n_accepted;

            if n_accepted == n_drafted {
                // All draft tokens were accepted — we get a bonus token from the
                // main model's prediction at position n_drafted (the last logit row)
                let bonus_offset = n_drafted * main_vocab;
                let bonus_logits = &all_logits[bonus_offset..bonus_offset + main_vocab];
                let bonus_token = sample_token(bonus_logits, config);
                generated.push(bonus_token);
                emit_token(tokenizer, bonus_token, &mut on_token);

                if bonus_token == EOS_TOKEN {
                    return;
                }

                last_main_token = bonus_token;
            }

            // Phase 4: Truncate KV caches to reflect the accepted sequence length.
            // Main model KV: we fed verify_len tokens, but only accepted n_accepted + 1
            // (the +1 is for last_main_token which was fed as part of the verify sequence).
            // After the verify forward, main KV len = old_len + verify_len.
            // We need it at old_len + n_accepted + 1:
            //   - old_len positions from before this step
            //   - 1 for last_main_token (fed as verify_seq[0])
            //   - n_accepted for the accepted draft tokens
            let main_kv_target = main_kv[0].cached_len() - verify_len + 1 + n_accepted;
            // If all were accepted, we also need the bonus token position, which
            // is already included (the main model saw verify_len tokens, and we
            // accepted n_accepted == n_drafted, so target = old + 1 + n_drafted = old + verify_len)
            // That means no truncation needed when all accepted. Only truncate on rejection.
            if n_accepted < n_drafted {
                truncate_kv_caches(&mut main_kv, main_kv_target);
            }

            // Draft model KV: we fed n_drafted tokens autoregressively.
            // Roll back to draft_cache_start + 1 + n_accepted:
            //   - draft_cache_start from before this step
            //   - 1 for the last_main_token we fed
            //   - n_accepted draft tokens that were accepted
            let draft_kv_target = draft_cache_start + 1 + n_accepted;
            truncate_kv_caches(&mut draft_kv, draft_kv_target);

            remaining = remaining.saturating_sub(n_accepted + 1);

            // Update draft's last prediction for synchronization tracking
            if n_accepted == n_drafted && n_drafted > 0 {
                last_draft_token = draft_candidates[n_drafted - 1];
            } else {
                last_draft_token = last_main_token;
            }
        }

        // Suppress "unused" — last_draft_token is tracked for synchronization
        // diagnostics and potential future adaptive draft length tuning.
        let _ = last_draft_token;
    });
}

/// Emit a token through the streaming callback, if present.
fn emit_token<F: FnMut(&str)>(tokenizer: &BpeTokenizer, token: u32, on_token: &mut Option<F>) {
    if let Some(ref mut callback) = on_token {
        let text = tokenizer.decode(&[token]);
        callback(&text);
    }
}

/// Truncate all KV caches in a layer stack to the given sequence length.
fn truncate_kv_caches(caches: &mut Vec<KvCache>, target_len: usize) {
    for cache in caches.iter_mut() {
        if cache.cached_len() > target_len {
            cache.truncate(target_len);
        }
    }
}

/// Greedy argmax over a logit slice. Returns the index of the maximum value.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
