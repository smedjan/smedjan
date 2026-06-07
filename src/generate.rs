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
    pub repetition_penalty: f32, // >1.0 penalizes repetition, 1.0 = disabled
    pub min_p: f32,              // keep tokens with p >= min_p * max_p; 0.0 = disabled
    pub typical_p: f32,          // locally-typical mass to keep; 1.0 = disabled
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            max_tokens: 256,
            repetition_penalty: 1.2,
            min_p: 0.0,
            typical_p: 1.0,
        }
    }
}

/// Renormalize a (token, prob) distribution to sum 1 (no-op if the mass is zero).
fn renorm(probs: &mut [(usize, f32)]) {
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            p.1 /= sum;
        }
    }
}

/// Apply min-p and locally-typical filtering to a normalized (token, prob) distribution, in place.
///
/// * **min-p** keeps tokens with `p >= min_p * max_p` — a relative floor that adapts to how peaked
///   the distribution is (unlike top-p's fixed cumulative mass). `0.0` disables it.
/// * **locally-typical** (Meister et al. 2022) keeps the smallest set of tokens whose surprisal
///   `−ln p` is closest to the distribution entropy `H`, until their mass reaches `typical_p` —
///   trimming both the over-confident head and the long tail. `1.0` disables it.
pub fn filter_min_p_typical(probs: &mut Vec<(usize, f32)>, min_p: f32, typical_p: f32) {
    if probs.is_empty() {
        return;
    }
    if min_p > 0.0 {
        let max_p = probs.iter().map(|x| x.1).fold(0.0f32, f32::max);
        let thresh = min_p * max_p;
        probs.retain(|&(_, p)| p >= thresh);
        renorm(probs);
    }
    if typical_p < 1.0 && probs.len() > 1 {
        let h: f32 = -probs.iter().map(|&(_, p)| if p > 0.0 { p * p.ln() } else { 0.0 }).sum::<f32>();
        let mut ranked = probs.clone();
        ranked.sort_by(|a, b| {
            let da = ((-a.1.ln()) - h).abs();
            let db = ((-b.1.ln()) - h).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumulative = 0.0;
        let mut cutoff = ranked.len();
        for (k, &(_, p)) in ranked.iter().enumerate() {
            cumulative += p;
            if cumulative >= typical_p {
                cutoff = k + 1;
                break;
            }
        }
        ranked.truncate(cutoff);
        *probs = ranked;
        renorm(probs);
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

        // Sample first token from prefill logits (last position predicts next token).
        // Zero-copy: as_slice() returns a reference to shared GPU/CPU memory.
        let all_logits = logits.as_slice();
        let last_logits = &all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];
        let mut next_token = sample_token(last_logits, config, &[]);

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
                sample_token_prescaled(token_logits, config, &generated)
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

/// Batched generation: decode `prompts.len()` continuations together through one batched KV cache,
/// for throughput (one forward over the whole batch per step instead of N separate forwards).
/// Requires the prompts to encode to the SAME length — variable lengths need padding masks /
/// continuous batching, a separate larger feature. Returns one decoded string per prompt.
pub fn generate_batch(
    ctx: &Arc<MetalContext>,
    model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompts: &[&str],
    config: &SamplingConfig,
) -> Vec<String> {
    assert!(!prompts.is_empty(), "generate_batch needs at least one prompt");
    autograd::no_grad(|| {
        let b = prompts.len();
        let encoded: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                let mut t = vec![BOS_TOKEN];
                t.extend(tokenizer.encode(p));
                t
            })
            .collect();
        let len = encoded[0].len();
        assert!(
            encoded.iter().all(|e| e.len() == len),
            "generate_batch requires equal token-length prompts (got {:?})",
            encoded.iter().map(|e| e.len()).collect::<Vec<_>>()
        );

        let vocab = model.config.vocab_size as usize;
        let greedy = config.temperature < 0.01;
        let argmax = |slice: &[f32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in slice.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best as u32
        };

        let mut kv = model.init_kv_caches_preallocated(b);

        // Prefill the whole batch [b, len] in one forward.
        let flat: Vec<u32> = encoded.iter().flatten().copied().collect();
        ctx.begin_batch();
        let logits = model.forward(&flat, b, len, Some(&mut kv), false);
        ctx.flush_batch();
        let all = logits.as_slice(); // [b * len, vocab]

        let mut seqs: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut cur: Vec<u32> = Vec::with_capacity(b);
        let mut done = vec![false; b];
        for (i, seq) in seqs.iter_mut().enumerate() {
            let lp = &all[(i * len + len - 1) * vocab..(i * len + len) * vocab];
            let t = if greedy { argmax(lp) } else { sample_token(lp, config, &[]) };
            seq.push(t);
            cur.push(t);
            if t == EOS_TOKEN {
                done[i] = true;
            }
        }

        // Decode the batch in lockstep: one token per sequence per step.
        for _ in 1..config.max_tokens {
            if done.iter().all(|&d| d) {
                break;
            }
            ctx.begin_batch();
            let logits = model.forward(&cur, b, 1, Some(&mut kv), false);
            ctx.flush_batch();
            let all = logits.as_slice(); // [b, vocab]
            for i in 0..b {
                if done[i] {
                    continue;
                }
                let lp = &all[i * vocab..(i + 1) * vocab];
                let t = if greedy { argmax(lp) } else { sample_token(lp, config, &seqs[i]) };
                cur[i] = t;
                seqs[i].push(t);
                if t == EOS_TOKEN {
                    done[i] = true;
                }
            }
        }

        seqs.into_iter()
            .map(|s| {
                let s: Vec<u32> = s.into_iter().take_while(|&t| t != EOS_TOKEN).collect();
                tokenizer.decode(&s)
            })
            .collect()
    })
}

/// Sample a token from pre-scaled logits (temperature already applied on GPU).
/// Uses repetition penalty, top-k, top-p filtering and softmax sampling.
fn sample_token_prescaled(logits: &[f32], config: &SamplingConfig, generated: &[u32]) -> u32 {
    let mut rng = rand::thread_rng();

    // Apply repetition penalty before top-k/softmax
    let mut penalized: Vec<f32> = logits.to_vec();
    if config.repetition_penalty != 1.0 {
        for &tok in generated {
            let idx = tok as usize;
            if idx < penalized.len() {
                if penalized[idx] > 0.0 {
                    penalized[idx] /= config.repetition_penalty;
                } else {
                    penalized[idx] *= config.repetition_penalty;
                }
            }
        }
    }

    let mut scaled: Vec<(usize, f32)> = penalized
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

    filter_min_p_typical(&mut probs, config.min_p, config.typical_p);

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

/// Sample a token from logits using temperature, top-k, top-p, and repetition penalty.
fn sample_token(logits: &[f32], config: &SamplingConfig, generated: &[u32]) -> u32 {
    let mut rng = rand::thread_rng();

    // Apply repetition penalty: penalize tokens that already appeared in generated text
    let mut penalized: Vec<f32> = logits.to_vec();
    if config.repetition_penalty != 1.0 {
        for &tok in generated {
            let idx = tok as usize;
            if idx < penalized.len() {
                if penalized[idx] > 0.0 {
                    penalized[idx] /= config.repetition_penalty;
                } else {
                    penalized[idx] *= config.repetition_penalty;
                }
            }
        }
    }

    // Apply temperature
    let mut scaled: Vec<(usize, f32)> = penalized
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

    filter_min_p_typical(&mut probs, config.min_p, config.typical_p);

    // Sample from the distribution
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if r < cumulative {
            return idx as u32;
        }
    }

    // Fallback: return the highest probability token (or token 0 if probs empty)
    probs.first().map_or(0, |p| p.0 as u32)
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

        let mut next_token = sample_token(last_logits, config, &[]);
        let mut generated: Vec<u32> = Vec::new();

        for _ in 0..config.max_tokens {
            if next_token == EOS_TOKEN {
                break;
            }

            generated.push(next_token);
            let text = tokenizer.decode(&[next_token]);
            on_token(&text);

            ctx.begin_batch();
            let logits = model.forward(&[next_token], 1, 1, Some(&mut kv_caches), false);
            ctx.flush_batch();

            next_token = if greedy {
                // PERF-5: GPU-side argmax — reads back 4 bytes instead of 128KB
                gpu_argmax(ctx, &logits.buffer, vocab_size as u32)
            } else {
                // Temperature scaling on GPU, then zero-copy read for CPU sampling
                gpu_temperature_scale(ctx, &logits.buffer, 0, vocab_size as u32, config.temperature);
                let token_logits = &logits.as_slice()[..vocab_size];
                sample_token_prescaled(token_logits, config, &generated)
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
        SpecModels { main: main_model, draft: draft_model, tokenizer },
        prompt,
        config,
        draft_tokens,
        SpecState { generated: &mut generated, total_accepted: &mut total_accepted, total_drafted: &mut total_drafted },
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
    models: SpecModels,
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
        models,
        prompt,
        config,
        draft_tokens,
        SpecState { generated: &mut generated, total_accepted: &mut total_accepted, total_drafted: &mut total_drafted },
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
/// The two models + tokenizer used by speculative decoding.
pub struct SpecModels<'a> {
    pub main: &'a Transformer,
    pub draft: &'a Transformer,
    pub tokenizer: &'a BpeTokenizer,
}

/// Mutable accounting threaded through the speculative loop.
struct SpecState<'a> {
    generated: &'a mut Vec<u32>,
    total_accepted: &'a mut usize,
    total_drafted: &'a mut usize,
}

fn generate_speculative_inner<F>(
    ctx: &Arc<MetalContext>,
    models: SpecModels,
    prompt: &str,
    config: &SamplingConfig,
    draft_tokens: usize,
    state: SpecState,
    mut on_token: Option<F>,
) where
    F: FnMut(&str),
{
    let SpecModels { main: main_model, draft: draft_model, tokenizer } = models;
    let SpecState { generated, total_accepted, total_drafted } = state;
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
        let mut last_main_token = sample_token(main_last, config, &[]);

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
            for (i, &draft) in draft_candidates.iter().enumerate().take(n_drafted) {
                // logits[i] is the main model's prediction for position i+1 in verify_seq
                let logit_offset = i * main_vocab;
                let position_logits = &all_logits[logit_offset..logit_offset + main_vocab];
                let main_argmax = argmax(position_logits);

                if main_argmax == draft {
                    // Draft token accepted
                    n_accepted += 1;
                    generated.push(draft);
                    emit_token(tokenizer, draft, &mut on_token);

                    if draft == EOS_TOKEN {
                        return;
                    }
                } else {
                    // Rejection: use main model's sampled token instead
                    let sampled = sample_token(position_logits, config, &[]);
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
                let bonus_token = sample_token(bonus_logits, config, &[]);
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
fn truncate_kv_caches(caches: &mut [KvCache], target_len: usize) {
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
