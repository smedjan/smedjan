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

impl SamplingConfig {
    pub fn default() -> Self {
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
        let logits = model.forward(&tokens, batch, seq_len, Some(&mut kv_caches), false);

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
            let logits = model.forward(&[next_token], 1, 1, Some(&mut kv_caches), false);

            next_token = if greedy {
                // PERF-5: GPU-side argmax — reads back 4 bytes instead of 128KB (vocab_size * 4)
                gpu_argmax(ctx, &logits.buffer, vocab_size as u32)
            } else {
                // Temperature scaling on GPU, then read back for CPU sampling
                gpu_temperature_scale(ctx, &logits.buffer, 0, vocab_size as u32, config.temperature);
                let logits_data = logits.to_vec();
                let token_logits = &logits_data[..vocab_size];
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
        let logits = model.forward(&tokens, 1, seq_len, Some(&mut kv_caches), false);
        let vocab_size = model.config.vocab_size as usize;
        let greedy = config.temperature < 0.01;
        let all_logits = logits.to_vec();
        let last_logits = &all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let mut next_token = sample_token(last_logits, config);

        for _ in 0..config.max_tokens {
            if next_token == EOS_TOKEN {
                break;
            }

            let text = tokenizer.decode(&[next_token]);
            on_token(&text);

            let logits = model.forward(&[next_token], 1, 1, Some(&mut kv_caches), false);

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
