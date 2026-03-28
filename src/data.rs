use crate::metal::MetalContext;
use crate::tokenizer::{BpeTokenizer, PAD_TOKEN};
use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

/// Pre-tokenized dataset stored as flat u32 array on disk.
pub struct Dataset {
    mmap: Mmap,
    len: usize, // number of tokens
}

impl Dataset {
    /// Load a pre-tokenized dataset from disk.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len() / 4; // u32 = 4 bytes
        Ok(Self { mmap, len })
    }

    /// Get token at index.
    pub fn get_token(&self, idx: usize) -> u32 {
        assert!(idx < self.len, "token index out of bounds");
        let offset = idx * 4;
        u32::from_le_bytes([
            self.mmap[offset],
            self.mmap[offset + 1],
            self.mmap[offset + 2],
            self.mmap[offset + 3],
        ])
    }

    /// Get a slice of tokens as a zero-copy reference into the mmap.
    /// The mmap bytes are reinterpreted as &[u32] (little-endian, aligned by construction).
    pub fn get_tokens_slice(&self, start: usize, len: usize) -> &[u32] {
        assert!(start + len <= self.len, "token slice out of bounds");
        let byte_offset = start * 4;
        let byte_len = len * 4;
        let bytes = &self.mmap[byte_offset..byte_offset + byte_len];
        // Safety: mmap is page-aligned (always aligned to at least 4096 bytes),
        // so the start of the mmap is u32-aligned. start*4 preserves alignment.
        // The data was written as little-endian u32s and we're on a little-endian platform.
        let ptr = bytes.as_ptr() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get a slice of tokens (copies into a new Vec). Use get_tokens_slice for zero-copy access.
    pub fn get_tokens(&self, start: usize, len: usize) -> Vec<u32> {
        // For small ranges, use single-token lookup to avoid alignment assumptions.
        // For larger ranges, use the zero-copy slice path.
        if len <= 4 {
            (start..start + len).map(|i| self.get_token(i)).collect()
        } else {
            self.get_tokens_slice(start, len).to_vec()
        }
    }

    /// Total number of tokens.
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Prepare a dataset: tokenize raw text and write as binary u32 array.
pub fn prepare_dataset(
    input_path: &str,
    tokenizer: &BpeTokenizer,
    output_path: &str,
) -> std::io::Result<usize> {
    let text = std::fs::read_to_string(input_path)?;
    let tokens = tokenizer.encode(&text);

    let mut file = File::create(output_path)?;
    for &token in &tokens {
        file.write_all(&token.to_le_bytes())?;
    }

    eprintln!(
        "Prepared dataset: {} bytes → {} tokens, saved to {}",
        text.len(),
        tokens.len(),
        output_path
    );
    Ok(tokens.len())
}

/// Data loader that yields (input, target) batches for next-token prediction.
/// Uses random offset sampling — each batch picks a random starting position in the dataset.
/// Wraps around for infinite iteration (no epoch boundary).
pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    seq_len: usize,
    position: usize,
    epoch: usize,
    inputs_buf: Vec<u32>,
    targets_buf: Vec<u32>,
}

impl DataLoader {
    pub fn new(dataset_path: &str, batch_size: usize, seq_len: usize) -> std::io::Result<Self> {
        let dataset = Dataset::load(dataset_path)?;
        let needed = batch_size * (seq_len + 1);
        assert!(
            dataset.len() >= needed,
            "Dataset too small: {} tokens, need at least {} (batch_size={} * (seq_len+1={}))",
            dataset.len(),
            needed,
            batch_size,
            seq_len + 1,
        );

        // Pre-allocate reusable buffers (avoid allocation every step)
        let cap = batch_size * seq_len;
        let inputs_buf = vec![0u32; cap];
        let targets_buf = vec![0u32; cap];

        Ok(Self {
            dataset,
            batch_size,
            seq_len,
            position: 0,
            epoch: 0,
            inputs_buf,
            targets_buf,
        })
    }

    /// Get next batch of (input_tokens, target_tokens).
    /// input: [batch_size * seq_len], target: [batch_size * seq_len]
    /// Target is input shifted right by 1. Wraps around at dataset end.
    pub fn next_batch(&mut self) -> (&[u32], &[u32]) {
        let needed = self.batch_size * (self.seq_len + 1);

        // Wrap around if we'd go past the end
        if self.position + needed > self.dataset.len() {
            self.position = 0;
            self.epoch += 1;
            use rand::Rng;
            let max_offset = self.dataset.len().saturating_sub(needed);
            if max_offset > 0 {
                self.position = rand::thread_rng().gen_range(0..max_offset);
            }
        }

        let tokens = self.dataset.get_tokens_slice(self.position, needed);

        // Copy into pre-allocated buffers (zero allocation)
        for b in 0..self.batch_size {
            let offset = b * (self.seq_len + 1);
            let dst_offset = b * self.seq_len;
            self.inputs_buf[dst_offset..dst_offset + self.seq_len]
                .copy_from_slice(&tokens[offset..offset + self.seq_len]);
            self.targets_buf[dst_offset..dst_offset + self.seq_len]
                .copy_from_slice(&tokens[offset + 1..offset + 1 + self.seq_len]);
        }

        self.position += needed;

        (&self.inputs_buf, &self.targets_buf)
    }

    /// Current epoch (how many times we've wrapped around the dataset).
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Approximate number of batches per epoch.
    pub fn batches_per_epoch(&self) -> usize {
        let tokens_per_batch = self.batch_size * (self.seq_len + 1);
        self.dataset.len() / tokens_per_batch
    }

    /// Total tokens in the underlying dataset.
    pub fn total_tokens(&self) -> usize {
        self.dataset.len()
    }
}

/// Pad a batch of variable-length token sequences to the same length using PAD_TOKEN.
/// Each sequence is padded on the right to `max_len`.
pub fn pad_sequences(sequences: &[Vec<u32>], max_len: usize) -> Vec<u32> {
    let mut padded = Vec::with_capacity(sequences.len() * max_len);
    for seq in sequences {
        let take = seq.len().min(max_len);
        padded.extend_from_slice(&seq[..take]);
        padded.extend(std::iter::repeat_n(PAD_TOKEN, max_len - take));
    }
    padded
}

/// Verify a dataset shard by round-tripping a sample through a GPU u32 buffer.
/// Also verifies the GPU transpose kernel by transposing a small matrix and checking the result.
/// Returns the number of verified tokens. Panics on mismatch.
pub fn verify_dataset_gpu(ctx: &Arc<MetalContext>, dataset_path: &str, sample_size: usize) -> usize {
    use crate::metal::compute;

    let dataset = Dataset::load(dataset_path).expect("Failed to load dataset for verification");
    let count = sample_size.min(dataset.len());
    let tokens = dataset.get_tokens(0, count);

    // Round-trip through Metal GPU buffer
    let gpu_buf = ctx.buffer_from_u32_slice(&tokens);
    let readback = MetalContext::read_buffer_u32(&gpu_buf, count);

    assert_eq!(tokens, readback, "GPU round-trip verification failed for dataset");

    // Verify GPU transpose kernel (sanity check that compute shaders are working correctly).
    // Transpose a 2x(count/2) float matrix and verify the result, ensuring the shader
    // pipeline is healthy before we start training on this dataset.
    if count >= 4 {
        let rows = 2u32;
        let cols = (count / 2) as u32;
        let n = (rows * cols) as usize;
        let float_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input_buf = ctx.buffer_from_slice(&float_data);
        let output_buf = ctx.alloc_buffer(n * 4);
        compute::gpu_transpose_2d(ctx, &input_buf, &output_buf, rows, cols);
        let transposed = MetalContext::read_buffer(&output_buf, n);

        // Verify: out[j * rows + i] == in[i * cols + j]
        for i in 0..rows as usize {
            for j in 0..cols as usize {
                let expected = float_data[i * cols as usize + j];
                let actual = transposed[j * rows as usize + i];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "GPU transpose verification failed at ({}, {}): expected {}, got {}",
                    i, j, expected, actual
                );
            }
        }
    }

    eprintln!("Dataset verification passed: {} tokens round-tripped through GPU (transpose OK)", count);
    count
}

/// Multi-source data mixer: samples from multiple datasets with configurable weights.
/// Useful for mixing code, text, math data in specific proportions.
/// weights[i] = relative probability of sampling from source i.
pub struct DataMixer {
    loaders: Vec<DataLoader>,
    weights: Vec<f32>,
    cumulative: Vec<f32>,
}

impl DataMixer {
    /// Create a mixer from multiple dataset paths and their sampling weights.
    /// Weights are normalized to sum to 1.0.
    pub fn new(
        paths: &[&str],
        weights: &[f32],
        batch_size: usize,
        seq_len: usize,
    ) -> std::io::Result<Self> {
        assert_eq!(paths.len(), weights.len(), "paths and weights must match");
        assert!(!paths.is_empty(), "need at least one data source");

        let total: f32 = weights.iter().sum();
        let norm_weights: Vec<f32> = weights.iter().map(|w| w / total).collect();
        let mut cumulative = Vec::with_capacity(norm_weights.len());
        let mut cum = 0.0f32;
        for w in &norm_weights {
            cum += w;
            cumulative.push(cum);
        }

        let loaders = paths.iter()
            .map(|p| DataLoader::new(p, batch_size, seq_len))
            .collect::<std::io::Result<Vec<_>>>()?;

        eprintln!("DataMixer: {} sources, weights={:?}", paths.len(),
            norm_weights.iter().map(|w| format!("{:.1}%", w * 100.0)).collect::<Vec<_>>());

        Ok(Self { loaders, weights: norm_weights, cumulative })
    }

    /// Get next batch from a randomly selected source (weighted).
    pub fn next_batch(&mut self) -> (&[u32], &[u32]) {
        let r: f32 = rand::random();
        let idx = self.cumulative.iter().position(|&c| r < c).unwrap_or(self.loaders.len() - 1);
        self.loaders[idx].next_batch()
    }

    /// Total tokens across all sources.
    pub fn total_tokens(&self) -> usize {
        self.loaders.iter().map(|l| l.total_tokens()).sum()
    }

    /// Get the normalized weights for each source.
    pub fn source_weights(&self) -> &[f32] {
        &self.weights
    }
}
