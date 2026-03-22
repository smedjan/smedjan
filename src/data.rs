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

    /// Get a slice of tokens.
    pub fn get_tokens(&self, start: usize, len: usize) -> Vec<u32> {
        (start..start + len).map(|i| self.get_token(i)).collect()
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

        Ok(Self {
            dataset,
            batch_size,
            seq_len,
            position: 0,
            epoch: 0,
        })
    }

    /// Get next batch of (input_tokens, target_tokens).
    /// input: [batch_size * seq_len], target: [batch_size * seq_len]
    /// Target is input shifted right by 1. Wraps around at dataset end.
    pub fn next_batch(&mut self) -> (Vec<u32>, Vec<u32>) {
        let needed = self.batch_size * (self.seq_len + 1);

        // Wrap around if we'd go past the end
        if self.position + needed > self.dataset.len() {
            self.position = 0;
            self.epoch += 1;
            // Shuffle: use a random offset within the first batch to avoid seeing the exact same data
            use rand::Rng;
            let max_offset = self.dataset.len().saturating_sub(needed);
            if max_offset > 0 {
                self.position = rand::thread_rng().gen_range(0..max_offset);
            }
        }

        let tokens = self.dataset.get_tokens(self.position, needed);

        let mut inputs = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);

        for b in 0..self.batch_size {
            let offset = b * (self.seq_len + 1);
            inputs.extend_from_slice(&tokens[offset..offset + self.seq_len]);
            targets.extend_from_slice(&tokens[offset + 1..offset + 1 + self.seq_len]);
        }

        self.position += needed;

        (inputs, targets)
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
/// Returns the number of verified tokens. Panics on mismatch.
pub fn verify_dataset_gpu(ctx: &Arc<MetalContext>, dataset_path: &str, sample_size: usize) -> usize {
    let dataset = Dataset::load(dataset_path).expect("Failed to load dataset for verification");
    let count = sample_size.min(dataset.len());
    let tokens = dataset.get_tokens(0, count);

    // Round-trip through Metal GPU buffer
    let gpu_buf = ctx.buffer_from_u32_slice(&tokens);
    let readback = MetalContext::read_buffer_u32(&gpu_buf, count);

    assert_eq!(tokens, readback, "GPU round-trip verification failed for dataset");
    eprintln!("Dataset verification passed: {} tokens round-tripped through GPU", count);
    count
}
