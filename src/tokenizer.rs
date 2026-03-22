use std::collections::HashMap;
use std::io::{Read, Write};

/// Special token IDs.
pub const PAD_TOKEN: u32 = 0;
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;
pub const SPECIAL_TOKENS: u32 = 3;

/// Byte-Pair Encoding tokenizer trained from scratch.
pub struct BpeTokenizer {
    /// Token string → token ID
    pub vocab: HashMap<Vec<u8>, u32>,
    /// Token ID → token bytes
    pub inverse_vocab: Vec<Vec<u8>>,
    /// Merge rules in priority order: (token_a, token_b) → merged_token
    pub merges: Vec<(u32, u32, u32)>,
    /// Fast lookup: (a, b) → priority (lower = higher priority)
    pub merge_priority: HashMap<(u32, u32), usize>,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer from a corpus.
    pub fn train(corpus: &[u8], vocab_size: u32) -> Self {
        assert!(vocab_size > 256 + SPECIAL_TOKENS, "vocab_size must be > 259");

        // Initialize vocab: special tokens + 256 byte-level tokens
        let mut inverse_vocab: Vec<Vec<u8>> = Vec::new();
        let mut vocab: HashMap<Vec<u8>, u32> = HashMap::new();

        // Special tokens
        inverse_vocab.push(b"<|pad|>".to_vec());
        vocab.insert(b"<|pad|>".to_vec(), 0);
        inverse_vocab.push(b"<|bos|>".to_vec());
        vocab.insert(b"<|bos|>".to_vec(), 1);
        inverse_vocab.push(b"<|eos|>".to_vec());
        vocab.insert(b"<|eos|>".to_vec(), 2);

        // Byte-level tokens (3..258)
        for byte in 0u8..=255 {
            let token = vec![byte];
            let id = inverse_vocab.len() as u32;
            vocab.insert(token.clone(), id);
            inverse_vocab.push(token);
        }

        // Convert corpus to initial token sequence (byte-level)
        let mut tokens: Vec<u32> = corpus.iter().map(|&b| b as u32 + SPECIAL_TOKENS).collect();

        let mut merges: Vec<(u32, u32, u32)> = Vec::new();

        let target_merges = vocab_size - (256 + SPECIAL_TOKENS);

        for merge_idx in 0..target_merges {
            if merge_idx % 100 == 0 {
                eprintln!("BPE merge {}/{} (vocab size: {})", merge_idx, target_merges, inverse_vocab.len());
            }

            // Count all adjacent pairs
            let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
            for window in tokens.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += 1;
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair
            let &best_pair = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, _)| pair)
                .unwrap();

            // Create new token for the merged pair
            let new_id = inverse_vocab.len() as u32;
            let mut new_token = inverse_vocab[best_pair.0 as usize].clone();
            new_token.extend_from_slice(&inverse_vocab[best_pair.1 as usize]);
            vocab.insert(new_token.clone(), new_id);
            inverse_vocab.push(new_token);

            merges.push((best_pair.0, best_pair.1, new_id));

            // Replace all occurrences of the pair in the token sequence
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        let mut merge_priority = HashMap::new();
        for (priority, &(a, b, _)) in merges.iter().enumerate() {
            merge_priority.insert((a, b), priority);
        }

        eprintln!("BPE training complete: {} tokens, {} merges", inverse_vocab.len(), merges.len());

        Self {
            vocab,
            inverse_vocab,
            merges,
            merge_priority,
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Start with byte-level tokens
        let mut tokens: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32 + SPECIAL_TOKENS).collect();

        // Apply merges in priority order
        loop {
            // Find the highest-priority (lowest index) merge that applies
            let mut best_merge: Option<(usize, usize)> = None; // (position, priority)

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&priority) = self.merge_priority.get(&pair) {
                    match best_merge {
                        None => best_merge = Some((i, priority)),
                        Some((_, best_pri)) if priority < best_pri => {
                            best_merge = Some((i, priority));
                        }
                        _ => {}
                    }
                }
            }

            match best_merge {
                None => break,
                Some((pos, _)) => {
                    let pair = (tokens[pos], tokens[pos + 1]);
                    // Look up the merged token
                    let merged = self.merges.iter()
                        .find(|&&(a, b, _)| a == pair.0 && b == pair.1)
                        .map(|&(_, _, new)| new)
                        .unwrap();
                    tokens[pos] = merged;
                    tokens.remove(pos + 1);
                }
            }
        }

        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &token_id in tokens {
            if (token_id as usize) < self.inverse_vocab.len() {
                bytes.extend_from_slice(&self.inverse_vocab[token_id as usize]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.inverse_vocab.len() as u32
    }

    /// Save tokenizer to a binary file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Magic + version
        file.write_all(b"ABPE")?;
        file.write_all(&1u32.to_le_bytes())?;

        // Vocab size
        let vocab_len = self.inverse_vocab.len() as u32;
        file.write_all(&vocab_len.to_le_bytes())?;

        // Each token: length (u32) + bytes
        for token_bytes in &self.inverse_vocab {
            let len = token_bytes.len() as u32;
            file.write_all(&len.to_le_bytes())?;
            file.write_all(token_bytes)?;
        }

        // Merge count + merges
        let merge_count = self.merges.len() as u32;
        file.write_all(&merge_count.to_le_bytes())?;
        for &(a, b, c) in &self.merges {
            file.write_all(&a.to_le_bytes())?;
            file.write_all(&b.to_le_bytes())?;
            file.write_all(&c.to_le_bytes())?;
        }

        Ok(())
    }

    /// Load tokenizer from a binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];

        // Magic
        file.read_exact(&mut buf4)?;
        assert_eq!(&buf4, b"ABPE", "Not a valid tokenizer file");

        // Version
        file.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);

        // Vocab size
        file.read_exact(&mut buf4)?;
        let vocab_len = u32::from_le_bytes(buf4) as usize;

        let mut inverse_vocab = Vec::with_capacity(vocab_len);
        let mut vocab = HashMap::new();
        for id in 0..vocab_len {
            file.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;
            let mut token_bytes = vec![0u8; len];
            file.read_exact(&mut token_bytes)?;
            vocab.insert(token_bytes.clone(), id as u32);
            inverse_vocab.push(token_bytes);
        }

        // Merges
        file.read_exact(&mut buf4)?;
        let merge_count = u32::from_le_bytes(buf4) as usize;
        let mut merges = Vec::with_capacity(merge_count);
        let mut merge_priority = HashMap::new();
        for priority in 0..merge_count {
            file.read_exact(&mut buf4)?;
            let a = u32::from_le_bytes(buf4);
            file.read_exact(&mut buf4)?;
            let b = u32::from_le_bytes(buf4);
            file.read_exact(&mut buf4)?;
            let c = u32::from_le_bytes(buf4);
            merges.push((a, b, c));
            merge_priority.insert((a, b), priority);
        }

        Ok(Self {
            vocab,
            inverse_vocab,
            merges,
            merge_priority,
        })
    }
}
