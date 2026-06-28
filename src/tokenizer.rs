use std::collections::HashMap;
use std::io::{Error, ErrorKind, Read, Write};

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

fn invalid_tokenizer_data(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

fn read_tokenizer_exact(
    file: &mut std::fs::File,
    buf: &mut [u8],
    consumed: &mut u64,
) -> std::io::Result<()> {
    file.read_exact(buf)?;
    *consumed += buf.len() as u64;
    Ok(())
}

fn read_tokenizer_u32(file: &mut std::fs::File, consumed: &mut u64) -> std::io::Result<u32> {
    let mut buf4 = [0u8; 4];
    read_tokenizer_exact(file, &mut buf4, consumed)?;
    Ok(u32::from_le_bytes(buf4))
}

fn ensure_tokenizer_remaining(
    file_len: u64,
    consumed: u64,
    needed: u64,
    context: impl Into<String>,
) -> std::io::Result<()> {
    let remaining = file_len.saturating_sub(consumed);
    if needed > remaining {
        return Err(invalid_tokenizer_data(format!(
            "{} exceeds remaining tokenizer file bytes: need {}, remaining {}",
            context.into(),
            needed,
            remaining
        )));
    }
    Ok(())
}

/// GPT-2's reversible byte→unicode map: bytes in the printable ranges map to themselves, the rest
/// to code points 256+n. Returns (byte, char) pairs — used to decode `merges.txt` token strings
/// (which live in this char space) back to their raw bytes.
fn gpt2_byte_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = Vec::new();
    for (lo, hi) in [
        (b'!' as u32, b'~' as u32),
        (0xA1u32, 0xACu32),
        (0xAEu32, 0xFFu32),
    ] {
        bs.extend(lo..=hi);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b as u8, char::from_u32(c).expect("valid code point")))
        .collect()
}

impl BpeTokenizer {
    /// Train a BPE tokenizer from a corpus.
    pub fn train(corpus: &[u8], vocab_size: u32) -> Self {
        assert!(
            vocab_size > 256 + SPECIAL_TOKENS,
            "vocab_size must be > 259"
        );

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
                eprintln!(
                    "BPE merge {}/{} (vocab size: {})",
                    merge_idx,
                    target_merges,
                    inverse_vocab.len()
                );
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
                .max_by_key(|&(_, &count)| count)
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
                if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1
                {
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

        eprintln!(
            "BPE training complete: {} tokens, {} merges",
            inverse_vocab.len(),
            merges.len()
        );

        Self {
            vocab,
            inverse_vocab,
            merges,
            merge_priority,
        }
    }

    /// Import a byte-level BPE from a GPT-2 / HuggingFace `merges.txt` (priority-ordered "A B" lines
    /// in GPT-2's byte-to-unicode char space). The merge RULES are reproduced exactly, so encoding
    /// matches the source tokenizer; IDs follow this harness's convention (3 special + 256 bytes +
    /// merges) rather than the source's, which is what a freshly-trained model needs. No vocab.json
    /// required — the vocab is rebuilt from the byte base by applying the merges in order.
    pub fn import_gpt2_merges(merges_text: &str) -> Self {
        let char2byte: HashMap<char, u8> = gpt2_byte_to_unicode()
            .into_iter()
            .map(|(b, c)| (c, b))
            .collect();
        // Decode a GPT-2 char-space token to its raw bytes (None if it contains an unmapped char).
        let decode = |s: &str| -> Option<Vec<u8>> {
            s.chars().map(|c| char2byte.get(&c).copied()).collect()
        };

        let mut inverse_vocab: Vec<Vec<u8>> = Vec::new();
        let mut vocab: HashMap<Vec<u8>, u32> = HashMap::new();
        // 3 special tokens (ids 0..3), then the 256 single bytes (id = byte + 3), matching `train`.
        for special in [&b"<|pad|>"[..], &b"<|bos|>"[..], &b"<|eos|>"[..]] {
            let id = inverse_vocab.len() as u32;
            vocab.insert(special.to_vec(), id);
            inverse_vocab.push(special.to_vec());
        }
        for b in 0u32..256 {
            let bytes = vec![b as u8];
            let id = inverse_vocab.len() as u32;
            vocab.insert(bytes.clone(), id);
            inverse_vocab.push(bytes);
        }

        let mut merges: Vec<(u32, u32, u32)> = Vec::new();
        let mut merge_priority: HashMap<(u32, u32), usize> = HashMap::new();
        for line in merges_text.lines() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let mut it = line.splitn(2, ' ');
            let (a, b) = match (it.next(), it.next()) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let (a_bytes, b_bytes) = match (decode(a), decode(b)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let (a_id, b_id) = match (vocab.get(&a_bytes), vocab.get(&b_bytes)) {
                (Some(&a), Some(&b)) => (a, b),
                _ => continue, // merge references a token not yet built — skip (malformed file)
            };
            let mut merged = a_bytes;
            merged.extend_from_slice(&b_bytes);
            if vocab.contains_key(&merged) {
                continue; // duplicate merge target
            }
            let merged_id = inverse_vocab.len() as u32;
            merge_priority.insert((a_id, b_id), merges.len());
            merges.push((a_id, b_id, merged_id));
            vocab.insert(merged.clone(), merged_id);
            inverse_vocab.push(merged);
        }
        BpeTokenizer {
            vocab,
            inverse_vocab,
            merges,
            merge_priority,
        }
    }

    /// Encode text to token IDs.
    /// Optimized: applies merges in priority order, scanning once per merge level.
    /// For long texts, splits into chunks to avoid O(n^2) behavior.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.len() > 10_000 {
            // Split long texts into overlapping chunks to avoid O(n^2) merge scanning.
            // Process each chunk independently then concatenate.
            return self.encode_chunked(text, 8_000);
        }
        self.encode_segment(text.as_bytes())
    }

    /// Encode a byte segment using BPE merges.
    ///
    /// Uses a linked-list + priority queue algorithm for O(n log n) encoding
    /// instead of the naive O(n × merges) approach. Each adjacent pair is
    /// inserted into a min-heap keyed by merge priority. We pop the
    /// highest-priority (lowest index) pair, merge it, and update neighbors.
    fn encode_segment(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return Vec::new();
        }

        let n = bytes.len();
        if n == 1 {
            return vec![bytes[0] as u32 + SPECIAL_TOKENS];
        }

        // Doubly-linked list: each node holds a token and links to prev/next.
        // Deleted nodes have next == usize::MAX.
        let mut token: Vec<u32> = bytes.iter().map(|&b| b as u32 + SPECIAL_TOKENS).collect();
        let mut next: Vec<usize> = (1..=n).collect(); // next[n-1] = n (sentinel)
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect(); // prev[0] = usize::MAX

        // Min-heap: (priority, left_index). Lower priority = merge first.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();

        // Seed the heap with all adjacent pairs that have a known merge.
        for i in 0..n - 1 {
            if let Some(&priority) = self.merge_priority.get(&(token[i], token[i + 1])) {
                heap.push(Reverse((priority, i)));
            }
        }

        while let Some(Reverse((priority, left))) = heap.pop() {
            // Validate: left must still be alive and its next must form the expected pair.
            let right = next[left];
            if right >= n {
                continue; // left is the last node or was deleted
            }
            // Check the pair still matches the priority we stored.
            let pair = (token[left], token[right]);
            match self.merge_priority.get(&pair) {
                Some(&p) if p == priority => {} // valid
                _ => continue,                  // pair changed since we enqueued
            }

            // Look up the merged token from the merge rules.
            let merged = self.merges[priority].2;

            // Merge: replace left\'s token, delete right.
            token[left] = merged;
            let right_next = next[right];
            next[left] = right_next;
            if right_next < n {
                prev[right_next] = left;
            }
            // Mark right as deleted.
            next[right] = usize::MAX;

            // Check new pair (prev_of_left, left).
            if prev[left] < n {
                let pl = prev[left];
                if let Some(&p) = self.merge_priority.get(&(token[pl], token[left])) {
                    heap.push(Reverse((p, pl)));
                }
            }

            // Check new pair (left, next_of_left).
            if next[left] < n {
                if let Some(&p) = self.merge_priority.get(&(token[left], token[next[left]])) {
                    heap.push(Reverse((p, left)));
                }
            }
        }

        // Collect surviving tokens by walking the linked list.
        let mut result = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            result.push(token[i]);
            i = next[i];
        }
        result
    }

    /// Encode long text by splitting into chunks, encoding each, and concatenating.
    fn encode_chunked(&self, text: &str, chunk_size: usize) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut all_tokens = Vec::new();

        let mut start = 0;
        while start < bytes.len() {
            // Find a safe split point (don't break UTF-8 or mid-word if possible)
            let end = if start + chunk_size >= bytes.len() {
                bytes.len()
            } else {
                // Try to split at a whitespace boundary
                let target = start + chunk_size;
                let mut split = target;
                // Search backward for whitespace
                while split > start + chunk_size / 2 {
                    if bytes[split] == b' ' || bytes[split] == b'\n' || bytes[split] == b'\t' {
                        split += 1; // include the whitespace in the previous chunk
                        break;
                    }
                    split -= 1;
                }
                if split <= start + chunk_size / 2 {
                    // No whitespace found — include the entire remaining text
                    // in this chunk rather than splitting mid-word, which would
                    // break BPE merges at the boundary.
                    bytes.len()
                } else {
                    split
                }
            };

            let chunk_tokens = self.encode_segment(&bytes[start..end]);
            all_tokens.extend_from_slice(&chunk_tokens);
            start = end;
        }

        all_tokens
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

    /// Check if a byte sequence is a known token in the vocabulary.
    pub fn contains_token(&self, token_bytes: &[u8]) -> bool {
        self.vocab.contains_key(token_bytes)
    }

    /// Print tokenizer statistics: vocab size, merge count, average token length.
    pub fn print_stats(&self) {
        let avg_len = if self.inverse_vocab.is_empty() {
            0.0
        } else {
            self.inverse_vocab.iter().map(|t| t.len()).sum::<usize>() as f64
                / self.inverse_vocab.len() as f64
        };
        eprintln!(
            "Tokenizer: {} tokens in vocab, {} merges, avg token {:.1} bytes",
            self.vocab.len(),
            self.merges.len(),
            avg_len,
        );
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
        let file_len = file.metadata()?.len();
        let mut consumed = 0u64;
        let mut buf4 = [0u8; 4];

        // Magic
        read_tokenizer_exact(&mut file, &mut buf4, &mut consumed)?;
        if &buf4 != b"ABPE" {
            return Err(invalid_tokenizer_data(format!(
                "not a valid tokenizer file: expected ABPE magic, got {:02x?}",
                buf4
            )));
        }

        // Version
        let version = read_tokenizer_u32(&mut file, &mut consumed)?;
        if !(1..=2).contains(&version) {
            return Err(invalid_tokenizer_data(format!(
                "unsupported tokenizer version: {version}"
            )));
        }

        // Vocab size
        let vocab_len = read_tokenizer_u32(&mut file, &mut consumed)? as usize;
        let min_vocab_len = (SPECIAL_TOKENS + 256) as usize;
        if vocab_len < min_vocab_len {
            return Err(invalid_tokenizer_data(format!(
                "tokenizer vocab must include {SPECIAL_TOKENS} special tokens and 256 byte tokens, got {vocab_len}"
            )));
        }
        let min_vocab_bytes = (vocab_len as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| invalid_tokenizer_data("tokenizer vocab section is too large"))?;
        ensure_tokenizer_remaining(
            file_len,
            consumed,
            min_vocab_bytes,
            format!("tokenizer vocab table for {vocab_len} entries"),
        )?;

        let mut inverse_vocab = Vec::with_capacity(vocab_len);
        let mut vocab = HashMap::new();
        for id in 0..vocab_len {
            let len = read_tokenizer_u32(&mut file, &mut consumed)? as usize;
            if len == 0 {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer token {id} has zero length"
                )));
            }
            ensure_tokenizer_remaining(
                file_len,
                consumed,
                len as u64,
                format!("tokenizer token {id} length {len}"),
            )?;
            let mut token_bytes = vec![0u8; len];
            read_tokenizer_exact(&mut file, &mut token_bytes, &mut consumed)?;
            if vocab.insert(token_bytes.clone(), id as u32).is_some() {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer contains duplicate token bytes at id {id}"
                )));
            }
            inverse_vocab.push(token_bytes);
        }
        for (id, expected) in [
            b"<|pad|>".as_slice(),
            b"<|bos|>".as_slice(),
            b"<|eos|>".as_slice(),
        ]
        .iter()
        .enumerate()
        {
            if inverse_vocab[id] != *expected {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer special token id {id} is invalid"
                )));
            }
        }
        for byte in 0u8..=255 {
            let id = byte as usize + SPECIAL_TOKENS as usize;
            if inverse_vocab[id] != [byte] {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer byte token id {id} is invalid"
                )));
            }
        }

        // Merges
        let merge_count = read_tokenizer_u32(&mut file, &mut consumed)? as usize;
        let merge_bytes = (merge_count as u64)
            .checked_mul(12)
            .ok_or_else(|| invalid_tokenizer_data("tokenizer merge section is too large"))?;
        ensure_tokenizer_remaining(
            file_len,
            consumed,
            merge_bytes,
            format!("tokenizer merge table for {merge_count} entries"),
        )?;
        let mut merges = Vec::with_capacity(merge_count);
        let mut merge_priority = HashMap::new();
        for priority in 0..merge_count {
            let a = read_tokenizer_u32(&mut file, &mut consumed)?;
            let b = read_tokenizer_u32(&mut file, &mut consumed)?;
            let c = read_tokenizer_u32(&mut file, &mut consumed)?;
            if a as usize >= vocab_len || b as usize >= vocab_len || c as usize >= vocab_len {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer merge {priority} references token ids ({a}, {b}, {c}) outside vocab size {vocab_len}"
                )));
            }
            if (c as usize) < min_vocab_len {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer merge {priority} output id {c} overlaps the base byte vocabulary"
                )));
            }
            if merge_priority.contains_key(&(a, b)) {
                return Err(invalid_tokenizer_data(format!(
                    "tokenizer merge {priority} duplicates pair ({a}, {b})"
                )));
            }
            merges.push((a, b, c));
            merge_priority.insert((a, b), priority);
        }
        if consumed != file_len {
            return Err(invalid_tokenizer_data(format!(
                "tokenizer file has {} trailing bytes",
                file_len - consumed
            )));
        }

        Ok(Self {
            vocab,
            inverse_vocab,
            merges,
            merge_priority,
        })
    }
}
