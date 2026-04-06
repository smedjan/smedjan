//! Data pipeline tools: download, clean, deduplicate, mix, tokenize.
//! Used to prepare training data from raw sources.

use crate::tokenizer::BpeTokenizer;
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// SHA-256 hash of a byte slice.
pub fn sha256(data: &[u8]) -> String {
    // Simple SHA-256 implementation (no external dependency)
    // We use a streaming approach for large files
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

/// Minimal SHA-256 implementation (FIPS 180-4).
/// No dependency needed for a hash function.
struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: Vec::new(),
            total_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.total_len += data.len() as u64;

        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.process_block(&block);
            self.buffer.drain(..64);
        }
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[i*4], block[i*4+1], block[i*4+2], block[i*4+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g; g = f; f = e; e = d.wrapping_add(temp1);
            d = c; c = b; b = a; a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    fn finalize(mut self) -> String {
        // Padding
        let bit_len = self.total_len * 8;
        self.buffer.push(0x80);
        while (self.buffer.len() % 64) != 56 {
            self.buffer.push(0x00);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());

        // Process remaining blocks
        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.process_block(&block);
            self.buffer.drain(..64);
        }

        // Output
        self.state.iter()
            .map(|s| format!("{:08x}", s))
            .collect()
    }
}

/// Hash a file and return SHA-256 hex string.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize())
}

/// Document quality filter — returns true if the document should be KEPT.
pub fn quality_filter(text: &str) -> bool {
    let len = text.len();

    // Too short or too long
    if !(50..=100_000).contains(&len) {
        return false;
    }

    // Must have reasonable alphanumeric ratio
    let alnum_count = text.chars().filter(|c| c.is_alphanumeric()).count();
    let alnum_ratio = alnum_count as f64 / text.chars().count().max(1) as f64;
    if alnum_ratio < 0.3 {
        return false;
    }

    // Check for excessive line repetition
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > 10 {
        let unique_lines: HashSet<&str> = lines.iter().copied().collect();
        let uniqueness = unique_lines.len() as f64 / lines.len() as f64;
        if uniqueness < 0.3 {
            return false; // >70% repeated lines = auto-generated junk
        }
    }

    // Must have some sentence structure (not just data)
    let has_punctuation = text.contains('.') || text.contains('?') || text.contains('!') || text.contains(':');
    let has_newlines = text.contains('\n');
    if !has_punctuation && !has_newlines {
        return false;
    }

    true
}

/// Normalize text: consistent whitespace, NFC Unicode, strip control chars.
pub fn normalize_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_blank = false;

    for line in text.lines() {
        let trimmed = line.trim_end(); // Remove trailing whitespace

        // Collapse multiple blank lines into one
        if trimmed.is_empty() {
            if !prev_blank {
                result.push('\n');
                prev_blank = true;
            }
            continue;
        }
        prev_blank = false;

        // Strip control characters except tab and newline
        for ch in trimmed.chars() {
            if ch == '\t' || ch >= ' ' {
                result.push(ch);
            }
        }
        result.push('\n');
    }

    // Remove trailing newlines
    while result.ends_with('\n') {
        result.pop();
    }
    result.push('\n');

    result
}

/// Exact deduplication using SHA-256. Returns indices of unique documents.
pub fn exact_dedup(documents: &[String]) -> Vec<usize> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique_indices = Vec::new();

    for (i, doc) in documents.iter().enumerate() {
        let hash = sha256(doc.as_bytes());
        if seen.insert(hash) {
            unique_indices.push(i);
        }
    }

    let removed = documents.len() - unique_indices.len();
    if removed > 0 {
        eprintln!("Exact dedup: removed {} duplicates ({:.1}%)", removed, removed as f64 / documents.len() as f64 * 100.0);
    }

    unique_indices
}

/// Simple n-gram based near-deduplication (simpler than MinHash but effective).
/// Documents with >80% 5-gram overlap are considered duplicates.
pub fn near_dedup(documents: &[String], threshold: f64) -> Vec<usize> {
    let ngram_size = 5;
    let mut unique_indices: Vec<usize> = Vec::new();
    let mut kept_ngrams: Vec<HashSet<u64>> = Vec::new();

    for (i, doc) in documents.iter().enumerate() {
        let ngrams = compute_ngrams(doc, ngram_size);
        if ngrams.is_empty() {
            continue;
        }

        // Check overlap with already-kept documents
        let mut is_duplicate = false;
        for kept in &kept_ngrams {
            let overlap = ngrams.intersection(kept).count();
            let similarity = overlap as f64 / ngrams.len().min(kept.len()).max(1) as f64;
            if similarity > threshold {
                is_duplicate = true;
                break;
            }
        }

        if !is_duplicate {
            unique_indices.push(i);
            kept_ngrams.push(ngrams);
        }
    }

    let removed = documents.len() - unique_indices.len();
    if removed > 0 {
        eprintln!("Near dedup: removed {} near-duplicates ({:.1}%)", removed, removed as f64 / documents.len() as f64 * 100.0);
    }

    unique_indices
}

/// Compute word n-gram hashes for a document.
fn compute_ngrams(text: &str, n: usize) -> HashSet<u64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return HashSet::new();
    }

    let mut ngrams = HashSet::new();
    for window in words.windows(n) {
        // Simple hash: FNV-1a on the concatenated n-gram
        let mut hash: u64 = 0xcbf29ce484222325;
        for word in window {
            for byte in word.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash ^= 0xff; // separator
        }
        ngrams.insert(hash);
    }
    ngrams
}

/// Data mix configuration.
pub struct DataMix {
    pub sources: Vec<DataSource>,
}

pub struct DataSource {
    pub name: String,
    pub path: PathBuf,
    pub weight: f32,      // sampling weight (higher = more frequent)
    pub upsample: usize,  // repeat count (1 = single pass)
}

/// Process a raw text file through the full cleaning pipeline:
/// 1. Split into documents (by separator)
/// 2. Quality filter
/// 3. Normalize
/// 4. Exact dedup
/// 5. Near dedup
/// 6. Tokenize
/// 7. Write as binary shard
pub fn process_source(
    input_path: &Path,
    output_path: &Path,
    tokenizer: &BpeTokenizer,
    separator: &str,
) -> std::io::Result<ProcessStats> {
    eprintln!("Processing: {}", input_path.display());

    // Read and split into documents
    let raw = std::fs::read_to_string(input_path)?;
    let total_bytes = raw.len();
    let documents: Vec<String> = if separator.is_empty() {
        vec![raw]
    } else {
        raw.split(separator)
            .map(|s| s.to_string())
            .filter(|s| !s.trim().is_empty())
            .collect()
    };
    let total_docs = documents.len();

    // Quality filter
    let filtered: Vec<String> = documents
        .into_iter()
        .filter(|d| quality_filter(d))
        .collect();
    let after_quality = filtered.len();

    // Normalize
    let normalized: Vec<String> = filtered
        .into_iter()
        .map(|d| normalize_text(&d))
        .collect();

    // Exact dedup
    let unique_idx = exact_dedup(&normalized);
    let deduped: Vec<String> = unique_idx.iter().map(|&i| normalized[i].clone()).collect();

    // Near dedup
    let near_unique_idx = near_dedup(&deduped, 0.8);
    let final_docs: Vec<String> = near_unique_idx.iter().map(|&i| deduped[i].clone()).collect();
    let final_count = final_docs.len();

    // Tokenize all documents
    let mut all_tokens: Vec<u32> = Vec::new();
    for doc in &final_docs {
        let tokens = tokenizer.encode(doc);
        all_tokens.extend_from_slice(&tokens);
        all_tokens.push(crate::tokenizer::EOS_TOKEN); // document separator
    }

    // Write binary shard (batch write — avoids syscall per token)
    let byte_data: Vec<u8> = all_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(output_path, &byte_data)?;

    // Write hash
    let hash = sha256_file(output_path)?;
    let hash_path = output_path.with_extension("sha256");
    std::fs::write(&hash_path, &hash)?;

    let stats = ProcessStats {
        input_bytes: total_bytes,
        input_docs: total_docs,
        after_quality_filter: after_quality,
        after_dedup: final_count,
        output_tokens: all_tokens.len(),
        output_bytes: all_tokens.len() * 4,
        sha256: hash,
    };

    eprintln!(
        "  {} docs → {} after quality → {} after dedup → {} tokens",
        stats.input_docs, stats.after_quality_filter, stats.after_dedup, stats.output_tokens
    );

    Ok(stats)
}

pub struct ProcessStats {
    pub input_bytes: usize,
    pub input_docs: usize,
    pub after_quality_filter: usize,
    pub after_dedup: usize,
    pub output_tokens: usize,
    pub output_bytes: usize,
    pub sha256: String,
}

/// Mix multiple tokenized shards into a single training dataset with specified weights.
/// Interleaves documents from different sources based on weight ratios.
pub fn mix_shards(
    shard_paths: &[(PathBuf, f32)], // (path, weight)
    output_path: &Path,
) -> std::io::Result<usize> {
    eprintln!("Mixing {} shards...", shard_paths.len());

    // Load all shards
    let mut shards: Vec<(Vec<u32>, f32)> = Vec::new();
    for (path, weight) in shard_paths {
        let data = std::fs::read(path)?;
        let tokens: Vec<u32> = data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("  {} — {} tokens, weight {:.2}", path.display(), tokens.len(), weight);
        shards.push((tokens, *weight));
    }

    // Normalize weights
    let total_weight: f32 = shards.iter().map(|(_, w)| w).sum();
    assert!(total_weight > 0.0, "Data mixing weights must sum to > 0 (got {})", total_weight);

    // Compute how many tokens to take from each shard
    let total_tokens: usize = shards.iter().map(|(t, _)| t.len()).sum();
    let mut output: Vec<u32> = Vec::with_capacity(total_tokens);

    // Round-robin sampling based on weights
    let mut positions: Vec<usize> = vec![0; shards.len()];
    let chunk_size = 1024; // tokens per chunk

    loop {
        let mut any_remaining = false;
        for (i, (tokens, weight)) in shards.iter().enumerate() {
            let proportion = weight / total_weight;
            let chunks_this_round = (proportion * 10.0).ceil() as usize; // proportional chunks

            for _ in 0..chunks_this_round {
                if positions[i] >= tokens.len() {
                    continue;
                }
                any_remaining = true;
                let end = (positions[i] + chunk_size).min(tokens.len());
                output.extend_from_slice(&tokens[positions[i]..end]);
                positions[i] = end;
            }
        }
        if !any_remaining {
            break;
        }
    }

    // Write output
    let mut file = std::fs::File::create(output_path)?;
    for &token in &output {
        file.write_all(&token.to_le_bytes())?;
    }

    let hash = sha256_file(output_path)?;
    let hash_path = output_path.with_extension("sha256");
    std::fs::write(&hash_path, &hash)?;

    eprintln!("Mixed dataset: {} tokens, hash: {}", output.len(), &hash[..16]);
    Ok(output.len())
}

/// Record data provenance: source URL, license, hash, token count.
pub fn record_provenance(
    log_path: &Path,
    source_name: &str,
    source_url: &str,
    license: &str,
    stats: &ProcessStats,
) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    writeln!(
        file,
        r#"{{"source":"{}","url":"{}","license":"{}","input_bytes":{},"input_docs":{},"output_tokens":{},"sha256":"{}"}}"#,
        source_name, source_url, license, stats.input_bytes, stats.input_docs, stats.output_tokens, stats.sha256
    )?;

    Ok(())
}

// ============================================================
// Phase 3: Data Quality Pipeline
// ============================================================

/// MinHash signature for near-duplicate detection.
/// Uses k independent hash functions to create a signature per document.
pub struct MinHashSignature {
    pub hashes: Vec<u64>,
}

/// Compute MinHash signature for a document (as list of shingle hashes).
pub fn minhash_signature(text: &str, num_hashes: usize, shingle_size: usize) -> MinHashSignature {
    // Generate shingles (n-grams of characters)
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < shingle_size {
        return MinHashSignature { hashes: vec![u64::MAX; num_hashes] };
    }

    let mut min_hashes = vec![u64::MAX; num_hashes];

    for i in 0..=(chars.len() - shingle_size) {
        let shingle: String = chars[i..i + shingle_size].iter().collect();
        let base_hash = fnv_hash(shingle.as_bytes());

        for (h, min_hash) in min_hashes.iter_mut().enumerate().take(num_hashes) {
            // Use different hash functions by XORing with seed
            let hash = base_hash.wrapping_mul(6364136223846793005u64.wrapping_add(h as u64 * 1442695040888963407));
            if hash < *min_hash {
                *min_hash = hash;
            }
        }
    }

    MinHashSignature { hashes: min_hashes }
}

/// Jaccard similarity estimate from MinHash signatures.
pub fn minhash_similarity(a: &MinHashSignature, b: &MinHashSignature) -> f32 {
    assert_eq!(a.hashes.len(), b.hashes.len());
    let matches = a.hashes.iter().zip(&b.hashes).filter(|(a, b)| a == b).count();
    matches as f32 / a.hashes.len() as f32
}

/// FNV-1a hash (fast, non-cryptographic).
fn fnv_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deduplicate documents using MinHash. Returns indices of documents to KEEP.
/// threshold: Jaccard similarity above which documents are considered duplicates (e.g. 0.8)
pub fn minhash_dedup(documents: &[String], threshold: f32, num_hashes: usize) -> Vec<usize> {
    let signatures: Vec<MinHashSignature> = documents.iter()
        .map(|doc| minhash_signature(doc, num_hashes, 5))
        .collect();

    let mut keep = Vec::new();
    let mut kept_sigs: Vec<&MinHashSignature> = Vec::new();

    for (i, sig) in signatures.iter().enumerate() {
        let is_dup = kept_sigs.iter().any(|kept| minhash_similarity(sig, kept) > threshold);
        if !is_dup {
            keep.push(i);
            kept_sigs.push(sig);
        }
    }

    keep
}

/// Score document quality by simple heuristics (no model needed).
/// Returns 0.0 (garbage) to 1.0 (high quality).
pub fn quality_score(text: &str) -> f32 {
    let len = text.len();
    if len < 50 { return 0.0; }   // too short
    if len > 100_000 { return 0.3; } // suspiciously long

    let mut score = 0.5f32;

    // Proportion of alphabetic characters (vs special chars, numbers)
    let alpha_ratio = text.chars().filter(|c| c.is_alphabetic() || c.is_whitespace()).count() as f32 / len as f32;
    if alpha_ratio > 0.7 { score += 0.2; }
    if alpha_ratio < 0.4 { score -= 0.3; }

    // Average word length (gibberish tends to have very long "words")
    let words: Vec<&str> = text.split_whitespace().collect();
    if !words.is_empty() {
        let avg_word_len = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
        if avg_word_len > 2.0 && avg_word_len < 15.0 { score += 0.1; }
        if avg_word_len > 30.0 { score -= 0.3; }
    }

    // Sentence structure (has periods, not all caps)
    if text.contains(". ") || text.contains(".\n") { score += 0.1; }
    let caps_ratio = text.chars().filter(|c| c.is_uppercase()).count() as f32 / len.max(1) as f32;
    if caps_ratio > 0.5 { score -= 0.2; } // mostly CAPS = low quality

    // Repetition check (same line repeated)
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > 3 {
        let unique_lines: HashSet<&str> = lines.iter().copied().collect();
        let unique_ratio = unique_lines.len() as f32 / lines.len() as f32;
        if unique_ratio < 0.5 { score -= 0.3; } // very repetitive
    }

    score.clamp(0.0, 1.0)
}

/// Filter documents by quality threshold.
pub fn quality_filter_batch(documents: &[String], min_quality: f32) -> Vec<usize> {
    documents.iter().enumerate()
        .filter(|(_, doc)| quality_score(doc) >= min_quality)
        .map(|(i, _)| i)
        .collect()
}
