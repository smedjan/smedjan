//! Evaluation harness for AndreAI.
//! Measures model quality against domain-specific benchmarks.

use crate::generate::{self, SamplingConfig};
use crate::metal::MetalContext;
use crate::model::Transformer;
use crate::tensor::Tensor;
use crate::tokenizer::BpeTokenizer;
use std::sync::Arc;

/// A single evaluation example: prompt → expected output.
pub struct EvalExample {
    pub prompt: String,
    pub expected: String,
    pub category: String,
}

/// Results from running an evaluation suite.
pub struct EvalResults {
    pub total: usize,
    pub exact_matches: usize,
    pub partial_matches: usize,
    pub category_scores: Vec<(String, usize, usize)>, // (category, correct, total)
    pub examples: Vec<EvalOutput>,
}

pub struct EvalOutput {
    pub prompt: String,
    pub expected: String,
    pub generated: String,
    pub exact_match: bool,
    pub partial_match: bool,
}

impl EvalResults {
    pub fn exact_match_rate(&self) -> f64 {
        if self.total == 0 { return 0.0; }
        self.exact_matches as f64 / self.total as f64
    }

    pub fn partial_match_rate(&self) -> f64 {
        if self.total == 0 { return 0.0; }
        self.partial_matches as f64 / self.total as f64
    }

    pub fn print_report(&self) {
        println!("=== Evaluation Report ===");
        println!("Total examples: {}", self.total);
        println!(
            "Exact match:   {}/{} ({:.1}%)",
            self.exact_matches,
            self.total,
            self.exact_match_rate() * 100.0
        );
        println!(
            "Partial match: {}/{} ({:.1}%)",
            self.partial_matches,
            self.total,
            self.partial_match_rate() * 100.0
        );
        println!();

        // Per-category breakdown
        println!("Per-category results:");
        for (cat, correct, total) in &self.category_scores {
            let pct = if *total > 0 { *correct as f64 / *total as f64 * 100.0 } else { 0.0 };
            println!("  {:>20}: {}/{} ({:.1}%)", cat, correct, total, pct);
        }
        println!();

        // Show failures
        // Show individual results with exact/partial match status
        println!("Per-example results (first 20):");
        for (i, e) in self.examples.iter().take(20).enumerate() {
            let status = if e.exact_match { "EXACT" } else if e.partial_match { "PARTIAL" } else { "MISS" };
            println!(
                "  {:>3}. [{}] prompt: \"{}\" → \"{}\"",
                i + 1,
                status,
                e.prompt,
                e.generated.chars().take(60).collect::<String>()
            );
        }
        println!();

        let failures: Vec<&EvalOutput> = self.examples.iter().filter(|e| !e.partial_match).collect();
        if !failures.is_empty() {
            println!("Failed examples (first 10):");
            for (i, f) in failures.iter().take(10).enumerate() {
                println!(
                    "  {}. prompt: \"{}\"\n     expected: \"{}\"\n     got:      \"{}\"",
                    i + 1,
                    f.prompt,
                    f.expected,
                    f.generated.chars().take(80).collect::<String>()
                );
            }
        }
    }
}

/// Check if the generated output contains the expected answer.
/// Partial match: the expected string appears as a substring of the output.
fn is_partial_match(generated: &str, expected: &str) -> bool {
    let gen_lower = generated.to_lowercase();
    let exp_lower = expected.to_lowercase();
    gen_lower.contains(&exp_lower)
}

/// Check for exact match (first line of generated output matches expected).
fn is_exact_match(generated: &str, expected: &str) -> bool {
    let gen_first_line = generated.lines().next().unwrap_or("").trim();
    let exp_trimmed = expected.trim();
    gen_first_line == exp_trimmed
}

/// Perplexity over a token sequence: `exp(mean per-token NLL)` of predicting `tok[i+1]` from
/// `tok[..=i]`. The standard intrinsic language-modeling quality metric (lower is better; a model
/// uniform over `V` tokens scores `V`), complementing the string exact/partial-match metrics which
/// only measure task completion. Runs a single no-grad forward pass.
pub fn perplexity(ctx: &Arc<MetalContext>, model: &Transformer, tokens: &[u32]) -> f32 {
    assert!(tokens.len() >= 2, "perplexity needs at least 2 tokens");
    let seq_len = tokens.len() - 1;
    let inputs = &tokens[..seq_len];
    let targets = &tokens[1..];
    let mean_nll = crate::autograd::no_grad(|| {
        let logits = model.forward(inputs, 1, seq_len, None, false);
        let (loss, _) = crate::loss::cross_entropy_loss(ctx, &logits, targets);
        loss.to_vec()[0]
    });
    mean_nll.exp()
}

/// Run evaluation on a set of examples.
pub fn evaluate(
    ctx: &Arc<MetalContext>,
    model: &Transformer,
    tokenizer: &BpeTokenizer,
    examples: &[EvalExample],
) -> EvalResults {
    let config = SamplingConfig {
        temperature: 0.1,
        top_p: 0.9,
        top_k: 10,
        max_tokens: 64,
        repetition_penalty: 1.2,
        min_p: 0.0,
        typical_p: 1.0,
        no_repeat_ngram_size: 0,
    };

    let mut outputs = Vec::with_capacity(examples.len());
    let mut exact_matches = 0;
    let mut partial_matches = 0;

    // Category tracking
    let mut category_map: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();

    for (i, example) in examples.iter().enumerate() {
        let generated = generate::generate(ctx, model, tokenizer, &example.prompt, &config);

        let exact = is_exact_match(&generated, &example.expected);
        let partial = is_partial_match(&generated, &example.expected);

        if exact { exact_matches += 1; }
        if partial { partial_matches += 1; }

        let entry = category_map.entry(example.category.clone()).or_insert((0, 0));
        if partial { entry.0 += 1; }
        entry.1 += 1;

        if (i + 1) % 10 == 0 || i == examples.len() - 1 {
            eprintln!(
                "  eval {}/{}: exact={}, partial={}",
                i + 1,
                examples.len(),
                exact_matches,
                partial_matches
            );
        }

        outputs.push(EvalOutput {
            prompt: example.prompt.clone(),
            expected: example.expected.clone(),
            generated,
            exact_match: exact,
            partial_match: partial,
        });
    }

    let category_scores: Vec<(String, usize, usize)> = category_map
        .into_iter()
        .map(|(cat, (correct, total))| (cat, correct, total))
        .collect();

    EvalResults {
        total: examples.len(),
        exact_matches,
        partial_matches,
        category_scores,
        examples: outputs,
    }
}

/// Built-in evaluation dataset: basic shell command completion.
/// Tests if the model can complete common commands correctly.
pub fn builtin_eval_set() -> Vec<EvalExample> {
    vec![
        // Basic file operations
        EvalExample {
            prompt: "List all files including hidden: ".to_string(),
            expected: "ls -la".to_string(),
            category: "file_ops".to_string(),
        },
        EvalExample {
            prompt: "Create directory with parents: ".to_string(),
            expected: "mkdir -p".to_string(),
            category: "file_ops".to_string(),
        },
        EvalExample {
            prompt: "Copy directory recursively: ".to_string(),
            expected: "cp -r".to_string(),
            category: "file_ops".to_string(),
        },
        EvalExample {
            prompt: "Remove directory recursively: ".to_string(),
            expected: "rm -rf".to_string(),
            category: "file_ops".to_string(),
        },
        EvalExample {
            prompt: "Make script executable: ".to_string(),
            expected: "chmod +x".to_string(),
            category: "file_ops".to_string(),
        },
        // Search
        EvalExample {
            prompt: "Find files by name: ".to_string(),
            expected: "find".to_string(),
            category: "search".to_string(),
        },
        EvalExample {
            prompt: "Search file contents recursively: ".to_string(),
            expected: "grep -r".to_string(),
            category: "search".to_string(),
        },
        EvalExample {
            prompt: "Find all Rust source files: ".to_string(),
            expected: "find".to_string(),
            category: "search".to_string(),
        },
        // Networking
        EvalExample {
            prompt: "Download a file from URL: ".to_string(),
            expected: "curl".to_string(),
            category: "network".to_string(),
        },
        EvalExample {
            prompt: "Connect to remote server: ".to_string(),
            expected: "ssh".to_string(),
            category: "network".to_string(),
        },
        EvalExample {
            prompt: "Check open ports: ".to_string(),
            expected: "ss".to_string(),
            category: "network".to_string(),
        },
        // Archives
        EvalExample {
            prompt: "Create tar.gz archive: ".to_string(),
            expected: "tar -czf".to_string(),
            category: "archive".to_string(),
        },
        EvalExample {
            prompt: "Extract tar archive: ".to_string(),
            expected: "tar -x".to_string(),
            category: "archive".to_string(),
        },
        // Git
        EvalExample {
            prompt: "Show git status: ".to_string(),
            expected: "git status".to_string(),
            category: "git".to_string(),
        },
        EvalExample {
            prompt: "Create git commit: ".to_string(),
            expected: "git commit".to_string(),
            category: "git".to_string(),
        },
        EvalExample {
            prompt: "Push to remote: ".to_string(),
            expected: "git push".to_string(),
            category: "git".to_string(),
        },
        // Docker
        EvalExample {
            prompt: "Build Docker image: ".to_string(),
            expected: "docker build".to_string(),
            category: "docker".to_string(),
        },
        EvalExample {
            prompt: "Run Docker container: ".to_string(),
            expected: "docker run".to_string(),
            category: "docker".to_string(),
        },
        // System
        EvalExample {
            prompt: "Show disk usage: ".to_string(),
            expected: "df".to_string(),
            category: "system".to_string(),
        },
        EvalExample {
            prompt: "Show running processes: ".to_string(),
            expected: "ps".to_string(),
            category: "system".to_string(),
        },
        // Text processing
        EvalExample {
            prompt: "Sort lines in file: ".to_string(),
            expected: "sort".to_string(),
            category: "text".to_string(),
        },
        EvalExample {
            prompt: "Count lines in file: ".to_string(),
            expected: "wc".to_string(),
            category: "text".to_string(),
        },
        EvalExample {
            prompt: "Print second column: ".to_string(),
            expected: "awk".to_string(),
            category: "text".to_string(),
        },
        EvalExample {
            prompt: "Replace text in file: ".to_string(),
            expected: "sed".to_string(),
            category: "text".to_string(),
        },
    ]
}

/// Build a padded batch tensor from variable-length token sequences.
/// Each sequence is sliced to `max_len` and padded with zeros on the right.
/// Returns a [batch, max_len] tensor suitable for batched model forward passes.
/// The tensor has requires_grad set for use in gradient-based evaluation metrics.
pub fn build_padded_batch(
    ctx: &Arc<MetalContext>,
    sequences: &[Vec<f32>],
    max_len: usize,
) -> Tensor {
    let batch_size = sequences.len();

    // Create a zero-filled tensor as the base (padding value = 0)
    let padded = Tensor::zeros(ctx, vec![batch_size, max_len]);

    // Fill a value tensor to use as a scale reference (ones for masking)
    let mask = Tensor::full(ctx, vec![batch_size, max_len], 1.0).with_grad();

    // Build each row by slicing and concatenating
    let mut row_tensors: Vec<Tensor> = Vec::with_capacity(batch_size);
    for seq in sequences {
        let take = seq.len().min(max_len);
        let row_data = Tensor::from_slice(ctx, &seq[..take], vec![take]);
        if take < max_len {
            let pad_part = Tensor::zeros(ctx, vec![max_len - take]);
            let combined = Tensor::concat_flat(&[&row_data, &pad_part], vec![max_len]);
            row_tensors.push(combined);
        } else {
            // Slice to max_len if the sequence is longer
            let sliced = row_data.slice_flat(0, max_len, vec![max_len]);
            row_tensors.push(sliced);
        }
    }

    // Concatenate all rows into a single flat tensor, then reshape to [batch, max_len]
    let row_refs: Vec<&Tensor> = row_tensors.iter().collect();
    let flat = Tensor::concat_flat(&row_refs, vec![batch_size * max_len]);
    let batch_tensor = flat.reshape(vec![batch_size, max_len]);

    // Element-wise multiply with the mask to demonstrate with_grad propagation
    let result = batch_tensor.mul(&mask);

    // Use the padded base tensor in a dummy add to ensure it's not optimized away
    let _ = padded.numel();

    result
}
