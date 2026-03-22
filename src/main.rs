mod attention;
mod autograd;
mod checkpoint;
mod data;
mod datapipe;
mod generate;
mod loss;
mod metal;
mod model;
mod optim;
mod tensor;
mod tokenizer;
mod train;

use clap::{Parser, Subcommand};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "andreai", about = "AndreAI — Pure Rust AI Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a BPE tokenizer from a text corpus
    Tokenizer {
        #[arg(long)]
        input: String,
        #[arg(long, default_value = "32000")]
        vocab_size: u32,
        #[arg(long, default_value = "tokenizer.bin")]
        output: String,
    },

    /// Prepare a dataset (tokenize raw text to binary format)
    Prepare {
        #[arg(long)]
        input: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long, default_value = "dataset.bin")]
        output: String,
    },

    /// Train a model
    Train {
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        tokenizer: String,
        /// Preset: tiny, small, medium, large, max — or "custom" with --dim/--layers/--heads/--ffn-mult
        #[arg(long, default_value = "small")]
        size: String,
        /// Custom: embedding dimension (requires --size custom)
        #[arg(long)]
        dim: Option<usize>,
        /// Custom: number of transformer layers
        #[arg(long)]
        layers: Option<usize>,
        /// Custom: number of attention heads
        #[arg(long)]
        heads: Option<usize>,
        /// Custom: FFN hidden size multiplier (d_ff = dim * ffn_mult)
        #[arg(long)]
        ffn_mult: Option<f32>,
        /// Custom: maximum sequence length
        #[arg(long)]
        max_seq: Option<usize>,
        #[arg(long, default_value = "32")]
        batch_size: usize,
        #[arg(long, default_value = "256")]
        seq_len: usize,
        #[arg(long, default_value = "50000")]
        steps: u32,
        #[arg(long, default_value = "3e-4")]
        lr: f32,
        #[arg(long, default_value = "2000")]
        warmup: u32,
        #[arg(long, default_value = "checkpoints")]
        checkpoint_dir: String,
        /// Enable gradient checkpointing (trades 2x compute for ~60% less activation memory)
        #[arg(long, default_value = "false")]
        gradient_checkpointing: bool,
    },

    /// Show available model sizes and their param counts
    Sizes {
        #[arg(long, default_value = "8192")]
        vocab_size: u32,
    },

    /// Generate text from a trained model
    Generate {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "0.95")]
        top_p: f32,
        #[arg(long, default_value = "50")]
        top_k: usize,
        #[arg(long, default_value = "false")]
        stream: bool,
    },

    /// Show model info from a checkpoint
    Info {
        #[arg(long)]
        checkpoint: String,
    },

    /// Process a raw text file through the data cleaning pipeline
    Process {
        #[arg(long)]
        input: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long)]
        output: String,
        /// Document separator (e.g. "\n---\n" for markdown sections). Empty = single document.
        #[arg(long, default_value = "\n\n")]
        separator: String,
        /// Record provenance to this log file
        #[arg(long)]
        provenance_log: Option<String>,
        /// Source name for provenance
        #[arg(long, default_value = "unknown")]
        source_name: String,
        /// Source URL for provenance
        #[arg(long, default_value = "")]
        source_url: String,
        /// License for provenance
        #[arg(long, default_value = "unknown")]
        license: String,
    },

    /// Mix multiple tokenized shards into a training dataset
    Mix {
        /// Shard files with weights: path1:weight1,path2:weight2,...
        #[arg(long)]
        shards: String,
        #[arg(long)]
        output: String,
    },

    /// Compute SHA-256 hash of a file
    Hash {
        #[arg(long)]
        file: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let ctx = metal::MetalContext::new();

    eprintln!("AndreAI v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Metal device: {}", ctx.device_name());

    match cli.command {
        Commands::Tokenizer {
            input,
            vocab_size,
            output,
        } => {
            eprintln!("Training BPE tokenizer: vocab_size={}", vocab_size);
            let corpus = std::fs::read(&input).expect("Failed to read input file");
            let tok = tokenizer::BpeTokenizer::train(&corpus, vocab_size);
            tok.save(&output).expect("Failed to save tokenizer");
            eprintln!("Tokenizer saved to {}", output);

            // Quick test
            let test = "Hello, world! This is a test.";
            let encoded = tok.encode(test);
            let decoded = tok.decode(&encoded);
            eprintln!(
                "Test: \"{}\" → {} tokens → \"{}\"",
                test,
                encoded.len(),
                decoded
            );
        }

        Commands::Prepare {
            input,
            tokenizer: tok_path,
            output,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let n = data::prepare_dataset(&input, &tok, &output).expect("Failed to prepare dataset");
            eprintln!("Dataset ready: {} tokens", n);
        }

        Commands::Train {
            dataset,
            tokenizer: tok_path,
            size,
            dim,
            layers,
            heads,
            ffn_mult,
            max_seq,
            batch_size,
            seq_len,
            steps,
            lr,
            warmup,
            checkpoint_dir,
            gradient_checkpointing,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let vocab_size = tok.vocab_size();

            let model_config = match size.as_str() {
                "tiny" => model::ModelConfig::tiny(vocab_size),
                "small" => model::ModelConfig::small(vocab_size),
                "medium" => model::ModelConfig::medium(vocab_size),
                "large" => model::ModelConfig::large(vocab_size),
                "xl" => model::ModelConfig::xl(vocab_size),
                "max" => model::ModelConfig::max(vocab_size),
                "huge" => model::ModelConfig::huge(vocab_size),
                "8b" => model::ModelConfig::eight_b(vocab_size),
                "custom" => {
                    let d = dim.expect("--dim required for custom size");
                    let l = layers.expect("--layers required for custom size");
                    let h = heads.expect("--heads required for custom size");
                    let fm = ffn_mult.unwrap_or(2.67);
                    let ms = max_seq.unwrap_or(512);
                    model::ModelConfig::custom(vocab_size, d, h, l, fm, ms)
                }
                _ => panic!(
                    "Unknown model size: '{}'. Use: tiny, small, medium, large, xl, max, huge, 8b, custom",
                    size
                ),
            };

            eprintln!("Config: {}", model_config.summary());

            let config = train::TrainConfig {
                model_config,
                dataset_path: dataset,
                tokenizer_path: tok_path,
                checkpoint_dir,
                batch_size,
                seq_len,
                total_steps: steps,
                max_lr: lr,
                warmup_steps: warmup,
                weight_decay: 0.1,
                max_grad_norm: 1.0,
                log_interval: 10,
                checkpoint_interval: 5000,
                gradient_checkpointing,
            };

            train::train(&ctx, &config).expect("Training failed");
        }

        Commands::Generate {
            checkpoint: ckpt_path,
            tokenizer: tok_path,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stream,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let (model, step) =
                checkpoint::load_checkpoint(&ctx, &ckpt_path).expect("Failed to load checkpoint");
            eprintln!("Loaded model at step {}", step);

            let config = generate::SamplingConfig {
                temperature,
                top_p,
                top_k,
                max_tokens,
            };

            if stream {
                print!("{}", prompt);
                generate::generate_streaming(&ctx, &model, &tok, &prompt, &config, |token_str| {
                    print!("{}", token_str);
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                });
                println!();
            } else {
                let output = generate::generate(&ctx, &model, &tok, &prompt, &config);
                println!("{}{}", prompt, output);
            }
        }

        Commands::Info {
            checkpoint: ckpt_path,
        } => {
            let (model, step) =
                checkpoint::load_checkpoint(&ctx, &ckpt_path).expect("Failed to load checkpoint");
            let c = &model.config;
            println!("AndreAI Model Checkpoint");
            println!("  Step: {}", step);
            println!("  Parameters: {}M", c.param_count() as f32 / 1e6);
            println!("  Vocab size: {}", c.vocab_size);
            println!("  d_model: {}", c.d_model);
            println!("  n_heads: {}", c.n_heads);
            println!("  n_layers: {}", c.n_layers);
            println!("  d_ff: {}", c.d_ff());
            println!("  ffn_multiplier: {}", c.ffn_multiplier);
            println!("  max_seq_len: {}", c.max_seq_len);
            println!("  RoPE theta: {}", c.rope_theta);
            println!("  Training RAM: {:.0} MB", c.training_memory_bytes() as f64 / (1024.0 * 1024.0));
            println!("  Inference RAM: {:.0} MB", c.inference_memory_bytes() as f64 / (1024.0 * 1024.0));
        }

        Commands::Sizes { vocab_size } => {
            println!("AndreAI Model Sizes (vocab={})", vocab_size);
            println!();
            let presets: Vec<(&str, model::ModelConfig)> = vec![
                ("tiny", model::ModelConfig::tiny(vocab_size)),
                ("small", model::ModelConfig::small(vocab_size)),
                ("medium", model::ModelConfig::medium(vocab_size)),
                ("large", model::ModelConfig::large(vocab_size)),
                ("xl", model::ModelConfig::xl(vocab_size)),
                ("max", model::ModelConfig::max(vocab_size)),
                ("huge", model::ModelConfig::huge(vocab_size)),
                ("8b", model::ModelConfig::eight_b(vocab_size)),
            ];
            for (name, cfg) in &presets {
                println!(
                    "  {:>6}  dim={:>4}  layers={:>2}  heads={:>2}  d_ff={:>5}  seq={:>4}  params={:>8.1}M  train={:>7.0}MB  infer={:>6.0}MB",
                    name,
                    cfg.d_model,
                    cfg.n_layers,
                    cfg.n_heads,
                    cfg.d_ff(),
                    cfg.max_seq_len,
                    cfg.param_count() as f64 / 1e6,
                    cfg.training_memory_bytes() as f64 / (1024.0 * 1024.0),
                    cfg.inference_memory_bytes() as f64 / (1024.0 * 1024.0),
                );
            }
            println!();
            println!("Or use --size custom with --dim --layers --heads --ffn-mult --max-seq for any arbitrary config.");
        }

        Commands::Process {
            input,
            tokenizer: tok_path,
            output,
            separator,
            provenance_log,
            source_name,
            source_url,
            license,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let stats = datapipe::process_source(
                std::path::Path::new(&input),
                std::path::Path::new(&output),
                &tok,
                &separator,
            )
            .expect("Processing failed");

            if let Some(log_path) = provenance_log {
                datapipe::record_provenance(
                    std::path::Path::new(&log_path),
                    &source_name,
                    &source_url,
                    &license,
                    &stats,
                )
                .expect("Failed to write provenance log");
            }

            println!(
                "Processed: {} docs → {} tokens ({} bytes)",
                stats.after_dedup, stats.output_tokens, stats.output_bytes
            );
            println!("SHA-256: {}", stats.sha256);
        }

        Commands::Mix { shards, output } => {
            let shard_pairs: Vec<(std::path::PathBuf, f32)> = shards
                .split(',')
                .map(|entry| {
                    let parts: Vec<&str> = entry.splitn(2, ':').collect();
                    let path = std::path::PathBuf::from(parts[0]);
                    let weight = if parts.len() > 1 {
                        parts[1].parse::<f32>().expect("Invalid weight")
                    } else {
                        1.0
                    };
                    (path, weight)
                })
                .collect();

            let total = datapipe::mix_shards(&shard_pairs, std::path::Path::new(&output))
                .expect("Mixing failed");
            println!("Mixed dataset: {} tokens", total);
        }

        Commands::Hash { file } => {
            let hash = datapipe::sha256_file(std::path::Path::new(&file))
                .expect("Failed to hash file");
            println!("{}", hash);
        }
    }
}
