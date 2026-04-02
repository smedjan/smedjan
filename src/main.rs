pub mod api;
mod attention;
mod autograd;
mod checkpoint;
mod data;
mod datapipe;
mod dpo;
mod eval;
mod generate;
mod loss;
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "andreos")]
mod andreos;
mod distill;
mod gpu;
mod model;
mod optim;
pub mod quantize;
mod tensor;
mod tokenizer;
mod sft;
mod train;

#[cfg(test)]
mod tests;

use clap::{Parser, Subcommand};

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
        /// Custom: number of key/value heads for Grouped Query Attention (defaults to --heads)
        #[arg(long)]
        kv_heads: Option<usize>,
        /// MoE: number of expert FFNs (1 = dense). Default: 1
        #[arg(long, default_value = "1")]
        n_experts: usize,
        /// MoE: top-K experts per token. Default: 1
        #[arg(long, default_value = "1")]
        top_k_experts: usize,
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
        /// Knowledge distillation: path to teacher model checkpoint
        #[arg(long)]
        teacher_checkpoint: Option<String>,
        /// Distillation temperature (softens distributions, higher = softer). Default: 4.0
        #[arg(long, default_value = "4.0")]
        distill_temperature: f32,
        /// Distillation alpha: loss = alpha * T^2 * KL + (1-alpha) * CE. Default: 0.5
        #[arg(long, default_value = "0.5")]
        distill_alpha: f32,
        /// Gradient accumulation steps. Effective batch = batch_size * grad_accum. Default: 1
        #[arg(long, default_value = "1")]
        grad_accum: u32,
        /// Resume training from a saved training state file
        #[arg(long)]
        resume: Option<String>,
        /// Validation dataset path (eval every checkpoint_interval steps)
        #[arg(long)]
        val_dataset: Option<String>,
        /// Dropout rate (0.0 = no dropout). Default: 0.0
        #[arg(long, default_value = "0.0")]
        dropout: f32,
        /// LR warm restart period (steps). 0 = standard cosine. Default: 0
        #[arg(long, default_value = "0")]
        lr_restart: u32,
        /// μP base width for hyperparameter transfer. 0 = disabled. Default: 0
        #[arg(long, default_value = "0")]
        mup_base: usize,
        /// BitNet: use ternary weights in FFN (no float multiply). Default: false
        #[arg(long)]
        bitnet: bool,
        /// Low-rank FFN training: decompose W=[d,ff] into U=[d,r]×V=[r,ff]. 0=full rank.
        #[arg(long, default_value = "0")]
        lowrank: usize,
        /// ALBERT: share weights across all layers (1 unique layer, N iterations)
        #[arg(long)]
        shared_layers: bool,
        /// Data pruning: skip batches where loss < threshold. 0.0=disabled. Try 8.0 after warmup.
        #[arg(long, default_value = "0.0")]
        prune_threshold: f32,
        /// GALORE: gradient low-rank projection rank. 0=disabled. Saves optimizer memory.
        #[arg(long, default_value = "0")]
        galore_rank: usize,
        /// Speculative pretraining: reference model checkpoint. Skip batches it already knows.
        #[arg(long)]
        reference_model: Option<String>,
        /// Speculative threshold: skip if reference loss < this value. Default: 7.0
        #[arg(long, default_value = "7.0")]
        speculative_threshold: f32,
        /// Optimizer: "adamw" or "sophia". Default: adamw
        #[arg(long, default_value = "adamw")]
        optimizer: String,
        /// Multi-token prediction: number of extra heads (0=standard, 4=recommended). 4x sample efficiency.
        #[arg(long, default_value = "0")]
        n_predict: usize,
        /// Curriculum learning: ramp seq_len from short→full over first 25% of training
        #[arg(long)]
        curriculum: bool,
        /// Z-loss coefficient: penalize large logits. 0=off, 1e-4=recommended for MoE
        #[arg(long, default_value = "0.0")]
        z_loss: f32,
        /// Stochastic depth: layer drop rate. 0=off, 0.1=recommended for deep models
        #[arg(long, default_value = "0.0")]
        stochastic_depth: f32,
        /// Sliding window attention size. 0=full causal, 1024=attend last 1024 tokens
        #[arg(long, default_value = "0")]
        sliding_window: usize,
        /// FP16 activation compression between layers. Halves inter-layer memory.
        #[arg(long)]
        fp16_activations: bool,
        /// LR schedule: "cosine" (default) or "wsd" (warmup-stable-decay, 5-10% better)
        #[arg(long, default_value = "cosine")]
        lr_schedule: String,
        /// Self-distillation EMA decay. 0=off, 0.999=recommended. EMA teacher improves sample efficiency.
        #[arg(long, default_value = "0.0")]
        ema_decay: f32,
        /// Anti-PGD noise scale. 0=off, 0.01=recommended. Anticorrelated noise for flatter minima.
        #[arg(long, default_value = "0.0")]
        noise_scale: f32,
        /// ReLoRA merge interval. 0=off, 5000=recommended. Merge lowrank weights for rank growth.
        #[arg(long, default_value = "0")]
        relora_interval: u32,
        /// Fused linear+cross-entropy: compute logits in chunks, save ~2GB peak memory
        #[arg(long)]
        fused_ce: bool,
        /// Progressive layer freezing fraction. 0=off, 0.5=freeze bottom 50% gradually.
        #[arg(long, default_value = "0.0")]
        freeze_fraction: f32,
        /// Load pretrained model checkpoint (weights only, fresh optimizer).
        /// For progressive training: grow a small model, then continue training larger.
        #[arg(long)]
        pretrained: Option<String>,
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
        /// Repetition penalty (>1.0 penalizes, 1.0 = disabled)
        #[arg(long, default_value = "1.2")]
        repetition_penalty: f32,
        #[arg(long, default_value = "false")]
        stream: bool,
        /// Enable speculative decoding with a smaller draft model
        #[arg(long, default_value = "false")]
        speculative: bool,
        /// Path to the draft model checkpoint (required when --speculative is set)
        #[arg(long)]
        draft_checkpoint: Option<String>,
        /// Number of speculative tokens per verification step
        #[arg(long, default_value = "8")]
        draft_tokens: usize,
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

    /// Evaluate model quality against built-in benchmarks
    Eval {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        tokenizer: String,
    },

    /// Supervised fine-tuning on instruction-response pairs
    Sft {
        /// Pre-trained model checkpoint to fine-tune from
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        tokenizer: String,
        /// JSONL file with {"prompt": "...", "response": "..."} per line
        #[arg(long)]
        data: String,
        #[arg(long, default_value = "1000")]
        steps: u32,
        #[arg(long, default_value = "2e-5")]
        lr: f32,
        #[arg(long, default_value = "8")]
        batch_size: usize,
        #[arg(long, default_value = "256")]
        seq_len: usize,
        #[arg(long, default_value = "100")]
        warmup: u32,
        #[arg(long, default_value = "sft_checkpoints")]
        output_dir: String,
    },

    /// Convert NL2Bash or paired text into JSONL SFT format
    SftPrepare {
        /// Input file: tab-separated or alternating-line pairs
        #[arg(long)]
        input: String,
        /// Output JSONL file
        #[arg(long)]
        output: String,
    },

    /// Quantize a model checkpoint to reduce size (Q8 = 4x smaller, Q4 = 8x smaller)
    Quantize {
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value = "model.qbin")]
        output: String,
        /// Quantization bits: 4 or 8
        #[arg(long, default_value = "4")]
        bits: u8,
    },

    /// Export model to GGUF format for llama.cpp inference
    ExportGguf {
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value = "model.gguf")]
        output: String,
        /// Quantization: "f32" or "q8_0"
        #[arg(long, default_value = "f32")]
        quant: String,
    },

    /// Export model to Safetensors format (HuggingFace ecosystem)
    ExportSafetensors {
        #[arg(long)]
        checkpoint: String,
        #[arg(long, default_value = "model.safetensors")]
        output: String,
    },

    /// Average multiple checkpoints (WSM — +3.5% benchmark improvement)
    Merge {
        /// Checkpoint files to average (2+)
        #[arg(long, num_args = 2..)]
        checkpoints: Vec<String>,
        /// Output averaged checkpoint
        #[arg(long, default_value = "merged.bin")]
        output: String,
    },

    /// Direct Preference Optimization — align a model using preference pairs
    Dpo {
        /// Pre-trained/SFT model checkpoint (policy — will be updated)
        #[arg(long)]
        checkpoint: String,
        /// Reference model checkpoint (frozen anchor — typically same as initial policy)
        #[arg(long)]
        ref_checkpoint: String,
        #[arg(long)]
        tokenizer: String,
        /// Binary preference dataset (.bin) — use `dpo-prepare` to create from JSONL
        #[arg(long)]
        dataset: String,
        /// DPO temperature beta (lower = more conservative). Default: 0.1
        #[arg(long, default_value = "0.1")]
        beta: f32,
        #[arg(long, default_value = "1e-6")]
        lr: f32,
        #[arg(long, default_value = "512")]
        max_seq_len: usize,
        #[arg(long, default_value = "1000")]
        steps: u32,
        #[arg(long, default_value = "100")]
        warmup: u32,
        #[arg(long, default_value = "dpo_checkpoints")]
        output_dir: String,
    },

    /// Convert JSONL preference pairs to binary DPO dataset format
    DpoPrepare {
        /// Input JSONL file: {"prompt": "...", "chosen": "...", "rejected": "..."}
        #[arg(long)]
        input: String,
        /// Output binary file
        #[arg(long)]
        output: String,
        #[arg(long)]
        tokenizer: String,
    },

    /// Generate training data from Claude/OpenAI/Ollama API (distillation)
    Distill {
        /// API endpoint URL
        #[arg(long, default_value = "http://localhost:11434/api/generate")]
        api_url: String,
        /// API key (for Claude/OpenAI, not needed for Ollama)
        #[arg(long, default_value = "")]
        api_key: String,
        /// Model name (e.g. "claude-sonnet-4-20250514", "qwen2.5:7b")
        #[arg(long, default_value = "qwen2.5:7b")]
        model: String,
        /// Output JSONL file
        #[arg(long)]
        output: String,
        /// Number of samples to generate
        #[arg(long, default_value = "100")]
        n_samples: usize,
        /// Max tokens per response
        #[arg(long, default_value = "512")]
        max_tokens: usize,
    },

    /// Deduplicate and filter training documents
    Dedup {
        /// Input file (one document per line)
        #[arg(long)]
        input: String,
        /// Output file (filtered)
        #[arg(long)]
        output: String,
        /// MinHash similarity threshold (0.0-1.0). Default: 0.8
        #[arg(long, default_value = "0.8")]
        threshold: f32,
        /// Minimum quality score (0.0-1.0). Default: 0.3
        #[arg(long, default_value = "0.3")]
        min_quality: f32,
    },
    /// Grow a small trained model into a larger architecture (progressive training)
    Grow {
        /// Input: small model checkpoint
        #[arg(long)]
        checkpoint: String,
        /// Output: grown model checkpoint
        #[arg(long)]
        output: String,
        /// Target d_model
        #[arg(long)]
        dim: usize,
        /// Target layers
        #[arg(long)]
        layers: usize,
        /// Target heads
        #[arg(long)]
        heads: usize,
    },

    /// Benchmark inference and training throughput with detailed metrics
    Bench {
        /// Model size preset: tiny, small, medium, large
        #[arg(long, default_value = "small")]
        size: String,
        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,
        /// Sequence length
        #[arg(long, default_value = "128")]
        seq_len: usize,
        /// Low-rank dimension (0 = full rank)
        #[arg(long, default_value = "0")]
        lowrank: usize,
        /// Number of warmup iterations
        #[arg(long, default_value = "5")]
        warmup: usize,
        /// Number of timed iterations
        #[arg(long, default_value = "20")]
        iters: usize,
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

            tok.print_stats();

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
            // Verify a known token is in the vocabulary
            eprintln!("Contains 'the': {}", tok.contains_token(b"the"));
        }

        Commands::Prepare {
            input,
            tokenizer: tok_path,
            output,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let n = data::prepare_dataset(&input, &tok, &output).expect("Failed to prepare dataset");

            // Demonstrate batch padding utility: encode a sample and pad to fixed length
            let sample_text = std::fs::read_to_string(&input)
                .map(|t| t.chars().take(200).collect::<String>())
                .unwrap_or_default();
            if !sample_text.is_empty() {
                let sample_tokens = tok.encode(&sample_text);
                let padded = data::pad_sequences(std::slice::from_ref(&sample_tokens), 64);
                eprintln!("Sample: {} tokens → {} padded to len 64", sample_tokens.len(), padded.len());
            }

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
            kv_heads,
            n_experts,
            top_k_experts,
            max_seq,
            batch_size,
            seq_len,
            steps,
            lr,
            warmup,
            checkpoint_dir,
            gradient_checkpointing,
            teacher_checkpoint,
            distill_temperature,
            distill_alpha,
            grad_accum,
            resume,
            val_dataset,
            dropout,
            lr_restart,
            mup_base,
            bitnet,
            lowrank,
            shared_layers,
            prune_threshold,
            galore_rank,
            reference_model,
            speculative_threshold,
            optimizer,
            n_predict,
            curriculum,
            z_loss,
            stochastic_depth,
            sliding_window,
            fp16_activations,
            lr_schedule,
            ema_decay,
            noise_scale,
            relora_interval,
            fused_ce,
            freeze_fraction,
            pretrained,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            tok.print_stats();
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
                    let kvh = kv_heads.unwrap_or(h);
                    let ms = max_seq.unwrap_or(512);
                    if n_experts > 1 {
                        model::ModelConfig::custom_moe(vocab_size, d, h, kvh, l, fm, ms, n_experts, top_k_experts)
                    } else {
                        model::ModelConfig::custom_gqa(vocab_size, d, h, kvh, l, fm, ms)
                    }
                }
                _ => panic!(
                    "Unknown model size: '{}'. Use: tiny, small, medium, large, xl, max, huge, 8b, custom",
                    size
                ),
            };

            eprintln!("Config: {}", model_config.summary());

            // Verify dataset integrity via GPU round-trip before training
            data::verify_dataset_gpu(&ctx, &dataset, 1024);

            // Use default_small as the base config, then override with CLI args.
            // This ensures all defaults are centralized in TrainConfig::default_small.
            let mut config = train::TrainConfig::default_small(&dataset, &tok_path);
            config.model_config = model_config;
            config.checkpoint_dir = checkpoint_dir;
            config.batch_size = batch_size;
            config.seq_len = seq_len;
            config.total_steps = steps;
            config.max_lr = lr;
            config.warmup_steps = warmup;
            config.gradient_checkpointing = gradient_checkpointing;
            config.teacher_checkpoint = teacher_checkpoint;
            config.distill_temperature = distill_temperature;
            config.distill_alpha = distill_alpha;
            config.grad_accum_steps = grad_accum;
            config.resume_from = resume;
            config.val_dataset = val_dataset;
            config.dropout = dropout;
            config.lr_restart_period = lr_restart;
            if mup_base > 0 {
                config.model_config.mup_base_width = mup_base;
            }
            config.model_config.bitnet = bitnet;
            config.model_config.lowrank = lowrank;
            config.model_config.shared_layers = shared_layers;
            config.prune_threshold = prune_threshold;
            config.galore_rank = galore_rank;
            config.optimizer_type = optimizer;
            config.reference_model = reference_model;
            config.speculative_threshold = speculative_threshold;
            config.model_config.n_predict = n_predict;
            config.curriculum = curriculum;
            config.z_loss_coefficient = z_loss;
            config.model_config.stochastic_depth = stochastic_depth;
            config.model_config.sliding_window = sliding_window;
            config.model_config.fp16_activations = fp16_activations;
            config.lr_schedule = lr_schedule;
            config.ema_decay = ema_decay;
            config.noise_scale = noise_scale;
            config.relora_interval = relora_interval;
            config.fused_ce = fused_ce;
            config.freeze_fraction = freeze_fraction;
            config.pretrained = pretrained;

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
            repetition_penalty,
            stream,
            speculative,
            draft_checkpoint,
            draft_tokens,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let (model, step) = if ckpt_path.ends_with(".qbin") {
                quantize::load_quantized(&ctx, &ckpt_path).expect("Failed to load quantized checkpoint")
            } else {
                checkpoint::load_checkpoint(&ctx, &ckpt_path).expect("Failed to load checkpoint")
            };
            eprintln!("Loaded main model at step {}", step);

            let mut config = generate::SamplingConfig::default();
            config.temperature = temperature;
            config.top_p = top_p;
            config.top_k = top_k;
            config.max_tokens = max_tokens;
            config.repetition_penalty = repetition_penalty;

            if speculative {
                let draft_ckpt = draft_checkpoint.expect(
                    "--draft-checkpoint is required when --speculative is set"
                );
                let (draft_model, draft_step) = if draft_ckpt.ends_with(".qbin") {
                    quantize::load_quantized(&ctx, &draft_ckpt)
                        .expect("Failed to load quantized draft checkpoint")
                } else {
                    checkpoint::load_checkpoint(&ctx, &draft_ckpt)
                        .expect("Failed to load draft checkpoint")
                };
                eprintln!("Loaded draft model at step {}", draft_step);

                assert_eq!(
                    model.config.vocab_size, draft_model.config.vocab_size,
                    "Main and draft models must have the same vocab_size (main={}, draft={})",
                    model.config.vocab_size, draft_model.config.vocab_size
                );

                if stream {
                    print!("{}", prompt);
                    generate::generate_speculative_streaming(
                        &ctx, &model, &draft_model, &tok, &prompt, &config, draft_tokens,
                        |token_str| {
                            print!("{}", token_str);
                            use std::io::Write;
                            std::io::stdout().flush().ok();
                        },
                    );
                    println!();
                } else {
                    let output = generate::generate_speculative(
                        &ctx, &model, &draft_model, &tok, &prompt, &config, draft_tokens,
                    );
                    println!("{}{}", prompt, output);
                }
            } else {
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
            println!("  n_kv_heads: {} (group_size={})", c.n_kv_heads, c.n_heads / c.n_kv_heads);
            println!("  n_layers: {}", c.n_layers);
            println!("  d_ff: {}", c.d_ff());
            println!("  ffn_multiplier: {}", c.ffn_multiplier);
            println!("  max_seq_len: {}", c.max_seq_len);
            println!("  RoPE theta: {}", c.rope_theta);
            println!("  Training RAM: {:.0} MB", c.training_memory_bytes() as f64 / (1024.0 * 1024.0));
            println!("  Inference RAM: {:.0} MB", c.inference_memory_bytes() as f64 / (1024.0 * 1024.0));

            // GPU diagnostic — verify all kernel variants
            let (n_tested, all_ok) = api::gpu_diagnostic(&ctx);
            println!("  GPU kernels: {} tested, {}", n_tested, if all_ok { "all passed" } else { "FAILURES" });
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
            println!("Or use --size custom with --dim --layers --heads --kv-heads --ffn-mult --max-seq for any arbitrary config.");
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
            // Parse CLI shard specs into a DataMix config
            let sources: Vec<datapipe::DataSource> = shards
                .split(',')
                .map(|entry| {
                    let parts: Vec<&str> = entry.splitn(2, ':').collect();
                    let path = std::path::PathBuf::from(parts[0]);
                    let weight = if parts.len() > 1 {
                        parts[1].parse::<f32>().expect("Invalid weight")
                    } else {
                        1.0
                    };
                    datapipe::DataSource {
                        name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                        path,
                        weight,
                        upsample: 1,
                    }
                })
                .collect();

            let mix = datapipe::DataMix { sources };
            eprintln!("Data mix: {} sources", mix.sources.len());
            for src in &mix.sources {
                eprintln!("  {} — weight {:.2}, upsample {}x", src.name, src.weight, src.upsample);
            }

            let shard_pairs: Vec<(std::path::PathBuf, f32)> = mix.sources
                .iter()
                .map(|s| (s.path.clone(), s.weight * s.upsample as f32))
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

        Commands::Eval {
            checkpoint: ckpt_path,
            tokenizer: tok_path,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let (model, step) =
                checkpoint::load_checkpoint(&ctx, &ckpt_path).expect("Failed to load checkpoint");
            eprintln!("Evaluating model at step {} ({:.1}M params)", step, model.config.param_count() as f64 / 1e6);

            let examples = eval::builtin_eval_set();
            eprintln!("Running {} evaluation examples...", examples.len());

            // Verify tensor batch utilities (zeros, full, with_grad, slice_flat, concat_flat)
            let sample_seqs: Vec<Vec<f32>> = examples.iter().take(4)
                .map(|e| e.prompt.bytes().map(|b| b as f32).collect())
                .collect();
            let batch_tensor = eval::build_padded_batch(&ctx, &sample_seqs, 32);
            eprintln!("Batch tensor check: {:?} ({} elements)", batch_tensor.shape, batch_tensor.numel());

            let results = eval::evaluate(&ctx, &model, &tok, &examples);
            results.print_report();
        }

        Commands::Sft {
            checkpoint: ckpt_path,
            tokenizer: tok_path,
            data,
            steps,
            lr,
            batch_size,
            seq_len,
            warmup,
            output_dir,
        } => {
            let mut config = sft::SftConfig::default_sft(&ckpt_path, &tok_path, &data);
            config.total_steps = steps;
            config.max_lr = lr;
            config.batch_size = batch_size;
            config.seq_len = seq_len;
            config.warmup_steps = warmup;
            config.output_dir = output_dir;

            sft::sft_train(&ctx, &config).expect("SFT training failed");
        }

        Commands::SftPrepare { input, output } => {
            let count = sft::generate_sft_dataset(&input, &output)
                .expect("SFT data preparation failed");
            println!("Generated {} instruction-response pairs", count);
        }

        Commands::Quantize {
            checkpoint: ckpt_path,
            output,
            bits,
        } => {
            quantize::quantize_checkpoint(&ckpt_path, &output, bits)
                .expect("Quantization failed");
        }

        Commands::ExportGguf {
            checkpoint: ckpt_path,
            output,
            quant,
        } => {
            let (model, step) = checkpoint::load_checkpoint(&ctx, &ckpt_path)
                .expect("Failed to load checkpoint");
            eprintln!("Loaded checkpoint: step {}, {}M params", step, model.config.param_count() as f32 / 1e6);
            quantize::export_gguf(&model, &output, &quant)
                .expect("GGUF export failed");
        }

        Commands::ExportSafetensors {
            checkpoint: ckpt_path,
            output,
        } => {
            let (model, step) = checkpoint::load_checkpoint(&ctx, &ckpt_path)
                .expect("Failed to load checkpoint");
            eprintln!("Loaded: step {}, {}M params", step, model.config.param_count() as f32 / 1e6);
            quantize::export_safetensors(&model, &output)
                .expect("Safetensors export failed");
        }

        Commands::Merge {
            checkpoints,
            output,
        } => {
            assert!(checkpoints.len() >= 2, "Need at least 2 checkpoints to merge");
            eprintln!("Merging {} checkpoints...", checkpoints.len());

            // Load first checkpoint as base
            let (base_model, base_step) = checkpoint::load_checkpoint(&ctx, &checkpoints[0])
                .expect("Failed to load first checkpoint");
            let n = checkpoints.len() as f32;
            eprintln!("  Base: {} (step {})", checkpoints[0], base_step);

            // Average weights: for each param, compute mean across all checkpoints
            let base_params = base_model.parameters();
            for ckpt_path in &checkpoints[1..] {
                let (other_model, other_step) = checkpoint::load_checkpoint(&ctx, ckpt_path)
                    .expect(&format!("Failed to load checkpoint: {}", ckpt_path));
                eprintln!("  + {} (step {})", ckpt_path, other_step);
                let other_params = other_model.parameters();
                assert_eq!(base_params.len(), other_params.len(), "Checkpoint param count mismatch");

                // Accumulate: base += other
                for (bp, op) in base_params.iter().zip(other_params.iter()) {
                    crate::metal::compute::gpu_add_inplace(&ctx, &bp.buffer, &op.buffer, bp.numel() as u32);
                }
            }

            // Divide by N to get mean
            for bp in &base_params {
                crate::metal::compute::gpu_scale(&ctx, &bp.buffer, bp.numel() as u32, 1.0 / n);
            }

            // Save merged checkpoint
            checkpoint::save_checkpoint(&output, &base_model, base_step)
                .expect("Failed to save merged checkpoint");
            eprintln!("Merged {} checkpoints → {} ({:.1} MB)",
                checkpoints.len(), output,
                std::fs::metadata(&output).map(|m| m.len() as f32 / 1e6).unwrap_or(0.0));
        }

        Commands::Dpo {
            checkpoint: ckpt_path,
            ref_checkpoint,
            tokenizer: tok_path,
            dataset,
            beta,
            lr,
            max_seq_len,
            steps,
            warmup,
            output_dir,
        } => {
            let mut config = dpo::DpoConfig::default_dpo(
                &ckpt_path, &ref_checkpoint, &tok_path, &dataset,
            );
            config.beta = beta;
            config.learning_rate = lr;
            config.max_seq_len = max_seq_len;
            config.total_steps = steps;
            config.warmup_steps = warmup;
            config.output_dir = output_dir;

            dpo::dpo_train(&ctx, &config).expect("DPO training failed");
        }

        Commands::DpoPrepare {
            input,
            output,
            tokenizer: tok_path,
        } => {
            let tok = tokenizer::BpeTokenizer::load(&tok_path).expect("Failed to load tokenizer");
            let count = dpo::prepare_dpo_dataset(&input, &output, &tok)
                .expect("DPO data preparation failed");
            println!("Generated {} preference pairs", count);
        }

        Commands::Distill {
            api_url, api_key, model, output, n_samples, max_tokens,
        } => {
            let config = distill::DistillConfig {
                api_url, api_key, model, output_path: output,
                n_samples, max_tokens, temperature: 0.7,
            };
            let count = distill::generate_training_data(&config)
                .expect("Data generation failed");
            println!("Generated {} training pairs", count);
        }

        Commands::Dedup {
            input, output, threshold, min_quality,
        } => {
            let docs: Vec<String> = std::fs::read_to_string(&input)
                .expect("Failed to read input")
                .lines()
                .map(|l| l.to_string())
                .collect();

            eprintln!("Input: {} documents", docs.len());

            // Quality filter
            let quality_keep = datapipe::quality_filter_batch(&docs, min_quality);
            eprintln!("After quality filter (>{}): {} docs", min_quality, quality_keep.len());

            let quality_docs: Vec<String> = quality_keep.iter().map(|&i| docs[i].clone()).collect();

            // MinHash dedup
            let dedup_keep = datapipe::minhash_dedup(&quality_docs, threshold, 128);
            eprintln!("After dedup (thresh={}): {} docs", threshold, dedup_keep.len());

            let mut out = std::fs::File::create(&output).expect("Failed to create output");
            for &i in &dedup_keep {
                use std::io::Write;
                writeln!(out, "{}", quality_docs[i]).expect("Write failed");
            }
            eprintln!("Output: {} → {} documents", docs.len(), dedup_keep.len());
        }

        Commands::Bench {
            size, batch_size, seq_len, lowrank, warmup, iters,
        } => {
            use std::time::Instant;

            let vocab: u32 = 8192;
            let mut config = match size.as_str() {
                "tiny" => model::ModelConfig::tiny(vocab),
                "small" => model::ModelConfig::small(vocab),
                "medium" => model::ModelConfig::medium(vocab),
                "large" => model::ModelConfig::large(vocab),
                _ => { eprintln!("Unknown size: {}. Use tiny/small/medium/large", size); return; }
            };
            config.lowrank = lowrank;
            config.max_seq_len = seq_len;
            let d = config.d_model;
            let ff = config.d_ff();
            let n_layers = config.n_layers;
            let params = config.param_count();
            let fused_eligible = config.n_experts <= 1 && !config.bitnet
                && d <= 256 && ff <= 1024;

            eprintln!("=== AndreAI Benchmark ===");
            eprintln!("Model: {}M params, d={}, ff={}, {}L, {}H, lowrank={}",
                params as f64 / 1e6, d, ff, n_layers, config.n_heads, lowrank);
            eprintln!("Batch: {}, Seq: {}, Tokens/step: {}",
                batch_size, seq_len, batch_size * seq_len);
            eprintln!("Fused kernels: {}", if fused_eligible { "ACTIVE (inference)" } else { "disabled" });
            eprintln!("Warmup: {}, Timed iters: {}", warmup, iters);
            eprintln!();

            let model = model::Transformer::new(&ctx, config.clone());

            // Random token input
            let tokens: Vec<u32> = (0..batch_size * seq_len)
                .map(|i| (i as u32 * 7 + 13) % vocab)
                .collect();

            // --- 1. Inference Forward (no_grad) ---
            eprintln!("--- Inference Forward (no_grad) ---");
            autograd::no_grad(|| {
                // Warmup
                for _ in 0..warmup {
                    ctx.begin_batch();
                    let _logits = model.forward(&tokens, batch_size, seq_len, None, false);
                    ctx.flush_batch();
                }

                // Timed
                let start = Instant::now();
                let mut dispatches = 0usize;
                for _ in 0..iters {
                    ctx.begin_batch();
                    let _logits = model.forward(&tokens, batch_size, seq_len, None, false);
                    dispatches += ctx.flush_batch();
                }
                let elapsed = start.elapsed().as_secs_f64();
                let total_tokens = iters * batch_size * seq_len;
                let tok_s = total_tokens as f64 / elapsed;
                let ms_per_step = elapsed * 1000.0 / iters as f64;
                let dispatches_per_step = dispatches / iters;
                let us_per_dispatch = ms_per_step * 1000.0 / dispatches_per_step as f64;
                eprintln!("  {:.0} tok/s | {:.2} ms/step | {} dispatches/step | {:.0} μs/dispatch",
                    tok_s, ms_per_step, dispatches_per_step, us_per_dispatch);
            });

            // --- 2. Inference Decode (single-token, KV cache) ---
            eprintln!("--- Inference Decode (single-token, KV cache) ---");
            autograd::no_grad(|| {
                let mut kv_caches = model.init_kv_caches_preallocated(1);

                // Prefill with 16 tokens
                let prompt: Vec<u32> = (0..16).map(|i| (i * 7 + 13) % vocab).collect();
                ctx.begin_batch();
                let _logits = model.forward(&prompt, 1, 16, Some(&mut kv_caches), false);
                ctx.flush_batch();

                // Warmup decode
                for i in 0..warmup {
                    let tok = [(i as u32 * 3 + 5) % vocab];
                    ctx.begin_batch();
                    let _logits = model.forward(&tok, 1, 1, Some(&mut kv_caches), false);
                    ctx.flush_batch();
                }

                // Timed decode
                let start = Instant::now();
                for i in 0..iters {
                    let tok = [(i as u32 * 3 + 5) % vocab];
                    ctx.begin_batch();
                    let _logits = model.forward(&tok, 1, 1, Some(&mut kv_caches), false);
                    ctx.flush_batch();
                }
                let elapsed = start.elapsed().as_secs_f64();
                let tok_s = iters as f64 / elapsed;
                let ms_per_token = elapsed * 1000.0 / iters as f64;
                eprintln!("  {:.0} tok/s | {:.2} ms/token | {:.1}s total ({} tokens)",
                    tok_s, ms_per_token, elapsed, iters);
            });

            // --- 3. Training Forward+Backward (single batch — matches real training loop) ---
            let targets: Vec<u32> = tokens.iter().map(|&t| (t + 1) % vocab).collect();
            eprintln!("--- Training Forward+Backward ---");
            {
                for _ in 0..warmup {
                    autograd::clear_tape();
                    ctx.begin_batch();
                    let logits = model.forward(&tokens, batch_size, seq_len, None, false);
                    let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                    autograd::backward(&ctx, loss.0.id);
                    ctx.flush_batch();
                    autograd::clear_tape();
                }

                let start = Instant::now();
                let mut tape_ops = 0usize;
                let mut tape_mem = 0usize;
                for _ in 0..iters {
                    autograd::clear_tape();
                    ctx.begin_batch();
                    let logits = model.forward(&tokens, batch_size, seq_len, None, false);
                    let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                    let (ops, mem) = autograd::tape_stats();
                    tape_ops = ops;
                    tape_mem = mem;
                    autograd::backward(&ctx, loss.0.id);
                    ctx.flush_batch();
                    autograd::clear_tape();
                }
                let elapsed = start.elapsed().as_secs_f64();
                let total_tokens = iters * batch_size * seq_len;
                let tok_s = total_tokens as f64 / elapsed;
                let ms_per_step = elapsed * 1000.0 / iters as f64;
                eprintln!("  {:.0} tok/s | {:.2} ms/step | {:.1}s total ({} iters)",
                    tok_s, ms_per_step, elapsed, iters);
                eprintln!("  Tape: {} ops, {:.1} MB activations", tape_ops, tape_mem as f64 / 1e6);
            }

            // --- 4. Training Forward+Backward with Gradient Checkpointing ---
            eprintln!("--- Training Forward+Backward (checkpointed) ---");
            {
                for _ in 0..warmup {
                    autograd::clear_tape();
                    autograd::clear_recompute_registry();
                    ctx.begin_batch();
                    let logits = model.forward(&tokens, batch_size, seq_len, None, true);
                    let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                    autograd::backward(&ctx, loss.0.id);
                    ctx.flush_batch();
                    autograd::clear_tape();
                    autograd::clear_recompute_registry();
                }

                let start = Instant::now();
                let mut tape_ops = 0usize;
                let mut tape_mem = 0usize;
                for _ in 0..iters {
                    autograd::clear_tape();
                    autograd::clear_recompute_registry();
                    ctx.begin_batch();
                    let logits = model.forward(&tokens, batch_size, seq_len, None, true);
                    let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                    let (ops, mem) = autograd::tape_stats();
                    tape_ops = ops;
                    tape_mem = mem;
                    autograd::backward(&ctx, loss.0.id);
                    ctx.flush_batch();
                    autograd::clear_tape();
                    autograd::clear_recompute_registry();
                }
                let elapsed = start.elapsed().as_secs_f64();
                let total_tokens = iters * batch_size * seq_len;
                let tok_s = total_tokens as f64 / elapsed;
                let ms_per_step = elapsed * 1000.0 / iters as f64;
                eprintln!("  {:.0} tok/s | {:.2} ms/step | {:.1}s total ({} iters)",
                    tok_s, ms_per_step, elapsed, iters);
                eprintln!("  Tape: {} ops, {:.1} MB activations (checkpointed)", tape_ops, tape_mem as f64 / 1e6);
            }

            // --- 5. Forward-only vs Backward-only breakdown ---
            eprintln!("--- Phase Breakdown (forward vs backward) ---");
            {
                // Measure forward-only time (with tape recording, no backward)
                let start = Instant::now();
                for _ in 0..iters {
                    autograd::clear_tape();
                    ctx.begin_batch();
                    let logits = model.forward(&tokens, batch_size, seq_len, None, false);
                    let _loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                    ctx.flush_batch();
                    autograd::clear_tape();
                }
                let fwd_elapsed = start.elapsed().as_secs_f64();
                let fwd_ms = fwd_elapsed * 1000.0 / iters as f64;

                // Measure full forward+backward
                let start = Instant::now();
                for _ in 0..iters {
                    autograd::clear_tape();
                    ctx.begin_batch();
                    let logits = model.forward(&tokens, batch_size, seq_len, None, false);
                    let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                    autograd::backward(&ctx, loss.0.id);
                    ctx.flush_batch();
                    autograd::clear_tape();
                }
                let total_elapsed = start.elapsed().as_secs_f64();
                let total_ms = total_elapsed * 1000.0 / iters as f64;
                let bwd_ms = total_ms - fwd_ms;
                let fwd_pct = fwd_ms / total_ms * 100.0;

                eprintln!("  Forward:  {:.2} ms ({:.0}%)", fwd_ms, fwd_pct);
                eprintln!("  Backward: {:.2} ms ({:.0}%)", bwd_ms, 100.0 - fwd_pct);
                eprintln!("  Total:    {:.2} ms/step", total_ms);
            }

            // --- Allocation profile ---
            eprintln!();
            eprintln!("--- Allocation Profile (1 forward+backward step) ---");
            metal::MetalContext::enable_alloc_log(true);
            {
                autograd::clear_tape();
                ctx.begin_batch();
                let logits = model.forward(&tokens, batch_size, seq_len, None, false);
                let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                autograd::backward(&ctx, loss.0.id);
                ctx.flush_batch();
                autograd::clear_tape();
            }
            metal::MetalContext::dump_alloc_log("1 train step");
            metal::MetalContext::enable_alloc_log(false);

            // --- 7. Memory & Pool ---
            eprintln!();
            eprintln!("--- Memory ---");
            let param_bytes = params * 4;
            eprintln!("  Model params: {:.1} MB ({:.2}M params × 4 bytes)",
                param_bytes as f64 / 1e6, params as f64 / 1e6);
            eprintln!("  Estimated training RAM: {:.0} MB", config.training_memory_bytes() as f64 / 1e6);
            eprintln!("  Estimated inference RAM: {:.0} MB", config.inference_memory_bytes() as f64 / 1e6);
            let (pool_hits, pool_misses) = metal::MetalContext::pool_stats();
            let pool_total = pool_hits + pool_misses;
            eprintln!("  Buffer pool: {}/{} hits ({:.0}% reuse), {} new allocs",
                pool_hits, pool_total,
                if pool_total > 0 { pool_hits as f64 / pool_total as f64 * 100.0 } else { 0.0 },
                pool_misses);

            // --- 8. Roofline Analysis ---
            eprintln!();
            eprintln!("--- Roofline Analysis ---");
            let mem_bw_gbs = 68.25; // M1 memory bandwidth
            let flops_tflops = 2.6; // M1 FP32 TFLOPS
            // Per forward pass: roughly 2 * params * batch * seq FLOPs (matmul dominated)
            let fwd_flops = 2.0 * params as f64 * (batch_size * seq_len) as f64;
            let fwd_bytes = params as f64 * 4.0 + (batch_size * seq_len) as f64 * d as f64 * 4.0 * n_layers as f64 * 2.0;
            let arithmetic_intensity = fwd_flops / fwd_bytes;
            let compute_bound_ms = fwd_flops / (flops_tflops * 1e9); // TFLOPS → ms
            let mem_bound_ms = fwd_bytes / (mem_bw_gbs * 1e6); // GB/s → ms
            let roofline_ms = compute_bound_ms.max(mem_bound_ms);
            eprintln!("  Arithmetic intensity: {:.1} FLOP/byte", arithmetic_intensity);
            eprintln!("  Compute-bound floor: {:.2} ms", compute_bound_ms);
            eprintln!("  Memory-bound floor:  {:.2} ms", mem_bound_ms);
            eprintln!("  Roofline (forward):  {:.2} ms ({:.0} tok/s theoretical)",
                roofline_ms, (batch_size * seq_len) as f64 / (roofline_ms / 1000.0));
        }

        Commands::Grow {
            checkpoint, output, dim, layers, heads,
        } => {
            let (small_model, step) = checkpoint::load_checkpoint(&ctx, &checkpoint)
                .expect("Failed to load small checkpoint");
            let large_config = model::ModelConfig::custom(
                small_model.config.vocab_size, dim, heads, layers,
                small_model.config.ffn_multiplier, small_model.config.max_seq_len,
            );
            let grown = model::grow_model(&ctx, &small_model, large_config);
            checkpoint::save_checkpoint(&output, &grown, step)
                .expect("Failed to save grown checkpoint");
        }
    }
}
