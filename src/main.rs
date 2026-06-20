#![cfg_attr(all(feature = "cuda", not(feature = "metal")), allow(dead_code))]

#[cfg(feature = "metal")]
pub mod api;
mod attention;
mod autograd;
mod checkpoint;
#[cfg(feature = "cuda")]
mod cuda;
mod data;
mod datapipe;
mod distill;
mod dpo;
mod eval;
mod generate;
mod gpu;
mod linear_attention;
mod loss;
#[cfg(feature = "metal")]
mod metal;
mod mla;
mod model;
mod optim;
pub mod quantize;
mod rwkv;
mod safetensors;
mod sft;
mod ssm;
mod tensor;
mod tokenizer;
mod train;

#[cfg(test)]
mod tests;

use clap::{Parser, Subcommand};
use std::fmt::Display;

#[derive(Parser)]
#[command(name = "andreai", about = "AndreAI — Pure Rust AI Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

fn exit_with_message(message: impl Display) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

fn exit_with_error(context: impl Display, err: impl Display) -> ! {
    eprintln!("{context}: {err}");
    std::process::exit(1);
}

fn result_or_exit<T, E: Display>(result: Result<T, E>, context: impl Display) -> T {
    match result {
        Ok(value) => value,
        Err(err) => exit_with_error(context, err),
    }
}

fn option_or_exit<T>(option: Option<T>, message: impl Display) -> T {
    match option {
        Some(value) => value,
        None => exit_with_message(message),
    }
}

fn parse_longctx_lengths(value: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for raw in value.split(',') {
        let part = raw.trim();
        if part.is_empty() {
            return Err("--longctx-lengths entries must not be empty".to_string());
        }
        let len = part
            .parse::<usize>()
            .map_err(|_| format!("invalid --longctx-lengths entry '{part}'"))?;
        if len == 0 {
            return Err("--longctx-lengths entries must be greater than 0".to_string());
        }
        out.push(len);
    }
    if out.is_empty() {
        Err("--longctx-lengths must include at least one length".to_string())
    } else {
        Ok(out)
    }
}

fn parse_longctx_depths(value: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    for raw in value.split(',') {
        let part = raw.trim();
        if part.is_empty() {
            return Err("--longctx-depths entries must not be empty".to_string());
        }
        let depth = part
            .parse::<f32>()
            .map_err(|_| format!("invalid --longctx-depths entry '{part}'"))?;
        if !depth.is_finite() || !(0.0..=1.0).contains(&depth) {
            return Err("--longctx-depths entries must be finite and in [0, 1]".to_string());
        }
        out.push(depth);
    }
    if out.is_empty() {
        Err("--longctx-depths must include at least one depth".to_string())
    } else {
        Ok(out)
    }
}

/// Arguments for the `train` subcommand (boxed in `Commands::Train` to keep the enum small).
#[derive(clap::Args)]
struct TrainArgs {
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
    /// MLA: Multi-head Latent Attention KV latent dim d_c. 0=off. e.g. 64 → 10-50× KV-cache shrink.
    #[arg(long, default_value = "0")]
    mla_latent_dim: usize,
    /// Block-sparse (MoBA/NSA) attention: # past blocks each query attends. 0=off. e.g. 8.
    #[arg(long, default_value = "0")]
    block_sparse_top_k: usize,
    /// Block-sparse attention block length. Default 64.
    #[arg(long, default_value = "64")]
    block_size: usize,
    /// YaRN RoPE context-extension factor (1.0 = off). e.g. 2.0 extends context 2x via NTK-by-parts.
    #[arg(long, default_value = "1.0")]
    yarn_scale: f32,
    /// Selective state-space (Mamba-2/SSD) mixer in every block instead of attention. O(N) sequence mixing.
    #[arg(long)]
    ssm: bool,
    /// RWKV-style time-mix (per-channel WKV + receptance) in every block instead of attention. O(N).
    /// Takes precedence over --ssm if both are set. EXPERIMENTAL: the materialised wkv path does not
    /// currently converge (loss stays flat even at short seq, token-shift omitted) — wired for
    /// development, not production. --ssm and --linear-attn do train.
    #[arg(long)]
    rwkv: bool,
    /// Linear (kernel) O(N) attention in every block instead of softmax attention.
    #[arg(long)]
    linear_attn: bool,
    /// Hybrid mixer cadence: every Nth layer is linear attention, the rest softmax. 0=off.
    /// e.g. 4 → "3 softmax : 1 linear". Ignored when --ssm/--rwkv/--linear-attn replace every block.
    #[arg(long, default_value = "0")]
    linear_attn_period: usize,
    /// Low-rank FFN training: decompose W=[d,ff] into U=[d,r]×V=[r,ff]. 0=full rank.
    #[arg(long, default_value = "0")]
    lowrank: usize,
    /// ALBERT: share weights across all layers (1 unique layer, N iterations)
    #[arg(long)]
    shared_layers: bool,
    /// Data pruning: skip batches where loss < threshold. 0.0=disabled. Try 8.0 after warmup.
    #[arg(long, default_value = "0.0")]
    prune_threshold: f32,
    /// Speculative pretraining: reference model checkpoint. Skip batches it already knows.
    #[arg(long)]
    reference_model: Option<String>,
    /// Speculative threshold: skip if reference loss < this value. Default: 7.0
    #[arg(long, default_value = "7.0")]
    speculative_threshold: f32,
    /// Optimizer: adamw, adamw-cpu, sophia, muon, hybrid/muon-adamw (Muon for 2-D matrices + AdamW
    /// for embeddings/head/routers/norms), or adamw-8bit (block-wise int8 moments). Default: adamw
    #[arg(long, default_value = "adamw")]
    optimizer: String,
    /// AdamW first-moment decay (beta1). Default: 0.9
    #[arg(long, default_value = "0.9")]
    adamw_beta1: f32,
    /// AdamW second-moment decay (beta2). Default: 0.95 (short memory, pairs with eps=1e-5)
    #[arg(long, default_value = "0.95")]
    adamw_beta2: f32,
    /// AdamW epsilon (update-denominator floor). Default: 1e-5 (the hardened value)
    #[arg(long, default_value = "0.00001")]
    adamw_eps: f32,
    /// Per-element clip on the normalized AdamW update m̂/(√v̂+ε). 0=off. Try 10 to catch spikes.
    #[arg(long, default_value = "0.0")]
    update_clip: f32,
    /// Clip gradients per-tensor (each to max_grad_norm) instead of by global norm.
    #[arg(long)]
    per_tensor_clip: bool,
    /// Hybrid optimizer: LR multiplier for the Muon (hidden-matrix) group. Default 1.0.
    #[arg(long, default_value = "1.0")]
    muon_lr_scale: f32,
    /// Hybrid optimizer: LR multiplier for the AdamW (embeddings/head/norms) group. Default 1.0.
    #[arg(long, default_value = "1.0")]
    adamw_lr_scale: f32,
    /// Disable the hardware simdgroup MMA matmul (ON by default; bit-identical, ~+31% training).
    /// Use only to fall back to scalar-MAC kernels or to enable --bf16-matmul.
    #[arg(long = "no-simdgroup-matmul")]
    no_simdgroup_matmul: bool,
    /// Route the default matmul through bf16: fp32 RANGE (no fp16 ±65504 clamp) but only ~7-bit
    /// mantissa (vs fp16's 10). Use ONLY when fp16 overflows (NaN at large activations) — its
    /// coarser precision DESTABILIZES otherwise (verified: diverged to ~475 on a real run where
    /// fp16 reached 1.56). For range AND precision use the fp32/simdgroup matmul paths.
    #[arg(long)]
    bf16_matmul: bool,
    /// Batch-size LR transfer reference batch. 0=off (use --lr as-is). When set, --lr is the LR tuned
    /// at THIS batch size and is scaled to the actual --batch-size by the √batch rule. Orthogonal to
    /// μP. For Muon, drop --muon-lr-scale as batch rises instead (see #6).
    #[arg(long, default_value = "0")]
    lr_ref_batch: usize,
    /// NorMuon: per-neuron (per-row) second-moment normalization of the Muon/hybrid orthogonalized
    /// update (~+11% over Muon). Only affects --optimizer muon / hybrid. Default false.
    #[arg(long)]
    normuon: bool,
    /// Cautious optimizer (Liang et al. 2024): mask Muon/hybrid orthogonalized-update components
    /// that disagree in sign with the gradient, then renormalize. Near-free convergence gain.
    /// Only affects `--optimizer muon` / `hybrid`. Composes with --normuon. Default false.
    #[arg(long)]
    cautious: bool,
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
    /// LR schedule: cosine, wsd, wso, invsqrt, or trapezoid. Default: cosine
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
    Train(Box<TrainArgs>),

    /// Show available model sizes and their param counts
    Sizes {
        #[arg(long, default_value = "8192")]
        vocab_size: u32,
    },

    /// Compute perplexity (exp mean NLL) of a model over a text string or file
    Perplexity {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        tokenizer: String,
        /// Text to score (overridden by --file when set)
        #[arg(long, default_value = "")]
        text: String,
        /// Read scoring text from a file instead of --text
        #[arg(long)]
        file: Option<String>,
    },

    /// Import a GPT-2 / HuggingFace merges.txt as a byte-level BPE tokenizer and save it
    ImportBpe {
        /// Path to a GPT-2/HF merges.txt
        #[arg(long)]
        merges: String,
        /// Output tokenizer path
        #[arg(long)]
        output: String,
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
        /// Min-p sampling: keep tokens with p >= min_p * max_p (0.0 = disabled)
        #[arg(long, default_value = "0.0")]
        min_p: f32,
        /// Locally-typical sampling: mass to keep (1.0 = disabled)
        #[arg(long, default_value = "1.0")]
        typical_p: f32,
        /// No-repeat-ngram: hard-ban any token completing an n-gram already generated (0 = off, 3 = good
        /// default for assistants — stops degenerate loops that repetition-penalty alone misses).
        #[arg(long, default_value = "0")]
        no_repeat_ngram_size: usize,
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
        /// Batch mode: read prompts (one per line, equal token length) from this file and decode
        /// them together through one batched KV cache. Overrides --prompt when set.
        #[arg(long)]
        batch_file: Option<String>,
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
        /// Run the synthetic long-context suite (NIAH + RULER-style retrieval/reasoning) instead
        /// of the builtin shell-command set.
        #[arg(long, default_value = "false")]
        longctx: bool,
        /// Long-context suite: comma-separated target context lengths in tokens.
        #[arg(long, default_value = "256,512,1024")]
        longctx_lengths: String,
        /// Long-context suite: comma-separated needle depths in [0,1].
        #[arg(long, default_value = "0.0,0.5,1.0")]
        longctx_depths: String,
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
        /// Route matmuls through the hardware simdgroup MMA units (bit-identical; measures the fast path).
        #[arg(long)]
        simdgroup_matmul: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let ctx = crate::gpu::MetalContext::new();

    eprintln!("AndreAI v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Metal device: {}", ctx.device_name());

    // Hardware matrix units (simdgroup MMA): bit-identical, ~+27% inference / +31% training. On by
    // default for every command; `bench` and `train` force it from their own flags below.
    crate::gpu::compute::set_simdgroup_matmul(true);

    match cli.command {
        Commands::Tokenizer {
            input,
            vocab_size,
            output,
        } => {
            eprintln!("Training BPE tokenizer: vocab_size={}", vocab_size);
            let corpus = result_or_exit(std::fs::read(&input), "Failed to read input file");
            let tok = tokenizer::BpeTokenizer::train(&corpus, vocab_size);
            result_or_exit(tok.save(&output), "Failed to save tokenizer");
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
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tok_path),
                "Failed to load tokenizer",
            );
            let n = result_or_exit(
                data::prepare_dataset(&input, &tok, &output),
                "Failed to prepare dataset",
            );

            // Demonstrate batch padding utility: encode a sample and pad to fixed length
            let sample_text = std::fs::read_to_string(&input)
                .map(|t| t.chars().take(200).collect::<String>())
                .unwrap_or_default();
            if !sample_text.is_empty() {
                let sample_tokens = tok.encode(&sample_text);
                let padded = data::pad_sequences(std::slice::from_ref(&sample_tokens), 64);
                eprintln!(
                    "Sample: {} tokens → {} padded to len 64",
                    sample_tokens.len(),
                    padded.len()
                );
            }

            eprintln!("Dataset ready: {} tokens", n);
        }

        Commands::Train(args) => {
            let TrainArgs {
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
                mla_latent_dim,
                block_sparse_top_k,
                block_size,
                yarn_scale,
                ssm,
                rwkv,
                linear_attn,
                linear_attn_period,
                lowrank,
                shared_layers,
                prune_threshold,
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
                adamw_beta1,
                adamw_beta2,
                adamw_eps,
                update_clip,
                per_tensor_clip,
                muon_lr_scale,
                adamw_lr_scale,
                no_simdgroup_matmul,
                bf16_matmul,
                lr_ref_batch,
                normuon,
                cautious,
            } = *args;
            if n_experts == 0 {
                exit_with_message("--n-experts must be greater than 0");
            }
            if top_k_experts == 0 || top_k_experts > n_experts {
                exit_with_message("--top-k-experts must be in 1..=--n-experts");
            }
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tok_path),
                "Failed to load tokenizer",
            );
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
                    let d = option_or_exit(dim, "--dim required for custom size");
                    let l = option_or_exit(layers, "--layers required for custom size");
                    let h = option_or_exit(heads, "--heads required for custom size");
                    let fm = ffn_mult.unwrap_or(2.67);
                    let kvh = kv_heads.unwrap_or(h);
                    let ms = max_seq.unwrap_or(512);
                    if d == 0 {
                        exit_with_message("--dim must be greater than 0");
                    }
                    if l == 0 {
                        exit_with_message("--layers must be greater than 0");
                    }
                    if h == 0 {
                        exit_with_message("--heads must be greater than 0");
                    }
                    if kvh == 0 {
                        exit_with_message("--kv-heads must be greater than 0");
                    }
                    if ms == 0 {
                        exit_with_message("--max-seq must be greater than 0");
                    }
                    if !fm.is_finite() || fm <= 0.0 {
                        exit_with_message("--ffn-mult must be finite and > 0");
                    }
                    if d % h != 0 {
                        exit_with_message("--dim must be divisible by --heads");
                    }
                    if kvh > h {
                        exit_with_message("--kv-heads must be <= --heads");
                    }
                    if h % kvh != 0 {
                        exit_with_message("--heads must be divisible by --kv-heads");
                    }
                    model::ModelConfig::custom_gqa(vocab_size, d, h, kvh, l, fm, ms)
                }
                _ => exit_with_message(format!(
                    "Unknown model size: '{}'. Use: tiny, small, medium, large, xl, max, huge, 8b, custom",
                    size
                )),
            };
            let model_config = if n_experts > 1 {
                model::ModelConfig::custom_moe(
                    model_config,
                    model::MoeSpec {
                        n_experts,
                        top_k_experts,
                    },
                )
            } else {
                model_config
            };

            if !yarn_scale.is_finite() || yarn_scale < 1.0 {
                exit_with_message("--yarn-scale must be finite and >= 1.0");
            }
            let scaled_max_seq = (model_config.max_seq_len as f64) * (yarn_scale as f64);
            if !scaled_max_seq.is_finite() || scaled_max_seq > (usize::MAX as f64) {
                exit_with_message("--yarn-scale makes max_seq_len overflow");
            }

            let model_config = if (yarn_scale - 1.0).abs() > f32::EPSILON {
                model_config.with_yarn(yarn_scale)
            } else {
                model_config
            };

            eprintln!("Config: {}", model_config.summary());

            // Verify dataset integrity via GPU round-trip before training
            result_or_exit(
                data::verify_dataset_gpu(&ctx, &dataset, 1024),
                "Failed to verify dataset",
            );

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
            config.model_config.mla_latent_dim = mla_latent_dim;
            config.model_config.block_sparse_top_k = block_sparse_top_k;
            config.model_config.block_size = block_size;
            config.model_config.ssm = ssm;
            config.model_config.rwkv = rwkv;
            config.model_config.linear_attn = linear_attn;
            config.model_config.linear_attn_period = linear_attn_period;
            config.model_config.lowrank = lowrank;
            config.model_config.shared_layers = shared_layers;
            config.prune_threshold = prune_threshold;
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
            config.adamw_beta1 = adamw_beta1;
            config.adamw_beta2 = adamw_beta2;
            config.adamw_eps = adamw_eps;
            config.update_clip = update_clip;
            config.per_tensor_clip = per_tensor_clip;
            config.muon_lr_scale = muon_lr_scale;
            config.adamw_lr_scale = adamw_lr_scale;
            config.simdgroup_matmul = !no_simdgroup_matmul;
            config.bf16_matmul = bf16_matmul;
            config.lr_ref_batch = lr_ref_batch;
            config.normuon = normuon;
            config.cautious = cautious;

            if let Err(e) = train::train(&ctx, &config) {
                eprintln!("Training failed: {e}");
                std::process::exit(1);
            }
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
            min_p,
            typical_p,
            no_repeat_ngram_size,
            stream,
            speculative,
            draft_checkpoint,
            draft_tokens,
            batch_file,
        } => {
            let config = generate::SamplingConfig {
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
                min_p,
                typical_p,
                no_repeat_ngram_size,
            };
            result_or_exit(config.validate(), "Invalid sampling config");
            if speculative && draft_tokens == 0 {
                exit_with_message(
                    "--draft-tokens must be greater than 0 when --speculative is set",
                );
            }
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tok_path),
                "Failed to load tokenizer",
            );
            let (model, step) = if ckpt_path.ends_with(".qbin") {
                result_or_exit(
                    quantize::load_quantized(&ctx, &ckpt_path),
                    "Failed to load quantized checkpoint",
                )
            } else {
                result_or_exit(
                    checkpoint::load_checkpoint(&ctx, &ckpt_path),
                    "Failed to load checkpoint",
                )
            };
            eprintln!("Loaded main model at step {}", step);

            if let Some(bf) = batch_file {
                let text =
                    result_or_exit(std::fs::read_to_string(&bf), "Failed to read --batch-file");
                let prompts: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
                if prompts.is_empty() {
                    exit_with_message("--batch-file must contain at least one non-empty prompt");
                }
                let token_lens: Vec<usize> =
                    prompts.iter().map(|p| 1 + tok.encode(p).len()).collect();
                let first_len = token_lens[0];
                if token_lens.iter().any(|&len| len != first_len) {
                    exit_with_message(format!(
                        "--batch-file prompts must encode to equal token lengths (got {:?})",
                        token_lens
                    ));
                }
                let outs = generate::generate_batch(&ctx, &model, &tok, &prompts, &config);
                for (p, o) in prompts.iter().zip(&outs) {
                    println!("[{}] => {}", p, o);
                }
            } else if speculative {
                let draft_ckpt = option_or_exit(
                    draft_checkpoint,
                    "--draft-checkpoint is required when --speculative is set",
                );
                let (draft_model, draft_step) = if draft_ckpt.ends_with(".qbin") {
                    result_or_exit(
                        quantize::load_quantized(&ctx, &draft_ckpt),
                        "Failed to load quantized draft checkpoint",
                    )
                } else {
                    result_or_exit(
                        checkpoint::load_checkpoint(&ctx, &draft_ckpt),
                        "Failed to load draft checkpoint",
                    )
                };
                eprintln!("Loaded draft model at step {}", draft_step);

                if model.config.vocab_size != draft_model.config.vocab_size {
                    exit_with_message(format!(
                        "Main and draft models must have the same vocab_size (main={}, draft={})",
                        model.config.vocab_size, draft_model.config.vocab_size
                    ));
                }

                if stream {
                    print!("{}", prompt);
                    generate::generate_speculative_streaming(
                        &ctx,
                        generate::SpecModels {
                            main: &model,
                            draft: &draft_model,
                            tokenizer: &tok,
                        },
                        &prompt,
                        &config,
                        draft_tokens,
                        |token_str| {
                            print!("{}", token_str);
                            use std::io::Write;
                            std::io::stdout().flush().ok();
                        },
                    );
                    println!();
                } else {
                    let output = generate::generate_speculative(
                        &ctx,
                        &model,
                        &draft_model,
                        &tok,
                        &prompt,
                        &config,
                        draft_tokens,
                    );
                    println!("{}{}", prompt, output);
                }
            } else if stream {
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
            let (model, step) = result_or_exit(
                checkpoint::load_checkpoint(&ctx, &ckpt_path),
                "Failed to load checkpoint",
            );
            let c = &model.config;
            println!("AndreAI Model Checkpoint");
            println!("  Step: {}", step);
            println!("  Parameters: {}M", c.param_count() as f32 / 1e6);
            println!("  Vocab size: {}", c.vocab_size);
            println!("  d_model: {}", c.d_model);
            println!("  n_heads: {}", c.n_heads);
            println!(
                "  n_kv_heads: {} (group_size={})",
                c.n_kv_heads,
                c.n_heads / c.n_kv_heads
            );
            println!("  n_layers: {}", c.n_layers);
            println!("  d_ff: {}", c.d_ff());
            println!("  ffn_multiplier: {}", c.ffn_multiplier);
            println!("  max_seq_len: {}", c.max_seq_len);
            println!("  RoPE theta: {}", c.rope_theta);
            if (c.yarn_scale - 1.0).abs() > f32::EPSILON {
                println!("  YaRN scale: {}", c.yarn_scale);
                println!("  YaRN original max seq: {}", c.yarn_orig_max_seq);
            }
            if c.n_experts > 1 {
                println!("  n_experts: {}", c.n_experts);
                println!("  top_k_experts: {}", c.top_k_experts);
            }
            println!(
                "  Training RAM: {:.0} MB",
                c.training_memory_bytes() as f64 / (1024.0 * 1024.0)
            );
            println!(
                "  Inference RAM: {:.0} MB",
                c.inference_memory_bytes() as f64 / (1024.0 * 1024.0)
            );

            // GPU diagnostic — verify all kernel variants (Metal-only parity harness)
            #[cfg(feature = "metal")]
            {
                let (n_tested, all_ok) = api::gpu_diagnostic(&ctx);
                println!(
                    "  GPU kernels: {} tested, {}",
                    n_tested,
                    if all_ok { "all passed" } else { "FAILURES" }
                );
            }
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

        Commands::Perplexity {
            checkpoint,
            tokenizer,
            text,
            file,
        } => {
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tokenizer),
                "Failed to load tokenizer",
            );
            let (model, step) = if checkpoint.ends_with(".qbin") {
                result_or_exit(
                    quantize::load_quantized(&ctx, &checkpoint),
                    "Failed to load quantized checkpoint",
                )
            } else {
                result_or_exit(
                    checkpoint::load_checkpoint(&ctx, &checkpoint),
                    "Failed to load checkpoint",
                )
            };
            let corpus = match file {
                Some(f) => String::from_utf8_lossy(&result_or_exit(
                    std::fs::read(&f),
                    "Failed to read --file",
                ))
                .into_owned(),
                None => text,
            };
            let tokens = tok.encode(&corpus);
            if tokens.len() < 2 {
                exit_with_message(format!(
                    "need >= 2 tokens to score perplexity (got {})",
                    tokens.len()
                ));
            }
            let ppl = eval::perplexity(&ctx, &model, &tokens);
            println!(
                "Perplexity: {:.3} over {} tokens (model step {})",
                ppl,
                tokens.len(),
                step
            );
        }

        Commands::ImportBpe { merges, output } => {
            let text = result_or_exit(
                std::fs::read_to_string(&merges),
                "Failed to read merges file",
            );
            let tok = tokenizer::BpeTokenizer::import_gpt2_merges(&text);
            result_or_exit(tok.save(&output), "Failed to save tokenizer");
            println!(
                "Imported {} merges → {} tokens, saved to {}",
                tok.merges.len(),
                tok.inverse_vocab.len(),
                output
            );
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
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tok_path),
                "Failed to load tokenizer",
            );
            let stats = result_or_exit(
                datapipe::process_source(
                    std::path::Path::new(&input),
                    std::path::Path::new(&output),
                    &tok,
                    &separator,
                ),
                "Processing failed",
            );

            if let Some(log_path) = provenance_log {
                result_or_exit(
                    datapipe::record_provenance(
                        std::path::Path::new(&log_path),
                        &source_name,
                        &source_url,
                        &license,
                        &stats,
                    ),
                    "Failed to write provenance log",
                );
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
                        result_or_exit(parts[1].parse::<f32>(), "Invalid weight")
                    } else {
                        1.0
                    };
                    datapipe::DataSource {
                        name: path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        path,
                        weight,
                        upsample: 1,
                    }
                })
                .collect();

            let mix = datapipe::DataMix { sources };
            eprintln!("Data mix: {} sources", mix.sources.len());
            for src in &mix.sources {
                eprintln!(
                    "  {} — weight {:.2}, upsample {}x",
                    src.name, src.weight, src.upsample
                );
            }

            let shard_pairs: Vec<(std::path::PathBuf, f32)> = mix
                .sources
                .iter()
                .map(|s| (s.path.clone(), s.weight * s.upsample as f32))
                .collect();

            let total = result_or_exit(
                datapipe::mix_shards(&shard_pairs, std::path::Path::new(&output)),
                "Mixing failed",
            );
            println!("Mixed dataset: {} tokens", total);
        }

        Commands::Hash { file } => {
            let hash = result_or_exit(
                datapipe::sha256_file(std::path::Path::new(&file)),
                "Failed to hash file",
            );
            println!("{}", hash);
        }

        Commands::Eval {
            checkpoint: ckpt_path,
            tokenizer: tok_path,
            longctx,
            longctx_lengths,
            longctx_depths,
        } => {
            let longctx_config = if longctx {
                Some((
                    result_or_exit(
                        parse_longctx_lengths(&longctx_lengths),
                        "Invalid long-context lengths",
                    ),
                    result_or_exit(
                        parse_longctx_depths(&longctx_depths),
                        "Invalid long-context depths",
                    ),
                ))
            } else {
                None
            };
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tok_path),
                "Failed to load tokenizer",
            );
            let (model, step) = result_or_exit(
                checkpoint::load_checkpoint(&ctx, &ckpt_path),
                "Failed to load checkpoint",
            );
            eprintln!(
                "Evaluating model at step {} ({:.1}M params)",
                step,
                model.config.param_count() as f64 / 1e6
            );

            let examples = if longctx {
                let (lengths, depths) =
                    longctx_config.expect("longctx config should be parsed when --longctx is set");
                eprintln!(
                    "Long-context suite: lengths={:?} tokens, depths={:?}",
                    lengths, depths
                );
                eval::longctx_eval_set(&tok, &lengths, &depths)
            } else {
                eval::builtin_eval_set()
            };
            eprintln!("Running {} evaluation examples...", examples.len());

            if !longctx {
                // Verify tensor batch utilities (zeros, full, with_grad, slice_flat, concat_flat)
                let sample_seqs: Vec<Vec<f32>> = examples
                    .iter()
                    .take(4)
                    .map(|e| e.prompt.bytes().map(|b| b as f32).collect())
                    .collect();
                let batch_tensor = eval::build_padded_batch(&ctx, &sample_seqs, 32);
                eprintln!(
                    "Batch tensor check: {:?} ({} elements)",
                    batch_tensor.shape,
                    batch_tensor.numel()
                );
            }

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

            result_or_exit(sft::sft_train(&ctx, &config), "SFT training failed");
        }

        Commands::SftPrepare { input, output } => {
            let count = result_or_exit(
                sft::generate_sft_dataset(&input, &output),
                "SFT data preparation failed",
            );
            println!("Generated {} instruction-response pairs", count);
        }

        Commands::Quantize {
            checkpoint: ckpt_path,
            output,
            bits,
        } => {
            result_or_exit(
                quantize::quantize_checkpoint(&ckpt_path, &output, bits),
                "Quantization failed",
            );
        }

        Commands::ExportGguf {
            checkpoint: ckpt_path,
            output,
            quant,
        } => {
            let (model, step) = result_or_exit(
                checkpoint::load_checkpoint(&ctx, &ckpt_path),
                "Failed to load checkpoint",
            );
            eprintln!(
                "Loaded checkpoint: step {}, {}M params",
                step,
                model.config.param_count() as f32 / 1e6
            );
            result_or_exit(
                quantize::export_gguf(&model, &output, &quant),
                "GGUF export failed",
            );
        }

        Commands::ExportSafetensors {
            checkpoint: ckpt_path,
            output,
        } => {
            let (model, step) = result_or_exit(
                checkpoint::load_checkpoint(&ctx, &ckpt_path),
                "Failed to load checkpoint",
            );
            eprintln!(
                "Loaded: step {}, {}M params",
                step,
                model.config.param_count() as f32 / 1e6
            );
            result_or_exit(
                quantize::export_safetensors(&model, &output),
                "Safetensors export failed",
            );
        }

        Commands::Merge {
            checkpoints,
            output,
        } => {
            if checkpoints.len() < 2 {
                exit_with_message("Need at least 2 checkpoints to merge");
            }
            eprintln!("Merging {} checkpoints...", checkpoints.len());

            // Load first checkpoint as base
            let (base_model, base_step) = result_or_exit(
                checkpoint::load_checkpoint(&ctx, &checkpoints[0]),
                "Failed to load first checkpoint",
            );
            let n = checkpoints.len() as f32;
            eprintln!("  Base: {} (step {})", checkpoints[0], base_step);

            // Average weights: for each param, compute mean across all checkpoints
            let base_params = base_model.parameters();
            for ckpt_path in &checkpoints[1..] {
                let (other_model, other_step) = result_or_exit(
                    checkpoint::load_checkpoint(&ctx, ckpt_path),
                    format!("Failed to load checkpoint: {}", ckpt_path),
                );
                eprintln!("  + {} (step {})", ckpt_path, other_step);
                let other_params = other_model.parameters();
                if base_params.len() != other_params.len() {
                    exit_with_message("Checkpoint param count mismatch");
                }
                for (idx, (bp, op)) in base_params.iter().zip(other_params.iter()).enumerate() {
                    if bp.shape != op.shape {
                        exit_with_message(format!(
                            "Checkpoint tensor {idx} shape mismatch: base {:?} vs {} {:?}",
                            bp.shape, ckpt_path, op.shape
                        ));
                    }
                    if bp.numel() != op.numel() {
                        exit_with_message(format!(
                            "Checkpoint tensor {idx} element count mismatch: base {} vs {} {}",
                            bp.numel(),
                            ckpt_path,
                            op.numel()
                        ));
                    }
                }

                // Accumulate: base += other
                for (bp, op) in base_params.iter().zip(other_params.iter()) {
                    crate::gpu::compute::gpu_add_inplace(
                        &ctx,
                        &bp.buffer,
                        &op.buffer,
                        bp.numel() as u32,
                    );
                }
            }

            // Divide by N to get mean
            for bp in &base_params {
                crate::gpu::compute::gpu_scale(&ctx, &bp.buffer, bp.numel() as u32, 1.0 / n);
            }

            // Save merged checkpoint
            result_or_exit(
                checkpoint::save_checkpoint(&output, &base_model, base_step),
                "Failed to save merged checkpoint",
            );
            eprintln!(
                "Merged {} checkpoints → {} ({:.1} MB)",
                checkpoints.len(),
                output,
                std::fs::metadata(&output)
                    .map(|m| m.len() as f32 / 1e6)
                    .unwrap_or(0.0)
            );
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
            let mut config =
                dpo::DpoConfig::default_dpo(&ckpt_path, &ref_checkpoint, &tok_path, &dataset);
            config.beta = beta;
            config.learning_rate = lr;
            config.max_seq_len = max_seq_len;
            config.total_steps = steps;
            config.warmup_steps = warmup;
            config.output_dir = output_dir;

            result_or_exit(dpo::dpo_train(&ctx, &config), "DPO training failed");
        }

        Commands::DpoPrepare {
            input,
            output,
            tokenizer: tok_path,
        } => {
            let tok = result_or_exit(
                tokenizer::BpeTokenizer::load(&tok_path),
                "Failed to load tokenizer",
            );
            let count = result_or_exit(
                dpo::prepare_dpo_dataset(&input, &output, &tok),
                "DPO data preparation failed",
            );
            println!("Generated {} preference pairs", count);
        }

        Commands::Distill {
            api_url,
            api_key,
            model,
            output,
            n_samples,
            max_tokens,
        } => {
            let config = distill::DistillConfig {
                api_url,
                api_key,
                model,
                output_path: output,
                n_samples,
                max_tokens,
                temperature: 0.7,
            };
            let count = result_or_exit(
                distill::generate_training_data(&config),
                "Data generation failed",
            );
            println!("Generated {} training pairs", count);
        }

        Commands::Dedup {
            input,
            output,
            threshold,
            min_quality,
        } => {
            let docs: Vec<String> =
                result_or_exit(std::fs::read_to_string(&input), "Failed to read input")
                    .lines()
                    .map(|l| l.to_string())
                    .collect();

            eprintln!("Input: {} documents", docs.len());

            // Quality filter
            let quality_keep = datapipe::quality_filter_batch(&docs, min_quality);
            eprintln!(
                "After quality filter (>{}): {} docs",
                min_quality,
                quality_keep.len()
            );

            let quality_docs: Vec<String> = quality_keep.iter().map(|&i| docs[i].clone()).collect();

            // MinHash dedup
            let dedup_keep = datapipe::minhash_dedup(&quality_docs, threshold, 128);
            eprintln!(
                "After dedup (thresh={}): {} docs",
                threshold,
                dedup_keep.len()
            );

            let mut out = result_or_exit(std::fs::File::create(&output), "Failed to create output");
            for &i in &dedup_keep {
                use std::io::Write;
                result_or_exit(writeln!(out, "{}", quality_docs[i]), "Write failed");
            }
            eprintln!("Output: {} → {} documents", docs.len(), dedup_keep.len());
        }

        Commands::Bench {
            size,
            batch_size,
            seq_len,
            lowrank,
            warmup,
            iters,
            simdgroup_matmul,
        } => {
            use std::time::Instant;

            let vocab: u32 = 8192;
            let mut config = match size.as_str() {
                "tiny" => model::ModelConfig::tiny(vocab),
                "small" => model::ModelConfig::small(vocab),
                "medium" => model::ModelConfig::medium(vocab),
                "large" => model::ModelConfig::large(vocab),
                _ => {
                    eprintln!("Unknown size: {}. Use tiny/small/medium/large", size);
                    return;
                }
            };
            config.lowrank = lowrank;
            config.max_seq_len = seq_len;
            let d = config.d_model;
            let ff = config.d_ff();
            let n_layers = config.n_layers;
            let params = config.param_count();
            let fused_eligible = config.n_experts <= 1 && !config.bitnet && d <= 256 && ff <= 1024;

            eprintln!("=== AndreAI Benchmark ===");
            eprintln!(
                "Model: {}M params, d={}, ff={}, {}L, {}H, lowrank={}",
                params as f64 / 1e6,
                d,
                ff,
                n_layers,
                config.n_heads,
                lowrank
            );
            eprintln!(
                "Batch: {}, Seq: {}, Tokens/step: {}",
                batch_size,
                seq_len,
                batch_size * seq_len
            );
            eprintln!(
                "Fused kernels: {}",
                if fused_eligible {
                    "ACTIVE (inference)"
                } else {
                    "disabled"
                }
            );
            eprintln!("Warmup: {}, Timed iters: {}", warmup, iters);
            crate::gpu::compute::set_simdgroup_matmul(simdgroup_matmul);
            eprintln!(
                "Matmul path: {}",
                if simdgroup_matmul {
                    "hardware simdgroup MMA"
                } else {
                    "scalar-MAC tiled"
                }
            );
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
                eprintln!(
                    "  {:.0} tok/s | {:.2} ms/step | {} dispatches/step | {:.0} μs/dispatch",
                    tok_s, ms_per_step, dispatches_per_step, us_per_dispatch
                );
            });

            // --- 2. Inference Decode (single-token, KV cache) ---
            eprintln!("--- Inference Decode (single-token, KV cache) ---");
            autograd::no_grad(|| {
                let mut kv_caches = model.init_kv_caches_preallocated(1);

                // Prefill must leave cache room for the (warmup + iters) decode steps that follow,
                // since cache capacity == config.max_seq_len (set to --seq-len above).
                let decode_budget = warmup + iters;
                let prefill_len = (seq_len.saturating_sub(decode_budget)).max(1);
                let prompt: Vec<u32> = (0..prefill_len)
                    .map(|i| (i as u32 * 7 + 13) % vocab)
                    .collect();
                ctx.begin_batch();
                let _logits = model.forward(&prompt, 1, prefill_len, Some(&mut kv_caches), false);
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
                eprintln!(
                    "  {:.0} tok/s | {:.2} ms/token | {:.1}s total ({} tokens)",
                    tok_s, ms_per_token, elapsed, iters
                );
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
                eprintln!(
                    "  {:.0} tok/s | {:.2} ms/step | {:.1}s total ({} iters)",
                    tok_s, ms_per_step, elapsed, iters
                );
                eprintln!(
                    "  Tape: {} ops, {:.1} MB activations",
                    tape_ops,
                    tape_mem as f64 / 1e6
                );
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
                eprintln!(
                    "  {:.0} tok/s | {:.2} ms/step | {:.1}s total ({} iters)",
                    tok_s, ms_per_step, elapsed, iters
                );
                eprintln!(
                    "  Tape: {} ops, {:.1} MB activations (checkpointed)",
                    tape_ops,
                    tape_mem as f64 / 1e6
                );
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
            crate::gpu::MetalContext::enable_alloc_log(true);
            {
                autograd::clear_tape();
                ctx.begin_batch();
                let logits = model.forward(&tokens, batch_size, seq_len, None, false);
                let loss = loss::cross_entropy_loss(&ctx, &logits, &targets);
                autograd::backward(&ctx, loss.0.id);
                ctx.flush_batch();
                autograd::clear_tape();
            }
            crate::gpu::MetalContext::dump_alloc_log("1 train step");
            crate::gpu::MetalContext::enable_alloc_log(false);

            // --- 7. Memory & Pool ---
            eprintln!();
            eprintln!("--- Memory ---");
            let param_bytes = params * 4;
            eprintln!(
                "  Model params: {:.1} MB ({:.2}M params × 4 bytes)",
                param_bytes as f64 / 1e6,
                params as f64 / 1e6
            );
            eprintln!(
                "  Estimated training RAM: {:.0} MB",
                config.training_memory_bytes() as f64 / 1e6
            );
            eprintln!(
                "  Estimated inference RAM: {:.0} MB",
                config.inference_memory_bytes() as f64 / 1e6
            );
            let (pool_hits, pool_misses) = crate::gpu::MetalContext::pool_stats();
            let pool_total = pool_hits + pool_misses;
            eprintln!(
                "  Buffer pool: {}/{} hits ({:.0}% reuse), {} new allocs",
                pool_hits,
                pool_total,
                if pool_total > 0 {
                    pool_hits as f64 / pool_total as f64 * 100.0
                } else {
                    0.0
                },
                pool_misses
            );

            // --- 8. Roofline Analysis ---
            eprintln!();
            eprintln!("--- Roofline Analysis ---");
            let mem_bw_gbs = 68.25; // M1 memory bandwidth
            let flops_tflops = 2.6; // M1 FP32 TFLOPS
                                    // Per forward pass: roughly 2 * params * batch * seq FLOPs (matmul dominated)
            let fwd_flops = 2.0 * params as f64 * (batch_size * seq_len) as f64;
            let fwd_bytes = params as f64 * 4.0
                + (batch_size * seq_len) as f64 * d as f64 * 4.0 * n_layers as f64 * 2.0;
            let arithmetic_intensity = fwd_flops / fwd_bytes;
            let compute_bound_ms = fwd_flops / (flops_tflops * 1e9); // TFLOPS → ms
            let mem_bound_ms = fwd_bytes / (mem_bw_gbs * 1e6); // GB/s → ms
            let roofline_ms = compute_bound_ms.max(mem_bound_ms);
            eprintln!(
                "  Arithmetic intensity: {:.1} FLOP/byte",
                arithmetic_intensity
            );
            eprintln!("  Compute-bound floor: {:.2} ms", compute_bound_ms);
            eprintln!("  Memory-bound floor:  {:.2} ms", mem_bound_ms);
            eprintln!(
                "  Roofline (forward):  {:.2} ms ({:.0} tok/s theoretical)",
                roofline_ms,
                (batch_size * seq_len) as f64 / (roofline_ms / 1000.0)
            );
        }

        Commands::Grow {
            checkpoint,
            output,
            dim,
            layers,
            heads,
        } => {
            let (small_model, step) = result_or_exit(
                checkpoint::load_checkpoint(&ctx, &checkpoint),
                "Failed to load small checkpoint",
            );
            let large_config = model::ModelConfig::custom(
                small_model.config.vocab_size,
                dim,
                heads,
                layers,
                small_model.config.ffn_multiplier,
                small_model.config.max_seq_len,
            );
            let grown = model::grow_model(&ctx, &small_model, large_config);
            result_or_exit(
                checkpoint::save_checkpoint(&output, &grown, step),
                "Failed to save grown checkpoint",
            );
        }
    }
}
