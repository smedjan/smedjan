use crate::gpu::MetalContext;
use crate::model::{ModelConfig, Transformer};
use crate::optim::AdamW;
use std::io::{Error, ErrorKind, Read, Write};
use std::sync::Arc;

/// Magic bytes for AndreAI checkpoint files.
const MAGIC: &[u8; 4] = b"AMDL";
const VERSION: u32 = 13; // v13: AMDT step is next training step to run. v12: sliding_window. v11: explicit optimizer-param count (0 for non-AdamW). v10: block-sparse. v9: MLA. v8: rwkv. v7: ssm. v6: linear_attn_period. v5: linear_attn. v4: ReLoRA base weights
const MAX_TENSOR_DIMS: usize = 8;

fn invalid_checkpoint_data(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

fn checked_tensor_byte_len(shape: &[usize], context: impl Into<String>) -> std::io::Result<usize> {
    let context = context.into();
    let n_elements = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            invalid_checkpoint_data(format!("{context} element count overflows usize"))
        })?;
    n_elements
        .checked_mul(4)
        .ok_or_else(|| invalid_checkpoint_data(format!("{context} byte length overflows usize")))
}

fn read_exact_count(
    file: &mut std::fs::File,
    buf: &mut [u8],
    consumed: &mut u64,
) -> std::io::Result<()> {
    file.read_exact(buf)?;
    *consumed += buf.len() as u64;
    Ok(())
}

fn ensure_remaining_bytes(
    file_len: u64,
    consumed: u64,
    needed: u64,
    context: impl Into<String>,
) -> std::io::Result<()> {
    let remaining = file_len.saturating_sub(consumed);
    if needed > remaining {
        return Err(invalid_checkpoint_data(format!(
            "{} exceeds remaining artifact bytes: need {}, remaining {}",
            context.into(),
            needed,
            remaining
        )));
    }
    Ok(())
}

/// Return type for load_training_state: (model, optimizer_states, next_step, opt_step, total_tokens)
pub type TrainingState = (Transformer, Vec<(Vec<f32>, Vec<f32>)>, u32, u32, u64);

/// Save model weights and config to a binary checkpoint file.
pub fn save_checkpoint(path: &str, model: &Transformer, step: u32) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Header
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&step.to_le_bytes())?;

    // Model config
    write_config(&mut file, &model.config)?;

    // Number of tensors (trainable params + base params for ReLoRA)
    let params = model.parameters();
    let base_params = model.base_parameters();
    let n_tensors = (params.len() + base_params.len()) as u32;
    file.write_all(&n_tensors.to_le_bytes())?;

    // Each tensor: shape + data (trainable first, then base weights)
    let all_params: Vec<&_> = params.iter().chain(base_params.iter()).copied().collect();
    for (i, param) in all_params.iter().enumerate() {
        // Shape
        let ndims = param.shape.len() as u32;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in &param.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }

        // Data (f32)
        let data = param.to_vec();
        let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&byte_data)?;

        if i % 10 == 0 {
            eprintln!(
                "  saving tensor {}/{} ({} elements)",
                i + 1,
                n_tensors,
                data.len()
            );
        }
    }

    let size_mb = std::fs::metadata(path)?.len() as f32 / (1024.0 * 1024.0);
    eprintln!(
        "Checkpoint saved: {} ({:.1} MB, step {})",
        path, size_mb, step
    );
    Ok(())
}

/// Save a checkpoint whose TRAINABLE weights come from the EMA buffers (the exponential moving
/// average maintained during training) instead of the live snapshot. The EMA is typically a better
/// model than the final step (BYOL / self-distillation result — "the EMA is always a better model
/// than the current snapshot"), so it's worth keeping rather than discarding. Format is byte-identical
/// to `save_checkpoint`, so it loads through the normal `load_checkpoint` path.
///
/// `ema_buffers` must be parallel to `model.parameters()` (same order/count — that's how the train
/// loop builds them). Frozen ReLoRA base params (which EMA does not track) are written from the model
/// as-is, since the EMA of a frozen weight is the weight itself.
pub fn save_checkpoint_ema(
    path: &str,
    model: &Transformer,
    ema_buffers: &[crate::gpu::Buf],
    step: u32,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&step.to_le_bytes())?;
    write_config(&mut file, &model.config)?;

    let params = model.parameters();
    let base_params = model.base_parameters();
    assert_eq!(
        ema_buffers.len(),
        params.len(),
        "save_checkpoint_ema: {} EMA buffers but {} trainable params",
        ema_buffers.len(),
        params.len()
    );
    let n_tensors = (params.len() + base_params.len()) as u32;
    file.write_all(&n_tensors.to_le_bytes())?;

    // Trainable tensors: shape from the param, DATA from the parallel EMA buffer.
    for (i, param) in params.iter().enumerate() {
        let ndims = param.shape.len() as u32;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in &param.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }
        let data = MetalContext::read_buffer(&ema_buffers[i], param.numel());
        let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&byte_data)?;
    }
    // Frozen base (ReLoRA) tensors: written from the model — EMA doesn't track them.
    for param in base_params.iter() {
        let ndims = param.shape.len() as u32;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in &param.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }
        let data = param.to_vec();
        let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&byte_data)?;
    }

    let size_mb = std::fs::metadata(path)?.len() as f32 / (1024.0 * 1024.0);
    eprintln!(
        "EMA checkpoint saved: {} ({:.1} MB, step {})",
        path, size_mb, step
    );
    Ok(())
}

/// Save full training state: model weights + optimizer state (m, v, next_step).
/// This allows resuming training exactly where it left off.
pub fn save_training_state(
    path: &str,
    model: &Transformer,
    optimizer: &AdamW,
    next_step: u32,
    total_tokens: u64,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Header: AMDT (AndreAI Model Training state)
    file.write_all(b"AMDT")?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&next_step.to_le_bytes())?;
    file.write_all(&total_tokens.to_le_bytes())?;

    // Model config
    write_config(&mut file, &model.config)?;

    // Model weights (trainable + base for ReLoRA)
    let params = model.parameters();
    let base_params = model.base_parameters();
    let all_params: Vec<&_> = params.iter().chain(base_params.iter()).copied().collect();
    let n_tensors = all_params.len() as u32;
    file.write_all(&n_tensors.to_le_bytes())?;

    for param in &all_params {
        let ndims = param.shape.len() as u32;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in &param.shape {
            file.write_all(&(dim as u32).to_le_bytes())?;
        }
        let data = param.to_vec();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
    }

    // Optimizer state: step, optimizer-param count (v11), then m and v for each. The count is 0 when
    // a non-AdamW optimizer is live (the fallback AdamW carries no state) — so resume reads none and
    // starts that optimizer's state fresh rather than choking on a size mismatch.
    file.write_all(&optimizer.step.to_le_bytes())?;
    file.write_all(&(optimizer.params.len() as u32).to_le_bytes())?;
    for ps in &optimizer.params {
        let m_data = MetalContext::read_buffer(&ps.m, ps.size);
        let v_data = MetalContext::read_buffer(&ps.v, ps.size);
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let v_bytes: Vec<u8> = v_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&m_bytes)?;
        file.write_all(&v_bytes)?;
    }

    let size_mb = std::fs::metadata(path)?.len() as f32 / (1024.0 * 1024.0);
    eprintln!(
        "Training state saved: {} ({:.1} MB, next_step {}, {} tokens)",
        path, size_mb, next_step, total_tokens
    );
    Ok(())
}

/// Load full training state for resume.
/// The optimizer_data is (m_buffers, v_buffers, opt_step) — caller creates the AdamW and loads these.
pub fn load_training_state(ctx: &Arc<MetalContext>, path: &str) -> std::io::Result<TrainingState> {
    let mut file = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Magic
    file.read_exact(&mut buf4)?;
    if &buf4 != b"AMDT" {
        return Err(invalid_checkpoint_data(format!(
            "not a valid AndreAI training state file: expected AMDT magic, got {:02x?}",
            buf4
        )));
    }

    // Version
    file.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if !(2..=13).contains(&version) {
        return Err(invalid_checkpoint_data(format!(
            "unsupported training state version: {version} (expected 2-13)"
        )));
    }

    // v13+: next training step to run. v12 and older stored the last completed loop step for
    // periodic states, while state_final.bin stored total_steps; normalize both to next_step.
    file.read_exact(&mut buf4)?;
    let raw_step = u32::from_le_bytes(buf4);
    let next_step = normalize_training_state_next_step(version, raw_step, path);
    file.read_exact(&mut buf8)?;
    let total_tokens = u64::from_le_bytes(buf8);

    // Config
    let config = read_config(&mut file, version)?;
    eprintln!(
        "Resuming: next_step {}, {}M params, {} tokens processed",
        next_step,
        config.param_count() as f32 / 1e6,
        total_tokens
    );

    // Model
    let model = Transformer::new(ctx, config);
    file.read_exact(&mut buf4)?;
    let n_tensors = u32::from_le_bytes(buf4) as usize;
    let params = model.parameters();
    let base_params = model.base_parameters();

    // v4 training states include base params (ReLoRA frozen weights) after trainable params.
    // v2/v3 states only include trainable params.
    let expected = if version >= 4 {
        params.len() + base_params.len()
    } else {
        params.len()
    };
    if n_tensors != expected {
        return Err(invalid_checkpoint_data(format!(
            "training state has {n_tensors} tensors, model expects {expected} (version {version})"
        )));
    }

    let all_params: Vec<&_> = if version >= 4 {
        params.iter().chain(base_params.iter()).copied().collect()
    } else {
        params.to_vec()
    };

    for (i, param) in all_params.iter().enumerate() {
        file.read_exact(&mut buf4)?;
        let ndims = u32::from_le_bytes(buf4) as usize;
        if ndims == 0 || ndims > MAX_TENSOR_DIMS {
            return Err(invalid_checkpoint_data(format!(
                "training state tensor {i} has invalid dimension count {ndims}"
            )));
        }
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            file.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }
        if shape != param.shape {
            return Err(invalid_checkpoint_data(format!(
                "shape mismatch for training state tensor {i}: state {:?} vs model {:?}",
                shape, param.shape
            )));
        }

        let byte_len = checked_tensor_byte_len(&shape, format!("training state tensor {i}"))?;
        let mut byte_data = vec![0u8; byte_len];
        file.read_exact(&mut byte_data)?;
        crate::gpu::buf_write_bytes(&param.buffer, &byte_data);
    }

    // Optimizer state (only for trainable params, not base params)
    file.read_exact(&mut buf4)?;
    let opt_step = u32::from_le_bytes(buf4);

    // v11+: explicit optimizer-param count (0 when a non-AdamW optimizer was live). Pre-v11 always
    // wrote one (m,v) per trainable param.
    let opt_count = if version >= 11 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) as usize
    } else {
        params.len()
    };
    if opt_count > params.len() {
        return Err(invalid_checkpoint_data(format!(
            "training state has optimizer state for {opt_count} params, model expects at most {}",
            params.len()
        )));
    }

    let mut opt_states = Vec::with_capacity(opt_count);
    for param in params.iter().take(opt_count) {
        let size = param.numel();
        let mut m_bytes = vec![0u8; size * 4];
        let mut v_bytes = vec![0u8; size * 4];
        file.read_exact(&mut m_bytes)?;
        file.read_exact(&mut v_bytes)?;
        let m: Vec<f32> = m_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let v: Vec<f32> = v_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        opt_states.push((m, v));
    }

    eprintln!(
        "Training state loaded: next_step {}, opt_step {}",
        next_step, opt_step
    );
    Ok((model, opt_states, next_step, opt_step, total_tokens))
}

pub(crate) fn normalize_training_state_next_step(version: u32, raw_step: u32, path: &str) -> u32 {
    if version >= 13 {
        return raw_step;
    }
    let is_legacy_final = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        == Some("state_final.bin");
    if is_legacy_final {
        raw_step
    } else {
        raw_step.saturating_add(1)
    }
}

/// Save a non-AdamW optimizer's state to a resume sidecar (`<state>.opt`). The main AMDT format
/// only carries AdamW m/v; muon/hybrid/8-bit state (momentum, int8 moments+scales) goes here so
/// resume restores it instead of restarting the optimizer fresh. Format: "AOPT" magic, opt_type
/// (len+utf8), step, n_blobs, then each blob (len + bytes).
pub fn save_opt_sidecar(
    path: &str,
    opt_type: &str,
    step: u32,
    blobs: &[Vec<u8>],
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"AOPT")?;
    let tb = opt_type.as_bytes();
    file.write_all(&(tb.len() as u32).to_le_bytes())?;
    file.write_all(tb)?;
    file.write_all(&step.to_le_bytes())?;
    file.write_all(&(blobs.len() as u32).to_le_bytes())?;
    for b in blobs {
        file.write_all(&(b.len() as u32).to_le_bytes())?;
        file.write_all(b)?;
    }
    Ok(())
}

/// Loaded optimizer sidecar: (opt_type, step, state blobs).
pub type OptSidecar = (String, u32, Vec<Vec<u8>>);

/// Load an optimizer-state sidecar if present. Returns (opt_type, step, blobs), or None if the file
/// doesn't exist (back-compat: pre-sidecar checkpoints just resume the optimizer fresh).
pub fn load_opt_sidecar(path: &str) -> std::io::Result<Option<OptSidecar>> {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let file_len = file.metadata()?.len();
    let mut consumed = 0u64;
    let mut buf4 = [0u8; 4];
    read_exact_count(&mut file, &mut buf4, &mut consumed)?;
    if &buf4 != b"AOPT" {
        return Err(invalid_checkpoint_data(format!(
            "not a valid AndreAI optimizer sidecar: expected AOPT magic, got {:02x?}",
            buf4
        )));
    }
    read_exact_count(&mut file, &mut buf4, &mut consumed)?;
    let type_len = u32::from_le_bytes(buf4) as usize;
    ensure_remaining_bytes(file_len, consumed, type_len as u64, "optimizer type")?;
    let mut tb = vec![0u8; type_len];
    read_exact_count(&mut file, &mut tb, &mut consumed)?;
    let opt_type = String::from_utf8(tb).map_err(|e| {
        invalid_checkpoint_data(format!("optimizer sidecar type is not valid UTF-8: {e}"))
    })?;
    read_exact_count(&mut file, &mut buf4, &mut consumed)?;
    let step = u32::from_le_bytes(buf4);
    read_exact_count(&mut file, &mut buf4, &mut consumed)?;
    let n_blobs = u32::from_le_bytes(buf4) as usize;
    ensure_remaining_bytes(
        file_len,
        consumed,
        (n_blobs as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid_checkpoint_data("optimizer sidecar blob table is too large"))?,
        format!("optimizer sidecar table for {n_blobs} blobs"),
    )?;
    let mut blobs = Vec::with_capacity(n_blobs);
    for i in 0..n_blobs {
        read_exact_count(&mut file, &mut buf4, &mut consumed)?;
        let len = u32::from_le_bytes(buf4) as usize;
        ensure_remaining_bytes(
            file_len,
            consumed,
            len as u64,
            format!("optimizer sidecar blob {i} length {len}"),
        )?;
        let mut b = vec![0u8; len];
        read_exact_count(&mut file, &mut b, &mut consumed)?;
        blobs.push(b);
    }
    if consumed != file_len {
        return Err(invalid_checkpoint_data(format!(
            "optimizer sidecar has {} trailing bytes",
            file_len - consumed
        )));
    }
    Ok(Some((opt_type, step, blobs)))
}

/// Load model from a checkpoint file.
pub fn load_checkpoint(ctx: &Arc<MetalContext>, path: &str) -> std::io::Result<(Transformer, u32)> {
    let mut file = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];

    // Magic
    file.read_exact(&mut buf4)?;
    if &buf4 != MAGIC {
        return Err(invalid_checkpoint_data(format!(
            "not a valid AndreAI checkpoint: expected AMDL magic, got {:02x?}",
            buf4
        )));
    }

    // Version
    file.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if !(1..=13).contains(&version) {
        return Err(invalid_checkpoint_data(format!(
            "unsupported checkpoint version: {version} (expected 1-13)"
        )));
    }

    // Step
    file.read_exact(&mut buf4)?;
    let step = u32::from_le_bytes(buf4);

    // Config
    let config = read_config(&mut file, version)?;
    eprintln!(
        "Loading checkpoint: step {}, {}M params",
        step,
        config.param_count() as f32 / 1e6
    );

    // Create model (random init, will be overwritten)
    let model = Transformer::new(ctx, config);

    // Number of tensors
    file.read_exact(&mut buf4)?;
    let n_tensors = u32::from_le_bytes(buf4) as usize;

    let params = model.parameters();
    let base_params = model.base_parameters();
    let all_params: Vec<&_> = params.iter().chain(base_params.iter()).copied().collect();

    // v3 checkpoints don't include base weights — allow loading with fewer tensors
    let expected = if version <= 3 {
        params.len()
    } else {
        all_params.len()
    };
    if n_tensors != expected {
        return Err(invalid_checkpoint_data(format!(
            "checkpoint has {n_tensors} tensors, model expects {expected}"
        )));
    }

    // Load each tensor (trainable params first, then base weights for v4+)
    let load_params = if version <= 3 {
        &params[..]
    } else {
        &all_params[..]
    };
    for (i, param) in load_params.iter().enumerate() {
        // Shape
        file.read_exact(&mut buf4)?;
        let ndims = u32::from_le_bytes(buf4) as usize;
        if ndims == 0 || ndims > MAX_TENSOR_DIMS {
            return Err(invalid_checkpoint_data(format!(
                "checkpoint tensor {i} has invalid dimension count {ndims}"
            )));
        }
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            file.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }

        if shape != param.shape {
            return Err(invalid_checkpoint_data(format!(
                "shape mismatch for checkpoint tensor {i}: checkpoint {:?} vs model {:?}",
                shape, param.shape
            )));
        }

        // Data
        let byte_len = checked_tensor_byte_len(&shape, format!("checkpoint tensor {i}"))?;
        let mut byte_data = vec![0u8; byte_len];
        file.read_exact(&mut byte_data)?;

        // Write directly to the Metal buffer (unified memory, zero-copy)
        crate::gpu::buf_write_bytes(&param.buffer, &byte_data);

        if i % 10 == 0 {
            eprintln!("  loaded tensor {}/{}", i + 1, n_tensors);
        }
    }

    eprintln!("Checkpoint loaded: {} (step {})", path, step);
    Ok((model, step))
}

fn write_config(file: &mut std::fs::File, config: &ModelConfig) -> std::io::Result<()> {
    file.write_all(&config.vocab_size.to_le_bytes())?;
    file.write_all(&(config.d_model as u32).to_le_bytes())?;
    file.write_all(&(config.n_heads as u32).to_le_bytes())?;
    file.write_all(&(config.n_layers as u32).to_le_bytes())?;
    file.write_all(&config.ffn_multiplier.to_le_bytes())?;
    file.write_all(&(config.max_seq_len as u32).to_le_bytes())?;
    file.write_all(&config.rope_theta.to_le_bytes())?;
    file.write_all(&config.norm_eps.to_le_bytes())?;
    // v2: GQA support
    file.write_all(&(config.n_kv_heads as u32).to_le_bytes())?;
    // v3: lowrank, MoE, bitnet, shared_layers, mup
    file.write_all(&(config.lowrank as u32).to_le_bytes())?;
    file.write_all(&(config.n_experts as u32).to_le_bytes())?;
    file.write_all(&(config.top_k_experts as u32).to_le_bytes())?;
    file.write_all(&(if config.bitnet { 1u32 } else { 0u32 }).to_le_bytes())?;
    file.write_all(&(if config.shared_layers { 1u32 } else { 0u32 }).to_le_bytes())?;
    file.write_all(&(config.mup_base_width as u32).to_le_bytes())?;
    file.write_all(&(config.n_predict as u32).to_le_bytes())?;
    // v5: linear (kernel) attention flag
    file.write_all(&(if config.linear_attn { 1u32 } else { 0u32 }).to_le_bytes())?;
    // v6: hybrid linear-attention period
    file.write_all(&(config.linear_attn_period as u32).to_le_bytes())?;
    // v7: SSM mixer flag
    file.write_all(&(if config.ssm { 1u32 } else { 0u32 }).to_le_bytes())?;
    // v8: RWKV mixer flag
    file.write_all(&(if config.rwkv { 1u32 } else { 0u32 }).to_le_bytes())?;
    // v9: MLA latent dim (0 = off)
    file.write_all(&(config.mla_latent_dim as u32).to_le_bytes())?;
    // v10: block-sparse attention top_k + block_size
    file.write_all(&(config.block_sparse_top_k as u32).to_le_bytes())?;
    file.write_all(&(config.block_size as u32).to_le_bytes())?;
    // v12: sliding-window size (0 = full causal). Forward-affecting — a windowed-trained model must
    // not silently load as full-causal. (stochastic_depth/fp16_activations are intentionally NOT
    // persisted: they're train-time-only knobs that should default off at inference/resume.)
    file.write_all(&(config.sliding_window as u32).to_le_bytes())?;
    Ok(())
}

fn read_config(file: &mut std::fs::File, version: u32) -> std::io::Result<ModelConfig> {
    let mut buf4 = [0u8; 4];

    file.read_exact(&mut buf4)?;
    let vocab_size = u32::from_le_bytes(buf4);

    file.read_exact(&mut buf4)?;
    let d_model = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let n_heads = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let n_layers = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let ffn_multiplier = f32::from_le_bytes(buf4);

    file.read_exact(&mut buf4)?;
    let max_seq_len = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf4)?;
    let rope_theta = f32::from_le_bytes(buf4);

    file.read_exact(&mut buf4)?;
    let norm_eps = f32::from_le_bytes(buf4);

    // v2: read n_kv_heads; v1: default to n_heads (standard MHA)
    let n_kv_heads = if version >= 2 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) as usize
    } else {
        n_heads
    };

    // v3: lowrank, MoE, bitnet, shared_layers, mup
    let (lowrank, n_experts, top_k_experts, bitnet, shared_layers, mup_base_width, n_predict) =
        if version >= 3 {
            file.read_exact(&mut buf4)?;
            let lr = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let ne = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let tk = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let bn = u32::from_le_bytes(buf4) != 0;
            file.read_exact(&mut buf4)?;
            let sl = u32::from_le_bytes(buf4) != 0;
            file.read_exact(&mut buf4)?;
            let mup = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let n_pred = u32::from_le_bytes(buf4) as usize;
            (lr, ne, tk, bn, sl, mup, n_pred)
        } else {
            (0, 1, 1, false, false, 0, 0) // defaults for v1/v2 checkpoints
        };

    // v5: linear (kernel) attention flag; older checkpoints default to softmax (false).
    let linear_attn = if version >= 5 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) != 0
    } else {
        false
    };

    // v6: hybrid linear-attention period; older checkpoints default to 0 (no hybrid schedule).
    let linear_attn_period = if version >= 6 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) as usize
    } else {
        0
    };

    // v7: SSM mixer flag; older checkpoints default to false.
    let ssm = if version >= 7 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) != 0
    } else {
        false
    };

    // v8: RWKV mixer flag; older checkpoints default to false.
    let rwkv = if version >= 8 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) != 0
    } else {
        false
    };

    // v9: MLA latent dim; older checkpoints default to 0 (off).
    let mla_latent_dim = if version >= 9 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) as usize
    } else {
        0
    };

    // v10: block-sparse top_k + block_size; older checkpoints default to off / 64.
    let (block_sparse_top_k, block_size) = if version >= 10 {
        file.read_exact(&mut buf4)?;
        let tk = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let bs = u32::from_le_bytes(buf4) as usize;
        (tk, bs)
    } else {
        (0, 64)
    };

    // v12: sliding-window size; older checkpoints default to 0 (full causal).
    let sliding_window = if version >= 12 {
        file.read_exact(&mut buf4)?;
        u32::from_le_bytes(buf4) as usize
    } else {
        0
    };

    Ok(ModelConfig {
        vocab_size,
        d_model,
        n_heads,
        n_kv_heads,
        n_layers,
        ffn_multiplier,
        max_seq_len,
        rope_theta,
        norm_eps,
        n_experts,
        top_k_experts,
        mup_base_width,
        shared_layers,
        bitnet,
        lowrank,
        n_predict,
        stochastic_depth: 0.0,
        sliding_window,
        fp16_activations: false,
        linear_attn,
        linear_attn_period,
        ssm,
        rwkv,
        mla_latent_dim,
        block_sparse_top_k,
        block_size,
        yarn_scale: 1.0,
        yarn_orig_max_seq: 0,
    })
}
