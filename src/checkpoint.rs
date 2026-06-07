use crate::metal::MetalContext;
use crate::model::{ModelConfig, Transformer};
use crate::optim::AdamW;
use objc2_metal::MTLBuffer;
use std::io::{Read, Write};
use std::sync::Arc;

/// Magic bytes for AndreAI checkpoint files.
const MAGIC: &[u8; 4] = b"AMDL";
const VERSION: u32 = 11; // v11: explicit optimizer-param count (0 for non-AdamW). v10: block-sparse. v9: MLA. v8: rwkv. v7: ssm. v6: linear_attn_period. v5: linear_attn. v4: ReLoRA base weights

/// Return type for load_training_state: (model, optimizer_states, step, opt_step, total_tokens)
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
            eprintln!("  saving tensor {}/{} ({} elements)", i + 1, n_tensors, data.len());
        }
    }

    let size_mb = std::fs::metadata(path)?.len() as f32 / (1024.0 * 1024.0);
    eprintln!("Checkpoint saved: {} ({:.1} MB, step {})", path, size_mb, step);
    Ok(())
}

/// Save full training state: model weights + optimizer state (m, v, step).
/// This allows resuming training exactly where it left off.
pub fn save_training_state(path: &str, model: &Transformer, optimizer: &AdamW, step: u32, total_tokens: u64) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Header: AMDT (AndreAI Model Training state)
    file.write_all(b"AMDT")?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&step.to_le_bytes())?;
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
    eprintln!("Training state saved: {} ({:.1} MB, step {}, {} tokens)", path, size_mb, step, total_tokens);
    Ok(())
}

/// Load full training state for resume. Returns (model, optimizer_data, step, total_tokens).
/// The optimizer_data is (m_buffers, v_buffers, opt_step) — caller creates the AdamW and loads these.
pub fn load_training_state(
    ctx: &Arc<MetalContext>,
    path: &str,
) -> std::io::Result<TrainingState> {
    let mut file = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Magic
    file.read_exact(&mut buf4)?;
    assert_eq!(&buf4, b"AMDT", "Not a valid AndreAI training state file");

    // Version
    file.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    assert!((2..=11).contains(&version), "Unsupported training state version: {}", version);

    // Step + total_tokens
    file.read_exact(&mut buf4)?;
    let step = u32::from_le_bytes(buf4);
    file.read_exact(&mut buf8)?;
    let total_tokens = u64::from_le_bytes(buf8);

    // Config
    let config = read_config(&mut file, version)?;
    eprintln!("Resuming: step {}, {}M params, {} tokens processed",
        step, config.param_count() as f32 / 1e6, total_tokens);

    // Model
    let model = Transformer::new(ctx, config);
    file.read_exact(&mut buf4)?;
    let n_tensors = u32::from_le_bytes(buf4) as usize;
    let params = model.parameters();
    let base_params = model.base_parameters();

    // v4 training states include base params (ReLoRA frozen weights) after trainable params.
    // v2/v3 states only include trainable params.
    let expected = if version >= 4 { params.len() + base_params.len() } else { params.len() };
    assert_eq!(n_tensors, expected,
        "Training state has {} tensors, model expects {} (version {})",
        n_tensors, expected, version);

    let all_params: Vec<&_> = if version >= 4 {
        params.iter().chain(base_params.iter()).copied().collect()
    } else {
        params.to_vec()
    };

    for (i, param) in all_params.iter().enumerate() {
        file.read_exact(&mut buf4)?;
        let ndims = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            file.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }
        assert_eq!(shape, param.shape, "Shape mismatch tensor {}", i);

        let n_elements: usize = shape.iter().product();
        let mut byte_data = vec![0u8; n_elements * 4];
        file.read_exact(&mut byte_data)?;
        unsafe {
            let ptr = param.buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(byte_data.as_ptr(), ptr, byte_data.len());
        }
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

    let mut opt_states = Vec::with_capacity(opt_count);
    for param in params.iter().take(opt_count) {
        let size = param.numel();
        let mut m_bytes = vec![0u8; size * 4];
        let mut v_bytes = vec![0u8; size * 4];
        file.read_exact(&mut m_bytes)?;
        file.read_exact(&mut v_bytes)?;
        let m: Vec<f32> = m_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let v: Vec<f32> = v_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        opt_states.push((m, v));
    }

    eprintln!("Training state loaded: step {}, opt_step {}", step, opt_step);
    Ok((model, opt_states, step, opt_step, total_tokens))
}

/// Load model from a checkpoint file.
pub fn load_checkpoint(
    ctx: &Arc<MetalContext>,
    path: &str,
) -> std::io::Result<(Transformer, u32)> {
    let mut file = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];

    // Magic
    file.read_exact(&mut buf4)?;
    assert_eq!(&buf4, MAGIC, "Not a valid AndreAI checkpoint");

    // Version
    file.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    assert!((1..=11).contains(&version), "Unsupported checkpoint version: {} (expected 1-11)", version);

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
    let expected = if version <= 3 { params.len() } else { all_params.len() };
    assert_eq!(
        n_tensors, expected,
        "Checkpoint has {} tensors, model expects {}",
        n_tensors, expected
    );

    // Load each tensor (trainable params first, then base weights for v4+)
    let load_params = if version <= 3 { &params[..] } else { &all_params[..] };
    for (i, param) in load_params.iter().enumerate() {
        // Shape
        file.read_exact(&mut buf4)?;
        let ndims = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            file.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }

        assert_eq!(
            shape, param.shape,
            "Shape mismatch for tensor {}: checkpoint {:?} vs model {:?}",
            i, shape, param.shape
        );

        // Data
        let n_elements: usize = shape.iter().product();
        let mut byte_data = vec![0u8; n_elements * 4];
        file.read_exact(&mut byte_data)?;

        // Write directly to the Metal buffer (unified memory, zero-copy)
        unsafe {
            let ptr = param.buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(byte_data.as_ptr(), ptr, byte_data.len());
        }

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
    let (lowrank, n_experts, top_k_experts, bitnet, shared_layers, mup_base_width, n_predict) = if version >= 3 {
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
        (0, 1, 1, false, false, 0, 0)  // defaults for v1/v2 checkpoints
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
        sliding_window: 0,
        fp16_activations: false,
        linear_attn,
        linear_attn_period,
        ssm,
        rwkv,
        mla_latent_dim,
        block_sparse_top_k,
        block_size,
    })
}
