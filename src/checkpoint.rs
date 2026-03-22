use crate::metal::MetalContext;
use crate::model::{ModelConfig, Transformer};
use objc2_metal::MTLBuffer;
use std::io::{Read, Write};
use std::sync::Arc;

/// Magic bytes for AndreAI checkpoint files.
const MAGIC: &[u8; 4] = b"AMDL";
const VERSION: u32 = 1;

/// Save model weights and config to a binary checkpoint file.
pub fn save_checkpoint(path: &str, model: &Transformer, step: u32) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Header
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&step.to_le_bytes())?;

    // Model config
    write_config(&mut file, &model.config)?;

    // Number of tensors
    let params = model.parameters();
    let n_tensors = params.len() as u32;
    file.write_all(&n_tensors.to_le_bytes())?;

    // Each tensor: shape + data
    for (i, param) in params.iter().enumerate() {
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
    assert_eq!(version, VERSION, "Unsupported checkpoint version: {}", version);

    // Step
    file.read_exact(&mut buf4)?;
    let step = u32::from_le_bytes(buf4);

    // Config
    let config = read_config(&mut file)?;
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
    assert_eq!(
        params.len(),
        n_tensors,
        "Checkpoint has {} tensors, model expects {}",
        n_tensors,
        params.len()
    );

    // Load each tensor
    for (i, param) in params.iter().enumerate() {
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
    Ok(())
}

fn read_config(file: &mut std::fs::File) -> std::io::Result<ModelConfig> {
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

    Ok(ModelConfig {
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        ffn_multiplier,
        max_seq_len,
        rope_theta,
        norm_eps,
    })
}
