use crate::checkpoint;
use crate::metal::MetalContext;
use crate::model::{ModelConfig, Transformer};
use objc2_metal::MTLBuffer;
use std::io::{Read, Write};
use std::sync::Arc;

/// Magic bytes for quantized AndreAI checkpoint files.
const QMAGIC: &[u8; 4] = b"AMQZ";
const QVERSION: u32 = 1;

/// A quantized representation of a float tensor.
///
/// Supports Q8 (8-bit) and Q4 (4-bit) quantization with per-group
/// scale factors and zero points for accurate reconstruction.
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
    pub shape: Vec<usize>,
    pub bits: u8,
    pub group_size: usize,
}

/// Quantize an f32 slice to Q8 or Q4 with per-group scale and zero point.
///
/// `bits` must be 4 or 8. `group_size` is the number of elements per
/// quantization group (typically 32 or 128). Smaller groups preserve
/// more precision at the cost of slightly more metadata overhead.
pub fn quantize(data: &[f32], shape: &[usize], bits: u8, group_size: usize) -> QuantizedTensor {
    assert!(bits == 4 || bits == 8, "bits must be 4 or 8, got {}", bits);
    assert!(group_size > 0, "group_size must be > 0");

    let n_elements = data.len();
    let n_groups = n_elements.div_ceil(group_size);
    let mut scales = Vec::with_capacity(n_groups);
    let mut zeros = Vec::with_capacity(n_groups);

    match bits {
        8 => {
            let mut quantized = Vec::with_capacity(n_elements);
            for group_idx in 0..n_groups {
                let start = group_idx * group_size;
                let end = (start + group_size).min(n_elements);
                let group = &data[start..end];

                let (scale, zero) = compute_scale_zero_q8(group);
                scales.push(scale);
                zeros.push(zero);

                for &val in group {
                    let q = if scale.abs() < 1e-10 {
                        0u8
                    } else {
                        ((val - zero) / scale).round().clamp(0.0, 255.0) as u8
                    };
                    quantized.push(q);
                }
            }
            QuantizedTensor {
                data: quantized,
                scales,
                zeros,
                shape: shape.to_vec(),
                bits: 8,
                group_size,
            }
        }
        4 => {
            // Q4: two values packed into one byte (low nibble first, high nibble second)
            let packed_size = n_elements.div_ceil(2);
            let mut quantized = vec![0u8; packed_size];

            for group_idx in 0..n_groups {
                let start = group_idx * group_size;
                let end = (start + group_size).min(n_elements);
                let group = &data[start..end];

                let (scale, zero) = compute_scale_zero_q4(group);
                scales.push(scale);
                zeros.push(zero);

                for (i, &val) in group.iter().enumerate() {
                    let global_idx = start + i;
                    let q = if scale.abs() < 1e-10 {
                        0u8
                    } else {
                        ((val - zero) / scale).round().clamp(0.0, 15.0) as u8
                    };
                    let byte_idx = global_idx / 2;
                    if global_idx.is_multiple_of(2) {
                        quantized[byte_idx] |= q & 0x0F;
                    } else {
                        quantized[byte_idx] |= (q & 0x0F) << 4;
                    }
                }
            }
            QuantizedTensor {
                data: quantized,
                scales,
                zeros,
                shape: shape.to_vec(),
                bits: 4,
                group_size,
            }
        }
        _ => unreachable!(),
    }
}

/// Dequantize a `QuantizedTensor` back to f32 for inference.
///
/// The reconstructed values are: `value = quantized * scale + zero`.
pub fn dequantize(qt: &QuantizedTensor) -> Vec<f32> {
    let n_elements: usize = qt.shape.iter().product();
    let mut output = Vec::with_capacity(n_elements);

    match qt.bits {
        8 => {
            for group_idx in 0..qt.scales.len() {
                let start = group_idx * qt.group_size;
                let end = (start + qt.group_size).min(n_elements);
                let scale = qt.scales[group_idx];
                let zero = qt.zeros[group_idx];

                for i in start..end {
                    let q = qt.data[i] as f32;
                    output.push(q * scale + zero);
                }
            }
        }
        4 => {
            for group_idx in 0..qt.scales.len() {
                let start = group_idx * qt.group_size;
                let end = (start + qt.group_size).min(n_elements);
                let scale = qt.scales[group_idx];
                let zero = qt.zeros[group_idx];

                for global_idx in start..end {
                    let byte_idx = global_idx / 2;
                    let q = if global_idx.is_multiple_of(2) {
                        qt.data[byte_idx] & 0x0F
                    } else {
                        (qt.data[byte_idx] >> 4) & 0x0F
                    } as f32;
                    output.push(q * scale + zero);
                }
            }
        }
        _ => panic!("Unsupported quantization bits: {}", qt.bits),
    }

    output
}

/// Quantize an entire model checkpoint and save as a `.qbin` file.
///
/// Reads a standard AndreAI checkpoint, quantizes every tensor to the
/// specified bit width (4 or 8), and writes the result.
pub fn quantize_checkpoint(
    input_path: &str,
    output_path: &str,
    bits: u8,
) -> std::io::Result<()> {
    assert!(bits == 4 || bits == 8, "bits must be 4 or 8");

    let ctx = MetalContext::new();
    let (model, step) =
        checkpoint::load_checkpoint(&ctx, input_path).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;

    let group_size: usize = if bits == 4 { 32 } else { 128 };

    let params = model.parameters();
    let n_tensors = params.len();
    eprintln!(
        "Quantizing {} tensors to Q{} (group_size={})",
        n_tensors, bits, group_size
    );

    let mut file = std::fs::File::create(output_path)?;

    // Header
    file.write_all(QMAGIC)?;
    file.write_all(&QVERSION.to_le_bytes())?;
    file.write_all(&step.to_le_bytes())?;

    // Model config
    write_config(&mut file, &model.config)?;

    // Quantization metadata
    file.write_all(&[bits])?;
    file.write_all(&(group_size as u32).to_le_bytes())?;

    // Number of tensors
    file.write_all(&(n_tensors as u32).to_le_bytes())?;

    let mut total_original: usize = 0;
    let mut total_quantized: usize = 0;

    for (i, param) in params.iter().enumerate() {
        let data = param.to_vec();
        let qt = quantize(&data, &param.shape, bits, group_size);

        write_quantized_tensor(&mut file, &qt)?;

        let original_bytes = data.len() * 4;
        let quant_bytes = qt.data.len() + qt.scales.len() * 4 + qt.zeros.len() * 4;
        total_original += original_bytes;
        total_quantized += quant_bytes;

        if i % 10 == 0 {
            eprintln!(
                "  quantized tensor {}/{} ({} elements, {:.1}x compression)",
                i + 1,
                n_tensors,
                data.len(),
                original_bytes as f64 / quant_bytes as f64
            );
        }
    }

    let file_size = std::fs::metadata(output_path)?.len();
    eprintln!(
        "Quantized checkpoint saved: {} ({:.1} MB)",
        output_path,
        file_size as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "Compression: {:.1} MB → {:.1} MB ({:.1}x)",
        total_original as f64 / (1024.0 * 1024.0),
        total_quantized as f64 / (1024.0 * 1024.0),
        total_original as f64 / total_quantized as f64
    );

    Ok(())
}

/// Load a quantized `.qbin` checkpoint, dequantize weights, and construct
/// a `Transformer` ready for inference.
pub fn load_quantized(
    ctx: &Arc<MetalContext>,
    path: &str,
) -> std::io::Result<(Transformer, u32)> {
    let mut file = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];

    // Magic
    file.read_exact(&mut buf4)?;
    if &buf4 != QMAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a valid quantized AndreAI checkpoint",
        ));
    }

    // Version
    file.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != QVERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported quantized checkpoint version: {}", version),
        ));
    }

    // Step
    file.read_exact(&mut buf4)?;
    let step = u32::from_le_bytes(buf4);

    // Config
    let config = read_config(&mut file)?;

    // Quantization metadata
    let mut buf1 = [0u8; 1];
    file.read_exact(&mut buf1)?;
    let bits = buf1[0];

    file.read_exact(&mut buf4)?;
    let group_size = u32::from_le_bytes(buf4) as usize;

    eprintln!(
        "Loading Q{} checkpoint: step {}, {:.1}M params, group_size={}",
        bits,
        step,
        config.param_count() as f64 / 1e6,
        group_size
    );

    // Number of tensors
    file.read_exact(&mut buf4)?;
    let n_tensors = u32::from_le_bytes(buf4) as usize;

    // Create model with random init (will overwrite)
    let model = Transformer::new(ctx, config);
    let params = model.parameters();

    if params.len() != n_tensors {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Checkpoint has {} tensors, model expects {}",
                n_tensors,
                params.len()
            ),
        ));
    }

    for (i, param) in params.iter().enumerate() {
        let qt = read_quantized_tensor(&mut file)?;

        if qt.shape != param.shape {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Shape mismatch for tensor {}: checkpoint {:?} vs model {:?}",
                    i, qt.shape, param.shape
                ),
            ));
        }

        let dequantized = dequantize(&qt);
        let byte_data: Vec<u8> = dequantized.iter().flat_map(|f| f.to_le_bytes()).collect();

        unsafe {
            let ptr = param.buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(byte_data.as_ptr(), ptr, byte_data.len());
        }

        if i % 10 == 0 {
            eprintln!("  loaded tensor {}/{}", i + 1, n_tensors);
        }
    }

    eprintln!("Quantized checkpoint loaded: {} (step {})", path, step);
    Ok((model, step))
}

// --- Internal helpers ---

/// Compute per-group scale and zero point for Q8 (affine mapping to 0..255).
fn compute_scale_zero_q8(group: &[f32]) -> (f32, f32) {
    let min = group.iter().copied().fold(f32::INFINITY, f32::min);
    let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range.abs() < 1e-10 {
        return (0.0, min);
    }
    let scale = range / 255.0;
    let zero = min;
    (scale, zero)
}

/// Compute per-group scale and zero point for Q4 (affine mapping to 0..15).
fn compute_scale_zero_q4(group: &[f32]) -> (f32, f32) {
    let min = group.iter().copied().fold(f32::INFINITY, f32::min);
    let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range.abs() < 1e-10 {
        return (0.0, min);
    }
    let scale = range / 15.0;
    let zero = min;
    (scale, zero)
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
    // v2: GQA support — read n_kv_heads if available, else default to n_heads
    let n_kv_heads = match file.read_exact(&mut buf4) {
        Ok(()) => u32::from_le_bytes(buf4) as usize,
        Err(_) => n_heads, // v1 checkpoint: no n_kv_heads field
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
        n_experts: 1,
        top_k_experts: 1,
        mup_base_width: 0,
        shared_layers: false,
        bitnet: false,
        lowrank: 0,
    })
}

fn write_quantized_tensor(
    file: &mut std::fs::File,
    qt: &QuantizedTensor,
) -> std::io::Result<()> {
    // Shape
    let ndims = qt.shape.len() as u32;
    file.write_all(&ndims.to_le_bytes())?;
    for &dim in &qt.shape {
        file.write_all(&(dim as u32).to_le_bytes())?;
    }

    // Number of groups
    let n_groups = qt.scales.len() as u32;
    file.write_all(&n_groups.to_le_bytes())?;

    // Scales
    for &s in &qt.scales {
        file.write_all(&s.to_le_bytes())?;
    }

    // Zeros
    for &z in &qt.zeros {
        file.write_all(&z.to_le_bytes())?;
    }

    // Quantized data length + data
    let data_len = qt.data.len() as u32;
    file.write_all(&data_len.to_le_bytes())?;
    file.write_all(&qt.data)?;

    Ok(())
}

fn read_quantized_tensor(file: &mut std::fs::File) -> std::io::Result<QuantizedTensor> {
    let mut buf4 = [0u8; 4];

    // Shape
    file.read_exact(&mut buf4)?;
    let ndims = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        file.read_exact(&mut buf4)?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }

    // Number of groups
    file.read_exact(&mut buf4)?;
    let n_groups = u32::from_le_bytes(buf4) as usize;

    // Scales
    let mut scales = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        file.read_exact(&mut buf4)?;
        scales.push(f32::from_le_bytes(buf4));
    }

    // Zeros
    let mut zeros = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        file.read_exact(&mut buf4)?;
        zeros.push(f32::from_le_bytes(buf4));
    }

    // Quantized data
    file.read_exact(&mut buf4)?;
    let data_len = u32::from_le_bytes(buf4) as usize;
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    // Infer bits and group_size from data length and element count
    let n_elements: usize = shape.iter().product();
    let bits = if data_len == n_elements { 8 } else { 4 };
    let group_size = if n_groups > 0 {
        n_elements.div_ceil(n_groups)
    } else {
        n_elements
    };

    Ok(QuantizedTensor {
        data,
        scales,
        zeros,
        shape,
        bits,
        group_size,
    })
}
