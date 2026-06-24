use crate::checkpoint;
use crate::gpu::MetalContext;
use crate::model::{ModelConfig, Transformer};
use std::io::{Error, ErrorKind, Read, Seek, Write};
use std::sync::Arc;

/// Magic bytes for quantized Smedjan checkpoint files.
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
        _ => panic!(
            "Unsupported quantization bits: {} (only 4 and 8 supported)",
            bits
        ),
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
/// Reads a standard Smedjan checkpoint, quantizes every tensor to the
/// specified bit width (4 or 8), and writes the result.
pub fn quantize_checkpoint(input_path: &str, output_path: &str, bits: u8) -> std::io::Result<()> {
    validate_quant_bits(bits)?;

    let ctx = MetalContext::new();
    let (model, step) = checkpoint::load_checkpoint(&ctx, input_path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

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
pub fn load_quantized(ctx: &Arc<MetalContext>, path: &str) -> std::io::Result<(Transformer, u32)> {
    let mut file = std::fs::File::open(path)?;
    let mut buf4 = [0u8; 4];

    // Magic
    file.read_exact(&mut buf4)?;
    if &buf4 != QMAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Not a valid quantized Smedjan checkpoint",
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
    if bits != 4 && bits != 8 {
        return Err(invalid_data(format!(
            "quantized checkpoint bits must be 4 or 8, got {bits}"
        )));
    }

    file.read_exact(&mut buf4)?;
    let group_size = u32::from_le_bytes(buf4) as usize;
    if group_size == 0 {
        return Err(invalid_data("quantized checkpoint group_size must be > 0"));
    }

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
        let qt = read_quantized_tensor(&mut file, bits, group_size)?;

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
        if dequantized.len() != param.numel() {
            return Err(invalid_data(format!(
                "Tensor {i} dequantized to {} values, expected {}",
                dequantized.len(),
                param.numel()
            )));
        }
        let byte_data: Vec<u8> = dequantized.iter().flat_map(|f| f.to_le_bytes()).collect();

        crate::gpu::buf_write_bytes(&param.buffer, &byte_data);

        if i % 10 == 0 {
            eprintln!("  loaded tensor {}/{}", i + 1, n_tensors);
        }
    }

    let pos = file.stream_position()?;
    let len = file.metadata()?.len();
    if pos != len {
        return Err(invalid_data(format!(
            "Quantized checkpoint has {} trailing bytes",
            len.saturating_sub(pos)
        )));
    }

    eprintln!("Quantized checkpoint loaded: {} (step {})", path, step);
    Ok((model, step))
}

// --- Internal helpers ---

fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

fn validate_quant_bits(bits: u8) -> std::io::Result<()> {
    if bits == 4 || bits == 8 {
        Ok(())
    } else {
        Err(invalid_input(format!("bits must be 4 or 8, got {bits}")))
    }
}

fn validate_gguf_quantize_type(quantize_type: &str) -> std::io::Result<()> {
    match quantize_type {
        "f32" | "q8_0" | "q4_0" => Ok(()),
        other => Err(invalid_input(format!(
            "unsupported GGUF quantization '{other}'; supported values: f32, q8_0, q4_0"
        ))),
    }
}

/// GGML quantization block length (both Q8_0 and Q4_0 quantize 32 values at a time).
const GGML_QK: usize = 32;
// GGML tensor type IDs (ggml.h).
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;

/// f32 -> IEEE-754 half (round-to-nearest-even). GGML stores each quant block's scale `d` as an
/// f16, so a faithful GGUF needs this exact encoding. Handles overflow→inf, subnormals, and NaN.
fn f32_to_f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        // Inf / NaN (preserve NaN-ness).
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> signed zero
        }
        // Subnormal: shift in the implicit leading 1 and round to nearest even.
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let round = u32::from(rem > halfway || (rem == halfway && (half & 1) == 1));
        return sign | (half + round) as u16;
    }
    // Normal: narrow the 23-bit mantissa to 10 bits with round-to-nearest-even.
    let mut half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1) {
        half_mant += 1;
        if half_mant == 0x0400 {
            half_mant = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | half_mant
}

/// Quantize one 32-value block to GGML Q8_0 layout: f16 scale `d = amax/127`, then 32 signed
/// int8 `round(x/d)`. 34 bytes/block. Matches `quantize_row_q8_0_ref` in llama.cpp.
fn quantize_q8_0_block(x: &[f32], out: &mut Vec<u8>) {
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    for &v in x {
        out.push(((v * id).round().clamp(-127.0, 127.0) as i8) as u8);
    }
}

/// Quantize one 32-value block to GGML Q4_0 layout: f16 scale `d = vmax/-8` (vmax = largest-
/// magnitude element, signed), then 16 bytes packing two 4-bit quants `clamp(trunc(x/d)+8.5,0,15)`,
/// pairing element `j` with `j+16`. 18 bytes/block. Matches `quantize_row_q4_0_ref` in llama.cpp.
fn quantize_q4_0_block(x: &[f32], out: &mut Vec<u8>) {
    let mut amax = 0f32;
    let mut vmax = 0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            vmax = v;
        }
    }
    let d = vmax / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    let q: Vec<u8> = x
        .iter()
        .map(|&v| (v * id + 8.5).floor().clamp(0.0, 15.0) as u8)
        .collect();
    for j in 0..GGML_QK / 2 {
        out.push(q[j] | (q[j + GGML_QK / 2] << 4));
    }
}

/// Whether a tensor is GGML-quantizable: 2-D+ weight matrix whose element count is a whole number
/// of 32-blocks. 1-D tensors (norms) and ragged shapes stay F32, exactly as llama.cpp's quantizer
/// keeps them — so a "quantized" GGUF is legitimately mixed-precision.
fn is_ggml_quantizable(shape: &[usize], numel: usize) -> bool {
    shape.len() >= 2 && numel % GGML_QK == 0 && numel > 0
}

/// Encode a tensor's f32 data into `(ggml_type, bytes)` for the requested GGUF quant. Non-quantizable
/// tensors (and the f32 export) pass through as raw little-endian f32.
fn encode_gguf_tensor(data: &[f32], shape: &[usize], quantize_type: &str) -> (u32, Vec<u8>) {
    let numel = data.len();
    let quantizable = quantize_type != "f32" && is_ggml_quantizable(shape, numel);
    if !quantizable {
        let bytes = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        return (GGML_TYPE_F32, bytes);
    }
    match quantize_type {
        "q8_0" => {
            let mut out = Vec::with_capacity(numel / GGML_QK * 34);
            for blk in data.chunks_exact(GGML_QK) {
                quantize_q8_0_block(blk, &mut out);
            }
            (GGML_TYPE_Q8_0, out)
        }
        "q4_0" => {
            let mut out = Vec::with_capacity(numel / GGML_QK * 18);
            for blk in data.chunks_exact(GGML_QK) {
                quantize_q4_0_block(blk, &mut out);
            }
            (GGML_TYPE_Q4_0, out)
        }
        _ => unreachable!("validated by validate_gguf_quantize_type"),
    }
}

fn ensure_tensor_names_match(
    context: &str,
    n_params: usize,
    n_names: usize,
) -> std::io::Result<()> {
    if n_params == n_names {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "{context} tensor count mismatch: model has {n_params} tensors but naming generates {n_names}"
        )))
    }
}

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
        n_predict: 0,
        stochastic_depth: 0.0,
        sliding_window: 0,
        fp16_activations: false,
        bitnet: false,
        lowrank: 0,
        linear_attn: false,
        linear_attn_period: 0,
        ssm: false,
        rwkv: false,
        mla_latent_dim: 0,
        block_sparse_top_k: 0,
        block_size: 64,
        yarn_scale: 1.0,
        yarn_orig_max_seq: 0,
    })
}

fn write_quantized_tensor(file: &mut std::fs::File, qt: &QuantizedTensor) -> std::io::Result<()> {
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

fn read_quantized_tensor(
    file: &mut std::fs::File,
    bits: u8,
    group_size: usize,
) -> std::io::Result<QuantizedTensor> {
    let mut buf4 = [0u8; 4];

    // Shape
    file.read_exact(&mut buf4)?;
    let ndims = u32::from_le_bytes(buf4) as usize;
    if ndims == 0 || ndims > 8 {
        return Err(invalid_data(format!(
            "quantized tensor has invalid rank {ndims}"
        )));
    }
    let mut shape = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        file.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;
        if dim == 0 {
            return Err(invalid_data("quantized tensor has a zero-sized dimension"));
        }
        shape.push(dim);
    }

    // Number of groups
    file.read_exact(&mut buf4)?;
    let n_groups = u32::from_le_bytes(buf4) as usize;
    let n_elements = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| invalid_data("quantized tensor element count overflows usize"))
    })?;
    let expected_groups = n_elements.div_ceil(group_size);
    if n_groups != expected_groups {
        return Err(invalid_data(format!(
            "quantized tensor group count mismatch: file has {n_groups}, expected {expected_groups} for {n_elements} elements with group_size {group_size}"
        )));
    }

    // Scales
    let mut scales = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        file.read_exact(&mut buf4)?;
        let value = f32::from_le_bytes(buf4);
        if !value.is_finite() {
            return Err(invalid_data("quantized tensor scale is not finite"));
        }
        scales.push(value);
    }

    // Zeros
    let mut zeros = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        file.read_exact(&mut buf4)?;
        let value = f32::from_le_bytes(buf4);
        if !value.is_finite() {
            return Err(invalid_data("quantized tensor zero point is not finite"));
        }
        zeros.push(value);
    }

    // Quantized data
    file.read_exact(&mut buf4)?;
    let data_len = u32::from_le_bytes(buf4) as usize;
    let expected_data_len = match bits {
        8 => n_elements,
        4 => n_elements.div_ceil(2),
        _ => return Err(invalid_input(format!("bits must be 4 or 8, got {bits}"))),
    };
    if data_len != expected_data_len {
        return Err(invalid_data(format!(
            "quantized tensor data length mismatch: file has {data_len} bytes, expected {expected_data_len} for Q{bits} and {n_elements} elements"
        )));
    }
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    Ok(QuantizedTensor {
        data,
        scales,
        zeros,
        shape,
        bits,
        group_size,
    })
}

/// Export model to GGUF format for llama.cpp inference.
/// Maps Smedjan tensor layout to GGUF's expected naming convention.
/// Supports F32, Q8_0, and Q4_0 (real GGML blocks; 1-D norm tensors stay F32).
pub fn export_gguf(
    model: &Transformer,
    output_path: &str,
    quantize_type: &str, // "f32", "q8_0", or "q4_0"
) -> std::io::Result<()> {
    use std::io::Write;
    validate_gguf_quantize_type(quantize_type)?;
    let config = &model.config;
    let params = model.parameters();
    let tensor_names = get_gguf_tensor_names(config);
    ensure_tensor_names_match("GGUF export", params.len(), tensor_names.len())?;

    let mut file = std::fs::File::create(output_path)?;

    // GGUF magic + version
    file.write_all(b"GGUF")?; // magic
    file.write_all(&3u32.to_le_bytes())?; // version 3
    let n_tensors = params.len() as u64;
    file.write_all(&n_tensors.to_le_bytes())?; // tensor count

    // Metadata KV pairs
    let metadata = vec![
        ("general.architecture", "llama"),
        ("general.name", "smedjan"),
    ];
    let n_kv = metadata.len() as u64 + 10; // base metadata + config values
    file.write_all(&n_kv.to_le_bytes())?;

    // Write string metadata
    for (key, val) in &metadata {
        write_gguf_string(&mut file, key)?;
        file.write_all(&8u32.to_le_bytes())?; // GGUF_TYPE_STRING
        write_gguf_string(&mut file, val)?;
    }

    // Write numeric metadata
    write_gguf_u32(&mut file, "llama.context_length", config.max_seq_len as u32)?;
    write_gguf_u32(&mut file, "llama.embedding_length", config.d_model as u32)?;
    write_gguf_u32(&mut file, "llama.block_count", config.n_layers as u32)?;
    write_gguf_u32(&mut file, "llama.feed_forward_length", config.d_ff() as u32)?;
    write_gguf_u32(
        &mut file,
        "llama.attention.head_count",
        config.n_heads as u32,
    )?;
    write_gguf_u32(
        &mut file,
        "llama.attention.head_count_kv",
        config.n_kv_heads as u32,
    )?;
    write_gguf_u32(&mut file, "llama.vocab_size", config.vocab_size)?;
    write_gguf_f32(&mut file, "llama.rope.freq_base", config.rope_theta)?;
    write_gguf_f32(
        &mut file,
        "llama.attention.layer_norm_rms_epsilon",
        config.norm_eps,
    )?;
    // llama.cpp ftype enum: ALL_F32=0, MOSTLY_Q4_0=2, MOSTLY_Q8_0=7.
    let file_type = match quantize_type {
        "q4_0" => 2u32,
        "q8_0" => 7u32,
        _ => 0u32,
    };
    write_gguf_u32(&mut file, "general.file_type", file_type)?;

    // Encode every tensor once into real GGML block bytes (mixed precision: norms/1-D stay F32).
    let encoded: Vec<(u32, Vec<u8>)> = params
        .iter()
        .map(|p| encode_gguf_tensor(&p.to_vec(), &p.shape, quantize_type))
        .collect();

    // Tensor info headers (name, shape, per-tensor type, offset).
    let mut data_offset: u64 = 0;
    for ((param, name), (gguf_type, bytes)) in
        params.iter().zip(tensor_names.iter()).zip(encoded.iter())
    {
        write_gguf_string(&mut file, name)?;
        let ndims = param.shape.len() as u32;
        file.write_all(&ndims.to_le_bytes())?;
        // GGUF stores dimensions in reverse order (innermost first).
        for &dim in param.shape.iter().rev() {
            file.write_all(&(dim as u64).to_le_bytes())?;
        }
        file.write_all(&gguf_type.to_le_bytes())?;
        file.write_all(&data_offset.to_le_bytes())?;
        data_offset += bytes.len() as u64;
    }

    // Alignment padding to 32 bytes.
    let pos = file.metadata()?.len();
    let aligned = (pos + 31) & !31;
    for _ in pos..aligned {
        file.write_all(&[0u8])?;
    }

    // Tensor data (GGML block bytes, in the same order as the info headers).
    for (_, bytes) in &encoded {
        file.write_all(bytes)?;
    }

    let size_mb = std::fs::metadata(output_path)?.len() as f32 / (1024.0 * 1024.0);
    eprintln!(
        "GGUF exported: {} ({:.1} MB, {} tensors, {})",
        output_path, size_mb, n_tensors, quantize_type
    );
    Ok(())
}

fn write_gguf_string(file: &mut std::fs::File, s: &str) -> std::io::Result<()> {
    use std::io::Write;
    file.write_all(&(s.len() as u64).to_le_bytes())?;
    file.write_all(s.as_bytes())
}

fn write_gguf_u32(file: &mut std::fs::File, key: &str, val: u32) -> std::io::Result<()> {
    use std::io::Write;
    write_gguf_string(file, key)?;
    file.write_all(&4u32.to_le_bytes())?; // GGUF_TYPE_UINT32
    file.write_all(&val.to_le_bytes())
}

fn write_gguf_f32(file: &mut std::fs::File, key: &str, val: f32) -> std::io::Result<()> {
    use std::io::Write;
    write_gguf_string(file, key)?;
    file.write_all(&6u32.to_le_bytes())?; // GGUF_TYPE_FLOAT32
    file.write_all(&val.to_le_bytes())
}

/// Map Smedjan tensor indices to GGUF-compatible names (llama architecture).
fn get_gguf_tensor_names(config: &ModelConfig) -> Vec<String> {
    let mut names = Vec::new();
    // Embedding
    names.push("token_embd.weight".to_string());
    // Final norm
    names.push("output_norm.weight".to_string());
    // Embed proj (if factored)
    if config.lowrank > 0 {
        names.push("token_embd_proj.weight".to_string());
    }
    // Per-layer tensors — when shared_layers, parameters() returns 1 unique layer
    let n_unique_layers = if config.shared_layers {
        1
    } else {
        config.n_layers
    };
    for i in 0..n_unique_layers {
        if config.lowrank > 0 {
            // Low-rank: U and V for each projection
            names.push(format!("blk.{}.attn_q_u.weight", i));
            names.push(format!("blk.{}.attn_q_v.weight", i));
            names.push(format!("blk.{}.attn_k_u.weight", i));
            names.push(format!("blk.{}.attn_k_v.weight", i));
            names.push(format!("blk.{}.attn_v_u.weight", i));
            names.push(format!("blk.{}.attn_v_v.weight", i));
            names.push(format!("blk.{}.attn_output_u.weight", i));
            names.push(format!("blk.{}.attn_output_v.weight", i));
        } else {
            names.push(format!("blk.{}.attn_q.weight", i));
            names.push(format!("blk.{}.attn_k.weight", i));
            names.push(format!("blk.{}.attn_v.weight", i));
            names.push(format!("blk.{}.attn_output.weight", i));
        }
        // QK-norm weight
        names.push(format!("blk.{}.attn_qk_norm.weight", i));
        if config.lowrank > 0 {
            names.push(format!("blk.{}.ffn_gate_u.weight", i));
            names.push(format!("blk.{}.ffn_gate_v.weight", i));
            names.push(format!("blk.{}.ffn_down_u.weight", i));
            names.push(format!("blk.{}.ffn_down_v.weight", i));
            names.push(format!("blk.{}.ffn_up_u.weight", i));
            names.push(format!("blk.{}.ffn_up_v.weight", i));
        } else {
            names.push(format!("blk.{}.ffn_gate.weight", i));
            names.push(format!("blk.{}.ffn_down.weight", i));
            names.push(format!("blk.{}.ffn_up.weight", i));
        }
        names.push(format!("blk.{}.attn_norm.weight", i));
        names.push(format!("blk.{}.ffn_norm.weight", i));
        // MoD router (if enabled — but always present in struct)
    }
    // MTP heads
    for k in 0..config.n_predict {
        names.push(format!("mtp.{}.proj.weight", k));
    }
    for k in 0..config.n_predict {
        names.push(format!("mtp.{}.norm.weight", k));
    }
    names
}

/// Export model to Safetensors format for HuggingFace ecosystem.
/// Safetensors is a simple, safe format: JSON header + raw tensor data.
pub fn export_safetensors(model: &Transformer, output_path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let config = &model.config;
    let params = model.parameters();
    let tensor_names = get_gguf_tensor_names(config); // reuse naming
    ensure_tensor_names_match("Safetensors export", params.len(), tensor_names.len())?;

    // Build JSON header: { "tensor_name": { "dtype": "F32", "shape": [...], "data_offsets": [start, end] } }
    let mut header = String::from("{");
    let mut offset: usize = 0;
    for (i, (param, name)) in params.iter().zip(tensor_names.iter()).enumerate() {
        if i > 0 {
            header.push(',');
        }
        let nbytes = param.numel() * 4;
        let shape_str: Vec<String> = param.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            shape_str.join(","),
            offset,
            offset + nbytes
        ));
        offset += nbytes;
    }
    // Add __metadata__
    header.push_str(",\"__metadata__\":{\"format\":\"smedjan\",\"description\":\"Smedjan model\"}");
    header.push('}');

    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file = std::fs::File::create(output_path)?;
    // Safetensors format: 8 bytes header length (LE u64) + header JSON + tensor data
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;

    // Write tensor data (F32, little-endian)
    for param in &params {
        let data = param.to_vec();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
    }

    let size_mb = std::fs::metadata(output_path)?.len() as f32 / (1024.0 * 1024.0);
    eprintln!(
        "Safetensors exported: {} ({:.1} MB, {} tensors)",
        output_path,
        size_mb,
        params.len()
    );
    Ok(())
}

#[cfg(test)]
mod gguf_block_tests {
    use super::*;

    /// Local f16 -> f32 decode (mirrors safetensors::f16_to_f32) so the block tests can verify the
    /// scale that quantize_q*_0_block writes without reaching across modules.
    fn f16_to_f32(h: u16) -> f32 {
        let sign = (h as u32 & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mant = h as u32 & 0x3ff;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e = -1i32;
                let mut m = mant;
                loop {
                    e += 1;
                    m <<= 1;
                    if m & 0x400 != 0 {
                        break;
                    }
                }
                sign | (((127 - 15 - e) as u32) << 23) | ((m & 0x3ff) << 13)
            }
        } else if exp == 0x1f {
            sign | (0xff << 23) | (mant << 13)
        } else {
            sign | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    #[test]
    fn f16_encoding_matches_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(-2.0), 0xC000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7BFF);
        for &v in &[0.1f32, -3.7, 12.25, -0.003, 100.0, 1e-4] {
            let r = f16_to_f32(f32_to_f16_bits(v));
            assert!((r - v).abs() <= 1e-2 * v.abs().max(1e-3), "f16 {v} -> {r}");
        }
        assert!(f16_to_f32(f32_to_f16_bits(1e30)).is_infinite());
    }

    #[test]
    fn q8_0_block_roundtrips_within_tolerance() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.37).collect();
        let mut out = Vec::new();
        quantize_q8_0_block(&x, &mut out);
        assert_eq!(out.len(), 34, "Q8_0 block = 2 (f16 d) + 32 int8");
        let d = f16_to_f32(u16::from_le_bytes([out[0], out[1]]));
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        for (i, &xi) in x.iter().enumerate() {
            let deq = (out[2 + i] as i8) as f32 * d;
            assert!((deq - xi).abs() <= amax / 127.0 + 1e-5, "q8_0 [{i}] {xi} -> {deq}");
        }
    }

    #[test]
    fn q4_0_block_roundtrips_within_tolerance() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.21).collect();
        let mut out = Vec::new();
        quantize_q4_0_block(&x, &mut out);
        assert_eq!(out.len(), 18, "Q4_0 block = 2 (f16 d) + 16 packed nibbles");
        let d = f16_to_f32(u16::from_le_bytes([out[0], out[1]]));
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        for j in 0..16 {
            let lo = (out[2 + j] & 0x0f) as i32 - 8;
            let hi = (out[2 + j] >> 4) as i32 - 8;
            assert!((lo as f32 * d - x[j]).abs() <= 1.5 * d.abs() + amax * 0.03, "q4_0 lo[{j}]");
            assert!((hi as f32 * d - x[j + 16]).abs() <= 1.5 * d.abs() + amax * 0.03, "q4_0 hi[{j}]");
        }
    }

    #[test]
    fn encode_keeps_1d_f32_and_quantizes_matrices() {
        let norm = vec![1.0f32; 64];
        let (t, bytes) = encode_gguf_tensor(&norm, &[64], "q8_0");
        assert_eq!(t, GGML_TYPE_F32, "1-D norm stays F32 (matches llama.cpp)");
        assert_eq!(bytes.len(), 64 * 4);

        let w: Vec<f32> = (0..64 * 32).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let (t8, b8) = encode_gguf_tensor(&w, &[64, 32], "q8_0");
        assert_eq!(t8, GGML_TYPE_Q8_0);
        assert_eq!(b8.len(), (64 * 32 / 32) * 34);
        let (t4, b4) = encode_gguf_tensor(&w, &[64, 32], "q4_0");
        assert_eq!(t4, GGML_TYPE_Q4_0);
        assert_eq!(b4.len(), (64 * 32 / 32) * 18);
        let (tf, bf) = encode_gguf_tensor(&w, &[64, 32], "f32");
        assert_eq!(tf, GGML_TYPE_F32);
        assert_eq!(bf.len(), 64 * 32 * 4);
    }
}
