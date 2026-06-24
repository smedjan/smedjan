//! safetensors I/O for Smedjan models (zero deps; hand-written). Layout: 8-byte little-endian
//! header length, a JSON header `{ "<name>": {dtype, shape, data_offsets:[begin,end]}, ... }`, then
//! the raw tensor blob. Tensors are written in `model.parameters()` order (trainable, then ReLoRA
//! base) under positional names `p{i}`, mirroring `checkpoint::save_checkpoint`, so an Smedjan model
//! round-trips through safetensors. `import_safetensors` rebuilds the model from a caller-supplied
//! config (the same flow a foreign HF import uses, sourcing config from the model's config.json) and
//! overwrites each parameter in order. Foreign HF->Smedjan name remap + `[out,in]`->`[in,out]` transpose
//! + RoPE permutation layer on top of this format machinery.
#![allow(dead_code)] // export/import are exercised by the round-trip test; CLI + foreign import wire them next.

use crate::gpu::MetalContext;
use crate::model::{ModelConfig, Transformer};
use std::io::{Error, ErrorKind, Read, Write};
use std::sync::Arc;

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

// --- minimal JSON, enough for safetensors headers (objects, arrays, strings, numbers) ---

enum Json {
    Obj(Vec<(String, Json)>),
    Arr(Vec<Json>),
    Str(String),
    Num(String),
    Bool(bool),
    Null,
}

impl Json {
    fn as_obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(e) = self {
            Some(e)
        } else {
            None
        }
    }
    fn get(&self, k: &str) -> Option<&Json> {
        self.as_obj()?.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }
    fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    fn as_u64(&self) -> Option<u64> {
        if let Json::Num(n) = self {
            n.parse::<u64>().ok()
        } else {
            None
        }
    }
    fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            n.parse::<f64>().ok()
        } else {
            None
        }
    }
    fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn value(&mut self) -> std::io::Result<Json> {
        self.ws();
        match self.b.get(self.i) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.lit("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.lit("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.lit("null")?;
                Ok(Json::Null)
            }
            Some(_) => self.number(),
            None => Err(invalid("safetensors header: unexpected end of JSON")),
        }
    }
    fn lit(&mut self, s: &str) -> std::io::Result<()> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(invalid("safetensors header: invalid literal"))
        }
    }
    fn object(&mut self) -> std::io::Result<Json> {
        self.i += 1; // consume {
        let mut entries = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(entries));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err(invalid("safetensors header: expected ':'"));
            }
            self.i += 1;
            let val = self.value()?;
            entries.push((key, val));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(invalid("safetensors header: expected ',' or '}'")),
            }
        }
        Ok(Json::Obj(entries))
    }
    fn array(&mut self) -> std::io::Result<Json> {
        self.i += 1; // consume [
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(invalid("safetensors header: expected ',' or ']'")),
            }
        }
        Ok(Json::Arr(items))
    }
    fn string(&mut self) -> std::io::Result<String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(invalid("safetensors header: expected string"));
        }
        self.i += 1;
        let mut s = String::new();
        while let Some(&c) = self.b.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = *self
                        .b
                        .get(self.i)
                        .ok_or_else(|| invalid("safetensors header: bad escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'u' => {
                            let hex = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or_else(|| invalid("safetensors header: bad \\u escape"))?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| invalid("bad \\u"))?,
                                16,
                            )
                            .map_err(|_| invalid("bad \\u"))?;
                            self.i += 4;
                            s.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(invalid("safetensors header: bad escape")),
                    }
                }
                _ => s.push(c as char), // safetensors names/dtypes are ASCII
            }
        }
        Err(invalid("safetensors header: unterminated string"))
    }
    fn number(&mut self) -> std::io::Result<Json> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok().map(|_| Json::Num(s.to_string())))
            .ok_or_else(|| invalid("safetensors header: bad number"))
    }
}

const ST_DTYPE: &str = "F32";
const MAX_HEADER_LEN: u64 = 100_000_000;

fn parse_header(header: &[u8]) -> std::io::Result<Json> {
    let mut parser = JsonParser::new(header);
    let json = parser.value()?;
    parser.ws();
    if parser.i != parser.b.len() {
        return Err(invalid("safetensors header: trailing data after JSON"));
    }
    Ok(json)
}

fn read_parts(path: &str) -> std::io::Result<(Json, Vec<u8>)> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut lenb = [0u8; 8];
    file.read_exact(&mut lenb)?;
    let header_len = u64::from_le_bytes(lenb);
    if header_len > MAX_HEADER_LEN {
        return Err(invalid(format!(
            "safetensors header too large: {header_len} bytes"
        )));
    }
    if header_len > file_len.saturating_sub(8) {
        return Err(invalid(format!(
            "safetensors header length {header_len} exceeds file payload"
        )));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| invalid("safetensors header length does not fit usize"))?;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let mut blob = Vec::new();
    file.read_to_end(&mut blob)?;
    Ok((parse_header(&header)?, blob))
}

fn json_usize(value: &Json, field: impl Into<String>) -> std::io::Result<usize> {
    let field = field.into();
    let n = value
        .as_u64()
        .ok_or_else(|| invalid(format!("{field}: expected non-negative integer")))?;
    usize::try_from(n).map_err(|_| invalid(format!("{field}: integer does not fit usize")))
}

/// On-disk bytes per element for a supported safetensors dtype. `F32` is native;
/// `BF16`/`F16` are half-width and converted to f32 on import (`bf16/f16 weight loading`).
fn dtype_elem_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "BF16" | "F16" => Some(2),
        _ => None,
    }
}

/// bf16 (truncated f32: sign+8 exp+7 mantissa) -> f32: the bf16 bit pattern is the high 16 bits
/// of the f32 with the same value, so widening is a left shift. Inf/NaN/subnormals carry through.
#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// IEEE-754 half (f16: sign+5 exp+10 mantissa) -> f32, exact for all inputs incl. subnormals,
/// inf and NaN (no `std` f16, so this is the hand-rolled widening every loader needs).
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let mant = h as u32 & 0x3FF;
    let bits = if exp == 0 {
        if mant == 0 {
            sign // signed zero
        } else {
            // Subnormal: renormalize into f32's normal range.
            let mut e = -1i32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let exp32 = (127 - 15 - e) as u32;
            sign | (exp32 << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        // Inf / NaN: max exponent, mantissa preserved (left-aligned).
        sign | (0xFF << 23) | (mant << 13)
    } else {
        // Normal: rebias exponent 15 -> 127, widen mantissa 10 -> 23 bits.
        let exp32 = (exp as i32 - 15 + 127) as u32;
        sign | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Decode a raw tensor byte range (`F32`/`BF16`/`F16`) to `Vec<f32>`, validating that the byte
/// length matches `numel` at the dtype's element width. Shared by the generic and HF importers.
fn decode_to_f32(bytes: &[u8], dtype: &str, numel: usize, name: &str) -> std::io::Result<Vec<f32>> {
    let elem = dtype_elem_bytes(dtype).ok_or_else(|| {
        invalid(format!(
            "{name}: dtype {dtype} unsupported (only F32, BF16, F16)"
        ))
    })?;
    let expect = numel
        .checked_mul(elem)
        .ok_or_else(|| invalid(format!("{name}: tensor byte length overflows usize")))?;
    if bytes.len() != expect {
        return Err(invalid(format!(
            "{name}: {} bytes for {numel} {dtype} elements (expected {expect})",
            bytes.len()
        )));
    }
    Ok(match dtype {
        "F32" => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        "BF16" => bytes
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        _ => unreachable!("dtype validated above"),
    })
}

fn shape_numel(shape: &[usize], name: &str) -> std::io::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| invalid(format!("{name}: shape product overflows usize")))
    })
}

fn shape_field(entry: &Json, name: &str) -> std::io::Result<Vec<usize>> {
    entry
        .get("shape")
        .and_then(|s| s.as_arr())
        .ok_or_else(|| invalid(format!("{name}: no shape")))?
        .iter()
        .enumerate()
        .map(|(i, d)| json_usize(d, format!("{name}: shape[{i}]")))
        .collect()
}

fn offsets_field(entry: &Json, name: &str) -> std::io::Result<(usize, usize)> {
    let offs = entry
        .get("data_offsets")
        .and_then(|o| o.as_arr())
        .ok_or_else(|| invalid(format!("{name}: no data_offsets")))?;
    if offs.len() != 2 {
        return Err(invalid(format!("{name}: data_offsets must have 2 entries")));
    }
    let start = json_usize(&offs[0], format!("{name}: data_offsets[0]"))?;
    let end = json_usize(&offs[1], format!("{name}: data_offsets[1]"))?;
    Ok((start, end))
}

/// Export an Smedjan model to a `.safetensors` file (F32). Tensors are named `p{i}` in
/// `parameters()`-then-`base_parameters()` order, the same order `import_safetensors` walks.
pub fn export_safetensors(path: &str, model: &Transformer) -> std::io::Result<()> {
    let params = model.parameters();
    let base = model.base_parameters();
    let all: Vec<&_> = params.iter().chain(base.iter()).copied().collect();

    let mut blob: Vec<u8> = Vec::new();
    let mut entries: Vec<String> = Vec::with_capacity(all.len());
    for (i, p) in all.iter().enumerate() {
        let data = p.to_vec();
        let start = blob.len();
        blob.extend(data.iter().flat_map(|f| f.to_le_bytes()));
        let end = blob.len();
        let shape = p
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"p{i}\":{{\"dtype\":\"{ST_DTYPE}\",\"shape\":[{shape}],\"data_offsets\":[{start},{end}]}}"
        ));
    }
    let mut header = String::from("{");
    header.push_str(&entries.join(","));
    if !entries.is_empty() {
        header.push(',');
    }
    header.push_str(&format!(
        "\"__metadata__\":{{\"format\":\"smedjan-safetensors-v1\",\"n_tensors\":\"{}\"}}}}",
        all.len()
    ));

    // pad header so the data blob starts on an 8-byte boundary (safetensors convention)
    let mut header_bytes = header.into_bytes();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    file.write_all(&blob)?;
    eprintln!(
        "safetensors exported: {} ({} tensors, {:.1} MB)",
        path,
        all.len(),
        (8 + header_bytes.len() + blob.len()) as f32 / 1_048_576.0
    );
    Ok(())
}

/// Import a `.safetensors` file into a fresh model built from `config`. Each `p{i}` tensor is
/// shape-checked against the model parameter at index `i` and written into its buffer.
pub fn import_safetensors(
    ctx: &Arc<MetalContext>,
    path: &str,
    config: ModelConfig,
) -> std::io::Result<Transformer> {
    let (json, blob) = read_parts(path)?;
    let obj = json
        .as_obj()
        .ok_or_else(|| invalid("safetensors header is not a JSON object"))?;
    let n_entries = obj.iter().filter(|(k, _)| k != "__metadata__").count();

    let model = Transformer::new(ctx, config);
    let params = model.parameters();
    let base = model.base_parameters();
    let targets: Vec<&_> = params.iter().chain(base.iter()).copied().collect();
    if n_entries != targets.len() {
        return Err(invalid(format!(
            "safetensors has {n_entries} tensors, model (from config) expects {}",
            targets.len()
        )));
    }

    for (i, p) in targets.iter().enumerate() {
        let name = format!("p{i}");
        let entry = json
            .get(&name)
            .ok_or_else(|| invalid(format!("safetensors: missing tensor {name}")))?;
        let dtype = entry
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| invalid(format!("{name}: no dtype")))?;
        let elem = dtype_elem_bytes(dtype).ok_or_else(|| {
            invalid(format!(
                "{name}: dtype {dtype} unsupported (only F32, BF16, F16)"
            ))
        })?;
        let shape = shape_field(entry, &name)?;
        if shape != p.shape {
            return Err(invalid(format!(
                "{name}: safetensors shape {:?} != model shape {:?}",
                shape, p.shape
            )));
        }
        let (start, end) = offsets_field(entry, &name)?;
        let expect = p
            .numel()
            .checked_mul(elem)
            .ok_or_else(|| invalid(format!("{name}: tensor byte length overflows usize")))?;
        if start > end || end > blob.len() || end - start != expect {
            return Err(invalid(format!(
                "{name}: byte range [{start},{end}) invalid (expected {expect} bytes, blob {})",
                blob.len()
            )));
        }
        // F32 writes straight through; BF16/F16 widen to f32 first.
        if dtype == "F32" {
            crate::gpu::buf_write_bytes(&p.buffer, &blob[start..end]);
        } else {
            let widened = decode_to_f32(&blob[start..end], dtype, p.numel(), &name)?;
            let f32_bytes: Vec<u8> = widened.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&p.buffer, &f32_bytes);
        }
    }
    eprintln!(
        "safetensors imported: {} tensors from {}",
        targets.len(),
        path
    );
    Ok(model)
}

// ============================================================================
// Foreign HF-Llama interop: load external safetensors weights as an Smedjan model INIT (for the
// SubQ-style retrofit / continued-training flow), and export an Smedjan model back to HF-Llama
// layout. Faithful HF *inference* parity is NOT the goal and is not reachable here — Smedjan applies
// a fixed QK-norm for d_model>=512 and uses interleaved RoPE where HF-Llama uses half-split; those
// divergences are adapted away by continued training. Shapes are mapped exactly: Linear weights
// transpose ([out,in]<->[in,out]) and Q/K rows are permuted per head between HF half-split and
// Smedjan interleaved RoPE order. The round-trip test proves the transforms are exact inverses.
// Supports the standard Llama-arch shape only (Softmax attn, dense FFN, full-rank, tied head).
// ============================================================================

const HF_BLOCK_PARAMS: usize = 10; // params per standard block in parameters() order:
                                   // [w_q, w_k, w_v, w_o, qk_norm, ffn_w1(gate), ffn_w2(down), ffn_w3(up), ln1, ln2]

/// Transpose a row-major [rows, cols] f32 matrix to [cols, rows].
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Permute the rows of a `[n_heads*head_dim, in_dim]` matrix between HF half-split RoPE order and
/// Smedjan interleaved-pair order, per head. `half_to_interleaved=true` maps HF->Smedjan
/// (row j -> 2j, row j+hd/2 -> 2j+1); `false` is the inverse.
fn rope_permute_rows(
    data: &[f32],
    n_heads: usize,
    head_dim: usize,
    in_dim: usize,
    half_to_interleaved: bool,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = vec![0.0f32; data.len()];
    let row = |r: usize| r * in_dim..r * in_dim + in_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for j in 0..half {
            let (a, b) = (base + j, base + j + half); // half-split rows
            let (x, y) = (base + 2 * j, base + 2 * j + 1); // interleaved rows
            if half_to_interleaved {
                out[row(x)].copy_from_slice(&data[row(a)]);
                out[row(y)].copy_from_slice(&data[row(b)]);
            } else {
                out[row(a)].copy_from_slice(&data[row(x)]);
                out[row(b)].copy_from_slice(&data[row(y)]);
            }
        }
    }
    out
}

/// Write a name-keyed F32 safetensors file (shared by the HF-layout export).
fn write_named(path: &str, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> std::io::Result<()> {
    let mut blob: Vec<u8> = Vec::new();
    let mut entries: Vec<String> = Vec::with_capacity(tensors.len());
    for (name, shape, data) in tensors {
        let expect = shape_numel(shape, name)?;
        if expect != data.len() {
            return Err(invalid(format!(
                "{name}: shape {:?} has {expect} elements but data has {}",
                shape,
                data.len()
            )));
        }
        let start = blob.len();
        blob.extend(data.iter().flat_map(|f| f.to_le_bytes()));
        let end = blob.len();
        let sh = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{sh}],\"data_offsets\":[{start},{end}]}}"
        ));
    }
    let mut header = String::from("{");
    header.push_str(&entries.join(","));
    if !entries.is_empty() {
        header.push(',');
    }
    header.push_str("\"__metadata__\":{\"format\":\"hf-llama-f32\"}}");
    let mut hb = header.into_bytes();
    while (8 + hb.len()) % 8 != 0 {
        hb.push(b' ');
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&(hb.len() as u64).to_le_bytes())?;
    file.write_all(&hb)?;
    file.write_all(&blob)?;
    Ok(())
}

fn ensure_standard(config: &ModelConfig, n_params: usize) -> std::io::Result<()> {
    let expected = 2 + config.n_layers * HF_BLOCK_PARAMS;
    let standard = config.n_experts <= 1
        && config.mla_latent_dim == 0
        && !config.ssm
        && !config.rwkv
        && config.block_sparse_top_k == 0
        && config.lowrank == 0
        && !config.shared_layers
        && config.n_predict == 0;
    let heads_ok = config.n_heads > 0
        && config.n_kv_heads > 0
        && config.d_model.is_multiple_of(config.n_heads)
        && (config.d_model / config.n_heads).is_multiple_of(2);
    if !standard || n_params != expected {
        return Err(invalid(format!(
            "HF interop supports only the standard Llama-arch shape (Softmax attn, dense FFN, full-rank, full embedding, tied head): expected {expected} params, got {n_params}. MoE/MLA/RWKV/SSM/low-rank are not HF-mappable."
        )));
    }
    if !heads_ok {
        return Err(invalid(format!(
            "HF interop requires non-zero heads and an even RoPE head_dim; got d_model={}, n_heads={}, n_kv_heads={}",
            config.d_model, config.n_heads, config.n_kv_heads
        )));
    }
    Ok(())
}

/// Export an Smedjan model to HF-Llama-named safetensors (inverse transforms of the importer).
pub fn export_hf_safetensors(path: &str, model: &Transformer) -> std::io::Result<()> {
    let c = &model.config;
    let p = model.parameters();
    ensure_standard(c, p.len())?;
    let (d, nh, kvh) = (c.d_model, c.n_heads, c.n_kv_heads);
    let hd = d / nh;

    let mut named: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    named.push((
        "model.embed_tokens.weight".into(),
        p[0].shape.clone(),
        p[0].to_vec(),
    ));
    named.push((
        "model.norm.weight".into(),
        p[1].shape.clone(),
        p[1].to_vec(),
    ));
    for b in 0..c.n_layers {
        let base = 2 + b * HF_BLOCK_PARAMS;
        let pfx = format!("model.layers.{b}");
        let lin = |t: &crate::tensor::Tensor| -> (Vec<usize>, Vec<f32>) {
            (
                vec![t.shape[1], t.shape[0]],
                transpose_2d(&t.to_vec(), t.shape[0], t.shape[1]),
            )
        };
        // q_proj/k_proj: transpose then un-permute (interleaved -> half-split)
        let (qs, qt) = lin(p[base]);
        named.push((
            format!("{pfx}.self_attn.q_proj.weight"),
            qs,
            rope_permute_rows(&qt, nh, hd, d, false),
        ));
        let (ks, kt) = lin(p[base + 1]);
        named.push((
            format!("{pfx}.self_attn.k_proj.weight"),
            ks,
            rope_permute_rows(&kt, kvh, hd, d, false),
        ));
        let (vs, vt) = lin(p[base + 2]);
        named.push((format!("{pfx}.self_attn.v_proj.weight"), vs, vt));
        let (os, ot) = lin(p[base + 3]);
        named.push((format!("{pfx}.self_attn.o_proj.weight"), os, ot));
        // p[base + 4] = qk_norm: not present in HF, dropped
        let (gs, gt) = lin(p[base + 5]);
        named.push((format!("{pfx}.mlp.gate_proj.weight"), gs, gt));
        let (ds, dt) = lin(p[base + 6]);
        named.push((format!("{pfx}.mlp.down_proj.weight"), ds, dt));
        let (us, ut) = lin(p[base + 7]);
        named.push((format!("{pfx}.mlp.up_proj.weight"), us, ut));
        named.push((
            format!("{pfx}.input_layernorm.weight"),
            p[base + 8].shape.clone(),
            p[base + 8].to_vec(),
        ));
        named.push((
            format!("{pfx}.post_attention_layernorm.weight"),
            p[base + 9].shape.clone(),
            p[base + 9].to_vec(),
        ));
    }
    write_named(path, &named)?;
    eprintln!(
        "HF-layout safetensors exported: {} ({} layers)",
        path, c.n_layers
    );
    Ok(())
}

/// Import HF-Llama safetensors as an Smedjan model init (build from `config`, map every weight).
/// Parse a HuggingFace Llama-style `config.json` into a Smedjan `ModelConfig` (dense Softmax-attn
/// Llama shape). Maps `hidden_size`, `num_attention_heads`, `num_key_value_heads` (MHA if absent),
/// `num_hidden_layers`, `vocab_size`, `max_position_embeddings`, `rope_theta`, and `rms_norm_eps`,
/// and derives `ffn_multiplier` from `intermediate_size`. Errors if the 256-aligned FFN hidden dim
/// cannot reproduce `intermediate_size` exactly (Smedjan rounds FFN hidden up to a multiple of 256).
///
/// This builds a model for the SubQ-style retrofit / continued-training flow. Faithful HF *inference*
/// parity (HF half-split vs Smedjan interleaved RoPE, the fixed QK-norm at d_model>=512) is adapted
/// away by continued training, not reproduced bit-for-bit — see the interop note above.
pub fn config_from_hf_json(path: &str) -> std::io::Result<ModelConfig> {
    let bytes = std::fs::read(path)?;
    let json = parse_header(&bytes)?;
    let req_u = |k: &str| -> std::io::Result<usize> {
        json.get(k)
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| invalid(format!("config.json: missing/invalid integer '{k}'")))
    };
    let opt_u = |k: &str, dflt: usize| -> usize {
        json.get(k)
            .and_then(|v| v.as_u64())
            .map_or(dflt, |n| n as usize)
    };
    let opt_f = |k: &str, dflt: f32| -> f32 {
        json.get(k)
            .and_then(|v| v.as_f64())
            .map_or(dflt, |n| n as f32)
    };

    if let Some(mt) = json.get("model_type").and_then(|v| v.as_str()) {
        if mt != "llama" && mt != "mistral" {
            eprintln!(
                "warning: config.json model_type='{mt}' (expected llama/mistral); mapping the standard \
                 Llama-arch fields — non-Llama architectures may not import faithfully."
            );
        }
    }

    let hidden = req_u("hidden_size")?;
    let n_heads = req_u("num_attention_heads")?;
    if hidden == 0 || n_heads == 0 {
        return Err(invalid(
            "config.json: hidden_size and num_attention_heads must be > 0",
        ));
    }
    let n_kv_heads = opt_u("num_key_value_heads", n_heads); // MHA when absent
    let n_layers = req_u("num_hidden_layers")?;
    let inter = req_u("intermediate_size")?;
    let vocab = req_u("vocab_size")? as u32;
    let max_seq = opt_u("max_position_embeddings", 2048);
    let ffn_multiplier = inter as f32 / hidden as f32;

    let mut config = ModelConfig::custom_gqa(
        vocab,
        hidden,
        n_heads,
        n_kv_heads,
        n_layers,
        ffn_multiplier,
        max_seq,
    );
    config.rope_theta = opt_f("rope_theta", 10000.0);
    config.norm_eps = opt_f("rms_norm_eps", 1e-5);

    let dff = config.d_ff();
    if dff != inter {
        return Err(invalid(format!(
            "config.json intermediate_size={inter} is not representable: Smedjan aligns FFN hidden to \
             a multiple of 256 (would build {dff}). Only multiples of 256 are supported."
        )));
    }
    Ok(config)
}

pub fn import_hf_safetensors(
    ctx: &Arc<MetalContext>,
    path: &str,
    config: ModelConfig,
) -> std::io::Result<Transformer> {
    let (json, blob) = read_parts(path)?;

    let c = config.clone();
    let (d, nh, kvh) = (c.d_model, c.n_heads, c.n_kv_heads);
    let hd = d / nh;
    let model = Transformer::new(ctx, config);
    let p = model.parameters();
    ensure_standard(&c, p.len())?;

    let fetch = |name: &str, expect_shape: &[usize]| -> std::io::Result<Vec<f32>> {
        let e = json
            .get(name)
            .ok_or_else(|| invalid(format!("HF safetensors: missing {name}")))?;
        let dtype = e
            .get("dtype")
            .and_then(|x| x.as_str())
            .ok_or_else(|| invalid(format!("{name}: no dtype")))?;
        let elem = dtype_elem_bytes(dtype).ok_or_else(|| {
            invalid(format!(
                "{name}: dtype {dtype} unsupported (only F32, BF16, F16)"
            ))
        })?;
        let shape = shape_field(e, name)?;
        if shape != expect_shape {
            return Err(invalid(format!(
                "{name}: HF shape {:?} != expected {:?}",
                shape, expect_shape
            )));
        }
        let numel = shape_numel(expect_shape, name)?;
        let (s, en) = offsets_field(e, name)?;
        let expect_bytes = numel
            .checked_mul(elem)
            .ok_or_else(|| invalid(format!("{name}: tensor byte length overflows usize")))?;
        if s > en || en > blob.len() || en - s != expect_bytes {
            return Err(invalid(format!(
                "{name}: byte range [{s},{en}) invalid (expected {expect_bytes} bytes, blob {})",
                blob.len()
            )));
        }
        decode_to_f32(&blob[s..en], dtype, numel, name)
    };
    let put = |t: &crate::tensor::Tensor, data: &[f32]| -> std::io::Result<()> {
        if data.len() != t.numel() {
            return Err(invalid(format!(
                "tensor size mismatch: got {} f32, model wants {}",
                data.len(),
                t.numel()
            )));
        }
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        crate::gpu::buf_write_bytes(&t.buffer, &bytes);
        Ok(())
    };

    put(p[0], &fetch("model.embed_tokens.weight", &p[0].shape)?)?;
    put(p[1], &fetch("model.norm.weight", &p[1].shape)?)?;
    for b in 0..c.n_layers {
        let base = 2 + b * HF_BLOCK_PARAMS;
        let pfx = format!("model.layers.{b}");
        let q_shape = vec![nh * hd, d];
        let k_shape = vec![kvh * hd, d];
        let v_shape = vec![kvh * hd, d];
        let o_shape = vec![d, nh * hd];
        let dff = p[base + 5].shape[1];
        let up_shape = vec![dff, d];
        let down_shape = vec![d, dff];
        let q = rope_permute_rows(
            &fetch(&format!("{pfx}.self_attn.q_proj.weight"), &q_shape)?,
            nh,
            hd,
            d,
            true,
        );
        put(p[base], &transpose_2d(&q, nh * hd, d))?;
        let k = rope_permute_rows(
            &fetch(&format!("{pfx}.self_attn.k_proj.weight"), &k_shape)?,
            kvh,
            hd,
            d,
            true,
        );
        put(p[base + 1], &transpose_2d(&k, kvh * hd, d))?;
        put(
            p[base + 2],
            &transpose_2d(
                &fetch(&format!("{pfx}.self_attn.v_proj.weight"), &v_shape)?,
                kvh * hd,
                d,
            ),
        )?;
        put(
            p[base + 3],
            &transpose_2d(
                &fetch(&format!("{pfx}.self_attn.o_proj.weight"), &o_shape)?,
                d,
                nh * hd,
            ),
        )?;
        // p[base + 4] = qk_norm: left at the Smedjan default (HF has none)
        put(
            p[base + 5],
            &transpose_2d(
                &fetch(&format!("{pfx}.mlp.gate_proj.weight"), &up_shape)?,
                dff,
                d,
            ),
        )?;
        put(
            p[base + 6],
            &transpose_2d(
                &fetch(&format!("{pfx}.mlp.down_proj.weight"), &down_shape)?,
                d,
                dff,
            ),
        )?;
        put(
            p[base + 7],
            &transpose_2d(
                &fetch(&format!("{pfx}.mlp.up_proj.weight"), &up_shape)?,
                dff,
                d,
            ),
        )?;
        put(
            p[base + 8],
            &fetch(&format!("{pfx}.input_layernorm.weight"), &p[base + 8].shape)?,
        )?;
        put(
            p[base + 9],
            &fetch(
                &format!("{pfx}.post_attention_layernorm.weight"),
                &p[base + 9].shape,
            )?,
        )?;
    }
    eprintln!(
        "HF safetensors imported as Smedjan init: {} layers from {}",
        c.n_layers, path
    );
    Ok(model)
}

#[cfg(test)]
mod dtype_tests {
    use super::{bf16_to_f32, decode_to_f32, f16_to_f32};

    #[test]
    fn bf16_widens_exactly() {
        // bf16 holds the high 16 bits of an f32, so any value whose low 16 mantissa bits are
        // zero round-trips exactly. 1.0=0x3F80, -2.0=0xC000, 0.0=0x0000.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x4049), f32::from_bits(0x4049u32 << 16)); // ~3.14
                                                                          // Range: bf16 keeps f32's 8-bit exponent, so values far above the f16 max survive.
        let big = bf16_to_f32(0x7149); // ~1e30
        assert!(big > 1e29, "bf16 must preserve large exponent: {big}");
    }

    #[test]
    fn f16_widens_exactly() {
        // IEEE half: 1.0=0x3C00, -2.0=0xC000, 0.5=0x3800, 0.0=0x0000.
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        // Largest normal half = 65504.
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        // Smallest positive subnormal half = 2^-24.
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-12);
        // Inf / NaN.
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn decode_dispatches_and_validates_length() {
        // F32 passthrough.
        let f32_bytes: Vec<u8> = [1.0f32, -2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(
            decode_to_f32(&f32_bytes, "F32", 2, "t").unwrap(),
            vec![1.0, -2.0]
        );
        // BF16 widening.
        let bf16_bytes: Vec<u8> = [0x3F80u16, 0xC000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        assert_eq!(
            decode_to_f32(&bf16_bytes, "BF16", 2, "t").unwrap(),
            vec![1.0, -2.0]
        );
        // F16 widening.
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x3800]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        assert_eq!(
            decode_to_f32(&f16_bytes, "F16", 2, "t").unwrap(),
            vec![1.0, 0.5]
        );
        // Wrong byte length is rejected (2 elems of bf16 = 4 bytes, not 8).
        assert!(decode_to_f32(&[0u8; 8], "BF16", 2, "t").is_err());
        // Unsupported dtype is rejected.
        assert!(decode_to_f32(&[0u8; 16], "F64", 2, "t").is_err());
    }
}
