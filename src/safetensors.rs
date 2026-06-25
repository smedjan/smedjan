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

/// Parse a Qwen3.5 / Qwen3-Next `config.json` (`model_type` "qwen3_5") into a `Qwen35Config`,
/// reading the nested `text_config` incl. the `layer_types` hybrid topology. Vision tower ignored.
pub fn config_from_hf_qwen35(path: &str) -> std::io::Result<crate::gated_deltanet::Qwen35Config> {
    let bytes = std::fs::read(path)?;
    let json = parse_header(&bytes)?;
    let tc = json
        .get("text_config")
        .ok_or_else(|| invalid("config.json: missing text_config"))?;
    let req_u = |k: &str| -> std::io::Result<usize> {
        tc.get(k)
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .ok_or_else(|| invalid(format!("text_config: missing/invalid integer '{k}'")))
    };
    let opt_f =
        |k: &str, d: f32| -> f32 { tc.get(k).and_then(|v| v.as_f64()).map_or(d, |n| n as f32) };
    let rope_theta = tc
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .map_or(10_000.0, |n| n as f32);
    let layer_types = tc
        .get("layer_types")
        .and_then(|v| v.as_arr())
        .ok_or_else(|| invalid("text_config: missing layer_types array"))?;
    let is_full_attention: Vec<bool> = layer_types
        .iter()
        .map(|lt| lt.as_str() == Some("full_attention"))
        .collect();
    Ok(crate::gated_deltanet::Qwen35Config {
        hidden_size: req_u("hidden_size")?,
        num_hidden_layers: req_u("num_hidden_layers")?,
        head_dim: req_u("head_dim")?,
        num_attention_heads: req_u("num_attention_heads")?,
        num_key_value_heads: req_u("num_key_value_heads")?,
        intermediate_size: req_u("intermediate_size")?,
        vocab_size: req_u("vocab_size")? as u32,
        rms_norm_eps: opt_f("rms_norm_eps", 1e-6),
        rope_theta,
        partial_rotary_factor: opt_f("partial_rotary_factor", 1.0),
        linear_num_key_heads: req_u("linear_num_key_heads")?,
        linear_num_value_heads: req_u("linear_num_value_heads")?,
        linear_key_head_dim: req_u("linear_key_head_dim")?,
        linear_value_head_dim: req_u("linear_value_head_dim")?,
        linear_conv_kernel_dim: req_u("linear_conv_kernel_dim")?,
        full_attention_interval: req_u("full_attention_interval")?,
        is_full_attention,
        strict_qwen35: false,
    })
}

/// MLX-style affine int4 (group-size 64) packed-weight dequantization.
///
/// Storage layout (per the Qwythos-9B-Claude-Mythos-5-1M Q4 artifact):
///   - `weight`: `U32` tensor of shape `[out, in_packed]` where `in_packed = in / 8` — each `U32`
///     packs 8 signed int4 nibbles, **little-endian nibble order** (nibble 0 = bits 0..3, etc.).
///   - `scales`: `BF16` shape `[out, in/64]` — one scale per group of 64 input features.
///   - `biases`: `BF16` same shape as `scales`.
///
/// Dequantization: `w[o, i] = nibble(o, i) · scales[o, i/64] + biases[o, i/64]`, where `nibble` is
/// sign-extended 4-bit (range `[-8, 7]`). `out` and `in` are the logical (dequantized) dims.
///
/// Returns row-major `[out * in]` f32. `name` is for diagnostics.
pub fn dequant_int4_affine(
    weight_u32: &[u8],
    scales_bf16: &[u8],
    biases_bf16: &[u8],
    out: usize,
    inp: usize,
    group_size: usize,
    name: &str,
) -> std::io::Result<Vec<f32>> {
    let in_packed = inp / 8;
    if in_packed * 8 != inp {
        return Err(invalid(format!(
            "{name}: in_dim {inp} not divisible by 8 (int4 packing)"
        )));
    }
    let n_groups = inp / group_size;
    if n_groups * group_size != inp {
        return Err(invalid(format!(
            "{name}: in_dim {inp} not divisible by group_size {group_size}"
        )));
    }
    if weight_u32.len() != out * in_packed * 4 {
        return Err(invalid(format!(
            "{name}: weight bytes {} != {} (out {out} × in_packed {in_packed} × 4)",
            weight_u32.len(),
            out * in_packed * 4
        )));
    }
    if scales_bf16.len() != out * n_groups * 2 {
        return Err(invalid(format!(
            "{name}: scales bytes {} != {} (out {out} × n_groups {n_groups} × 2)",
            scales_bf16.len(),
            out * n_groups * 2
        )));
    }
    if biases_bf16.len() != out * n_groups * 2 {
        return Err(invalid(format!(
            "{name}: biases bytes {} != {} (out {out} × n_groups {n_groups} × 2)",
            biases_bf16.len(),
            out * n_groups * 2
        )));
    }
    let mut dst = vec![0.0f32; out * inp];
    for o in 0..out {
        let w_row_off = o * in_packed * 4;
        let s_row_off = o * n_groups * 2;
        for ig in 0..n_groups {
            let scale = bf16_to_f32(u16::from_le_bytes([
                scales_bf16[s_row_off + ig * 2],
                scales_bf16[s_row_off + ig * 2 + 1],
            ]));
            let bias = bf16_to_f32(u16::from_le_bytes([
                biases_bf16[s_row_off + ig * 2],
                biases_bf16[s_row_off + ig * 2 + 1],
            ]));
            for j in 0..group_size {
                let i = ig * group_size + j;
                let pack_idx = i / 8;
                let nibble_idx = i % 8;
                let u = u32::from_le_bytes([
                    weight_u32[w_row_off + pack_idx * 4],
                    weight_u32[w_row_off + pack_idx * 4 + 1],
                    weight_u32[w_row_off + pack_idx * 4 + 2],
                    weight_u32[w_row_off + pack_idx * 4 + 3],
                ]);
                let raw = ((u >> (nibble_idx * 4)) & 0xF) as i8;
                let nibble = if raw & 0x8 != 0 { raw - 16 } else { raw };
                dst[o * inp + i] = nibble as f32 * scale + bias;
            }
        }
    }
    Ok(dst)
}

/// Fetch one triplet `{stem}.weight / {stem}.scales / {stem}.biases` and dequantize to `Vec<f32>`
/// (row-major `[out, in]` → smedjan convention `[in, out]` happens in the caller, not here).
/// `expect_out`/`expect_in` are the logical (dequantized) dims; group_size is 64 for this artifact.
fn fetch_q4(
    json: &Json,
    blob: &[u8],
    stem: &str,
    expect_out: usize,
    expect_in: usize,
    group_size: usize,
) -> std::io::Result<Vec<f32>> {
    let entry = |suffix: &str| -> std::io::Result<(&Json, &[u8])> {
        let name = format!("{stem}.{suffix}");
        let e = json
            .get(&name)
            .ok_or_else(|| invalid(format!("Q4 fetch: missing {name}")))?;
        let shape = shape_field(e, &name)?;
        let (s, en) = offsets_field(e, &name)?;
        if s > en || en > blob.len() {
            return Err(invalid(format!(
                "{name}: bad offsets [{s},{en}) in blob {}",
                blob.len()
            )));
        }
        Ok((e, &blob[s..en]))
    };
    let (we, wbytes) = entry("weight")?;
    let (se, sbytes) = entry("scales")?;
    let (be, bbytes) = entry("biases")?;
    // Validate the stored shapes match expectations for this artifact.
    let w_shape = shape_field(we, stem)?;
    if w_shape[0] != expect_out || w_shape[1] * 8 != expect_in {
        return Err(invalid(format!(
            "{stem}.weight shape {w_shape:?} != expected [{expect_out}, {}]",
            expect_in / 8
        )));
    }
    let _ = shape_field(se, stem)?;
    let _ = shape_field(be, stem)?;
    dequant_int4_affine(
        wbytes, sbytes, bbytes, expect_out, expect_in, group_size, stem,
    )
}

/// Decode a plain (non-quantized) BF16/F32 tensor by name → `Vec<f32>`.
fn fetch_plain(json: &Json, blob: &[u8], name: &str, numel: usize) -> std::io::Result<Vec<f32>> {
    let e = json
        .get(name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    let dtype = e
        .get("dtype")
        .and_then(|x| x.as_str())
        .ok_or_else(|| invalid(format!("{name}: no dtype")))?;
    let shape = shape_field(e, name)?;
    let stored_numel = shape_numel(&shape, name)?;
    if stored_numel != numel {
        return Err(invalid(format!(
            "{name}: numel {stored_numel} != expected {numel}"
        )));
    }
    let (s, en) = offsets_field(e, name)?;
    if s > en || en > blob.len() {
        return Err(invalid(format!("{name}: bad offsets")));
    }
    decode_to_f32(&blob[s..en], dtype, numel, name)
}

/// Put `data` (row-major f32) into a model-resident `Tensor`, transposing `[out, in] → [in, out]`
/// (HF stores weights as `[out, in]`; smedjan `matmul` expects `[in, out]`).
fn put_transposed(
    t: &crate::tensor::Tensor,
    data: &[f32],
    out: usize,
    inp: usize,
) -> std::io::Result<()> {
    if data.len() != out * inp {
        return Err(invalid(format!(
            "put_transposed: {} elements, expected {out}×{inp} = {}",
            data.len(),
            out * inp
        )));
    }
    let mut t_data = vec![0.0f32; out * inp];
    for o in 0..out {
        for i in 0..inp {
            t_data[i * out + o] = data[o * inp + i];
        }
    }
    if t_data.len() != t.numel() {
        return Err(invalid(format!(
            "put_transposed: target tensor numel {} != transposed {}",
            t.numel(),
            t_data.len()
        )));
    }
    let bytes: Vec<u8> = t_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    crate::gpu::buf_write_bytes(&t.buffer, &bytes);
    Ok(())
}

/// Load an MLX-affine-int4 Qwen3.5 / Qwen3-Next safetensors file (single-shard `.safetensors`)
/// into a freshly-allocated `Qwen35Model` (smedjan).
///
/// Maps the 927 quantized tensors (names prefixed `language_model.`) onto the hybrid topology:
///   - 24 Gated-DeltaNet layers (combined `in_proj_qkv` split into q/k/v, `in_proj_z` → z_gate,
///     `in_proj_a/b` → gate/beta pre-acts, `A_log`+`dt_bias`, conv1d, out-norm, out-proj).
///   - 8 full-attention layers (`q_proj` doubled → q + output-gate, `q_norm`+`k_norm` separate).
///   - Shared `embed_tokens`, `model.norm`, `lm_head`, per-layer `input_layernorm` /
///     `post_attention_layernorm` / MLP `gate_proj`/`up_proj`/`down_proj`.
///
/// The forward stays on the placeholder activations (the model's existing behaviour); flipping
/// `cfg.strict_qwen35 = true` (caller's choice) is what wires `A_log`/`dt_bias`/`RMSNormGated`.
pub fn import_qwen35_safetensors(
    ctx: &Arc<MetalContext>,
    path: &str,
    cfg: crate::gated_deltanet::Qwen35Config,
    group_size: usize,
) -> std::io::Result<crate::gated_deltanet::Qwen35Model> {
    use crate::gated_deltanet::{Mixer, OwnedDelta, OwnedFull, Qwen35Layer, Qwen35Model};
    let (json, blob) = read_parts(path)?;
    let d = cfg.hidden_size;
    let _eps = cfg.rms_norm_eps;

    let mk_w = |rows: usize, cols: usize| -> crate::tensor::Tensor {
        crate::tensor::Tensor::zeros(ctx, vec![rows, cols])
    };
    let mk_v = |n: usize| -> crate::tensor::Tensor { crate::tensor::Tensor::zeros(ctx, vec![n]) };

    // Embeddings [vocab, d]. Stored quantized; dequant once into a model-resident tensor.
    let embed_data = fetch_q4(
        &json,
        &blob,
        "language_model.model.embed_tokens",
        cfg.vocab_size as usize,
        d,
        group_size,
    )?;
    let embed = crate::tensor::Tensor::zeros(ctx, vec![cfg.vocab_size as usize, d]);
    let bytes: Vec<u8> = embed_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    crate::gpu::buf_write_bytes(&embed.buffer, &bytes);

    // Final norm [d] (BF16, plain).
    let final_norm = crate::tensor::Tensor::zeros(ctx, vec![d]);
    let fn_data = fetch_plain(&json, &blob, "language_model.model.norm.weight", d)?;
    let bytes: Vec<u8> = fn_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    crate::gpu::buf_write_bytes(&final_norm.buffer, &bytes);

    // lm_head [d, vocab] — stored quantized; we keep it row-major [d, vocab].
    let lm_raw = fetch_q4(
        &json,
        &blob,
        "language_model.lm_head",
        cfg.vocab_size as usize,
        d,
        group_size,
    )?;
    // HF stores [out=vocab, in=d]; smedjan matmul wants [in=d, out=vocab] → transpose.
    let lm_head = mk_w(d, cfg.vocab_size as usize);
    put_transposed(&lm_head, &lm_raw, cfg.vocab_size as usize, d)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for b in 0..cfg.num_hidden_layers {
        let pfx = format!("language_model.model.layers.{b}");
        let ln1 = mk_v(d);
        let ln1_data = fetch_plain(&json, &blob, &format!("{pfx}.input_layernorm.weight"), d)?;
        let bytes: Vec<u8> = ln1_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        crate::gpu::buf_write_bytes(&ln1.buffer, &bytes);
        let ln2 = mk_v(d);
        let ln2_data = fetch_plain(
            &json,
            &blob,
            &format!("{pfx}.post_attention_layernorm.weight"),
            d,
        )?;
        let bytes: Vec<u8> = ln2_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        crate::gpu::buf_write_bytes(&ln2.buffer, &bytes);

        // MLP — gate/up/down are all quantized. Smedjan stores FFN as [d, inter]/[d, inter]/[inter, d]
        // (matmul convention); HF stores [inter, d]/[inter, d]/[d, inter].
        let inter = cfg.intermediate_size;
        let gate_raw = fetch_q4(
            &json,
            &blob,
            &format!("{pfx}.mlp.gate_proj"),
            inter,
            d,
            group_size,
        )?;
        let up_raw = fetch_q4(
            &json,
            &blob,
            &format!("{pfx}.mlp.up_proj"),
            inter,
            d,
            group_size,
        )?;
        let down_raw = fetch_q4(
            &json,
            &blob,
            &format!("{pfx}.mlp.down_proj"),
            d,
            inter,
            group_size,
        )?;
        let ffn_gate = mk_w(d, inter);
        put_transposed(&ffn_gate, &gate_raw, inter, d)?;
        let ffn_up = mk_w(d, inter);
        put_transposed(&ffn_up, &up_raw, inter, d)?;
        let ffn_down = mk_w(inter, d);
        put_transposed(&ffn_down, &down_raw, d, inter)?;

        let is_full = cfg.is_full_attention.get(b).copied().unwrap_or(false);
        let mixer = if is_full {
            let n_h = cfg.num_attention_heads;
            let n_kv = cfg.num_key_value_heads;
            let hd = cfg.head_dim;
            // q_proj is doubled: [n_h*hd*2, d] → split into w_q [n_h*hd, d] and gate [n_h*hd, d].
            let q_full = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.self_attn.q_proj"),
                n_h * hd * 2,
                d,
                group_size,
            )?;
            let (q_half, gate_half) = q_full.split_at(n_h * hd * d);
            let w_q = mk_w(d, n_h * hd);
            put_transposed(&w_q, q_half, n_h * hd, d)?;
            let w_gate = mk_w(d, n_h * hd);
            put_transposed(&w_gate, gate_half, n_h * hd, d)?;
            let k_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.self_attn.k_proj"),
                n_kv * hd,
                d,
                group_size,
            )?;
            let w_k = mk_w(d, n_kv * hd);
            put_transposed(&w_k, &k_raw, n_kv * hd, d)?;
            let v_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.self_attn.v_proj"),
                n_kv * hd,
                d,
                group_size,
            )?;
            let w_v = mk_w(d, n_kv * hd);
            put_transposed(&w_v, &v_raw, n_kv * hd, d)?;
            let o_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.self_attn.o_proj"),
                d,
                n_h * hd,
                group_size,
            )?;
            let w_o = mk_w(n_h * hd, d);
            put_transposed(&w_o, &o_raw, d, n_h * hd)?;
            let qk_norm = mk_v(hd);
            let qn = fetch_plain(&json, &blob, &format!("{pfx}.self_attn.q_norm.weight"), hd)?;
            let bytes: Vec<u8> = qn.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&qk_norm.buffer, &bytes);
            let k_norm = mk_v(hd);
            let kn = fetch_plain(&json, &blob, &format!("{pfx}.self_attn.k_norm.weight"), hd)?;
            let bytes: Vec<u8> = kn.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&k_norm.buffer, &bytes);
            // Store the doubled q_proj for the strict path; placeholder path uses w_q.
            let q_proj_out = {
                let t = mk_w(d, n_h * hd * 2);
                put_transposed(&t, &q_full, n_h * hd * 2, d)?;
                Some(t)
            };
            Mixer::Full(OwnedFull {
                w_q,
                w_k,
                w_v,
                qk_norm,
                w_gate,
                w_o,
                k_norm,
                q_proj_out,
            })
        } else {
            let n_k = cfg.linear_num_key_heads;
            let n_v = cfg.linear_num_value_heads;
            let ldh = cfg.linear_key_head_dim;
            let lvh = cfg.linear_value_head_dim;
            let kw = cfg.linear_conv_kernel_dim;
            let qkv_out = 2 * n_k * ldh + n_v * lvh; // q+k+v combined
            let qkv_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.linear_attn.in_proj_qkv"),
                qkv_out,
                d,
                group_size,
            )?;
            let q_len = n_k * ldh;
            let k_len = n_k * ldh;
            let v_len = n_v * lvh;
            let q_raw = &qkv_raw[..q_len * d];
            let k_raw = &qkv_raw[q_len * d..(q_len + k_len) * d];
            let v_raw = &qkv_raw[(q_len + k_len) * d..(q_len + k_len + v_len) * d];
            let w_q = mk_w(d, q_len);
            put_transposed(&w_q, q_raw, q_len, d)?;
            let w_k = mk_w(d, k_len);
            put_transposed(&w_k, k_raw, k_len, d)?;
            let w_v = mk_w(d, v_len);
            put_transposed(&w_v, v_raw, v_len, d)?;
            // conv1d: stored as BF16 [conv_dim, kw, 1] (MLX layout) = row-major [conv_dim, kw].
            // smedjan's placeholder forward takes separate conv_q/conv_k/conv_v each [n_* * ldh, kw];
            // split the combined conv raw data on CPU (slice_cols works on dim 1, but the split axis
            // here is dim 0 = conv_dim, so we slice the f32 buffer directly before upload).
            let conv_dim = 2 * n_k * ldh + n_v * lvh;
            let conv_raw = fetch_plain(
                &json,
                &blob,
                &format!("{pfx}.linear_attn.conv1d.weight"),
                conv_dim * kw,
            )?;
            let q_chans = n_k * ldh;
            let k_chans = n_k * ldh;
            let v_chans = n_v * lvh;
            let conv_q = mk_w(q_chans, kw);
            let conv_q_data: Vec<f32> = conv_raw[..q_chans * kw].to_vec();
            let bytes: Vec<u8> = conv_q_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&conv_q.buffer, &bytes);
            let conv_k = mk_w(k_chans, kw);
            let conv_k_data: Vec<f32> = conv_raw[q_chans * kw..(q_chans + k_chans) * kw].to_vec();
            let bytes: Vec<u8> = conv_k_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&conv_k.buffer, &bytes);
            let conv_v = mk_w(v_chans, kw);
            let conv_v_data: Vec<f32> = conv_raw[(q_chans + k_chans) * kw..].to_vec();
            let bytes: Vec<u8> = conv_v_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&conv_v.buffer, &bytes);
            // in_proj_a/b → gate/beta pre-activations ([n_v] each, quantized).
            let a_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.linear_attn.in_proj_a"),
                n_v,
                d,
                group_size,
            )?;
            let w_a = mk_w(d, n_v);
            put_transposed(&w_a, &a_raw, n_v, d)?;
            let b_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.linear_attn.in_proj_b"),
                n_v,
                d,
                group_size,
            )?;
            let w_b = mk_w(d, n_v);
            put_transposed(&w_b, &b_raw, n_v, d)?;
            // in_proj_z → z_gate (the output-gate projection).
            let z_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.linear_attn.in_proj_z"),
                n_v * lvh,
                d,
                group_size,
            )?;
            let z_gate = mk_w(d, n_v * lvh);
            put_transposed(&z_gate, &z_raw, n_v * lvh, d)?;
            // w_gate (placeholder path) ← z_gate copy; they hold the same data semantically.
            let w_gate = {
                let t = mk_w(d, n_v * lvh);
                let bytes: Vec<u8> = z_raw.iter().flat_map(|f| f.to_le_bytes()).collect();
                // Transpose z_raw [out=n_v*lvh, in=d] → [d, n_v*lvh] into w_gate.
                put_transposed(&t, &z_raw, n_v * lvh, d)?;
                t
            };
            // out_norm [lvh] (BF16, plain).
            let out_norm = mk_v(lvh);
            let on = fetch_plain(&json, &blob, &format!("{pfx}.linear_attn.norm.weight"), lvh)?;
            let bytes: Vec<u8> = on.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&out_norm.buffer, &bytes);
            let out_proj_out = n_v * lvh;
            let o_raw = fetch_q4(
                &json,
                &blob,
                &format!("{pfx}.linear_attn.out_proj"),
                out_proj_out,
                d,
                group_size,
            )?;
            let w_o = mk_w(out_proj_out, d);
            put_transposed(&w_o, &o_raw, out_proj_out, d)?;
            // A_log, dt_bias [n_v] (BF16).
            let a_log = mk_v(n_v);
            let al = fetch_plain(&json, &blob, &format!("{pfx}.linear_attn.A_log"), n_v)?;
            let bytes: Vec<u8> = al.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&a_log.buffer, &bytes);
            let dt_bias = mk_v(n_v);
            let dt = fetch_plain(&json, &blob, &format!("{pfx}.linear_attn.dt_bias"), n_v)?;
            let bytes: Vec<u8> = dt.iter().flat_map(|f| f.to_le_bytes()).collect();
            crate::gpu::buf_write_bytes(&dt_bias.buffer, &bytes);
            Mixer::Delta(OwnedDelta {
                w_q,
                w_k,
                w_v,
                conv_q,
                conv_k,
                conv_v,
                w_a,
                w_b,
                w_gate,
                out_norm,
                w_o,
                a_log,
                dt_bias,
                z_gate,
            })
        };

        layers.push(Qwen35Layer {
            ln1,
            ln2,
            mixer,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    eprintln!(
        "Qwen3.5 safetensors imported: {} layers ({}/{}/{} tensors) from {}",
        cfg.num_hidden_layers, 0, 0, 0, path
    );
    Ok(Qwen35Model {
        layers,
        final_norm,
        lm_head,
        cfg,
        embed: Some(embed),
    })
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

    #[test]
    fn int4_affine_dequant_roundtrips_known_values() {
        use super::dequant_int4_affine;
        // 1 output row, 8 inputs (1 group of 8, group_size=8 here for a tight test).
        // Pack 8 nibbles into one U32: nibble i = (-1)^i * i, i.e. [0,1,-2,3,-4,5,-6,7].
        let nibbles: [i8; 8] = [0, 1, -2, 3, -4, 5, -6, 7];
        let mut packed: u32 = 0;
        for (i, n) in nibbles.iter().enumerate() {
            let raw: u32 = if *n < 0 {
                ((*n as u8) & 0xF) as u32
            } else {
                *n as u32
            };
            packed |= (raw & 0xF) << (i * 4);
        }
        let weight: Vec<u8> = packed.to_le_bytes().to_vec(); // 4 bytes
                                                             // scale=2.0, bias=0.5 (bf16). 1 group, 1 row.
        let scale_bf16 = (0x4000u16).to_le_bytes().to_vec(); // bf16 2.0
        let bias_bf16 = (0x3F00u16).to_le_bytes().to_vec(); // bf16 0.5
                                                            // group_size=8 → n_groups=1.
        let out = dequant_int4_affine(&weight, &scale_bf16, &bias_bf16, 1, 8, 8, "test").unwrap();
        let want: Vec<f32> = nibbles.iter().map(|n| *n as f32 * 2.0 + 0.5).collect();
        for (i, (g, w)) in out.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "nibble {i}: got {g} want {w}");
        }
    }

    #[test]
    fn int4_affine_rejects_bad_shapes() {
        use super::dequant_int4_affine;
        // in_dim not divisible by 8.
        assert!(dequant_int4_affine(&[], &[], &[], 1, 7, 8, "t").is_err());
        // in_dim not divisible by group_size.
        assert!(dequant_int4_affine(&[0u8; 4], &[0u8; 2], &[0u8; 2], 1, 8, 3, "t").is_err());
        // weight bytes mismatch (expected 1*1*4=4, got 8).
        assert!(dequant_int4_affine(&[0u8; 8], &[0u8; 2], &[0u8; 2], 1, 8, 8, "t").is_err());
    }

    /// End-to-end loader test: synthesize a tiny Q4 safetensors with the real Qwen3.5 tensor-name
    /// layout (1 DeltaNet + 1 full-attention layer, tiny dims), import it via
    /// `import_qwen35_safetensors`, and verify the model loads + forward produces finite logits of
    /// the right shape. Exercises every name-mapping branch + the dequant path + transpose, without
    /// needing the 5 GB real artifact.
    #[test]
    fn qwen35_loader_maps_synthetic_q4_artifact() {
        use super::*;
        use crate::gated_deltanet::Qwen35Config;
        use std::sync::Arc;

        let ctx = Arc::new(MetalContext::new());
        // Tiny config that still exercises both layer types.
        let (d, n_h, n_kv, hd, inter, vocab) = (8usize, 4, 2, 4, 16, 12);
        let (lnk, lnv, ldh, lvh, kw) = (2usize, 4, 4, 4, 2);
        let cfg = Qwen35Config {
            hidden_size: d,
            num_hidden_layers: 2,
            head_dim: hd,
            num_attention_heads: n_h,
            num_key_value_heads: n_kv,
            intermediate_size: inter,
            vocab_size: vocab as u32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            linear_num_key_heads: lnk,
            linear_num_value_heads: lnv,
            linear_key_head_dim: ldh,
            linear_value_head_dim: lvh,
            linear_conv_kernel_dim: kw,
            full_attention_interval: 2,
            is_full_attention: vec![false, true],
            strict_qwen35: false,
        };
        let group_size = 8; // tiny group so the synthetic file stays compact

        // Build entries: {name: {dtype, shape, data_offsets}}.
        let mut blob: Vec<u8> = Vec::new();
        let mut entries: Vec<String> = Vec::new();
        fn add_u32(
            blob: &mut Vec<u8>,
            entries: &mut Vec<String>,
            name: &str,
            shape: Vec<usize>,
            data: &[u8],
        ) {
            let start = blob.len();
            blob.extend_from_slice(data);
            let end = blob.len();
            let shape_s = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"U32\",\"shape\":[{shape_s}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        fn add_bf16(
            blob: &mut Vec<u8>,
            entries: &mut Vec<String>,
            name: &str,
            shape: Vec<usize>,
            data: &[u8],
        ) {
            let start = blob.len();
            blob.extend_from_slice(data);
            let end = blob.len();
            let shape_s = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{shape_s}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        // Helper: encode one row of int4-packed weights given logical [out, in] f32 values.
        fn encode_q(vals: &[f32], out: usize, inp: usize, group_size: usize) -> Vec<u8> {
            assert_eq!(vals.len(), out * inp);
            let in_packed = inp / 8;
            let mut bytes = vec![0u8; out * in_packed * 4];
            for o in 0..out {
                for ig in 0..inp / group_size {
                    for j in 0..group_size {
                        let i = ig * group_size + j;
                        let raw = (vals[o * inp + i] as i32) as i8;
                        let nibble: u8 = if raw < 0 {
                            (raw as u8) & 0xF
                        } else {
                            (raw as u8) & 0xF
                        };
                        let pack_idx = i / 8;
                        let off = (o * in_packed + pack_idx) * 4;
                        let mut u = u32::from_le_bytes([
                            bytes[off],
                            bytes[off + 1],
                            bytes[off + 2],
                            bytes[off + 3],
                        ]);
                        u |= (nibble as u32 & 0xF) << ((i % 8) * 4);
                        bytes[off..off + 4].copy_from_slice(&u.to_le_bytes());
                    }
                }
            }
            bytes
        }
        fn bf16(f: f32) -> Vec<u8> {
            let bits = (f.to_bits() >> 16) as u16;
            bits.to_le_bytes().to_vec()
        }
        fn q4_triplet(
            blob: &mut Vec<u8>,
            entries: &mut Vec<String>,
            stem: &str,
            out: usize,
            inp: usize,
            vals: &[f32],
            group_size: usize,
        ) {
            let weight = encode_q(vals, out, inp, group_size);
            let in_packed = inp / 8;
            let n_groups = inp / group_size;
            let mut scales = Vec::new();
            let mut biases = Vec::new();
            for _ in 0..(out * n_groups) {
                scales.extend(bf16(1.0));
                biases.extend(bf16(0.0));
            }
            add_u32(
                blob,
                entries,
                &format!("{stem}.weight"),
                vec![out, in_packed],
                &weight,
            );
            add_bf16(
                blob,
                entries,
                &format!("{stem}.scales"),
                vec![out, n_groups],
                &scales,
            );
            add_bf16(
                blob,
                entries,
                &format!("{stem}.biases"),
                vec![out, n_groups],
                &biases,
            );
        }
        fn bf16_vec(blob: &mut Vec<u8>, entries: &mut Vec<String>, name: &str, n: usize, val: f32) {
            let mut data = Vec::new();
            for _ in 0..n {
                data.extend(bf16(val));
            }
            add_bf16(blob, entries, name, vec![n], &data);
        }

        // Top-level.
        let embed_vals: Vec<f32> = (0..(vocab * d)).map(|i| (i % 7) as f32 - 3.0).collect();
        q4_triplet(
            &mut blob,
            &mut entries,
            "language_model.model.embed_tokens",
            vocab,
            d,
            &embed_vals,
            group_size,
        );
        bf16_vec(
            &mut blob,
            &mut entries,
            "language_model.model.norm.weight",
            d,
            1.0,
        );
        let lm_vals: Vec<f32> = (0..(vocab * d)).map(|i| (i % 5) as f32 - 2.0).collect();
        q4_triplet(
            &mut blob,
            &mut entries,
            "language_model.lm_head",
            vocab,
            d,
            &lm_vals,
            group_size,
        );

        // Per-layer.
        for b in 0..2 {
            let pfx = format!("language_model.model.layers.{b}");
            bf16_vec(
                &mut blob,
                &mut entries,
                &format!("{pfx}.input_layernorm.weight"),
                d,
                1.0,
            );
            bf16_vec(
                &mut blob,
                &mut entries,
                &format!("{pfx}.post_attention_layernorm.weight"),
                d,
                1.0,
            );
            let gate_vals: Vec<f32> = (0..(inter * d)).map(|i| (i % 3) as f32 - 1.0).collect();
            q4_triplet(
                &mut blob,
                &mut entries,
                &format!("{pfx}.mlp.gate_proj"),
                inter,
                d,
                &gate_vals,
                group_size,
            );
            let up_vals: Vec<f32> = (0..(inter * d)).map(|i| (i % 4) as f32 - 1.0).collect();
            q4_triplet(
                &mut blob,
                &mut entries,
                &format!("{pfx}.mlp.up_proj"),
                inter,
                d,
                &up_vals,
                group_size,
            );
            let down_vals: Vec<f32> = (0..(d * inter)).map(|i| (i % 5) as f32 - 2.0).collect();
            q4_triplet(
                &mut blob,
                &mut entries,
                &format!("{pfx}.mlp.down_proj"),
                d,
                inter,
                &down_vals,
                group_size,
            );

            let is_full = b == 1;
            if is_full {
                let q_full_out = n_h * hd * 2;
                let q_vals: Vec<f32> = (0..(q_full_out * d))
                    .map(|i| (i % 5) as f32 - 2.0)
                    .collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.self_attn.q_proj"),
                    q_full_out,
                    d,
                    &q_vals,
                    group_size,
                );
                let k_out = n_kv * hd;
                let k_vals: Vec<f32> = (0..(k_out * d)).map(|i| (i % 3) as f32 - 1.0).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.self_attn.k_proj"),
                    k_out,
                    d,
                    &k_vals,
                    group_size,
                );
                let v_vals: Vec<f32> = (0..(k_out * d)).map(|i| (i % 4) as f32 - 1.0).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.self_attn.v_proj"),
                    k_out,
                    d,
                    &v_vals,
                    group_size,
                );
                let o_vals: Vec<f32> = (0..(d * n_h * hd)).map(|i| (i % 6) as f32 - 3.0).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.self_attn.o_proj"),
                    d,
                    n_h * hd,
                    &o_vals,
                    group_size,
                );
                bf16_vec(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.self_attn.q_norm.weight"),
                    hd,
                    1.0,
                );
                bf16_vec(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.self_attn.k_norm.weight"),
                    hd,
                    1.0,
                );
            } else {
                let qkv_out = 2 * lnk * ldh + lnv * lvh;
                let qkv_vals: Vec<f32> = (0..(qkv_out * d)).map(|i| (i % 5) as f32 - 2.0).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.in_proj_qkv"),
                    qkv_out,
                    d,
                    &qkv_vals,
                    group_size,
                );
                let a_vals: Vec<f32> = (0..(lnv * d)).map(|i| (i % 3) as f32).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.in_proj_a"),
                    lnv,
                    d,
                    &a_vals,
                    group_size,
                );
                let b_vals: Vec<f32> = (0..(lnv * d)).map(|i| (i % 3) as f32).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.in_proj_b"),
                    lnv,
                    d,
                    &b_vals,
                    group_size,
                );
                let z_vals: Vec<f32> = (0..(lnv * lvh * d)).map(|i| (i % 4) as f32).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.in_proj_z"),
                    lnv * lvh,
                    d,
                    &z_vals,
                    group_size,
                );
                let o_vals: Vec<f32> = (0..(lnv * lvh * d)).map(|i| (i % 5) as f32 - 2.0).collect();
                q4_triplet(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.out_proj"),
                    lnv * lvh,
                    d,
                    &o_vals,
                    group_size,
                );
                let conv_dim = 2 * lnk * ldh + lnv * lvh;
                let conv_data: Vec<u8> = (0..(conv_dim * kw)).flat_map(|_| bf16(0.5)).collect();
                add_bf16(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.conv1d.weight"),
                    vec![conv_dim, kw, 1],
                    &conv_data,
                );
                bf16_vec(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.norm.weight"),
                    lvh,
                    1.0,
                );
                bf16_vec(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.A_log"),
                    lnv,
                    0.0,
                );
                bf16_vec(
                    &mut blob,
                    &mut entries,
                    &format!("{pfx}.linear_attn.dt_bias"),
                    lnv,
                    1.0,
                );
            }
        }

        // Serialize header.
        let header = format!("{{{}}}", entries.join(","));
        let path = std::env::temp_dir().join("qwen35_loader_test.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&blob).unwrap();
        drop(file);

        // Load and forward.
        let model =
            import_qwen35_safetensors(&ctx, path.to_str().unwrap(), cfg.clone(), group_size)
                .unwrap();
        assert_eq!(model.layers.len(), 2, "two layers loaded");
        assert!(model.embed.is_some(), "embed populated");
        let (batch, seq) = (1usize, 3);
        let token_ids: Vec<u32> = vec![0, 5, 11]; // within vocab=12
        let x = model.embed_tokens(&token_ids, batch, seq);
        assert_eq!(x.shape, vec![batch, seq, d], "embedded input shape");
        let logits = crate::autograd::no_grad(|| model.forward(&x));
        assert_eq!(logits.shape, vec![batch, seq, vocab], "logits shape");
        let lv = logits.to_vec();
        assert!(lv.iter().all(|v| v.is_finite()), "logits finite");
        std::fs::remove_file(&path).ok();
    }
}
