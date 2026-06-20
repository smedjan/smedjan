//! safetensors I/O for AndreAI models (zero deps; hand-written). Layout: 8-byte little-endian
//! header length, a JSON header `{ "<name>": {dtype, shape, data_offsets:[begin,end]}, ... }`, then
//! the raw tensor blob. Tensors are written in `model.parameters()` order (trainable, then ReLoRA
//! base) under positional names `p{i}`, mirroring `checkpoint::save_checkpoint`, so an AndreAI model
//! round-trips through safetensors. `import_safetensors` rebuilds the model from a caller-supplied
//! config (the same flow a foreign HF import uses, sourcing config from the model's config.json) and
//! overwrites each parameter in order. Foreign HF->AndreAI name remap + [out,in]->[in,out] transpose
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
    Num(f64),
    Bool(bool),
    Null,
}

impl Json {
    fn as_obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(e) = self { Some(e) } else { None }
    }
    fn get(&self, k: &str) -> Option<&Json> {
        self.as_obj()?.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }
    fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }
    fn as_u64(&self) -> Option<u64> {
        if let Json::Num(n) = self { Some(*n as u64) } else { None }
    }
    fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self { Some(a) } else { None }
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
            Some(b't') => { self.lit("true")?; Ok(Json::Bool(true)) }
            Some(b'f') => { self.lit("false")?; Ok(Json::Bool(false)) }
            Some(b'n') => { self.lit("null")?; Ok(Json::Null) }
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
                Some(b'}') => { self.i += 1; break; }
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
                Some(b']') => { self.i += 1; break; }
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
                    let e = *self.b.get(self.i).ok_or_else(|| invalid("safetensors header: bad escape"))?;
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
            && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Json::Num)
            .ok_or_else(|| invalid("safetensors header: bad number"))
    }
}

const ST_DTYPE: &str = "F32";

/// Export an AndreAI model to a `.safetensors` file (F32). Tensors are named `p{i}` in
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
        "\"__metadata__\":{{\"format\":\"andreai-safetensors-v1\",\"n_tensors\":\"{}\"}}}}",
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
    let mut file = std::fs::File::open(path)?;
    let mut lenb = [0u8; 8];
    file.read_exact(&mut lenb)?;
    let header_len = u64::from_le_bytes(lenb) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let mut blob = Vec::new();
    file.read_to_end(&mut blob)?;

    let json = JsonParser::new(&header).value()?;
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
        if dtype != "F32" {
            return Err(invalid(format!(
                "{name}: dtype {dtype} unsupported (only F32; bf16/f16 import is a follow-up)"
            )));
        }
        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(|s| s.as_arr())
            .ok_or_else(|| invalid(format!("{name}: no shape")))?
            .iter()
            .map(|d| d.as_u64().map(|n| n as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| invalid(format!("{name}: bad shape")))?;
        if shape != p.shape {
            return Err(invalid(format!(
                "{name}: safetensors shape {:?} != model shape {:?}",
                shape, p.shape
            )));
        }
        let offs = entry
            .get("data_offsets")
            .and_then(|o| o.as_arr())
            .ok_or_else(|| invalid(format!("{name}: no data_offsets")))?;
        let start = offs
            .first()
            .and_then(|x| x.as_u64())
            .ok_or_else(|| invalid(format!("{name}: bad data_offsets")))? as usize;
        let end = offs
            .get(1)
            .and_then(|x| x.as_u64())
            .ok_or_else(|| invalid(format!("{name}: bad data_offsets")))? as usize;
        let expect = p.numel() * 4;
        if start > end || end > blob.len() || end - start != expect {
            return Err(invalid(format!(
                "{name}: byte range [{start},{end}) invalid (expected {expect} bytes, blob {})",
                blob.len()
            )));
        }
        crate::gpu::buf_write_bytes(&p.buffer, &blob[start..end]);
    }
    eprintln!("safetensors imported: {} tensors from {}", targets.len(), path);
    Ok(model)
}
