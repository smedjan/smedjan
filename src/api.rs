use crate::checkpoint;
use crate::generate::{self, SamplingConfig};
use crate::gpu::GpuContext as MetalContext; // backend-agnostic import
use crate::model::Transformer;
use crate::tokenizer::BpeTokenizer;
use std::sync::Arc;

/// Model metadata exposed through the public API.
pub struct ModelInfo {
    pub params: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: u32,
    pub step: u32,
}

/// High-level API for using AndreAI as a library.
///
/// Encapsulates model loading, tokenization, and generation into a single
/// struct suitable for embedding into other Rust projects (e.g. AndreOS).
pub struct AndreAI {
    ctx: Arc<MetalContext>,
    model: Transformer,
    tokenizer: BpeTokenizer,
    step: u32,
}

impl AndreAI {
    /// Load a model from checkpoint and tokenizer files.
    ///
    /// The checkpoint can be either a standard `.bin` checkpoint or a
    /// quantized `.qbin` checkpoint (auto-detected by extension).
    pub fn load(checkpoint_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        let ctx = MetalContext::new();
        let tokenizer = BpeTokenizer::load(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer '{}': {}", tokenizer_path, e))?;

        let (model, step) = if checkpoint_path.ends_with(".qbin") {
            crate::quantize::load_quantized(&ctx, checkpoint_path)
                .map_err(|e| format!("Failed to load quantized checkpoint '{}': {}", checkpoint_path, e))?
        } else {
            checkpoint::load_checkpoint(&ctx, checkpoint_path)
                .map_err(|e| format!("Failed to load checkpoint '{}': {}", checkpoint_path, e))?
        };

        Ok(Self {
            ctx,
            model,
            tokenizer,
            step,
        })
    }

    /// Generate a response to a prompt using default sampling parameters.
    pub fn generate(&self, prompt: &str) -> String {
        let config = SamplingConfig::default();
        generate::generate(&self.ctx, &self.model, &self.tokenizer, prompt, &config)
    }

    /// Generate a response with custom sampling parameters.
    pub fn generate_with_config(&self, prompt: &str, config: &SamplingConfig) -> String {
        generate::generate(&self.ctx, &self.model, &self.tokenizer, prompt, config)
    }

    /// Generate streaming, calling the callback for each produced token.
    pub fn generate_streaming<F: FnMut(&str)>(&self, prompt: &str, on_token: F) {
        let config = SamplingConfig::default();
        generate::generate_streaming(
            &self.ctx,
            &self.model,
            &self.tokenizer,
            prompt,
            &config,
            on_token,
        );
    }

    /// Generate streaming with custom sampling parameters.
    pub fn generate_streaming_with_config<F: FnMut(&str)>(
        &self,
        prompt: &str,
        config: &SamplingConfig,
        on_token: F,
    ) {
        generate::generate_streaming(
            &self.ctx,
            &self.model,
            &self.tokenizer,
            prompt,
            config,
            on_token,
        );
    }

    /// Get model metadata.
    pub fn model_info(&self) -> ModelInfo {
        let c = &self.model.config;
        ModelInfo {
            params: c.param_count(),
            d_model: c.d_model,
            n_layers: c.n_layers,
            n_heads: c.n_heads,
            n_kv_heads: c.n_kv_heads,
            vocab_size: c.vocab_size,
            step: self.step,
        }
    }

    /// Get a reference to the underlying tokenizer for direct token manipulation.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }
}
