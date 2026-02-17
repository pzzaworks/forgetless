//! Token counting and management
//!
//! Provides accurate token counting for various LLM tokenizers.
//! Supports text, images, and multi-modal content.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};

/// Supported tokenizer models (February 2026)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TokenizerModel {
    // OpenAI Models
    /// GPT-5.3 Codex - OpenAI's latest self-improving model (o200k_base tokenizer)
    Gpt53Codex,
    /// GPT-5.2 - 400K context with reduced hallucination (o200k_base tokenizer)
    Gpt52,
    /// GPT-4o / GPT-4o-mini (o200k_base tokenizer)
    Gpt4o,
    /// gpt-oss-120b - Open-weight 117B MoE model (o200k_base tokenizer)
    GptOss120b,

    // Anthropic Models
    /// Claude Opus 4.6 - Latest Claude model (February 2026)
    ClaudeOpus46,
    /// Claude Sonnet 4.5
    ClaudeSonnet45,
    /// Claude Haiku 4.5
    ClaudeHaiku45,

    // Google Models
    /// Gemini 3 Pro/Flash - 1M token context
    Gemini3,
    /// Gemini 2.5 Flash Lite
    Gemini25,

    // xAI Models
    /// Grok 4.1 Fast
    Grok41,

    // DeepSeek Models
    /// DeepSeek V3.2 - Beats GPT-4 at 1/40th cost, 256K context
    DeepSeekV32,

    // Alibaba Models
    /// Qwen3-Coder-480B - MoE code generation model
    Qwen3Coder,

    // Meta Models
    /// Llama 4 - Meta's latest open model
    Llama4,

    // Mistral Models
    /// Mistral Large 2 / Medium
    MistralLarge,

    /// Custom token-per-char ratio (stored as ratio * 100 for precision)
    Custom {
        /// Average characters per token (multiplied by 100, e.g., 400 = 4.0 chars/token)
        chars_per_token_x100: u32,
    },
}

impl Default for TokenizerModel {
    fn default() -> Self {
        Self::ClaudeSonnet45
    }
}

/// Image detail level for vision models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageDetail {
    /// Low detail - fixed 85 tokens
    Low,
    /// High detail - calculated based on image size
    High,
    /// Auto - let the model decide
    Auto,
}

impl Default for ImageDetail {
    fn default() -> Self {
        Self::Auto
    }
}

/// Image dimensions for token calculation
#[derive(Debug, Clone, Copy)]
pub struct ImageDimensions {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
}

/// Token counter for accurate token budget management
pub struct TokenCounter {
    model: TokenizerModel,
    bpe: Option<CoreBPE>,
}

impl TokenCounter {
    /// Create a new token counter for the specified model
    pub fn new(model: TokenizerModel) -> Result<Self> {
        let bpe = match model {
            // OpenAI models - use o200k_base approximation via cl100k
            TokenizerModel::Gpt53Codex
            | TokenizerModel::Gpt52
            | TokenizerModel::Gpt4o
            | TokenizerModel::GptOss120b => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            // Anthropic Claude - similar tokenization to OpenAI
            TokenizerModel::ClaudeOpus46
            | TokenizerModel::ClaudeSonnet45
            | TokenizerModel::ClaudeHaiku45 => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            // Google Gemini - approximate with cl100k
            TokenizerModel::Gemini3 | TokenizerModel::Gemini25 => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            // xAI Grok
            TokenizerModel::Grok41 => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            // DeepSeek - uses similar tokenization
            TokenizerModel::DeepSeekV32 => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            // Qwen/Alibaba
            TokenizerModel::Qwen3Coder => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            // Open source models - use approximation
            TokenizerModel::Llama4 | TokenizerModel::MistralLarge => {
                let bpe = get_bpe_from_model("gpt-4")
                    .map_err(|e| Error::TokenCountError(e.to_string()))?;
                Some(bpe)
            }
            TokenizerModel::Custom { .. } => None,
        };

        Ok(Self { model, bpe })
    }

    /// Count tokens in text
    pub fn count(&self, text: &str) -> usize {
        match (&self.bpe, self.model) {
            (Some(bpe), _) => bpe.encode_with_special_tokens(text).len(),
            (None, TokenizerModel::Custom { chars_per_token_x100 }) => {
                let chars_per_token = chars_per_token_x100 as f32 / 100.0;
                (text.chars().count() as f32 / chars_per_token).ceil() as usize
            }
            _ => text.split_whitespace().count(), // Fallback
        }
    }

    /// Count tokens for an image based on dimensions and detail level
    /// Based on OpenAI's vision token calculation
    pub fn count_image(&self, dimensions: ImageDimensions, detail: ImageDetail) -> usize {
        match detail {
            ImageDetail::Low => 85, // Fixed cost for low detail
            ImageDetail::High | ImageDetail::Auto => {
                // Scale image to fit in 2048x2048
                let (w, h) = Self::scale_to_fit(dimensions.width, dimensions.height, 2048);
                // Scale shortest side to 768
                let (w, h) = Self::scale_shortest_side(w, h, 768);
                // Calculate number of 512x512 tiles
                let tiles_x = (w as f32 / 512.0).ceil() as usize;
                let tiles_y = (h as f32 / 512.0).ceil() as usize;
                let num_tiles = tiles_x * tiles_y;
                // 170 tokens per tile + 85 base
                170 * num_tiles + 85
            }
        }
    }

    /// Count tokens for multiple images
    pub fn count_images(&self, images: &[(ImageDimensions, ImageDetail)]) -> usize {
        images.iter().map(|(dim, detail)| self.count_image(*dim, *detail)).sum()
    }

    /// Count tokens for multiple texts
    pub fn count_many(&self, texts: &[&str]) -> usize {
        texts.iter().map(|t| self.count(t)).sum()
    }

    fn scale_to_fit(width: u32, height: u32, max_size: u32) -> (u32, u32) {
        if width <= max_size && height <= max_size {
            return (width, height);
        }
        let ratio = (max_size as f32) / (width.max(height) as f32);
        ((width as f32 * ratio) as u32, (height as f32 * ratio) as u32)
    }

    fn scale_shortest_side(width: u32, height: u32, target: u32) -> (u32, u32) {
        let shortest = width.min(height);
        if shortest <= target {
            return (width, height);
        }
        let ratio = target as f32 / shortest as f32;
        ((width as f32 * ratio) as u32, (height as f32 * ratio) as u32)
    }

    /// Estimate if text fits within token budget
    pub fn fits_budget(&self, text: &str, budget: usize) -> bool {
        self.count(text) <= budget
    }

    /// Truncate text to fit within token budget
    pub fn truncate_to_budget(&self, text: &str, budget: usize) -> String {
        if let Some(bpe) = &self.bpe {
            let tokens = bpe.encode_with_special_tokens(text);
            if tokens.len() <= budget {
                return text.to_string();
            }
            let truncated_tokens: Vec<_> = tokens.into_iter().take(budget).collect();
            bpe.decode(truncated_tokens).unwrap_or_default()
        } else {
            // Simple char-based truncation for custom tokenizers
            let chars_per_token = match self.model {
                TokenizerModel::Custom { chars_per_token_x100 } => chars_per_token_x100 as f32 / 100.0,
                _ => 4.0,
            };
            let max_chars = (budget as f32 * chars_per_token) as usize;
            text.chars().take(max_chars).collect()
        }
    }

    /// Get the tokenizer model
    pub fn model(&self) -> TokenizerModel {
        self.model
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::new(TokenizerModel::default()).expect("Default tokenizer should always work")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counting() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let text = "Hello, world!";
        let count = counter.count(text);
        assert!(count > 0);
        assert!(count < text.len());
    }

    #[test]
    fn test_tokenizer_model_default() {
        let model = TokenizerModel::default();
        assert_eq!(model, TokenizerModel::ClaudeSonnet45);
    }

    #[test]
    fn test_token_counter_default() {
        let counter = TokenCounter::default();
        assert_eq!(counter.model(), TokenizerModel::ClaudeSonnet45);
    }

    #[test]
    fn test_gpt53_codex_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Gpt53Codex).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Gpt53Codex);
    }

    #[test]
    fn test_claude_opus46_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::ClaudeOpus46).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::ClaudeOpus46);
    }

    #[test]
    fn test_custom_tokenizer() {
        // 400 = 4.0 chars per token
        let counter = TokenCounter::new(TokenizerModel::Custom {
            chars_per_token_x100: 400,
        })
        .unwrap();

        // "Hello" = 5 chars, 4 chars/token = 2 tokens (ceil)
        let count = counter.count("Hello");
        assert_eq!(count, 2);

        // "Hi" = 2 chars, 4 chars/token = 1 token (ceil)
        let count2 = counter.count("Hi");
        assert_eq!(count2, 1);

        assert_eq!(
            counter.model(),
            TokenizerModel::Custom {
                chars_per_token_x100: 400
            }
        );
    }

    #[test]
    fn test_count_many() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let texts = vec!["Hello", "World", "Test"];
        let total = counter.count_many(&texts);
        let individual: usize = texts.iter().map(|t| counter.count(t)).sum();
        assert_eq!(total, individual);
    }

    #[test]
    fn test_budget_check() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        assert!(counter.fits_budget("Hello", 100));
        assert!(!counter.fits_budget("Hello ".repeat(1000).as_str(), 10));
    }

    #[test]
    fn test_truncation_with_bpe() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let long_text = "This is a long text that needs to be truncated. ".repeat(100);
        let truncated = counter.truncate_to_budget(&long_text, 10);
        assert!(counter.count(&truncated) <= 10);
    }

    #[test]
    fn test_truncation_short_text() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let short_text = "Hi";
        let truncated = counter.truncate_to_budget(short_text, 100);
        assert_eq!(truncated, short_text);
    }

    #[test]
    fn test_truncation_custom_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Custom {
            chars_per_token_x100: 400,
        })
        .unwrap();

        let long_text = "This is a very long text that should be truncated";
        let truncated = counter.truncate_to_budget(long_text, 3);
        // 3 tokens * 4 chars = 12 chars max
        assert!(truncated.chars().count() <= 12);
    }

    #[test]
    fn test_empty_text() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        assert_eq!(counter.count(""), 0);
        assert!(counter.fits_budget("", 0));
    }

    #[test]
    fn test_unicode_text() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let unicode_text = "こんにちは世界! 🌍";
        let count = counter.count(unicode_text);
        assert!(count > 0);
    }

    #[test]
    fn test_tokenizer_model_equality() {
        assert_eq!(TokenizerModel::Gpt4o, TokenizerModel::Gpt4o);
        assert_ne!(TokenizerModel::Gpt4o, TokenizerModel::ClaudeSonnet45);
        assert_ne!(
            TokenizerModel::Custom {
                chars_per_token_x100: 400
            },
            TokenizerModel::Custom {
                chars_per_token_x100: 300
            }
        );
    }

    #[test]
    fn test_image_token_counting_low() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let dims = ImageDimensions { width: 1024, height: 768 };
        let tokens = counter.count_image(dims, ImageDetail::Low);
        assert_eq!(tokens, 85);
    }

    #[test]
    fn test_image_token_counting_high() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let dims = ImageDimensions { width: 1024, height: 1024 };
        let tokens = counter.count_image(dims, ImageDetail::High);
        assert!(tokens > 85);
    }

    #[test]
    fn test_count_images() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let images = vec![
            (ImageDimensions { width: 512, height: 512 }, ImageDetail::Low),
            (ImageDimensions { width: 512, height: 512 }, ImageDetail::Low),
        ];
        let tokens = counter.count_images(&images);
        assert_eq!(tokens, 170); // 85 * 2
    }

    #[test]
    fn test_gemini3_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Gemini3).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Gemini3);
    }

    #[test]
    fn test_llama4_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Llama4).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Llama4);
    }

    #[test]
    fn test_deepseek_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::DeepSeekV32).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::DeepSeekV32);
    }

    #[test]
    fn test_grok41_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Grok41).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Grok41);
    }

    #[test]
    fn test_qwen3_coder_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Qwen3Coder).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Qwen3Coder);
    }
}
