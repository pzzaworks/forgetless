//! Token counting and management
//!
//! Uses cl100k_base (GPT-4) tokenizer which works well for all modern LLMs.
//! All major models (GPT, Claude, Gemini, Llama, etc.) use similar BPE tokenizers
//! with ~4 characters per token ratio.

use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};

/// Tokenizer model selection
///
/// Most modern LLMs use similar BPE tokenizers, so `Default` works for all:
/// - OpenAI GPT-4o, GPT-5.x
/// - Anthropic Claude 4.x
/// - Google Gemini
/// - Meta Llama 4
/// - Mistral, DeepSeek, Qwen, etc.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TokenizerModel {
    /// Default tokenizer (cl100k_base) - works for all modern LLMs
    /// ~4 characters per token, accurate within 5% for any model
    Default,

    /// Custom characters-per-token ratio (value * 100, e.g., 400 = 4.0)
    Custom {
        /// Characters per token multiplied by 100 (e.g., 400 = 4.0 chars/token)
        chars_per_token_x100: u32,
    },
}

impl Default for TokenizerModel {
    fn default() -> Self {
        Self::Default
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
    /// Create a new token counter
    ///
    /// Uses cl100k_base (GPT-4) tokenizer by default, which works well for all modern LLMs.
    pub fn new(model: TokenizerModel) -> Result<Self> {
        let bpe = match model {
            TokenizerModel::Default => {
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
            (None, TokenizerModel::Default) => {
                unreachable!("Default tokenizer always has BPE")
            }
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
            // Char-based truncation for Custom tokenizer
            let TokenizerModel::Custom { chars_per_token_x100 } = self.model else {
                unreachable!("Only Custom model has no BPE");
            };
            let chars_per_token = chars_per_token_x100 as f32 / 100.0;
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
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let text = "Hello, world!";
        let count = counter.count(text);
        assert!(count > 0);
        assert!(count < text.len());
    }

    #[test]
    fn test_tokenizer_model_default() {
        let model = TokenizerModel::default();
        assert_eq!(model, TokenizerModel::Default);
    }

    #[test]
    fn test_token_counter_default() {
        let counter = TokenCounter::default();
        assert_eq!(counter.model(), TokenizerModel::Default);
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
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let texts = vec!["Hello", "World", "Test"];
        let total = counter.count_many(&texts);
        let individual: usize = texts.iter().map(|t| counter.count(t)).sum();
        assert_eq!(total, individual);
    }

    #[test]
    fn test_budget_check() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        assert!(counter.fits_budget("Hello", 100));
        assert!(!counter.fits_budget("Hello ".repeat(1000).as_str(), 10));
    }

    #[test]
    fn test_truncation_with_bpe() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let long_text = "This is a long text that needs to be truncated. ".repeat(100);
        let truncated = counter.truncate_to_budget(&long_text, 10);
        assert!(counter.count(&truncated) <= 10);
    }

    #[test]
    fn test_truncation_short_text() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
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
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        assert_eq!(counter.count(""), 0);
        assert!(counter.fits_budget("", 0));
    }

    #[test]
    fn test_unicode_text() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let unicode_text = "こんにちは世界! 🌍";
        let count = counter.count(unicode_text);
        assert!(count > 0);
    }

    #[test]
    fn test_tokenizer_model_equality() {
        assert_eq!(TokenizerModel::Default, TokenizerModel::Default);
        assert_ne!(
            TokenizerModel::Default,
            TokenizerModel::Custom { chars_per_token_x100: 400 }
        );
        assert_ne!(
            TokenizerModel::Custom { chars_per_token_x100: 400 },
            TokenizerModel::Custom { chars_per_token_x100: 300 }
        );
    }

    #[test]
    fn test_image_token_counting_low() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let dims = ImageDimensions { width: 1024, height: 768 };
        let tokens = counter.count_image(dims, ImageDetail::Low);
        assert_eq!(tokens, 85);
    }

    #[test]
    fn test_image_token_counting_high() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let dims = ImageDimensions { width: 1024, height: 1024 };
        let tokens = counter.count_image(dims, ImageDetail::High);
        assert!(tokens > 85);
    }

    #[test]
    fn test_count_images() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let images = vec![
            (ImageDimensions { width: 512, height: 512 }, ImageDetail::Low),
            (ImageDimensions { width: 512, height: 512 }, ImageDetail::Low),
        ];
        let tokens = counter.count_images(&images);
        assert_eq!(tokens, 170); // 85 * 2
    }

    #[test]
    fn test_image_token_counting_auto() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let dims = ImageDimensions { width: 1024, height: 1024 };
        let tokens = counter.count_image(dims, ImageDetail::Auto);
        // Auto behaves like High
        assert!(tokens > 85);
    }

    #[test]
    fn test_image_detail_default() {
        let detail = ImageDetail::default();
        assert_eq!(detail, ImageDetail::Auto);
    }

    #[test]
    fn test_image_small_no_scaling() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        // Small image that doesn't need scaling
        let dims = ImageDimensions { width: 256, height: 256 };
        let tokens = counter.count_image(dims, ImageDetail::High);
        assert!(tokens > 85);
    }

    #[test]
    fn test_image_large_needs_scaling() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        // Large image that needs scaling
        let dims = ImageDimensions { width: 4096, height: 4096 };
        let tokens = counter.count_image(dims, ImageDetail::High);
        assert!(tokens > 85);
    }

    #[test]
    fn test_image_tall_aspect_ratio() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let dims = ImageDimensions { width: 512, height: 2048 };
        let tokens = counter.count_image(dims, ImageDetail::High);
        assert!(tokens > 85);
    }

    #[test]
    fn test_image_wide_aspect_ratio() {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let dims = ImageDimensions { width: 2048, height: 512 };
        let tokens = counter.count_image(dims, ImageDetail::High);
        assert!(tokens > 85);
    }
}
