//! Token counting and management
//!
//! Provides accurate token counting for various LLM tokenizers.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};

/// Supported tokenizer models
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TokenizerModel {
    /// GPT-4 and GPT-4 Turbo (cl100k_base)
    Gpt4,
    /// GPT-3.5 Turbo (cl100k_base)
    Gpt35Turbo,
    /// Claude models (approximate, uses cl100k_base)
    Claude,
    /// Custom token-per-char ratio (stored as ratio * 100 for precision)
    Custom {
        /// Average characters per token (multiplied by 100, e.g., 400 = 4.0 chars/token)
        chars_per_token_x100: u32,
    },
}

impl Default for TokenizerModel {
    fn default() -> Self {
        Self::Gpt4
    }
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
            TokenizerModel::Gpt4 | TokenizerModel::Gpt35Turbo | TokenizerModel::Claude => {
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

    /// Count tokens for multiple texts
    pub fn count_many(&self, texts: &[&str]) -> usize {
        texts.iter().map(|t| self.count(t)).sum()
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
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
        let text = "Hello, world!";
        let count = counter.count(text);
        assert!(count > 0);
        assert!(count < text.len());
    }

    #[test]
    fn test_tokenizer_model_default() {
        let model = TokenizerModel::default();
        assert_eq!(model, TokenizerModel::Gpt4);
    }

    #[test]
    fn test_token_counter_default() {
        let counter = TokenCounter::default();
        assert_eq!(counter.model(), TokenizerModel::Gpt4);
    }

    #[test]
    fn test_gpt35_turbo_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Gpt35Turbo).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Gpt35Turbo);
    }

    #[test]
    fn test_claude_tokenizer() {
        let counter = TokenCounter::new(TokenizerModel::Claude).unwrap();
        let count = counter.count("Hello, world!");
        assert!(count > 0);
        assert_eq!(counter.model(), TokenizerModel::Claude);
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
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
        let texts = vec!["Hello", "World", "Test"];
        let total = counter.count_many(&texts);
        let individual: usize = texts.iter().map(|t| counter.count(t)).sum();
        assert_eq!(total, individual);
    }

    #[test]
    fn test_budget_check() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
        assert!(counter.fits_budget("Hello", 100));
        assert!(!counter.fits_budget("Hello ".repeat(1000).as_str(), 10));
    }

    #[test]
    fn test_truncation_with_bpe() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
        let long_text = "This is a long text that needs to be truncated. ".repeat(100);
        let truncated = counter.truncate_to_budget(&long_text, 10);
        assert!(counter.count(&truncated) <= 10);
    }

    #[test]
    fn test_truncation_short_text() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
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
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
        assert_eq!(counter.count(""), 0);
        assert!(counter.fits_budget("", 0));
    }

    #[test]
    fn test_unicode_text() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4).unwrap();
        let unicode_text = "こんにちは世界! 🌍";
        let count = counter.count(unicode_text);
        assert!(count > 0);
    }

    #[test]
    fn test_tokenizer_model_equality() {
        assert_eq!(TokenizerModel::Gpt4, TokenizerModel::Gpt4);
        assert_ne!(TokenizerModel::Gpt4, TokenizerModel::Claude);
        assert_ne!(
            TokenizerModel::Custom {
                chars_per_token_x100: 400
            },
            TokenizerModel::Custom {
                chars_per_token_x100: 300
            }
        );
    }
}
