//! Configuration for the Forgetless optimizer

use crate::processing::chunking::ChunkConfig;
use crate::processing::token::TokenizerModel;
use serde::{Deserialize, Serialize};

/// Configuration for the Forgetless optimizer (internal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetlessConfig {
    /// Runtime options
    pub options: Config,
    /// Tokenizer model for token counting
    pub tokenizer: TokenizerModel,
    /// Chunking configuration
    pub chunk: ChunkConfig,
    /// Scoring weights
    pub scoring: ScoringConfig,
}

impl Default for ForgetlessConfig {
    fn default() -> Self {
        Self {
            options: Config::default(),
            tokenizer: TokenizerModel::default(),
            chunk: ChunkConfig::default(),
            scoring: ScoringConfig::default(),
        }
    }
}

impl ForgetlessConfig {
    /// Create config with options
    pub fn new(options: Config) -> Self {
        let mut config = Self {
            options,
            ..Default::default()
        };
        // Sync chunk size from options
        config.chunk.target_tokens = config.options.chunk_size;
        config
    }

    /// Set the tokenizer model
    pub fn with_tokenizer(mut self, tokenizer: TokenizerModel) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set chunk configuration
    pub fn with_chunk(mut self, chunk: ChunkConfig) -> Self {
        self.chunk = chunk;
        self
    }

    /// Set scoring configuration
    pub fn with_scoring(mut self, scoring: ScoringConfig) -> Self {
        self.scoring = scoring;
        self
    }
}

/// Runtime options for Forgetless
///
/// # Example
/// ```ignore
/// Forgetless::new()
///     .options(Config::default()
///         .context_limit(128_000)
///         .vision_llm(true)
///         .context_llm(true))
///     .add("content")
///     .run()
///     .await?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum tokens in output (default: 128_000)
    pub context_limit: usize,

    /// Use LLM for image understanding (default: false)
    /// Generates descriptions for images instead of just metadata
    pub vision_llm: bool,

    /// Use LLM for context optimization (default: false)
    /// Enables smart scoring and summarization
    pub context_llm: bool,

    /// Target chunk size in tokens (default: 512)
    pub chunk_size: usize,

    /// Enable parallel file processing (default: true)
    pub parallel: bool,

    /// Enable embedding cache (default: true)
    pub cache: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            context_limit: 128_000,
            vision_llm: false,
            context_llm: false,
            chunk_size: 512,
            parallel: true,
            cache: true,
        }
    }
}

impl Config {
    /// Set maximum output tokens
    pub fn context_limit(mut self, limit: usize) -> Self {
        self.context_limit = limit;
        self
    }

    /// Enable LLM for image descriptions
    pub fn vision_llm(mut self, enabled: bool) -> Self {
        self.vision_llm = enabled;
        self
    }

    /// Enable LLM for smart scoring/summarization
    pub fn context_llm(mut self, enabled: bool) -> Self {
        self.context_llm = enabled;
        self
    }

    /// Set target chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Enable/disable embedding cache
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache = enabled;
        self
    }
}

/// Configuration for scoring weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Weight for semantic similarity (0.0 - 1.0)
    pub semantic_weight: f32,
    /// Weight for keyword matching (0.0 - 1.0)
    pub keyword_weight: f32,
    /// Weight for priority boost (0.0 - 1.0)
    pub priority_weight: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.5,
            keyword_weight: 0.3,
            priority_weight: 0.2,
        }
    }
}

impl ScoringConfig {
    /// Validate that weights sum to 1.0
    pub fn validate(&self) -> bool {
        let sum = self.semantic_weight + self.keyword_weight + self.priority_weight;
        (sum - 1.0).abs() < 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ForgetlessConfig::default();
        assert_eq!(config.options.context_limit, 128_000);
        assert!(config.scoring.validate());
    }

    #[test]
    fn test_config_new() {
        let opts = Config::default().context_limit(4096);
        let config = ForgetlessConfig::new(opts);
        assert_eq!(config.options.context_limit, 4096);
    }

    #[test]
    fn test_config_builder() {
        let opts = Config::default().context_limit(64_000);
        let config = ForgetlessConfig::new(opts).with_tokenizer(TokenizerModel::Default);

        assert_eq!(config.options.context_limit, 64_000);
    }

    #[test]
    fn test_with_chunk() {
        let chunk = ChunkConfig {
            target_tokens: 256,
            max_tokens: 512,
            min_tokens: 30,
            overlap_tokens: 25,
            ..Default::default()
        };

        let config = ForgetlessConfig::default().with_chunk(chunk);
        assert_eq!(config.chunk.target_tokens, 256);
        assert_eq!(config.chunk.max_tokens, 512);
    }

    #[test]
    fn test_with_scoring() {
        let scoring = ScoringConfig {
            semantic_weight: 0.6,
            keyword_weight: 0.3,
            priority_weight: 0.1,
        };

        let config = ForgetlessConfig::default().with_scoring(scoring);
        assert_eq!(config.scoring.semantic_weight, 0.6);
    }

    #[test]
    fn test_options() {
        let opts = Config::default()
            .context_limit(64_000)
            .vision_llm(true)
            .context_llm(true)
            .chunk_size(256)
            .parallel(false)
            .cache(false);

        assert_eq!(opts.context_limit, 64_000);
        assert!(opts.vision_llm);
        assert!(opts.context_llm);
        assert_eq!(opts.chunk_size, 256);
        assert!(!opts.parallel);
        assert!(!opts.cache);
    }

    #[test]
    fn test_scoring_validate() {
        let valid = ScoringConfig::default();
        assert!(valid.validate());

        let invalid = ScoringConfig {
            semantic_weight: 0.5,
            keyword_weight: 0.5,
            priority_weight: 0.5,
        };
        assert!(!invalid.validate());
    }

    #[test]
    fn test_config_serialization() {
        let opts = Config::default().context_limit(4096);
        let config = ForgetlessConfig::new(opts);
        let json = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: ForgetlessConfig =
            serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.options.context_limit, 4096);
    }
}
