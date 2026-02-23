//! Chunking configuration

use super::types::ContentType;
use serde::{Deserialize, Serialize};

/// Configuration for chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Target chunk size in tokens
    pub target_tokens: usize,
    /// Maximum chunk size in tokens
    pub max_tokens: usize,
    /// Minimum chunk size in tokens (chunks smaller than this are skipped)
    pub min_tokens: usize,
    /// Overlap between chunks in tokens
    pub overlap_tokens: usize,
    /// Content type to optimize for
    pub content_type: ContentType,
    /// Enable deduplication (removes duplicate chunks)
    pub deduplicate: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 512,
            max_tokens: 1024,
            min_tokens: 10,
            overlap_tokens: 50,
            content_type: ContentType::Text,
            deduplicate: true,
        }
    }
}

impl ChunkConfig {
    /// Create config for code
    pub fn for_code() -> Self {
        Self {
            target_tokens: 256,
            max_tokens: 512,
            min_tokens: 10,
            overlap_tokens: 20,
            content_type: ContentType::Code,
            deduplicate: true,
        }
    }

    /// Create config for conversation
    pub fn for_conversation() -> Self {
        Self {
            target_tokens: 200,
            max_tokens: 400,
            min_tokens: 10,
            overlap_tokens: 0,
            content_type: ContentType::Conversation,
            deduplicate: true,
        }
    }

    /// Create config optimized for speed (larger chunks)
    pub fn for_speed() -> Self {
        Self {
            target_tokens: 1000,
            max_tokens: 2000,
            min_tokens: 20,
            overlap_tokens: 0,
            content_type: ContentType::Text,
            deduplicate: true,
        }
    }

    /// Create config optimized for quality (smaller chunks)
    pub fn for_quality() -> Self {
        Self {
            target_tokens: 256,
            max_tokens: 512,
            min_tokens: 10,
            overlap_tokens: 50,
            content_type: ContentType::Text,
            deduplicate: true,
        }
    }

    /// Set content type
    pub fn with_content_type(mut self, content_type: ContentType) -> Self {
        self.content_type = content_type;
        self
    }

    /// Set target tokens
    pub fn with_target_tokens(mut self, tokens: usize) -> Self {
        self.target_tokens = tokens;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set min tokens
    pub fn with_min_tokens(mut self, tokens: usize) -> Self {
        self.min_tokens = tokens;
        self
    }

    /// Enable/disable deduplication
    pub fn with_deduplication(mut self, enabled: bool) -> Self {
        self.deduplicate = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.target_tokens, 512);
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.min_tokens, 10);
        assert_eq!(config.overlap_tokens, 50);
        assert!(config.deduplicate);
    }

    #[test]
    fn test_chunk_config_for_code() {
        let config = ChunkConfig::for_code();
        assert_eq!(config.content_type, ContentType::Code);
        assert_eq!(config.target_tokens, 256);
    }

    #[test]
    fn test_chunk_config_for_conversation() {
        let config = ChunkConfig::for_conversation();
        assert_eq!(config.content_type, ContentType::Conversation);
        assert_eq!(config.overlap_tokens, 0);
    }

    #[test]
    fn test_chunk_config_for_speed() {
        let config = ChunkConfig::for_speed();
        assert_eq!(config.target_tokens, 1000);
        assert_eq!(config.max_tokens, 2000);
    }

    #[test]
    fn test_chunk_config_for_quality() {
        let config = ChunkConfig::for_quality();
        assert_eq!(config.target_tokens, 256);
        assert_eq!(config.overlap_tokens, 50);
    }

    #[test]
    fn test_chunk_config_builders() {
        let config = ChunkConfig::default()
            .with_content_type(ContentType::Code)
            .with_target_tokens(200)
            .with_max_tokens(400)
            .with_min_tokens(5)
            .with_deduplication(false);

        assert_eq!(config.content_type, ContentType::Code);
        assert_eq!(config.target_tokens, 200);
        assert_eq!(config.max_tokens, 400);
        assert_eq!(config.min_tokens, 5);
        assert!(!config.deduplicate);
    }
}
