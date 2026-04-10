//! Core types for context optimization

use crate::processing::chunking::{Chunk, ContentType};
use crate::processing::scoring::Priority;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A scored chunk ready for selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
    /// The chunk data
    pub chunk: Chunk,
    /// Combined score (0.0 - 1.0)
    pub score: f32,
    /// Component scores for debugging
    pub breakdown: ScoreBreakdown,
}

/// Breakdown of how the final score was calculated
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Keyword matching score
    pub algorithmic: f32,
    /// Semantic similarity to query
    pub semantic: f32,
    /// LLM-based importance score (unused, kept for compatibility)
    pub llm: f32,
    /// Priority boost factor
    pub priority_boost: f32,
    /// Recency factor
    pub recency_factor: f32,
}

/// The result of context optimization
#[derive(Debug, Clone)]
pub struct OptimizedContext {
    /// The optimized content string
    pub content: String,
    /// Individual chunks that were included
    pub chunks: Vec<ScoredChunk>,
    /// Total tokens in the optimized output
    pub total_tokens: usize,
    /// Statistics about the optimization process
    pub stats: OptimizationStats,
}

impl OptimizedContext {
    /// Get compression ratio (input tokens / output tokens)
    pub fn compression_ratio(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.stats.input_tokens as f32 / self.total_tokens as f32
    }
}

/// The result of optimization with LLM polish
#[derive(Debug, Clone)]
pub struct PolishedContext {
    /// The polished content (reorganized by LLM)
    pub content: String,
    /// The raw content before polishing
    pub raw_content: String,
    /// Individual chunks that were included
    pub chunks: Vec<ScoredChunk>,
    /// Total tokens in the output
    pub total_tokens: usize,
    /// Statistics about the optimization process
    pub stats: OptimizationStats,
}

impl PolishedContext {
    /// Get compression ratio (input tokens / output tokens)
    pub fn compression_ratio(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.stats.input_tokens as f32 / self.total_tokens as f32
    }

    /// Get input tokens
    pub fn input_tokens(&self) -> usize {
        self.stats.input_tokens
    }

    /// Get output tokens
    pub fn output_tokens(&self) -> usize {
        self.stats.output_tokens
    }
}

/// Statistics about the optimization process
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Total input tokens before optimization
    pub input_tokens: usize,
    /// Total output tokens after optimization
    pub output_tokens: usize,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Number of chunks selected
    pub chunks_selected: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Compression ratio (input / output)
    pub compression_ratio: f32,
}

/// Input content item
#[derive(Debug, Clone)]
pub struct ContentItem {
    /// Unique identifier
    pub id: String,
    /// The actual content
    pub content: String,
    /// Content type
    pub content_type: ContentType,
    /// Priority level
    pub priority: Priority,
    /// Whether this content is pinned (must be included)
    pub pinned: bool,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ContentItem {
    /// Create a new content item
    pub fn new(content: impl Into<String>) -> Self {
        let content = content.into();
        let id = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(content.as_bytes()));
        Self {
            id,
            content,
            content_type: ContentType::default(),
            priority: Priority::default(),
            pinned: false,
            metadata: HashMap::new(),
        }
    }

    /// Set custom ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set content type
    pub fn with_content_type(mut self, content_type: ContentType) -> Self {
        self.content_type = content_type;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as pinned
    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self.priority = Priority::Critical;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_item_new() {
        let item = ContentItem::new("Hello world");
        assert_eq!(item.content, "Hello world");
        assert!(!item.id.is_empty());
        assert_eq!(item.priority, Priority::Medium);
        assert!(!item.pinned);
    }

    #[test]
    fn test_content_item_with_id() {
        let item = ContentItem::new("content").with_id("custom-id");
        assert_eq!(item.id, "custom-id");
    }

    #[test]
    fn test_content_item_with_priority() {
        let item = ContentItem::new("content").with_priority(Priority::High);
        assert_eq!(item.priority, Priority::High);
    }

    #[test]
    fn test_content_item_pinned() {
        let item = ContentItem::new("content").pinned();
        assert!(item.pinned);
        assert_eq!(item.priority, Priority::Critical);
    }

    #[test]
    fn test_content_item_with_metadata() {
        let item = ContentItem::new("content").with_metadata("key", "value");
        assert_eq!(item.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_content_item_with_content_type() {
        let item = ContentItem::new("fn main() {}").with_content_type(ContentType::Code);
        assert_eq!(item.content_type, ContentType::Code);
    }

    #[test]
    fn test_content_item_deterministic_id() {
        let item1 = ContentItem::new("same content");
        let item2 = ContentItem::new("same content");
        assert_eq!(item1.id, item2.id);
    }

    #[test]
    fn test_content_item_builder_chain() {
        let item = ContentItem::new("content")
            .with_id("id")
            .with_priority(Priority::High)
            .with_content_type(ContentType::Code)
            .with_metadata("source", "test.rs");

        assert_eq!(item.id, "id");
        assert_eq!(item.priority, Priority::High);
        assert_eq!(item.content_type, ContentType::Code);
        assert_eq!(item.metadata.get("source"), Some(&"test.rs".to_string()));
    }

    #[test]
    fn test_optimization_stats_default() {
        let stats = OptimizationStats::default();
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
        assert_eq!(stats.compression_ratio, 0.0);
    }

    #[test]
    fn test_optimized_context_compression_ratio() {
        let ctx = OptimizedContext {
            content: String::new(),
            chunks: Vec::new(),
            total_tokens: 100,
            stats: OptimizationStats {
                input_tokens: 1000,
                ..Default::default()
            },
        };
        assert!((ctx.compression_ratio() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_optimized_context_compression_ratio_zero() {
        let ctx = OptimizedContext {
            content: String::new(),
            chunks: Vec::new(),
            total_tokens: 0,
            stats: OptimizationStats::default(),
        };
        assert_eq!(ctx.compression_ratio(), 0.0);
    }

    #[test]
    fn test_score_breakdown_default() {
        let breakdown = ScoreBreakdown::default();
        assert_eq!(breakdown.algorithmic, 0.0);
        assert_eq!(breakdown.semantic, 0.0);
        assert_eq!(breakdown.llm, 0.0);
    }

    #[test]
    fn test_polished_context_compression_ratio() {
        let ctx = PolishedContext {
            content: String::new(),
            raw_content: String::new(),
            chunks: Vec::new(),
            total_tokens: 100,
            stats: OptimizationStats {
                input_tokens: 500,
                ..Default::default()
            },
        };
        assert!((ctx.compression_ratio() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_polished_context_compression_ratio_zero() {
        let ctx = PolishedContext {
            content: String::new(),
            raw_content: String::new(),
            chunks: Vec::new(),
            total_tokens: 0,
            stats: OptimizationStats::default(),
        };
        assert_eq!(ctx.compression_ratio(), 0.0);
    }

    #[test]
    fn test_polished_context_input_output_tokens() {
        let ctx = PolishedContext {
            content: String::new(),
            raw_content: String::new(),
            chunks: Vec::new(),
            total_tokens: 0,
            stats: OptimizationStats {
                input_tokens: 1000,
                output_tokens: 200,
                ..Default::default()
            },
        };
        assert_eq!(ctx.input_tokens(), 1000);
        assert_eq!(ctx.output_tokens(), 200);
    }
}
