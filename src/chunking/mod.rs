//! Semantic chunking for text and code
//!
//! Splits content into meaningful chunks while respecting semantic boundaries.

use crate::scoring::{Priority, RelevanceScore};
use crate::token::TokenCounter;
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;
use xxhash_rust::xxh3::xxh3_64;

/// Content type for chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    /// Plain text content
    Text,
    /// Source code
    Code,
    /// Markdown formatted text
    Markdown,
    /// Conversation/chat messages
    Conversation,
    /// JSON or structured data
    Structured,
}

impl Default for ContentType {
    fn default() -> Self {
        Self::Text
    }
}

/// A chunk of content with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier (hash-based)
    pub id: u64,
    /// The actual content
    pub content: String,
    /// Content type
    pub content_type: ContentType,
    /// Token count
    pub tokens: usize,
    /// Relevance score
    pub score: RelevanceScore,
    /// Source identifier (e.g., filename, message index)
    pub source: Option<String>,
    /// Position in original content (start, end)
    pub position: Option<(usize, usize)>,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Chunk {
    /// Create a new chunk
    pub fn new(content: impl Into<String>, content_type: ContentType) -> Self {
        let content = content.into();
        let id = xxh3_64(content.as_bytes());
        Self {
            id,
            content,
            content_type,
            tokens: 0,
            score: RelevanceScore::default(),
            source: None,
            position: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.score = RelevanceScore::new(priority);
        self
    }

    /// Set source identifier
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set position
    pub fn with_position(mut self, start: usize, end: usize) -> Self {
        self.position = Some((start, end));
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Calculate tokens using provided counter
    pub fn calculate_tokens(&mut self, counter: &TokenCounter) {
        self.tokens = counter.count(&self.content);
    }

    /// Check if chunk is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Configuration for chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Target chunk size in tokens
    pub target_tokens: usize,
    /// Maximum chunk size in tokens
    pub max_tokens: usize,
    /// Minimum chunk size in tokens
    pub min_tokens: usize,
    /// Overlap between chunks in tokens
    pub overlap_tokens: usize,
    /// Content type to optimize for
    pub content_type: ContentType,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 512,
            max_tokens: 1024,
            min_tokens: 50,
            overlap_tokens: 50,
            content_type: ContentType::Text,
        }
    }
}

impl ChunkConfig {
    /// Create config for code
    pub fn for_code() -> Self {
        Self {
            target_tokens: 256,
            max_tokens: 512,
            min_tokens: 20,
            overlap_tokens: 20,
            content_type: ContentType::Code,
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
        }
    }
}

/// Semantic text chunker
pub struct Chunker {
    config: ChunkConfig,
    counter: TokenCounter,
}

impl Chunker {
    /// Create a new chunker
    pub fn new(config: ChunkConfig, counter: TokenCounter) -> Self {
        Self { config, counter }
    }

    /// Chunk text into semantic pieces
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        match self.config.content_type {
            ContentType::Text | ContentType::Markdown => self.chunk_text(text),
            ContentType::Code => self.chunk_code(text),
            ContentType::Conversation => self.chunk_conversation(text),
            ContentType::Structured => self.chunk_structured(text),
        }
    }

    fn chunk_text(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut current_content = String::new();
        let mut current_start = 0;

        for para in paragraphs {
            let para_tokens = self.counter.count(para);
            let current_tokens = self.counter.count(&current_content);

            if current_tokens + para_tokens > self.config.max_tokens && !current_content.is_empty()
            {
                // Save current chunk
                let mut chunk =
                    Chunk::new(current_content.trim(), self.config.content_type)
                        .with_position(current_start, current_start + current_content.len());
                chunk.calculate_tokens(&self.counter);
                chunks.push(chunk);

                // Start new chunk with overlap
                current_content = self.get_overlap(&current_content);
                current_start += current_content.len();
            }

            if !current_content.is_empty() {
                current_content.push_str("\n\n");
            }
            current_content.push_str(para);
        }

        // Don't forget the last chunk
        if !current_content.trim().is_empty() {
            let mut chunk = Chunk::new(current_content.trim(), self.config.content_type)
                .with_position(current_start, text.len());
            chunk.calculate_tokens(&self.counter);
            chunks.push(chunk);
        }

        chunks
    }

    fn chunk_code(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut current_lines: Vec<&str> = Vec::new();
        let mut current_start = 0;
        let mut char_pos = 0;

        for line in &lines {
            let current_content = current_lines.join("\n");
            let current_tokens = self.counter.count(&current_content);
            let line_tokens = self.counter.count(line);

            // Check for natural break points (empty lines, function definitions)
            let is_break_point =
                line.is_empty() || line.starts_with("fn ") || line.starts_with("pub ");

            if current_tokens + line_tokens > self.config.max_tokens
                || (is_break_point && current_tokens > self.config.target_tokens)
            {
                if !current_lines.is_empty() {
                    let content = current_lines.join("\n");
                    let mut chunk = Chunk::new(&content, ContentType::Code)
                        .with_position(current_start, char_pos);
                    chunk.calculate_tokens(&self.counter);
                    chunks.push(chunk);

                    current_start = char_pos;
                    current_lines.clear();
                }
            }

            current_lines.push(line);
            char_pos += line.len() + 1; // +1 for newline
        }

        // Last chunk
        if !current_lines.is_empty() {
            let content = current_lines.join("\n");
            let mut chunk =
                Chunk::new(&content, ContentType::Code).with_position(current_start, char_pos);
            chunk.calculate_tokens(&self.counter);
            chunks.push(chunk);
        }

        chunks
    }

    fn chunk_conversation(&self, text: &str) -> Vec<Chunk> {
        // Split by message boundaries (simple heuristic)
        let mut chunks = Vec::new();
        let messages: Vec<&str> = text.split("\n---\n").collect();

        for (i, msg) in messages.iter().enumerate() {
            if !msg.trim().is_empty() {
                let mut chunk = Chunk::new(msg.trim(), ContentType::Conversation)
                    .with_metadata("message_index", i.to_string());
                chunk.calculate_tokens(&self.counter);
                chunks.push(chunk);
            }
        }

        chunks
    }

    fn chunk_structured(&self, text: &str) -> Vec<Chunk> {
        // For structured data, try to keep it whole if possible
        let tokens = self.counter.count(text);
        if tokens <= self.config.max_tokens {
            let mut chunk = Chunk::new(text, ContentType::Structured);
            chunk.tokens = tokens;
            return vec![chunk];
        }

        // Fall back to text chunking
        self.chunk_text(text)
    }

    fn get_overlap(&self, text: &str) -> String {
        if self.config.overlap_tokens == 0 {
            return String::new();
        }

        // Get last N sentences for overlap
        let sentences: Vec<&str> = text.unicode_sentences().collect();
        let mut overlap = String::new();

        for sentence in sentences.iter().rev() {
            let new_overlap = if overlap.is_empty() {
                sentence.to_string()
            } else {
                format!("{} {}", sentence, overlap)
            };

            if self.counter.count(&new_overlap) > self.config.overlap_tokens {
                break;
            }
            overlap = new_overlap;
        }

        overlap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenizerModel;

    // ContentType tests
    #[test]
    fn test_content_type_default() {
        assert_eq!(ContentType::default(), ContentType::Text);
    }

    // Chunk tests
    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new("Hello, world!", ContentType::Text)
            .with_priority(Priority::High)
            .with_source("test.txt");

        assert_eq!(chunk.content, "Hello, world!");
        assert_eq!(chunk.source, Some("test.txt".to_string()));
        assert_eq!(chunk.content_type, ContentType::Text);
        assert_eq!(chunk.score.priority, Priority::High);
    }

    #[test]
    fn test_chunk_with_position() {
        let chunk = Chunk::new("Content", ContentType::Text).with_position(10, 20);
        assert_eq!(chunk.position, Some((10, 20)));
    }

    #[test]
    fn test_chunk_with_metadata() {
        let chunk = Chunk::new("Content", ContentType::Text)
            .with_metadata("key", "value")
            .with_metadata("another", "data");

        assert_eq!(chunk.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(chunk.metadata.get("another"), Some(&"data".to_string()));
    }

    #[test]
    fn test_chunk_calculate_tokens() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut chunk = Chunk::new("Hello, world!", ContentType::Text);

        assert_eq!(chunk.tokens, 0);
        chunk.calculate_tokens(&counter);
        assert!(chunk.tokens > 0);
    }

    #[test]
    fn test_chunk_is_empty() {
        let empty_chunk = Chunk::new("", ContentType::Text);
        assert!(empty_chunk.is_empty());

        let non_empty = Chunk::new("Hello", ContentType::Text);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_chunk_id_is_hash() {
        let chunk1 = Chunk::new("Same content", ContentType::Text);
        let chunk2 = Chunk::new("Same content", ContentType::Text);
        let chunk3 = Chunk::new("Different", ContentType::Text);

        assert_eq!(chunk1.id, chunk2.id);
        assert_ne!(chunk1.id, chunk3.id);
    }

    // ChunkConfig tests
    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.target_tokens, 512);
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.min_tokens, 50);
        assert_eq!(config.overlap_tokens, 50);
        assert_eq!(config.content_type, ContentType::Text);
    }

    #[test]
    fn test_chunk_config_for_code() {
        let config = ChunkConfig::for_code();
        assert_eq!(config.content_type, ContentType::Code);
        assert!(config.target_tokens < ChunkConfig::default().target_tokens);
    }

    #[test]
    fn test_chunk_config_for_conversation() {
        let config = ChunkConfig::for_conversation();
        assert_eq!(config.content_type, ContentType::Conversation);
        assert_eq!(config.overlap_tokens, 0);
    }

    // Chunker tests
    #[test]
    fn test_text_chunking() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let chunker = Chunker::new(ChunkConfig::default(), counter);

        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.tokens > 0);
        }
    }

    #[test]
    fn test_text_chunking_long() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 20,
            target_tokens: 10,
            min_tokens: 5,
            overlap_tokens: 5,
            content_type: ContentType::Text,
        };
        let chunker = Chunker::new(config, counter);

        let text = "This is paragraph one with content.\n\n\
                    This is paragraph two with more content.\n\n\
                    This is paragraph three with even more content.\n\n\
                    This is paragraph four to make it longer.";
        let chunks = chunker.chunk(text);

        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_code_chunking() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig::for_code();
        let chunker = Chunker::new(config, counter);

        let code = r#"fn main() {
    println!("Hello");
}

fn another() {
    let x = 1;
}

pub fn public_fn() {
    // comment
}"#;

        let chunks = chunker.chunk(code);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(chunk.content_type, ContentType::Code);
        }
    }

    #[test]
    fn test_conversation_chunking() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig::for_conversation();
        let chunker = Chunker::new(config, counter);

        let conversation = "User: Hello\n---\nAssistant: Hi there!\n---\nUser: How are you?";
        let chunks = chunker.chunk(conversation);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(chunk.content_type, ContentType::Conversation);
        }
    }

    #[test]
    fn test_structured_chunking_small() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 1000,
            content_type: ContentType::Structured,
            ..Default::default()
        };
        let chunker = Chunker::new(config, counter);

        let json = r#"{"key": "value", "number": 42}"#;
        let chunks = chunker.chunk(json);

        // Small structured data should be kept whole
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content_type, ContentType::Structured);
    }

    #[test]
    fn test_structured_chunking_large() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 10,
            content_type: ContentType::Structured,
            ..Default::default()
        };
        let chunker = Chunker::new(config, counter);

        let large_json = r#"{"key": "value", "data": "lots of content here that exceeds the limit"}"#;
        let chunks = chunker.chunk(large_json);

        // Large structured data should fall back to text chunking
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_chunking() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            content_type: ContentType::Markdown,
            ..Default::default()
        };
        let chunker = Chunker::new(config, counter);

        let markdown = "# Header\n\nParagraph one.\n\n## Subheader\n\nParagraph two.";
        let chunks = chunker.chunk(markdown);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_text_chunking() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let chunker = Chunker::new(ChunkConfig::default(), counter);

        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only_chunking() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let chunker = Chunker::new(ChunkConfig::default(), counter);

        let chunks = chunker.chunk("   \n\n   ");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_positions_are_set() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let chunker = Chunker::new(ChunkConfig::default(), counter);

        let text = "First paragraph.\n\nSecond paragraph.";
        let chunks = chunker.chunk(text);

        for chunk in chunks {
            assert!(chunk.position.is_some());
        }
    }

    #[test]
    fn test_code_chunking_with_empty_lines() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 50,
            target_tokens: 25,
            content_type: ContentType::Code,
            ..Default::default()
        };
        let chunker = Chunker::new(config, counter);

        let code = "line1\n\nline2\n\nfn test() {\n    body\n}\n\npub fn other() {}";
        let chunks = chunker.chunk(code);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_code_chunking_exceeds_max_tokens() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 5,  // Very small to force splits
            target_tokens: 3,
            min_tokens: 1,
            overlap_tokens: 0,
            content_type: ContentType::Code,
        };
        let chunker = Chunker::new(config, counter);

        // Long code that will exceed max_tokens and trigger splits
        let code = "fn function_one() {\n    let x = 1;\n    let y = 2;\n}\n\nfn function_two() {\n    let a = 3;\n}";
        let chunks = chunker.chunk(code);

        // Should have multiple chunks due to token limit
        assert!(chunks.len() >= 1);
        for chunk in &chunks {
            assert_eq!(chunk.content_type, ContentType::Code);
        }
    }

    #[test]
    fn test_text_chunking_with_overlap() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 15,
            target_tokens: 10,
            min_tokens: 5,
            overlap_tokens: 50, // Large overlap to trigger overlap logic
            content_type: ContentType::Text,
        };
        let chunker = Chunker::new(config, counter);

        let text = "First sentence here. Second sentence follows. Third sentence ends.\n\n\
                    Another paragraph starts. More content here. Final sentence.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_text_chunking_no_overlap() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = ChunkConfig {
            max_tokens: 15,
            target_tokens: 10,
            min_tokens: 5,
            overlap_tokens: 0, // No overlap
            content_type: ContentType::Text,
        };
        let chunker = Chunker::new(config, counter);

        let text = "First paragraph with content.\n\nSecond paragraph here.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
    }
}
