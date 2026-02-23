//! Content types and chunks

use crate::processing::scoring::Priority;
use crate::processing::token::TokenCounter;
use serde::{Deserialize, Serialize};
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

impl ContentType {
    /// Detect content type from file path/extension
    pub fn detect_from_path(path: &str) -> Self {
        let path_lower = path.to_lowercase();

        // Code files
        if path_lower.ends_with(".rs")
            || path_lower.ends_with(".py")
            || path_lower.ends_with(".js")
            || path_lower.ends_with(".ts")
            || path_lower.ends_with(".tsx")
            || path_lower.ends_with(".jsx")
            || path_lower.ends_with(".go")
            || path_lower.ends_with(".c")
            || path_lower.ends_with(".cpp")
            || path_lower.ends_with(".h")
            || path_lower.ends_with(".java")
            || path_lower.ends_with(".rb")
            || path_lower.ends_with(".php")
            || path_lower.ends_with(".swift")
            || path_lower.ends_with(".kt")
            || path_lower.ends_with(".scala")
            || path_lower.ends_with(".sh")
            || path_lower.ends_with(".bash")
        {
            return Self::Code;
        }

        // Markdown
        if path_lower.ends_with(".md") || path_lower.ends_with(".markdown") {
            return Self::Markdown;
        }

        // Structured data
        if path_lower.ends_with(".json")
            || path_lower.ends_with(".yaml")
            || path_lower.ends_with(".yml")
            || path_lower.ends_with(".toml")
            || path_lower.ends_with(".xml")
        {
            return Self::Structured;
        }

        // Default to text
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
    /// Priority level
    pub priority: Priority,
    /// Whether this chunk is pinned
    pub pinned: bool,
    /// Source identifier (e.g., filename, message index)
    pub source: Option<String>,
    /// Position index for ordering
    pub position: usize,
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
            priority: Priority::default(),
            pinned: false,
            source: None,
            position: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set source identifier
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set position index
    pub fn with_position(mut self, position: usize) -> Self {
        self.position = position;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_default() {
        assert_eq!(ContentType::default(), ContentType::Text);
    }

    #[test]
    fn test_content_type_detect() {
        assert_eq!(ContentType::detect_from_path("test.rs"), ContentType::Code);
        assert_eq!(ContentType::detect_from_path("test.py"), ContentType::Code);
        assert_eq!(ContentType::detect_from_path("test.md"), ContentType::Markdown);
        assert_eq!(ContentType::detect_from_path("test.json"), ContentType::Structured);
        assert_eq!(ContentType::detect_from_path("test.txt"), ContentType::Text);
    }

    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new("Hello, world!", ContentType::Text)
            .with_priority(Priority::High)
            .with_source("test.txt");

        assert_eq!(chunk.content, "Hello, world!");
        assert_eq!(chunk.source, Some("test.txt".to_string()));
        assert_eq!(chunk.content_type, ContentType::Text);
        assert_eq!(chunk.priority, Priority::High);
    }

    #[test]
    fn test_chunk_with_position() {
        let chunk = Chunk::new("Content", ContentType::Text).with_position(5);
        assert_eq!(chunk.position, 5);
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
    fn test_chunk_is_empty() {
        let empty = Chunk::new("", ContentType::Text);
        let not_empty = Chunk::new("content", ContentType::Text);
        assert!(empty.is_empty());
        assert!(!not_empty.is_empty());
    }

    #[test]
    fn test_chunk_deterministic_id() {
        let chunk1 = Chunk::new("same content", ContentType::Text);
        let chunk2 = Chunk::new("same content", ContentType::Text);
        assert_eq!(chunk1.id, chunk2.id);

        let chunk3 = Chunk::new("different", ContentType::Text);
        assert_ne!(chunk1.id, chunk3.id);
    }

    #[test]
    fn test_content_type_detect_all_code() {
        let code_extensions = [
            ".rs", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".c", ".cpp",
            ".h", ".java", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
        ];

        for ext in code_extensions {
            assert_eq!(
                ContentType::detect_from_path(&format!("test{ext}")),
                ContentType::Code,
                "Failed for extension: {ext}"
            );
        }
    }

    #[test]
    fn test_content_type_detect_structured() {
        let structured = [".json", ".yaml", ".yml", ".toml", ".xml"];
        for ext in structured {
            assert_eq!(
                ContentType::detect_from_path(&format!("config{ext}")),
                ContentType::Structured
            );
        }
    }
}
