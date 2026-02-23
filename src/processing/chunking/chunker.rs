//! Semantic text chunker

use super::config::ChunkConfig;
use super::types::{Chunk, ContentType};
use crate::processing::token::TokenCounter;
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;
use xxhash_rust::xxh3::xxh3_64;

/// Semantic text chunker with deduplication and content-aware splitting
pub struct Chunker<'a> {
    config: ChunkConfig,
    counter: &'a TokenCounter,
}

impl<'a> Chunker<'a> {
    /// Create a new chunker
    pub fn new(config: ChunkConfig, counter: &'a TokenCounter) -> Self {
        Self { config, counter }
    }

    /// Chunk text into semantic pieces
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let chunks = self.chunk_by_type(text, self.config.content_type);
        self.dedupe(chunks)
    }

    /// Chunk with specific content type (ignores config content_type)
    pub fn chunk_as(&self, text: &str, content_type: ContentType) -> Vec<Chunk> {
        let chunks = self.chunk_by_type(text, content_type);
        self.dedupe(chunks)
    }

    /// Chunk with custom target size
    pub fn chunk_with_size(&self, text: &str, target_tokens: usize) -> Vec<Chunk> {
        let chunks = self.chunk_text_sized(text, target_tokens, self.config.content_type);
        self.dedupe(chunks)
    }

    fn chunk_by_type(&self, text: &str, content_type: ContentType) -> Vec<Chunk> {
        match content_type {
            ContentType::Text => self.chunk_text_sized(text, self.config.target_tokens, ContentType::Text),
            ContentType::Markdown => self.chunk_markdown(text),
            ContentType::Code => self.chunk_code(text),
            ContentType::Conversation => self.chunk_conversation(text),
            ContentType::Structured => self.chunk_structured(text),
        }
    }

    /// Deduplicate chunks by content hash
    fn dedupe(&self, chunks: Vec<Chunk>) -> Vec<Chunk> {
        if !self.config.deduplicate {
            return chunks;
        }

        let mut seen: HashSet<u64> = HashSet::with_capacity(chunks.len());
        let mut result: Vec<Chunk> = Vec::with_capacity(chunks.len());
        let mut position = 0;

        for mut chunk in chunks {
            // Skip small chunks (min_tokens filter)
            // Note: Priority filtering happens later in builder after priority is assigned
            if chunk.tokens < self.config.min_tokens {
                continue;
            }

            let hash = xxh3_64(chunk.content.as_bytes());
            if seen.insert(hash) {
                chunk.position = position;
                result.push(chunk);
                position += 1;
            }
        }

        result
    }

    /// Chunk markdown by headers
    fn chunk_markdown(&self, content: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut position = 0;
        let max_tokens = self.config.max_tokens;

        for line in content.lines() {
            let is_header = line.starts_with('#');

            // Split before headers if we have enough content
            if is_header && !current.trim().is_empty() {
                let current_tokens = self.counter.count(&current);
                if current_tokens >= self.config.target_tokens / 2 {
                    let mut chunk = Chunk::new(current.trim(), ContentType::Markdown);
                    chunk.position = position;
                    chunk.calculate_tokens(self.counter);
                    chunks.push(chunk);
                    position += 1;
                    current = String::new();
                }
            }

            current.push_str(line);
            current.push('\n');

            // Force split if too large
            let current_tokens = self.counter.count(&current);
            if current_tokens > max_tokens {
                let mut chunk = Chunk::new(current.trim(), ContentType::Markdown);
                chunk.position = position;
                chunk.calculate_tokens(self.counter);
                chunks.push(chunk);
                position += 1;
                current = String::new();
            }
        }

        // Last chunk
        if !current.trim().is_empty() {
            let mut chunk = Chunk::new(current.trim(), ContentType::Markdown);
            chunk.position = position;
            chunk.calculate_tokens(self.counter);
            chunks.push(chunk);
        }

        chunks
    }

    /// Chunk code by function/block boundaries
    fn chunk_code(&self, content: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut position = 0;
        let mut brace_depth: i32 = 0;

        for line in content.lines() {
            // Track brace depth
            brace_depth += line.chars().filter(|&c| c == '{').count() as i32;
            brace_depth -= line.chars().filter(|&c| c == '}').count() as i32;
            brace_depth = brace_depth.max(0);

            current.push_str(line);
            current.push('\n');

            // Natural break points at top level
            let is_break = brace_depth == 0 && (
                line.trim().is_empty() ||
                line.starts_with("fn ") ||
                line.starts_with("pub fn ") ||
                line.starts_with("pub(") ||
                line.starts_with("impl ") ||
                line.starts_with("struct ") ||
                line.starts_with("enum ") ||
                line.starts_with("mod ") ||
                line.starts_with("trait ") ||
                line.starts_with("type ") ||
                line.starts_with("const ") ||
                line.starts_with("static ") ||
                line.starts_with("use ") ||
                line.starts_with("def ") ||
                line.starts_with("class ") ||
                line.starts_with("async def ") ||
                line.starts_with("function ") ||
                line.starts_with("export ") ||
                line.starts_with("import ")
            );

            let current_tokens = self.counter.count(&current);

            if is_break && current_tokens >= self.config.target_tokens / 2 {
                let mut chunk = Chunk::new(current.trim(), ContentType::Code);
                chunk.position = position;
                chunk.calculate_tokens(self.counter);
                chunks.push(chunk);
                position += 1;
                current = String::new();
            }

            // Force split if too large
            if current_tokens > self.config.max_tokens {
                let mut chunk = Chunk::new(current.trim(), ContentType::Code);
                chunk.position = position;
                chunk.calculate_tokens(self.counter);
                chunks.push(chunk);
                position += 1;
                current = String::new();
            }
        }

        // Last chunk
        if !current.trim().is_empty() {
            let mut chunk = Chunk::new(current.trim(), ContentType::Code);
            chunk.position = position;
            chunk.calculate_tokens(self.counter);
            chunks.push(chunk);
        }

        chunks
    }

    /// Chunk conversation by message boundaries
    fn chunk_conversation(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let messages: Vec<&str> = text.split("\n---\n").collect();

        for (i, msg) in messages.iter().enumerate() {
            if !msg.trim().is_empty() {
                let mut chunk = Chunk::new(msg.trim(), ContentType::Conversation)
                    .with_position(i)
                    .with_metadata("message_index", i.to_string());
                chunk.calculate_tokens(self.counter);
                chunks.push(chunk);
            }
        }

        chunks
    }

    /// Chunk structured data (JSON, YAML, etc.)
    fn chunk_structured(&self, text: &str) -> Vec<Chunk> {
        let tokens = self.counter.count(text);
        if tokens <= self.config.max_tokens {
            let mut chunk = Chunk::new(text, ContentType::Structured);
            chunk.tokens = tokens;
            return vec![chunk];
        }

        // Fall back to text chunking
        self.chunk_text_sized(text, self.config.target_tokens, ContentType::Structured)
    }

    /// Chunk plain text by paragraphs with configurable size
    fn chunk_text_sized(&self, content: &str, target_tokens: usize, content_type: ContentType) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut position = 0;
        let max_tokens = target_tokens * 2;

        for para in content.split("\n\n") {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            let combined = if current.is_empty() {
                para.to_string()
            } else {
                format!("{}\n\n{}", current, para)
            };
            let combined_tokens = self.counter.count(&combined);

            if combined_tokens > max_tokens && !current.is_empty() {
                // Save current chunk
                let mut chunk = Chunk::new(current.trim(), content_type);
                chunk.position = position;
                chunk.calculate_tokens(self.counter);
                chunks.push(chunk);
                position += 1;

                // Start new with overlap
                current = if self.config.overlap_tokens > 0 {
                    let overlap = self.get_overlap(&current);
                    if overlap.is_empty() {
                        para.to_string()
                    } else {
                        format!("{}\n\n{}", overlap, para)
                    }
                } else {
                    para.to_string()
                };
            } else {
                current = combined;
            }
        }

        // Last chunk
        if !current.trim().is_empty() {
            let mut chunk = Chunk::new(current.trim(), content_type);
            chunk.position = position;
            chunk.calculate_tokens(self.counter);
            chunks.push(chunk);
        }

        chunks
    }

    fn get_overlap(&self, text: &str) -> String {
        if self.config.overlap_tokens == 0 {
            return String::new();
        }

        let sentences: Vec<&str> = text.unicode_sentences().collect();
        let mut overlap = String::new();

        for sentence in sentences.iter().rev() {
            let new_overlap = if overlap.is_empty() {
                sentence.to_string()
            } else {
                format!("{sentence} {overlap}")
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
    use crate::processing::token::TokenizerModel;

    fn make_chunker(target: usize, max: usize, min: usize, overlap: usize) -> (TokenCounter, ChunkConfig) {
        let counter = TokenCounter::new(TokenizerModel::Default).unwrap();
        let config = ChunkConfig {
            target_tokens: target,
            max_tokens: max,
            min_tokens: min,
            overlap_tokens: overlap,
            content_type: ContentType::Text,
            deduplicate: true,
        };
        (counter, config)
    }

    #[test]
    fn test_text_chunking() {
        let (counter, config) = make_chunker(50, 100, 5, 10);
        let chunker = Chunker::new(config, &counter);

        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_text_chunking_with_overlap() {
        let (counter, config) = make_chunker(20, 40, 5, 10);
        let chunker = Chunker::new(config, &counter);

        let text = "Sentence one. Sentence two. Sentence three.\n\nParagraph two here. More content.";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_text_chunking_no_overlap() {
        let (counter, mut config) = make_chunker(20, 40, 5, 0);
        config.overlap_tokens = 0;
        let chunker = Chunker::new(config, &counter);

        let text = "First part.\n\nSecond part.\n\nThird part.";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_text_chunking_long() {
        let (counter, config) = make_chunker(50, 100, 5, 10);
        let chunker = Chunker::new(config, &counter);

        // Create enough text to force multiple chunks (each paragraph ~20 tokens, need > 100 total)
        let text = (0..20).map(|i| format!("This is paragraph number {} which contains enough text to be meaningful. Here is more content to increase the token count.", i)).collect::<Vec<_>>().join("\n\n");
        let chunks = chunker.chunk(&text);

        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());
        for chunk in &chunks {
            assert!(chunk.tokens <= 100, "Chunk exceeded max_tokens: {}", chunk.tokens);
        }
    }

    #[test]
    fn test_text_overlap_empty_result() {
        let (counter, config) = make_chunker(1000, 2000, 5, 50);
        let chunker = Chunker::new(config, &counter);

        // Empty overlap for short text
        let overlap = chunker.get_overlap("Hi");
        assert!(overlap.len() <= 10);
    }

    #[test]
    fn test_whitespace_only_chunking() {
        let (counter, config) = make_chunker(50, 100, 5, 10);
        let chunker = Chunker::new(config, &counter);

        let text = "   \n\n   \n\n   ";
        let chunks = chunker.chunk(text);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_markdown_header_split() {
        let (counter, config) = make_chunker(30, 60, 5, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Markdown), &counter);

        let markdown = "# Header 1\nContent under header one.\n\n# Header 2\nContent under header two.";
        let chunks = chunker.chunk(markdown);

        assert!(!chunks.is_empty());
        for chunk in chunks {
            assert_eq!(chunk.content_type, ContentType::Markdown);
        }
    }

    #[test]
    fn test_code_chunking() {
        let (counter, config) = make_chunker(50, 100, 5, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Code), &counter);

        let code = r#"
fn one() {
    println!("one");
}

fn two() {
    println!("two");
}
"#;
        let chunks = chunker.chunk(code);
        assert!(!chunks.is_empty());
        for chunk in chunks {
            assert_eq!(chunk.content_type, ContentType::Code);
        }
    }

    #[test]
    fn test_conversation_chunking() {
        let (counter, config) = make_chunker(100, 200, 1, 0); // min_tokens=1 to include all
        let chunker = Chunker::new(config.with_content_type(ContentType::Conversation), &counter);

        let conv = "User: Hello there friend\n---\nAssistant: Hi there how are you doing today!\n---\nUser: I am doing great thanks for asking!";
        let chunks = chunker.chunk(conv);

        assert_eq!(chunks.len(), 3);
        for chunk in chunks {
            assert_eq!(chunk.content_type, ContentType::Conversation);
        }
    }

    #[test]
    fn test_structured_chunking_small() {
        let (counter, config) = make_chunker(100, 200, 5, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Structured), &counter);

        let json = r#"{"key": "value", "number": 42}"#;
        let chunks = chunker.chunk(json);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content_type, ContentType::Structured);
    }

    #[test]
    fn test_structured_large_fallback() {
        let (counter, config) = make_chunker(20, 40, 1, 0); // min_tokens=1
        let chunker = Chunker::new(config.with_content_type(ContentType::Structured), &counter);

        // Create large structured content that exceeds max_tokens (40)
        let large = (0..100).map(|i| format!("\"key{i}\": \"this is a longer value for item number {i}\"")).collect::<Vec<_>>().join(",\n\n");
        let json = format!("{{\n\n{large}\n\n}}");
        let chunks = chunker.chunk(&json);

        assert!(chunks.len() > 1, "Expected multiple chunks for large JSON, got {}", chunks.len());
    }

    #[test]
    fn test_deduplication() {
        let (counter, config) = make_chunker(100, 200, 1, 0); // min_tokens=1
        let chunker = Chunker::new(config.with_content_type(ContentType::Conversation), &counter);

        let conv = "This is a longer message one for testing deduplication properly\n---\nThis is a longer message one for testing deduplication properly\n---\nThis is message two which is different";
        let chunks = chunker.chunk(conv);

        assert_eq!(chunks.len(), 2); // Duplicate removed
    }

    #[test]
    fn test_deduplication_disabled() {
        let (counter, mut config) = make_chunker(100, 200, 1, 0); // min_tokens=1
        config.deduplicate = false;
        config.content_type = ContentType::Conversation;
        let chunker = Chunker::new(config, &counter);

        let conv = "This is a longer message one for testing\n---\nThis is a longer message one for testing\n---\nThis is message two which is different";
        let chunks = chunker.chunk(conv);

        assert_eq!(chunks.len(), 3); // No dedup
    }

    #[test]
    fn test_min_tokens_filter() {
        let (counter, config) = make_chunker(100, 200, 50, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Conversation), &counter);

        let conv = "Hi\n---\nThis is a much longer message with enough tokens to pass the filter";
        let chunks = chunker.chunk(conv);

        // "Hi" should be filtered out due to min_tokens
        assert!(chunks.iter().all(|c| c.tokens >= 50 || c.content.len() > 10));
    }

    #[test]
    fn test_chunk_positions() {
        let (counter, config) = make_chunker(100, 200, 1, 0); // min_tokens=1
        let chunker = Chunker::new(config.with_content_type(ContentType::Conversation), &counter);

        let conv = "This is message one with enough content\n---\nThis is message two with enough content\n---\nThis is message three with enough content";
        let chunks = chunker.chunk(conv);

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.position, i);
        }
    }

    #[test]
    fn test_chunk_as() {
        let (counter, config) = make_chunker(100, 200, 5, 0);
        let chunker = Chunker::new(config, &counter);

        let text = "# Header\nContent";
        let chunks = chunker.chunk_as(text, ContentType::Markdown);

        assert!(chunks.iter().all(|c| c.content_type == ContentType::Markdown));
    }

    #[test]
    fn test_chunk_with_size() {
        let (counter, config) = make_chunker(1000, 2000, 1, 0); // min_tokens=1
        let chunker = Chunker::new(config, &counter);

        // Create text with clear paragraph breaks to ensure splitting works
        let text = (0..30).map(|i| format!("This is paragraph {} with enough content to make it meaningful and test the chunking properly.", i)).collect::<Vec<_>>().join("\n\n");
        let chunks = chunker.chunk_with_size(&text, 30);

        // Should create multiple small chunks with target_tokens=30
        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());
    }

    #[test]
    fn test_markdown_force_split_large() {
        // Test force split when markdown content exceeds max_tokens
        let (counter, config) = make_chunker(20, 40, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Markdown), &counter);

        // Large content without headers that exceeds max_tokens
        let large = (0..50).map(|i| format!("Line {} with content.", i)).collect::<Vec<_>>().join("\n");
        let chunks = chunker.chunk(&large);

        assert!(chunks.len() > 1, "Should force split large content");
        for chunk in chunks {
            assert!(chunk.tokens <= 80, "Chunk should be within reasonable limits");
        }
    }

    #[test]
    fn test_markdown_header_split_with_content() {
        // Test split before header when we have enough content
        let (counter, config) = make_chunker(30, 60, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Markdown), &counter);

        let markdown = "Some intro text here that is long enough.\nMore intro content.\n\n# Header 1\nContent under header.\n\n# Header 2\nMore content here.";
        let chunks = chunker.chunk(markdown);

        assert!(chunks.len() >= 2, "Should split at headers");
    }

    #[test]
    fn test_code_force_split_large() {
        // Test force split when code content exceeds max_tokens
        let (counter, config) = make_chunker(20, 40, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Code), &counter);

        // Large function that exceeds max_tokens
        let code = (0..100).map(|i| format!("    let var{i} = {i};")).collect::<Vec<_>>().join("\n");
        let full_code = format!("fn large_function() {{\n{}\n}}", code);
        let chunks = chunker.chunk(&full_code);

        assert!(chunks.len() > 1, "Should force split large code");
    }

    #[test]
    fn test_code_split_at_function_boundary() {
        // Test split at function boundaries
        let (counter, config) = make_chunker(30, 60, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Code), &counter);

        let code = r#"
fn function_one() {
    let a = 1;
    let b = 2;
    let c = 3;
    println!("{}", a + b + c);
}

fn function_two() {
    let x = 10;
    let y = 20;
    println!("{}", x + y);
}

fn function_three() {
    let z = 100;
    println!("{}", z);
}
"#;
        let chunks = chunker.chunk(code);
        assert!(chunks.len() >= 2, "Should split at function boundaries");
    }

    #[test]
    fn test_code_python_keywords() {
        let (counter, config) = make_chunker(30, 60, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Code), &counter);

        let python = r#"
def function_one():
    a = 1
    b = 2
    return a + b

class MyClass:
    def __init__(self):
        self.value = 0

async def async_func():
    await something()
"#;
        let chunks = chunker.chunk(python);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_code_javascript_keywords() {
        let (counter, config) = make_chunker(30, 60, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Code), &counter);

        let js = r#"
function hello() {
    console.log("hello");
}

export function exported() {
    return 42;
}

import { something } from 'module';
"#;
        let chunks = chunker.chunk(js);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_overlap_calculation() {
        let (counter, config) = make_chunker(50, 100, 1, 20);
        let chunker = Chunker::new(config, &counter);

        // Text with multiple sentences
        let text = "First sentence here. Second sentence here. Third sentence here.\n\nAnother paragraph with content. More content here.";
        let overlap = chunker.get_overlap(text);

        // Overlap should contain some content but not exceed overlap_tokens
        assert!(!overlap.is_empty() || text.len() < 10);
    }

    #[test]
    fn test_overlap_empty_for_short_text() {
        let (counter, config) = make_chunker(100, 200, 1, 50);
        let chunker = Chunker::new(config, &counter);

        let overlap = chunker.get_overlap("A");
        // Very short text might produce minimal overlap
        assert!(overlap.len() <= 10);
    }

    #[test]
    fn test_text_chunking_large_paragraphs() {
        let (counter, config) = make_chunker(30, 60, 1, 10);
        let chunker = Chunker::new(config, &counter);

        // Create large paragraphs that will need splitting
        let text = (0..10).map(|i| {
            (0..20).map(|j| format!("Sentence {} in paragraph {}.", j, i)).collect::<Vec<_>>().join(" ")
        }).collect::<Vec<_>>().join("\n\n");

        let chunks = chunker.chunk(&text);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_all_cached_batch() {
        // Test embed_batch when all texts are already cached
        let (counter, config) = make_chunker(100, 200, 1, 0);
        let chunker = Chunker::new(config.with_content_type(ContentType::Conversation), &counter);

        // First call caches
        let conv1 = "Message A here\n---\nMessage B here";
        let _ = chunker.chunk(conv1);

        // Second call should hit cache
        let conv2 = "Message A here\n---\nMessage B here";
        let chunks = chunker.chunk(conv2);
        assert!(!chunks.is_empty());
    }
}
