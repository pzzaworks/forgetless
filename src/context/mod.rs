//! Context management and optimization
//!
//! The main module that brings everything together for intelligent context management.

use crate::chunking::{ChunkConfig, Chunker, ContentType};
use crate::memory::{ConversationMemory, MemoryConfig, Message, Role};
use crate::scoring::{Priority, RelevanceScore};
use crate::token::{TokenCounter, TokenizerModel};
use crate::Result;
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::HashMap;

/// Context item that can be added to the context window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    /// Unique identifier
    pub id: String,
    /// Content
    pub content: String,
    /// Content type
    pub content_type: ContentType,
    /// Token count
    pub tokens: usize,
    /// Relevance score
    pub score: RelevanceScore,
    /// Whether this item is pinned (always included)
    pub pinned: bool,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ContextItem {
    /// Create a new context item
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            content_type: ContentType::Text,
            tokens: 0,
            score: RelevanceScore::default(),
            pinned: false,
            metadata: HashMap::new(),
        }
    }

    /// Set content type
    pub fn with_type(mut self, content_type: ContentType) -> Self {
        self.content_type = content_type;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.score = RelevanceScore::new(priority);
        self
    }

    /// Pin this item (always include in context)
    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Configuration for context manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum tokens for the context window
    pub max_tokens: usize,
    /// Reserved tokens for the response
    pub reserved_for_response: usize,
    /// Model name for tokenization
    pub model: String,
    /// Tokenizer model
    pub tokenizer: TokenizerModel,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Chunking configuration
    pub chunk_config: ChunkConfig,
    /// Whether to include long-term memory
    pub include_long_term: bool,
    /// Maximum items from long-term memory
    pub max_long_term_items: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200_000, // Claude Sonnet 4.5 supports 200k context
            reserved_for_response: 8192,
            model: "claude-sonnet-4.5".to_string(),
            tokenizer: TokenizerModel::ClaudeSonnet45,
            memory_config: MemoryConfig::default(),
            chunk_config: ChunkConfig::default(),
            include_long_term: true,
            max_long_term_items: 10,
        }
    }
}

impl ContextConfig {
    /// Set maximum tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        self.tokenizer = match model.as_str() {
            // OpenAI GPT-5.3 Codex (latest)
            "gpt-5.3-codex" | "gpt-5.3" | "openai/gpt-5.3-codex" => TokenizerModel::Gpt53Codex,
            // OpenAI GPT-5.2
            "gpt-5.2" | "openai/gpt-5.2" => TokenizerModel::Gpt52,
            // OpenAI GPT-4o series
            "gpt-4o" | "gpt-4o-mini" | "openai/gpt-4o" => TokenizerModel::Gpt4o,
            // OpenAI gpt-oss-120b (open weight)
            "gpt-oss-120b" | "openai/gpt-oss-120b" => TokenizerModel::GptOss120b,
            // Anthropic Claude Opus 4.6 (latest)
            "claude-opus-4.6" | "claude-4.6" | "anthropic/claude-opus-4.6" => TokenizerModel::ClaudeOpus46,
            // Anthropic Claude Sonnet 4.5
            "claude-sonnet-4.5" | "claude-4.5" | "anthropic/claude-sonnet-4.5" => TokenizerModel::ClaudeSonnet45,
            // Anthropic Claude Haiku 4.5
            "claude-haiku-4.5" | "anthropic/claude-haiku-4.5" => TokenizerModel::ClaudeHaiku45,
            // Google Gemini 3
            "gemini-3" | "gemini-3-pro" | "gemini-3-flash" | "google/gemini-3-pro-preview" => TokenizerModel::Gemini3,
            // Google Gemini 2.5
            "gemini-2.5" | "gemini-2.5-flash-lite" | "google/gemini-2.5-flash-lite" => TokenizerModel::Gemini25,
            // xAI Grok 4.1
            "grok-4.1" | "grok-4.1-fast" | "x-ai/grok-4.1-fast" => TokenizerModel::Grok41,
            // DeepSeek V3.2
            "deepseek-v3.2" | "deepseek/deepseek-v3.2" => TokenizerModel::DeepSeekV32,
            // Qwen3 Coder
            "qwen3-coder" | "qwen/qwen3-coder-480b" => TokenizerModel::Qwen3Coder,
            // Meta Llama 4
            "llama-4" | "llama-4-405b" | "llama-4-70b" | "meta-llama/llama-4-405b" => TokenizerModel::Llama4,
            // Mistral
            "mistral-large" | "mistral-large-2" | "mistral/mistral-large-2" => TokenizerModel::MistralLarge,
            // Default to Claude Sonnet 4.5
            _ => TokenizerModel::ClaudeSonnet45,
        };
        self.model = model;
        self
    }

    /// Set reserved tokens for response
    pub fn with_reserved_tokens(mut self, tokens: usize) -> Self {
        self.reserved_for_response = tokens;
        self
    }

    /// Available tokens for context (max - reserved)
    pub fn available_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserved_for_response)
    }
}

/// Built context ready for use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltContext {
    /// System prompt
    pub system: Option<String>,
    /// Messages
    pub messages: Vec<Message>,
    /// Additional context items
    pub items: Vec<ContextItem>,
    /// Total tokens used
    pub total_tokens: usize,
    /// Available tokens
    pub available_tokens: usize,
    /// Items that were excluded due to budget
    pub excluded_count: usize,
}

impl BuiltContext {
    /// Get remaining tokens for response
    pub fn remaining_tokens(&self) -> usize {
        self.available_tokens.saturating_sub(self.total_tokens)
    }

    /// Check if context is within budget
    pub fn is_within_budget(&self) -> bool {
        self.total_tokens <= self.available_tokens
    }

    /// Get all content as a single string
    pub fn to_string(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref sys) = self.system {
            parts.push(format!("[System]\n{}", sys));
        }

        for msg in &self.messages {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                Role::Tool => "Tool",
            };
            parts.push(format!("[{}]\n{}", role, msg.content));
        }

        for item in &self.items {
            parts.push(format!("[Context: {}]\n{}", item.id, item.content));
        }

        parts.join("\n\n")
    }
}

/// Main context manager
pub struct ContextManager {
    /// Configuration
    config: ContextConfig,
    /// Token counter
    counter: TokenCounter,
    /// Conversation memory
    memory: ConversationMemory,
    /// Additional context items
    items: HashMap<String, ContextItem>,
    /// Chunker for processing documents
    chunker: Chunker,
}

impl ContextManager {
    /// Create a new context manager
    pub fn new(config: ContextConfig) -> Result<Self> {
        let counter = TokenCounter::new(config.tokenizer)?;
        let memory = ConversationMemory::new(config.memory_config.clone(),
            TokenCounter::new(config.tokenizer)?);
        let chunker = Chunker::new(config.chunk_config.clone(),
            TokenCounter::new(config.tokenizer)?);

        Ok(Self {
            config,
            counter,
            memory,
            items: HashMap::new(),
            chunker,
        })
    }

    /// Set system prompt
    pub fn set_system(&mut self, content: impl Into<String>) {
        self.memory.set_system_prompt(content);
    }

    /// Add a user message
    pub fn add_user(&mut self, content: impl Into<String>) -> Result<()> {
        self.memory.add_user(content)
    }

    /// Add an assistant message
    pub fn add_assistant(&mut self, content: impl Into<String>) -> Result<()> {
        self.memory.add_assistant(content)
    }

    /// Add a context item
    pub fn add_item(&mut self, mut item: ContextItem) {
        item.tokens = self.counter.count(&item.content);
        self.items.insert(item.id.clone(), item);
    }

    /// Add content with priority
    pub fn add(&mut self, id: impl Into<String>, content: impl Into<String>, priority: Priority) {
        let item = ContextItem::new(id, content).with_priority(priority);
        self.add_item(item);
    }

    /// Add a document (will be chunked)
    pub fn add_document(&mut self, id: impl Into<String>, content: &str, content_type: ContentType) {
        let id = id.into();
        let chunks = self.chunker.chunk(content);

        for (i, chunk) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}:{}", id, i);
            let item = ContextItem {
                id: chunk_id,
                content: chunk.content,
                content_type,
                tokens: chunk.tokens,
                score: chunk.score,
                pinned: false,
                metadata: chunk.metadata,
            };
            self.items.insert(item.id.clone(), item);
        }
    }

    /// Remove a context item
    pub fn remove_item(&mut self, id: &str) -> Option<ContextItem> {
        self.items.remove(id)
    }

    /// Pin an item (will always be included)
    pub fn pin_item(&mut self, id: &str) -> bool {
        if let Some(item) = self.items.get_mut(id) {
            item.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpin an item
    pub fn unpin_item(&mut self, id: &str) -> bool {
        if let Some(item) = self.items.get_mut(id) {
            item.pinned = false;
            true
        } else {
            false
        }
    }

    /// Build optimized context within token budget
    pub fn build(&self) -> Result<BuiltContext> {
        let available = self.config.available_tokens();
        let mut total_tokens = 0;
        let mut excluded_count = 0;

        // Start with system prompt and messages
        let messages: Vec<Message> = self
            .memory
            .get_messages_within_budget(available)
            .into_iter()
            .cloned()
            .collect();

        total_tokens += messages.iter().map(|m| m.tokens).sum::<usize>();

        // Get system separately
        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone());

        // Filter out system from messages
        let messages: Vec<Message> = messages
            .into_iter()
            .filter(|m| m.role != Role::System)
            .collect();

        // Remaining budget for items
        let remaining = available.saturating_sub(total_tokens);

        // Build priority queue of items
        let mut queue: PriorityQueue<String, Reverse<i64>> = PriorityQueue::new();

        for (id, item) in &self.items {
            let priority = if item.pinned {
                i64::MAX
            } else {
                (item.score.final_score() * 1000.0) as i64
            };
            queue.push(id.clone(), Reverse(-priority)); // Reverse for max-heap behavior
        }

        // Select items within budget
        let mut selected_items = Vec::new();
        let mut items_budget = remaining;

        // First, add all pinned items
        for (_id, item) in &self.items {
            if item.pinned {
                if item.tokens <= items_budget {
                    selected_items.push(item.clone());
                    items_budget -= item.tokens;
                    total_tokens += item.tokens;
                } else {
                    excluded_count += 1;
                }
            }
        }

        // Then add highest priority items
        while let Some((id, _)) = queue.pop() {
            if let Some(item) = self.items.get(&id) {
                if item.pinned {
                    continue; // Already added
                }
                if item.tokens <= items_budget {
                    selected_items.push(item.clone());
                    items_budget -= item.tokens;
                    total_tokens += item.tokens;
                } else {
                    excluded_count += 1;
                }
            }
        }

        Ok(BuiltContext {
            system,
            messages,
            items: selected_items,
            total_tokens,
            available_tokens: available,
            excluded_count,
        })
    }

    /// Get current token usage
    pub fn token_usage(&self) -> usize {
        let memory_tokens = self.memory.total_tokens();
        let items_tokens: usize = self.items.values().map(|i| i.tokens).sum();
        memory_tokens + items_tokens
    }

    /// Get available token budget
    pub fn available_budget(&self) -> usize {
        self.config.available_tokens().saturating_sub(self.token_usage())
    }

    /// Clear all context (keeps system prompt)
    pub fn clear(&mut self) {
        self.memory.clear();
        self.items.clear();
    }

    /// Get config reference
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Get token counter reference
    pub fn token_counter(&self) -> &TokenCounter {
        &self.counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenizerModel;

    // ContextItem tests
    #[test]
    fn test_context_item_new() {
        let item = ContextItem::new("id1", "content");
        assert_eq!(item.id, "id1");
        assert_eq!(item.content, "content");
        assert_eq!(item.content_type, ContentType::Text);
        assert!(!item.pinned);
    }

    #[test]
    fn test_context_item_with_type() {
        let item = ContextItem::new("id1", "code").with_type(ContentType::Code);
        assert_eq!(item.content_type, ContentType::Code);
    }

    #[test]
    fn test_context_item_with_priority() {
        let item = ContextItem::new("id1", "content").with_priority(Priority::Critical);
        assert_eq!(item.score.priority, Priority::Critical);
    }

    #[test]
    fn test_context_item_pinned() {
        let item = ContextItem::new("id1", "content").pinned();
        assert!(item.pinned);
    }

    #[test]
    fn test_context_item_with_metadata() {
        let item = ContextItem::new("id1", "content")
            .with_metadata("key", "value")
            .with_metadata("key2", "value2");
        assert_eq!(item.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(item.metadata.get("key2"), Some(&"value2".to_string()));
    }

    // ContextConfig tests
    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();
        assert_eq!(config.max_tokens, 200_000);
        assert_eq!(config.reserved_for_response, 8192);
        assert_eq!(config.model, "claude-sonnet-4.5");
    }

    #[test]
    fn test_context_config() {
        let config = ContextConfig::default()
            .with_max_tokens(16000)
            .with_model("claude-opus-4.6");

        assert_eq!(config.max_tokens, 16000);
        assert_eq!(config.model, "claude-opus-4.6");
    }

    #[test]
    fn test_context_config_with_reserved_tokens() {
        let config = ContextConfig::default().with_reserved_tokens(2000);
        assert_eq!(config.reserved_for_response, 2000);
    }

    #[test]
    fn test_context_config_available_tokens() {
        let config = ContextConfig::default()
            .with_max_tokens(10000)
            .with_reserved_tokens(2000);
        assert_eq!(config.available_tokens(), 8000);
    }

    #[test]
    fn test_context_config_model_tokenizer_mapping() {
        let gpt53 = ContextConfig::default().with_model("gpt-5.3-codex");
        assert_eq!(gpt53.tokenizer, TokenizerModel::Gpt53Codex);

        let gpt4o = ContextConfig::default().with_model("gpt-4o");
        assert_eq!(gpt4o.tokenizer, TokenizerModel::Gpt4o);

        let claude_opus = ContextConfig::default().with_model("claude-opus-4.6");
        assert_eq!(claude_opus.tokenizer, TokenizerModel::ClaudeOpus46);

        let claude_sonnet = ContextConfig::default().with_model("claude-sonnet-4.5");
        assert_eq!(claude_sonnet.tokenizer, TokenizerModel::ClaudeSonnet45);

        let gemini = ContextConfig::default().with_model("gemini-3-pro");
        assert_eq!(gemini.tokenizer, TokenizerModel::Gemini3);

        let deepseek = ContextConfig::default().with_model("deepseek-v3.2");
        assert_eq!(deepseek.tokenizer, TokenizerModel::DeepSeekV32);

        let grok = ContextConfig::default().with_model("grok-4.1-fast");
        assert_eq!(grok.tokenizer, TokenizerModel::Grok41);

        let llama = ContextConfig::default().with_model("llama-4-405b");
        assert_eq!(llama.tokenizer, TokenizerModel::Llama4);

        let unknown = ContextConfig::default().with_model("unknown-model");
        assert_eq!(unknown.tokenizer, TokenizerModel::ClaudeSonnet45); // Default
    }

    // BuiltContext tests
    #[test]
    fn test_built_context_remaining_tokens() {
        let context = BuiltContext {
            system: None,
            messages: vec![],
            items: vec![],
            total_tokens: 100,
            available_tokens: 1000,
            excluded_count: 0,
        };
        assert_eq!(context.remaining_tokens(), 900);
    }

    #[test]
    fn test_built_context_is_within_budget() {
        let within = BuiltContext {
            system: None,
            messages: vec![],
            items: vec![],
            total_tokens: 100,
            available_tokens: 1000,
            excluded_count: 0,
        };
        assert!(within.is_within_budget());

        let over = BuiltContext {
            system: None,
            messages: vec![],
            items: vec![],
            total_tokens: 1500,
            available_tokens: 1000,
            excluded_count: 0,
        };
        assert!(!over.is_within_budget());
    }

    #[test]
    fn test_built_context_to_string() {
        let context = BuiltContext {
            system: Some("System prompt".to_string()),
            messages: vec![
                Message::new(Role::User, "Hello"),
                Message::new(Role::Assistant, "Hi"),
            ],
            items: vec![],
            total_tokens: 50,
            available_tokens: 1000,
            excluded_count: 0,
        };

        let s = context.to_string();
        assert!(s.contains("System prompt"));
        assert!(s.contains("[User]"));
        assert!(s.contains("[Assistant]"));
    }

    // ContextManager tests
    #[test]
    fn test_context_manager() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        manager.set_system("You are a helpful assistant.");
        manager.add_user("Hello!").unwrap();
        manager.add_assistant("Hi there!").unwrap();

        manager.add("doc1", "Some important context", Priority::High);

        let context = manager.build().unwrap();
        assert!(context.is_within_budget());
        assert!(context.total_tokens > 0);
    }

    #[test]
    fn test_context_manager_add_item() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        let item = ContextItem::new("id1", "content")
            .with_priority(Priority::High)
            .with_type(ContentType::Code);
        manager.add_item(item);

        let context = manager.build().unwrap();
        assert_eq!(context.items.len(), 1);
        assert_eq!(context.items[0].id, "id1");
    }

    #[test]
    fn test_context_manager_remove_item() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        manager.add("id1", "content", Priority::High);
        assert!(manager.remove_item("id1").is_some());
        assert!(manager.remove_item("id1").is_none()); // Already removed

        let context = manager.build().unwrap();
        assert_eq!(context.items.len(), 0);
    }

    #[test]
    fn test_context_manager_pin_unpin() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        manager.add("id1", "content", Priority::Low);

        assert!(manager.pin_item("id1"));
        assert!(!manager.pin_item("nonexistent"));

        assert!(manager.unpin_item("id1"));
        assert!(!manager.unpin_item("nonexistent"));
    }

    #[test]
    fn test_context_manager_add_document() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        let long_doc = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        manager.add_document("doc", long_doc, ContentType::Markdown);

        let context = manager.build().unwrap();
        assert!(!context.items.is_empty());
    }

    #[test]
    fn test_context_manager_token_usage() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        let initial = manager.token_usage();
        manager.add("id1", "Some content here", Priority::High);
        let after = manager.token_usage();

        assert!(after > initial);
    }

    #[test]
    fn test_context_manager_available_budget() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        let initial = manager.available_budget();
        assert_eq!(initial, 7000);

        manager.add("id1", "Some content", Priority::High);
        let after = manager.available_budget();
        assert!(after < initial);
    }

    #[test]
    fn test_context_manager_clear() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        manager.add_user("Hello").unwrap();
        manager.add("id1", "content", Priority::High);

        manager.clear();

        let context = manager.build().unwrap();
        assert_eq!(context.messages.len(), 0);
        assert_eq!(context.items.len(), 0);
    }

    #[test]
    fn test_context_manager_config_access() {
        let config = ContextConfig::default().with_max_tokens(10000);
        let manager = ContextManager::new(config).unwrap();

        assert_eq!(manager.config().max_tokens, 10000);
    }

    #[test]
    fn test_context_manager_token_counter_access() {
        let config = ContextConfig::default();
        let manager = ContextManager::new(config).unwrap();

        let count = manager.token_counter().count("Hello");
        assert!(count > 0);
    }

    #[test]
    fn test_item_prioritization() {
        let config = ContextConfig::default()
            .with_max_tokens(4000)
            .with_reserved_tokens(500);
        let mut manager = ContextManager::new(config).unwrap();

        manager.add("low", "Low priority content", Priority::Low);
        manager.add("high", "High priority content", Priority::High);
        manager.add("critical", "Critical content", Priority::Critical);

        let context = manager.build().unwrap();

        // All should fit in available budget
        assert_eq!(context.items.len(), 3);
    }

    #[test]
    fn test_pinned_items_always_included() {
        let config = ContextConfig::default()
            .with_max_tokens(4000)
            .with_reserved_tokens(500);
        let mut manager = ContextManager::new(config).unwrap();

        manager.add("low", "Low priority but pinned", Priority::Minimal);
        manager.pin_item("low");

        manager.add("high", "High priority", Priority::Critical);

        let context = manager.build().unwrap();

        // Pinned item should be included
        assert!(context.items.iter().any(|i| i.id == "low"));
    }

    #[test]
    fn test_items_excluded_when_over_budget() {
        let config = ContextConfig::default()
            .with_max_tokens(100)
            .with_reserved_tokens(50);
        let mut manager = ContextManager::new(config).unwrap();

        // Add items that exceed the budget
        for i in 0..10 {
            manager.add(
                format!("item{}", i),
                "This is some content that takes tokens".repeat(5),
                Priority::Medium,
            );
        }

        let context = manager.build().unwrap();

        // Some items should be excluded
        assert!(context.excluded_count > 0);
    }

    #[test]
    fn test_built_context_with_system() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        manager.set_system("You are helpful.");
        manager.add_user("Hello").unwrap();

        let context = manager.build().unwrap();

        assert!(context.system.is_some());
        assert_eq!(context.system.unwrap(), "You are helpful.");
    }

    #[test]
    fn test_built_context_messages_exclude_system() {
        let config = ContextConfig::default()
            .with_max_tokens(8000)
            .with_reserved_tokens(1000);
        let mut manager = ContextManager::new(config).unwrap();

        manager.set_system("System");
        manager.add_user("User message").unwrap();
        manager.add_assistant("Assistant message").unwrap();

        let context = manager.build().unwrap();

        // Messages should not include system (it's in context.system)
        assert!(context.messages.iter().all(|m| m.role != Role::System));
        assert_eq!(context.messages.len(), 2);
    }

    #[test]
    fn test_context_item_to_string_with_items() {
        let item = ContextItem::new("doc1", "Document content").with_priority(Priority::High);

        let context = BuiltContext {
            system: None,
            messages: vec![],
            items: vec![item],
            total_tokens: 10,
            available_tokens: 1000,
            excluded_count: 0,
        };

        let s = context.to_string();
        assert!(s.contains("[Context: doc1]"));
        assert!(s.contains("Document content"));
    }

    #[test]
    fn test_built_context_to_string_all_roles() {
        // Test that all role types are properly formatted in to_string
        let context = BuiltContext {
            system: Some("System prompt".to_string()),
            messages: vec![
                Message::new(Role::User, "User message"),
                Message::new(Role::Assistant, "Assistant message"),
                Message::new(Role::Tool, "Tool output"),
            ],
            items: vec![],
            total_tokens: 50,
            available_tokens: 1000,
            excluded_count: 0,
        };

        let s = context.to_string();
        assert!(s.contains("[System]"));
        assert!(s.contains("[User]"));
        assert!(s.contains("[Assistant]"));
        assert!(s.contains("[Tool]"));
    }

    #[test]
    fn test_pinned_item_exceeds_budget() {
        // Test that pinned items that exceed budget are excluded and counted
        let config = ContextConfig::default()
            .with_max_tokens(50)
            .with_reserved_tokens(40); // Only 10 tokens available
        let mut manager = ContextManager::new(config).unwrap();

        // Add a pinned item that's too large
        let large_content = "This is a very long piece of content that will definitely exceed the tiny token budget we have set for this test";
        manager.add("large", large_content, Priority::Critical);
        manager.pin_item("large");

        let context = manager.build().unwrap();

        // The pinned item should be excluded due to budget
        assert_eq!(context.excluded_count, 1);
    }
}
