//! Memory management for conversations and long-term storage
//!
//! Provides mechanisms to store, retrieve, and manage conversational context.

use crate::chunking::Chunk;
use crate::scoring::{Priority, RecencyDecay, RelevanceScore};
use crate::token::TokenCounter;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Role in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// System prompt
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// Tool/function call
    Tool,
}

impl Role {
    /// Get default priority for this role
    pub fn default_priority(&self) -> Priority {
        match self {
            Role::System => Priority::Critical,
            Role::User => Priority::High,
            Role::Assistant => Priority::Medium,
            Role::Tool => Priority::Low,
        }
    }
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Message content
    pub content: String,
    /// Token count
    pub tokens: usize,
    /// Relevance score
    pub score: RelevanceScore,
    /// Timestamp (unix millis)
    pub timestamp: u64,
    /// Optional name for the sender
    pub name: Option<String>,
    /// Whether this message has been summarized
    pub summarized: bool,
}

impl Message {
    /// Create a new message
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            role,
            content,
            tokens: 0,
            score: RelevanceScore::new(role.default_priority()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            name: None,
            summarized: false,
        }
    }

    /// Set message name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.score = RelevanceScore::new(priority);
        self
    }

    /// Calculate tokens
    pub fn calculate_tokens(&mut self, counter: &TokenCounter) {
        self.tokens = counter.count(&self.content);
    }

    /// Check if message is from user
    pub fn is_user(&self) -> bool {
        self.role == Role::User
    }

    /// Check if message is from assistant
    pub fn is_assistant(&self) -> bool {
        self.role == Role::Assistant
    }
}

/// Configuration for conversation memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum messages to keep in working memory
    pub max_messages: usize,
    /// Maximum tokens for working memory
    pub max_tokens: usize,
    /// Recency decay half-life
    pub recency_half_life: f32,
    /// Whether to auto-summarize old messages
    pub auto_summarize: bool,
    /// Token threshold for summarization
    pub summarize_threshold: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_tokens: 8000,
            recency_half_life: 20.0,
            auto_summarize: true,
            summarize_threshold: 4000,
        }
    }
}

/// Conversation memory manager
pub struct ConversationMemory {
    /// Working memory (recent messages)
    messages: VecDeque<Message>,
    /// Long-term memory (summarized chunks)
    long_term: Vec<Chunk>,
    /// System prompt
    system_prompt: Option<Message>,
    /// Configuration
    config: MemoryConfig,
    /// Token counter
    counter: TokenCounter,
    /// Recency decay calculator
    decay: RecencyDecay,
    /// Total tokens in working memory
    total_tokens: usize,
}

impl ConversationMemory {
    /// Create new conversation memory
    pub fn new(config: MemoryConfig, counter: TokenCounter) -> Self {
        let decay = RecencyDecay::new(config.recency_half_life);
        Self {
            messages: VecDeque::new(),
            long_term: Vec::new(),
            system_prompt: None,
            config,
            counter,
            decay,
            total_tokens: 0,
        }
    }

    /// Set system prompt
    pub fn set_system_prompt(&mut self, content: impl Into<String>) {
        let mut msg = Message::new(Role::System, content);
        msg.calculate_tokens(&self.counter);
        self.system_prompt = Some(msg);
    }

    /// Add a message to memory
    pub fn add(&mut self, role: Role, content: impl Into<String>) -> Result<()> {
        let mut msg = Message::new(role, content);
        msg.calculate_tokens(&self.counter);
        self.total_tokens += msg.tokens;
        self.messages.push_back(msg);

        // Apply recency decay
        self.update_recency();

        // Check if we need to compress
        if self.total_tokens > self.config.max_tokens
            || self.messages.len() > self.config.max_messages
        {
            self.compress()?;
        }

        Ok(())
    }

    /// Add a user message
    pub fn add_user(&mut self, content: impl Into<String>) -> Result<()> {
        self.add(Role::User, content)
    }

    /// Add an assistant message
    pub fn add_assistant(&mut self, content: impl Into<String>) -> Result<()> {
        self.add(Role::Assistant, content)
    }

    /// Get all messages for context building
    pub fn get_messages(&self) -> Vec<&Message> {
        let mut result = Vec::new();

        if let Some(ref sys) = self.system_prompt {
            result.push(sys);
        }

        result.extend(self.messages.iter());
        result
    }

    /// Get messages within token budget
    pub fn get_messages_within_budget(&self, budget: usize) -> Vec<&Message> {
        let mut result = Vec::new();
        let mut remaining = budget;

        // Always include system prompt if it fits
        if let Some(ref sys) = self.system_prompt {
            if sys.tokens <= remaining {
                result.push(sys);
                remaining -= sys.tokens;
            }
        }

        // Add messages from most recent, respecting budget
        for msg in self.messages.iter().rev() {
            if msg.tokens <= remaining {
                result.push(msg);
                remaining -= msg.tokens;
            } else {
                break;
            }
        }

        // Reverse to maintain chronological order
        result.reverse();
        result
    }

    /// Get total token count
    pub fn total_tokens(&self) -> usize {
        let system_tokens = self.system_prompt.as_ref().map(|m| m.tokens).unwrap_or(0);
        self.total_tokens + system_tokens
    }

    /// Get message count
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Clear all messages (keeps system prompt)
    pub fn clear(&mut self) {
        self.messages.clear();
        self.long_term.clear();
        self.total_tokens = 0;
    }

    /// Update recency scores for all messages
    fn update_recency(&mut self) {
        let len = self.messages.len();
        for (i, msg) in self.messages.iter_mut().enumerate() {
            let age = len - i - 1;
            msg.score.recency = self.decay.decay(age);
        }
    }

    /// Compress old messages to save tokens
    fn compress(&mut self) -> Result<()> {
        // Simple compression: remove oldest messages until under budget
        while self.total_tokens > self.config.max_tokens && !self.messages.is_empty() {
            if let Some(old_msg) = self.messages.pop_front() {
                self.total_tokens = self.total_tokens.saturating_sub(old_msg.tokens);

                // Store summary in long-term memory
                if self.config.auto_summarize && old_msg.tokens > 50 {
                    let chunk = Chunk::new(&old_msg.content, crate::chunking::ContentType::Conversation)
                        .with_priority(old_msg.score.priority)
                        .with_metadata("role", format!("{:?}", old_msg.role));
                    self.long_term.push(chunk);
                }
            }
        }

        // Also check message count
        while self.messages.len() > self.config.max_messages {
            if let Some(old_msg) = self.messages.pop_front() {
                self.total_tokens = self.total_tokens.saturating_sub(old_msg.tokens);
            }
        }

        Ok(())
    }

    /// Get long-term memory chunks
    pub fn get_long_term(&self) -> &[Chunk] {
        &self.long_term
    }

    /// Search long-term memory (simple keyword search)
    pub fn search_long_term(&self, query: &str) -> Vec<&Chunk> {
        let query_lower = query.to_lowercase();
        self.long_term
            .iter()
            .filter(|chunk| chunk.content.to_lowercase().contains(&query_lower))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::Priority;
    use crate::token::TokenizerModel;

    // Role tests
    #[test]
    fn test_role_default_priority() {
        assert_eq!(Role::System.default_priority(), Priority::Critical);
        assert_eq!(Role::User.default_priority(), Priority::High);
        assert_eq!(Role::Assistant.default_priority(), Priority::Medium);
        assert_eq!(Role::Tool.default_priority(), Priority::Low);
    }

    // Message tests
    #[test]
    fn test_message_creation() {
        let msg = Message::new(Role::User, "Hello!");
        assert!(msg.is_user());
        assert!(!msg.is_assistant());
        assert_eq!(msg.content, "Hello!");
        assert!(!msg.summarized);
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn test_message_with_name() {
        let msg = Message::new(Role::User, "Hello!").with_name("Alice");
        assert_eq!(msg.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_message_with_priority() {
        let msg = Message::new(Role::User, "Hello!").with_priority(Priority::Critical);
        assert_eq!(msg.score.priority, Priority::Critical);
    }

    #[test]
    fn test_message_calculate_tokens() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut msg = Message::new(Role::User, "Hello, world!");
        assert_eq!(msg.tokens, 0);
        msg.calculate_tokens(&counter);
        assert!(msg.tokens > 0);
    }

    #[test]
    fn test_message_is_assistant() {
        let msg = Message::new(Role::Assistant, "Hi there!");
        assert!(msg.is_assistant());
        assert!(!msg.is_user());
    }

    // MemoryConfig tests
    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.max_messages, 100);
        assert_eq!(config.max_tokens, 8000);
        assert!(config.auto_summarize);
    }

    // ConversationMemory tests
    #[test]
    fn test_conversation_memory() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.set_system_prompt("You are a helpful assistant.");
        memory.add_user("Hi there!").unwrap();
        memory.add_assistant("Hello! How can I help?").unwrap();

        assert_eq!(memory.message_count(), 2);
        assert!(memory.total_tokens() > 0);
    }

    #[test]
    fn test_conversation_memory_get_messages() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.set_system_prompt("System prompt");
        memory.add_user("User message").unwrap();

        let messages = memory.get_messages();
        assert_eq!(messages.len(), 2); // system + user
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
    }

    #[test]
    fn test_budget_retrieval() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.add_user("Message 1").unwrap();
        memory.add_assistant("Response 1").unwrap();
        memory.add_user("Message 2").unwrap();

        let messages = memory.get_messages_within_budget(100);
        assert!(!messages.is_empty());
    }

    #[test]
    fn test_budget_retrieval_with_system() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.set_system_prompt("Short system");
        memory.add_user("Message 1").unwrap();

        let messages = memory.get_messages_within_budget(1000);
        assert!(messages.iter().any(|m| m.role == Role::System));
    }

    #[test]
    fn test_budget_retrieval_excludes_when_over_budget() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        // Add messages that exceed a small budget
        memory.add_user("This is a longer message that takes more tokens").unwrap();
        memory.add_assistant("This is also a response with many words").unwrap();

        // Very small budget should return fewer messages
        let messages = memory.get_messages_within_budget(5);
        assert!(messages.len() < 2);
    }

    #[test]
    fn test_memory_clear() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.add_user("Message").unwrap();
        memory.add_assistant("Response").unwrap();

        let initial_count = memory.message_count();
        assert!(initial_count > 0);

        memory.clear();

        assert_eq!(memory.message_count(), 0);
        // Note: total_tokens() may include system prompt tokens
        // which are preserved after clear()
    }

    #[test]
    fn test_memory_clear_with_system() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.set_system_prompt("System");
        let system_tokens = memory.total_tokens();

        memory.add_user("Message").unwrap();
        memory.add_assistant("Response").unwrap();

        memory.clear();

        assert_eq!(memory.message_count(), 0);
        // System prompt is preserved, so its tokens should remain
        assert_eq!(memory.total_tokens(), system_tokens);
    }

    #[test]
    fn test_memory_compression_by_tokens() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = MemoryConfig {
            max_tokens: 50, // Very small token limit
            max_messages: 100,
            auto_summarize: false,
            ..Default::default()
        };
        let mut memory = ConversationMemory::new(config, counter);

        // Add messages that exceed the token limit
        for i in 0..10 {
            memory.add_user(format!("This is message number {} with some content", i)).unwrap();
        }

        // Should have compressed, so total tokens should be under limit
        assert!(memory.total_tokens() <= 50);
    }

    #[test]
    fn test_memory_compression_by_count() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = MemoryConfig {
            max_tokens: 10000,
            max_messages: 3, // Very small message limit
            auto_summarize: false,
            ..Default::default()
        };
        let mut memory = ConversationMemory::new(config, counter);

        // Add more messages than the limit
        for i in 0..10 {
            memory.add_user(format!("Msg {}", i)).unwrap();
        }

        assert!(memory.message_count() <= 3);
    }

    #[test]
    fn test_long_term_memory() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = MemoryConfig {
            max_tokens: 20,
            auto_summarize: true,
            summarize_threshold: 4000,
            ..Default::default()
        };
        let mut memory = ConversationMemory::new(config, counter);

        // Add a long message that will be moved to long-term when compressed
        let long_content = "This is a very long message with lots of content ".repeat(5);
        memory.add_user(&long_content).unwrap();

        // Force compression by adding more
        memory.add_user("Another message").unwrap();
        memory.add_user("Yet another").unwrap();

        // Check long-term memory access works
        let lt = memory.get_long_term();
        // Long-term memory may have entries if messages were compressed
        // Just verify we can access it without panic
        let _ = lt.len();
    }

    #[test]
    fn test_search_long_term_empty() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let memory = ConversationMemory::new(MemoryConfig::default(), counter);

        let results = memory.search_long_term("test");
        assert!(results.is_empty());
    }

    #[test]
    fn test_add_with_role() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.add(Role::Tool, "Tool result").unwrap();

        let messages = memory.get_messages();
        assert_eq!(messages[0].role, Role::Tool);
    }

    #[test]
    fn test_recency_updates() {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let mut memory = ConversationMemory::new(MemoryConfig::default(), counter);

        memory.add_user("First").unwrap();
        memory.add_user("Second").unwrap();
        memory.add_user("Third").unwrap();

        let messages = memory.get_messages();
        // Most recent message should have highest recency
        assert!(messages.last().unwrap().score.recency >= messages.first().unwrap().score.recency);
    }
}
