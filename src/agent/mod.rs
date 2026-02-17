//! Agent Memory Patterns
//!
//! Implements cognitive-inspired memory architecture for AI agents:
//! - Working Memory: Current task context, limited capacity
//! - Episodic Memory: Past interactions and events
//! - Semantic Memory: Long-term knowledge and facts

use crate::scoring::{Priority, RelevanceScore};
use crate::token::TokenCounter;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Short-term, high-relevance, limited capacity
    Working,
    /// Event-based, timestamped experiences
    Episodic,
    /// Factual knowledge, concepts, relationships
    Semantic,
}

impl Default for MemoryType {
    fn default() -> Self {
        Self::Working
    }
}

/// A memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,
    /// Memory type
    pub memory_type: MemoryType,
    /// Content
    pub content: String,
    /// Token count
    pub tokens: usize,
    /// Relevance score
    pub score: RelevanceScore,
    /// Timestamp (unix millis)
    pub timestamp: u64,
    /// Access count (for importance calculation)
    pub access_count: u32,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Linked memory IDs
    pub links: Vec<String>,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        memory_type: MemoryType,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: id.into(),
            memory_type,
            content: content.into(),
            tokens: 0,
            score: RelevanceScore::new(Priority::Medium),
            timestamp: now,
            access_count: 0,
            last_accessed: now,
            metadata: HashMap::new(),
            links: Vec::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.score = RelevanceScore::new(priority);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Link to another memory
    pub fn with_link(mut self, memory_id: impl Into<String>) -> Self {
        self.links.push(memory_id.into());
        self
    }

    /// Calculate tokens
    pub fn calculate_tokens(&mut self, counter: &TokenCounter) {
        self.tokens = counter.count(&self.content);
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.score.add_reference();
    }

    /// Calculate importance score (for consolidation decisions)
    pub fn importance(&self) -> f32 {
        let recency = self.score.recency;
        let access_factor = (self.access_count as f32).ln_1p() * 0.1;
        let priority_factor = self.score.priority.score() as f32 / 100.0;

        recency * 0.3 + access_factor * 0.3 + priority_factor * 0.4
    }
}

/// Agent memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemoryConfig {
    /// Maximum tokens for working memory
    pub working_memory_tokens: usize,
    /// Maximum entries in working memory
    pub working_memory_size: usize,
    /// Maximum entries in episodic memory
    pub episodic_memory_size: usize,
    /// Maximum entries in semantic memory
    pub semantic_memory_size: usize,
    /// Importance threshold for consolidation
    pub consolidation_threshold: f32,
    /// Enable automatic consolidation
    pub auto_consolidate: bool,
}

impl Default for AgentMemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_tokens: 8000,
            working_memory_size: 20,
            episodic_memory_size: 1000,
            semantic_memory_size: 500,
            consolidation_threshold: 0.5,
            auto_consolidate: true,
        }
    }
}

/// Agent memory system
pub struct AgentMemory {
    /// Working memory (current context)
    working: VecDeque<MemoryEntry>,
    /// Episodic memory (past events)
    episodic: Vec<MemoryEntry>,
    /// Semantic memory (knowledge)
    semantic: HashMap<String, MemoryEntry>,
    /// Configuration
    config: AgentMemoryConfig,
    /// Token counter
    counter: TokenCounter,
    /// Current working memory token usage
    working_tokens: usize,
}

impl AgentMemory {
    /// Create a new agent memory system
    pub fn new(config: AgentMemoryConfig, counter: TokenCounter) -> Self {
        Self {
            working: VecDeque::new(),
            episodic: Vec::new(),
            semantic: HashMap::new(),
            config,
            counter,
            working_tokens: 0,
        }
    }

    /// Add to working memory
    pub fn add_working(&mut self, mut entry: MemoryEntry) {
        entry.memory_type = MemoryType::Working;
        entry.calculate_tokens(&self.counter);

        // Check if we need to make room
        while self.working_tokens + entry.tokens > self.config.working_memory_tokens
            || self.working.len() >= self.config.working_memory_size
        {
            if let Some(old) = self.working.pop_front() {
                self.working_tokens = self.working_tokens.saturating_sub(old.tokens);

                // Optionally move to episodic memory
                if self.config.auto_consolidate && old.importance() >= self.config.consolidation_threshold {
                    self.add_episodic(old);
                }
            } else {
                break;
            }
        }

        self.working_tokens += entry.tokens;
        self.working.push_back(entry);
    }

    /// Add to episodic memory
    pub fn add_episodic(&mut self, mut entry: MemoryEntry) {
        entry.memory_type = MemoryType::Episodic;
        entry.calculate_tokens(&self.counter);

        // Enforce size limit
        if self.episodic.len() >= self.config.episodic_memory_size {
            // Remove least important entry
            if let Some(min_idx) = self
                .episodic
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.importance()
                        .partial_cmp(&b.importance())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.episodic.remove(min_idx);
            }
        }

        self.episodic.push(entry);
    }

    /// Add to semantic memory (knowledge)
    pub fn add_semantic(&mut self, mut entry: MemoryEntry) {
        entry.memory_type = MemoryType::Semantic;
        entry.calculate_tokens(&self.counter);

        // Semantic memory uses id as key, overwrites existing
        self.semantic.insert(entry.id.clone(), entry);

        // Enforce size limit
        while self.semantic.len() > self.config.semantic_memory_size {
            // Remove least important
            if let Some(key) = self
                .semantic
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.importance()
                        .partial_cmp(&b.importance())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone())
            {
                self.semantic.remove(&key);
            }
        }
    }

    /// Get working memory entries
    pub fn get_working(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.working.iter()
    }

    /// Get episodic memory entries
    pub fn get_episodic(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.episodic.iter()
    }

    /// Get semantic memory entries
    pub fn get_semantic(&self) -> impl Iterator<Item = &MemoryEntry> {
        self.semantic.values()
    }

    /// Get a specific semantic memory by id
    pub fn get_knowledge(&self, id: &str) -> Option<&MemoryEntry> {
        self.semantic.get(id)
    }

    /// Search memories by content (simple keyword search)
    pub fn search(&self, query: &str, memory_types: &[MemoryType]) -> Vec<&MemoryEntry> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for memory_type in memory_types {
            match memory_type {
                MemoryType::Working => {
                    results.extend(
                        self.working
                            .iter()
                            .filter(|e| e.content.to_lowercase().contains(&query_lower)),
                    );
                }
                MemoryType::Episodic => {
                    results.extend(
                        self.episodic
                            .iter()
                            .filter(|e| e.content.to_lowercase().contains(&query_lower)),
                    );
                }
                MemoryType::Semantic => {
                    results.extend(
                        self.semantic
                            .values()
                            .filter(|e| e.content.to_lowercase().contains(&query_lower)),
                    );
                }
            }
        }

        // Sort by importance
        results.sort_by(|a, b| {
            b.importance()
                .partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Get recent episodic memories
    pub fn get_recent_episodes(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut episodes: Vec<_> = self.episodic.iter().collect();
        episodes.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        episodes.into_iter().take(limit).collect()
    }

    /// Consolidate: move important working memories to episodic/semantic
    pub fn consolidate(&mut self) {
        let to_consolidate: Vec<_> = self
            .working
            .iter()
            .filter(|e| e.importance() >= self.config.consolidation_threshold)
            .cloned()
            .collect();

        for entry in to_consolidate {
            self.add_episodic(entry);
        }
    }

    /// Clear working memory
    pub fn clear_working(&mut self) {
        if self.config.auto_consolidate {
            self.consolidate();
        }
        self.working.clear();
        self.working_tokens = 0;
    }

    /// Get total token usage
    pub fn total_tokens(&self) -> usize {
        let episodic: usize = self.episodic.iter().map(|e| e.tokens).sum();
        let semantic: usize = self.semantic.values().map(|e| e.tokens).sum();
        self.working_tokens + episodic + semantic
    }

    /// Get working memory token usage
    pub fn working_tokens(&self) -> usize {
        self.working_tokens
    }

    /// Get memory counts
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.working.len(), self.episodic.len(), self.semantic.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenizerModel;

    fn create_memory() -> AgentMemory {
        let counter = TokenCounter::new(TokenizerModel::Gpt4o).unwrap();
        let config = AgentMemoryConfig {
            working_memory_tokens: 1000,
            working_memory_size: 5,
            episodic_memory_size: 10,
            semantic_memory_size: 10,
            ..Default::default()
        };
        AgentMemory::new(config, counter)
    }

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("id1", "content", MemoryType::Working)
            .with_priority(Priority::High)
            .with_metadata("key", "value");

        assert_eq!(entry.id, "id1");
        assert_eq!(entry.memory_type, MemoryType::Working);
        assert_eq!(entry.score.priority, Priority::High);
    }

    #[test]
    fn test_working_memory() {
        let mut memory = create_memory();

        memory.add_working(MemoryEntry::new("1", "First entry", MemoryType::Working));
        memory.add_working(MemoryEntry::new("2", "Second entry", MemoryType::Working));

        let working: Vec<_> = memory.get_working().collect();
        assert_eq!(working.len(), 2);
    }

    #[test]
    fn test_working_memory_overflow() {
        let mut memory = create_memory();

        // Add more than limit
        for i in 0..10 {
            memory.add_working(MemoryEntry::new(
                format!("{}", i),
                "Entry content",
                MemoryType::Working,
            ));
        }

        // Should be limited to max size
        let working: Vec<_> = memory.get_working().collect();
        assert!(working.len() <= 5);
    }

    #[test]
    fn test_episodic_memory() {
        let mut memory = create_memory();

        memory.add_episodic(MemoryEntry::new("ep1", "Episode 1", MemoryType::Episodic));
        memory.add_episodic(MemoryEntry::new("ep2", "Episode 2", MemoryType::Episodic));

        let episodes: Vec<_> = memory.get_episodic().collect();
        assert_eq!(episodes.len(), 2);
    }

    #[test]
    fn test_semantic_memory() {
        let mut memory = create_memory();

        memory.add_semantic(
            MemoryEntry::new("fact1", "The sky is blue", MemoryType::Semantic)
                .with_priority(Priority::High),
        );

        let fact = memory.get_knowledge("fact1");
        assert!(fact.is_some());
        assert_eq!(fact.unwrap().content, "The sky is blue");
    }

    #[test]
    fn test_search() {
        let mut memory = create_memory();

        memory.add_working(MemoryEntry::new("1", "Hello world", MemoryType::Working));
        memory.add_episodic(MemoryEntry::new("2", "Goodbye world", MemoryType::Episodic));
        memory.add_semantic(MemoryEntry::new("3", "World facts", MemoryType::Semantic));

        let results = memory.search("world", &[MemoryType::Working, MemoryType::Episodic]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_importance_calculation() {
        let mut entry = MemoryEntry::new("1", "content", MemoryType::Working)
            .with_priority(Priority::Critical);

        entry.record_access();
        entry.record_access();

        let importance = entry.importance();
        assert!(importance > 0.0);
    }

    #[test]
    fn test_memory_counts() {
        let mut memory = create_memory();

        memory.add_working(MemoryEntry::new("1", "w", MemoryType::Working));
        memory.add_episodic(MemoryEntry::new("2", "e", MemoryType::Episodic));
        memory.add_semantic(MemoryEntry::new("3", "s", MemoryType::Semantic));

        let (w, e, s) = memory.counts();
        assert_eq!(w, 1);
        assert_eq!(e, 1);
        assert_eq!(s, 1);
    }

    #[test]
    fn test_clear_working() {
        let mut memory = create_memory();

        memory.add_working(MemoryEntry::new("1", "entry", MemoryType::Working));
        assert!(memory.get_working().count() > 0);

        memory.clear_working();
        assert_eq!(memory.get_working().count(), 0);
    }
}
