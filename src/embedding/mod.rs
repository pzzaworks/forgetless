//! Embedding and semantic similarity
//!
//! Provides embedding-based semantic scoring for intelligent context selection.

use crate::scoring::{Priority, RelevanceScore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedding vector type (f32 for efficiency)
pub type Embedding = Vec<f32>;

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModel {
    /// OpenAI text-embedding-3-large (3072 dimensions)
    OpenAILarge,
    /// OpenAI text-embedding-3-small (1536 dimensions)
    OpenAISmall,
    /// Anthropic Voyage (1024 dimensions)
    Voyage,
    /// Cohere embed-v4 (1024 dimensions)
    CohereV4,
    /// Local/custom model
    Custom {
        /// Embedding vector dimensions
        dimensions: usize,
    },
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::OpenAISmall
    }
}

impl EmbeddingModel {
    /// Get the embedding dimensions for this model
    pub fn dimensions(&self) -> usize {
        match self {
            EmbeddingModel::OpenAILarge => 3072,
            EmbeddingModel::OpenAISmall => 1536,
            EmbeddingModel::Voyage => 1024,
            EmbeddingModel::CohereV4 => 1024,
            EmbeddingModel::Custom { dimensions } => *dimensions,
        }
    }
}

/// Embedded content item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedItem {
    /// Unique identifier
    pub id: String,
    /// Original content
    pub content: String,
    /// Embedding vector
    pub embedding: Embedding,
    /// Token count
    pub tokens: usize,
    /// Base priority
    pub priority: Priority,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl EmbeddedItem {
    /// Create a new embedded item
    pub fn new(id: impl Into<String>, content: impl Into<String>, embedding: Embedding) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding,
            tokens: 0,
            priority: Priority::Medium,
            metadata: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set token count
    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.tokens = tokens;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Semantic similarity calculator
pub struct SemanticScorer {
    /// Query embedding for comparison
    query_embedding: Option<Embedding>,
    /// Similarity threshold (0.0 - 1.0)
    threshold: f32,
    /// Weight for semantic score in final ranking
    semantic_weight: f32,
}

impl SemanticScorer {
    /// Create a new semantic scorer
    pub fn new() -> Self {
        Self {
            query_embedding: None,
            threshold: 0.3,
            semantic_weight: 0.5,
        }
    }

    /// Set the query embedding
    pub fn with_query(mut self, embedding: Embedding) -> Self {
        self.query_embedding = Some(embedding);
        self
    }

    /// Set similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set semantic weight in final score
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.semantic_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Calculate semantic similarity for an item
    pub fn score(&self, item: &EmbeddedItem) -> f32 {
        match &self.query_embedding {
            Some(query) => Self::cosine_similarity(query, &item.embedding),
            None => 0.5, // Default neutral score
        }
    }

    /// Check if item passes the similarity threshold
    pub fn passes_threshold(&self, item: &EmbeddedItem) -> bool {
        self.score(item) >= self.threshold
    }

    /// Calculate combined relevance score
    pub fn relevance_score(&self, item: &EmbeddedItem, base_score: &RelevanceScore) -> f32 {
        let semantic = self.score(item);
        let base = base_score.final_score();

        // Weighted combination
        (1.0 - self.semantic_weight) * base + self.semantic_weight * (semantic * 100.0)
    }

    /// Rank items by semantic similarity
    pub fn rank_by_similarity<'a>(&self, items: &'a [EmbeddedItem]) -> Vec<&'a EmbeddedItem> {
        let mut scored: Vec<_> = items.iter().map(|item| (item, self.score(item))).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(item, _)| item).collect()
    }

    /// Filter items above threshold and rank by similarity
    pub fn filter_and_rank<'a>(&self, items: &'a [EmbeddedItem]) -> Vec<&'a EmbeddedItem> {
        let mut scored: Vec<_> = items
            .iter()
            .filter_map(|item| {
                let score = self.score(item);
                if score >= self.threshold {
                    Some((item, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(item, _)| item).collect()
    }
}

impl Default for SemanticScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize an embedding vector to unit length
pub fn normalize(embedding: &mut Embedding) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in embedding.iter_mut() {
            *val /= norm;
        }
    }
}

/// Average multiple embeddings
pub fn average_embeddings(embeddings: &[Embedding]) -> Option<Embedding> {
    if embeddings.is_empty() {
        return None;
    }

    let dim = embeddings[0].len();
    if embeddings.iter().any(|e| e.len() != dim) {
        return None;
    }

    let mut result = vec![0.0f32; dim];
    let count = embeddings.len() as f32;

    for embedding in embeddings {
        for (i, val) in embedding.iter().enumerate() {
            result[i] += val / count;
        }
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_model_dimensions() {
        assert_eq!(EmbeddingModel::OpenAILarge.dimensions(), 3072);
        assert_eq!(EmbeddingModel::OpenAISmall.dimensions(), 1536);
        assert_eq!(EmbeddingModel::Voyage.dimensions(), 1024);
        assert_eq!(EmbeddingModel::CohereV4.dimensions(), 1024);
        assert_eq!(
            EmbeddingModel::Custom { dimensions: 512 }.dimensions(),
            512
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = SemanticScorer::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = SemanticScorer::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = SemanticScorer::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let mut embedding = vec![3.0, 4.0];
        normalize(&mut embedding);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_average_embeddings() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let avg = average_embeddings(&embeddings).unwrap();
        assert_eq!(avg, vec![2.0, 3.0]);
    }

    #[test]
    fn test_semantic_scorer() {
        let query = vec![1.0, 0.0, 0.0];
        let scorer = SemanticScorer::new().with_query(query).with_threshold(0.5);

        let similar = EmbeddedItem::new("1", "similar", vec![0.9, 0.1, 0.0]);
        let different = EmbeddedItem::new("2", "different", vec![0.0, 1.0, 0.0]);

        assert!(scorer.passes_threshold(&similar));
        assert!(!scorer.passes_threshold(&different));
    }

    #[test]
    fn test_rank_by_similarity() {
        let query = vec![1.0, 0.0];
        let scorer = SemanticScorer::new().with_query(query);

        let items = vec![
            EmbeddedItem::new("1", "far", vec![0.0, 1.0]),
            EmbeddedItem::new("2", "close", vec![0.9, 0.1]),
            EmbeddedItem::new("3", "medium", vec![0.5, 0.5]),
        ];

        let ranked = scorer.rank_by_similarity(&items);
        assert_eq!(ranked[0].id, "2"); // closest
        assert_eq!(ranked[2].id, "1"); // farthest
    }

    #[test]
    fn test_embedded_item_builder() {
        let item = EmbeddedItem::new("id", "content", vec![1.0, 2.0])
            .with_priority(Priority::High)
            .with_tokens(100)
            .with_metadata("key", "value");

        assert_eq!(item.id, "id");
        assert_eq!(item.priority, Priority::High);
        assert_eq!(item.tokens, 100);
        assert_eq!(item.metadata.get("key"), Some(&"value".to_string()));
    }
}
