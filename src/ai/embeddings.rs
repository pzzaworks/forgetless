//! Embedding module for semantic similarity
//!
//! Uses FastEmbed for local, fast ML embeddings with LRU caching.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};
use xxhash_rust::xxh3::xxh3_64;

use crate::core::error::{Error, Result};

// ============================================================================
// Cosine Similarity
// ============================================================================

/// Calculate cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ============================================================================
// Embedding Cache
// ============================================================================

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in cache
    pub size: usize,
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f32,
}

/// LRU cache for embedding vectors
pub struct EmbeddingCache {
    cache: LruCache<u64, Vec<f32>>,
    hits: usize,
    misses: usize,
}

impl EmbeddingCache {
    /// Create a new cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            cache: LruCache::new(cap),
            hits: 0,
            misses: 0,
        }
    }

    /// Get an embedding from the cache
    pub fn get(&mut self, text: &str) -> Option<Vec<f32>> {
        let key = self.hash_key(text);
        match self.cache.get(&key) {
            Some(embedding) => {
                self.hits += 1;
                Some(embedding.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert an embedding into the cache
    pub fn insert(&mut self, text: &str, embedding: Vec<f32>) {
        let key = self.hash_key(text);
        self.cache.put(key, embedding);
    }

    /// Get the cache hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f32 / total as f32
    }

    /// Get the number of cached entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
        }
    }

    fn hash_key(&self, text: &str) -> u64 {
        xxh3_64(text.as_bytes())
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(10_000)
    }
}

// ============================================================================
// Global Embedder Singleton
// ============================================================================

/// Global embedding model - loaded once per process
static GLOBAL_EMBEDDER: OnceLock<Mutex<GlobalEmbedder>> = OnceLock::new();

/// Thread-safe global embedder with caching
pub struct GlobalEmbedder {
    model: TextEmbedding,
    cache: EmbeddingCache,
}

impl GlobalEmbedder {
    fn new() -> std::result::Result<Self, String> {
        // Use quantized model for speed
        let init_options =
            InitOptions::new(EmbeddingModel::AllMiniLML6V2Q).with_show_download_progress(false); // Quiet mode

        let model = TextEmbedding::try_new(init_options)
            .map_err(|e| format!("Failed to initialize fastembed: {e}"))?;

        Ok(Self {
            model,
            cache: EmbeddingCache::new(10_000),
        })
    }

    /// Embed a single text
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(embedding) = self.cache.get(text) {
            return Ok(embedding);
        }

        // Generate embedding
        let embeddings = self
            .model
            .embed(vec![text.to_string()], None)
            .map_err(|e| Error::ContextBuildError(format!("Embedding failed: {e}")))?;

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::ContextBuildError("No embedding returned".to_string()))?;

        // Cache and return
        self.cache.insert(text, embedding.clone());
        Ok(embedding)
    }

    /// Embed a batch of texts
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Separate cached and uncached
        let mut results = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.cache.get(text) {
                results[i] = Some(embedding);
            } else {
                uncached_indices.push(i);
                uncached_texts.push(text.to_string());
            }
        }

        // Embed uncached texts
        if !uncached_texts.is_empty() {
            let new_embeddings = self
                .model
                .embed(uncached_texts, None)
                .map_err(|e| Error::ContextBuildError(format!("Embedding failed: {e}")))?;

            for (idx, embedding) in uncached_indices.into_iter().zip(new_embeddings) {
                self.cache.insert(texts[idx], embedding.clone());
                results[idx] = Some(embedding);
            }
        }

        // Unwrap all results
        results
            .into_iter()
            .map(|r| r.ok_or_else(|| Error::ContextBuildError("Missing embedding".to_string())))
            .collect()
    }
}

/// Initialize the global embedder (call once at startup if you want to control timing)
fn init_global_embedder() -> &'static Mutex<GlobalEmbedder> {
    GLOBAL_EMBEDDER.get_or_init(|| match GlobalEmbedder::new() {
        Ok(embedder) => Mutex::new(embedder),
        Err(e) => panic!("Failed to initialize global embedder: {e}"),
    })
}

/// Quick function to embed using global singleton
pub fn embed_text(text: &str) -> Result<Vec<f32>> {
    let embedder = init_global_embedder();
    let mut guard = embedder
        .lock()
        .map_err(|e| Error::ContextBuildError(format!("Lock poisoned: {e}")))?;
    guard.embed(text)
}

/// Quick function to batch embed using global singleton
pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let embedder = init_global_embedder();
    let mut guard = embedder
        .lock()
        .map_err(|e| Error::ContextBuildError(format!("Lock poisoned: {e}")))?;
    guard.embed_batch(texts)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Cosine similarity tests
    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // Cache tests
    #[test]
    fn test_cache_insert_get() {
        let mut cache = EmbeddingCache::new(100);
        cache.insert("hello", vec![1.0, 2.0, 3.0]);
        let result = cache.get("hello");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = EmbeddingCache::new(100);
        let result = cache.get("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = EmbeddingCache::new(100);
        cache.insert("a", vec![1.0]);
        cache.insert("b", vec![2.0]);
        cache.get("a"); // Hit
        cache.get("b"); // Hit
        cache.get("c"); // Miss
        let rate = cache.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = EmbeddingCache::new(2);
        cache.insert("a", vec![1.0]);
        cache.insert("b", vec![2.0]);
        cache.insert("c", vec![3.0]); // Should evict "a"
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = EmbeddingCache::new(100);
        cache.insert("a", vec![1.0]);
        cache.get("a");
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = EmbeddingCache::new(100);
        cache.insert("a", vec![1.0]);
        cache.get("a"); // Hit
        cache.get("b"); // Miss
        let stats = cache.stats();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_len() {
        let mut cache = EmbeddingCache::new(100);
        assert_eq!(cache.len(), 0);
        cache.insert("a", vec![1.0]);
        assert_eq!(cache.len(), 1);
        cache.insert("b", vec![2.0]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_default() {
        let cache = EmbeddingCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector_a() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector_b() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_partial() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // cos(45°) ≈ 0.707
        assert!(sim > 0.7 && sim < 0.72);
    }

    #[test]
    fn test_embed_text_basic() {
        // This test requires the fastembed model to be available
        let result = embed_text("hello world");
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert!(!embedding.is_empty());
        // all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_embed_batch_basic() {
        let result = embed_batch(&["hello", "world"]);
        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
        for emb in embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    fn test_embed_text_caching() {
        // First embed
        let result1 = embed_text("cached test text");
        assert!(result1.is_ok());

        // Second embed of same text should hit cache
        let result2 = embed_text("cached test text");
        assert!(result2.is_ok());

        // Results should be identical
        assert_eq!(result1.unwrap(), result2.unwrap());
    }

    #[test]
    fn test_embed_batch_with_duplicates() {
        let result = embed_batch(&["same", "same", "different"]);
        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
        // First two should be identical
        assert_eq!(embeddings[0], embeddings[1]);
        // Third should be different
        assert_ne!(embeddings[0], embeddings[2]);
    }

    #[test]
    fn test_embed_empty_string() {
        let result = embed_text("");
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 384);
    }
}
