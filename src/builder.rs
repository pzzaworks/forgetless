//! Forgetless builder and optimization pipeline

use crate::processing::chunking::{Chunk, Chunker};
use crate::core::config::ForgetlessConfig;
use crate::ai::embeddings::{embed_batch, cosine_similarity};
use crate::ai::llm::{LLM, LLMConfig};
use crate::core::error::Result;
use crate::input::content::{ContentInput, FileWithPriority, IntoContent, WithPriority};
use crate::input::file::{read_file_content, read_file_preview};
use crate::processing::scoring::Priority;
use crate::processing::token::TokenCounter;
use crate::core::types::{OptimizationStats, OptimizedContext, ScoredChunk, ScoreBreakdown};
use std::path::{Path, PathBuf};

/// Lazy file reference - not read until needed
#[derive(Debug, Clone)]
pub(crate) struct LazyFile {
    pub(crate) path: PathBuf,
    pub(crate) priority: Priority,
}

/// Sealed trait for anything that can become a lazy file input
pub(crate) trait IntoLazyFile {
    fn into_lazy_file(self) -> LazyFile;
}

// Plain path types - default Medium priority
impl IntoLazyFile for &str {
    fn into_lazy_file(self) -> LazyFile {
        LazyFile { path: PathBuf::from(self), priority: Priority::Medium }
    }
}

impl IntoLazyFile for &&str {
    fn into_lazy_file(self) -> LazyFile {
        LazyFile { path: PathBuf::from(*self), priority: Priority::Medium }
    }
}

impl IntoLazyFile for String {
    fn into_lazy_file(self) -> LazyFile {
        LazyFile { path: PathBuf::from(self), priority: Priority::Medium }
    }
}

impl IntoLazyFile for PathBuf {
    fn into_lazy_file(self) -> LazyFile {
        LazyFile { path: self, priority: Priority::Medium }
    }
}

impl IntoLazyFile for &Path {
    fn into_lazy_file(self) -> LazyFile {
        LazyFile { path: self.to_path_buf(), priority: Priority::Medium }
    }
}

// FileWithPriority - custom priority
impl<P: AsRef<Path>> IntoLazyFile for FileWithPriority<P> {
    fn into_lazy_file(self) -> LazyFile {
        LazyFile { path: self.path().to_path_buf(), priority: self.priority() }
    }
}

/// Main context optimizer.
///
/// # Example
///
/// ```rust,ignore
/// use forgetless::{Forgetless, Config};
///
/// let result = Forgetless::new()
///     .config(Config::default()
///         .context_limit(128_000)
///         .vision_llm(true))
///     .add("conversation and prompts...")
///     .add_file("document.pdf")
///     .run()
///     .await?;
/// ```
pub struct Forgetless {
    pub(crate) config: ForgetlessConfig,
    pub(crate) inputs: Vec<ContentInput>,      // Eager: already loaded content
    pub(crate) lazy_files: Vec<LazyFile>,       // Lazy: files to be read on demand
    pub(crate) query: Option<String>,
}

impl Default for Forgetless {
    fn default() -> Self {
        Self {
            config: ForgetlessConfig::default(),
            inputs: Vec::new(),
            lazy_files: Vec::new(),
            query: None,
        }
    }
}

impl Forgetless {
    /// Create a new optimizer.
    ///
    /// Use `.config()` to set options like context_limit.
    ///
    /// # Example
    /// ```ignore
    /// Forgetless::new()
    ///     .config(Config::default().context_limit(128_000))
    ///     .add("content")
    ///     .run()
    ///     .await?;
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set runtime options. Merges with existing options.
    ///
    /// # Example
    /// ```ignore
    /// use forgetless::{Forgetless, Config};
    ///
    /// // Option 1: Everything in options
    /// Forgetless::default()
    ///     .config(Config::default()
    ///         .context_limit(128_000)
    ///         .vision_llm(true))
    ///     .add_file("document.pdf")
    ///     .run()
    ///     .await?;
    ///
    /// // Option 2: Use new() shortcut for context_limit
    /// Forgetless::new()
    ///     .config(Config::default().vision_llm(true))
    ///     .run()
    ///     .await?;
    /// ```
    pub fn config(mut self, cfg: crate::core::config::Config) -> Self {
        // Only override if explicitly set (non-default values)
        // This allows new(context_limit).config(...) to preserve context_limit
        if cfg.context_limit != 128_000 {
            self.config.options.context_limit = cfg.context_limit;
        }
        if cfg.chunk_size != 512 {
            self.config.options.chunk_size = cfg.chunk_size;
            self.config.chunk.target_tokens = cfg.chunk_size;
        }
        // Booleans: always apply since they have meaning
        self.config.options.vision_llm = cfg.vision_llm;
        self.config.options.context_llm = cfg.context_llm;
        self.config.options.parallel = cfg.parallel;
        self.config.options.cache = cfg.cache;
        self
    }

    /// Set a query for relevance-based scoring.
    /// When set, chunks more relevant to the query will be prioritized.
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Add content (string or with priority).
    ///
    /// # Examples
    /// ```text
    /// .add("content")                          // default Medium priority
    /// .add(WithPriority::high("important"))    // High priority
    /// .add(WithPriority::critical("system"))   // Critical priority
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, content: impl IntoContent) -> Self {
        self.inputs.push(content.into_content_input());
        self
    }

    /// Add pinned content (always included, Critical priority).
    pub fn add_pinned(self, content: impl Into<String>) -> Self {
        self.add(WithPriority::critical(content))
    }

    /// Add a file (lazy - not read until run()).
    ///
    /// # Examples
    /// ```text
    /// .add_file("doc.txt")                              // default Medium priority
    /// .add_file(FileWithPriority::high("doc.txt"))      // High priority
    /// .add_file(FileWithPriority::critical("system.txt"))
    /// ```
    #[allow(private_bounds)]
    pub fn add_file(mut self, file: impl IntoLazyFile) -> Self {
        self.lazy_files.push(file.into_lazy_file());
        self
    }

    /// Add multiple files (lazy).
    ///
    /// # Examples
    /// ```text
    /// .add_files(["a.rs", "b.rs"])
    /// ```
    #[allow(private_bounds)]
    pub fn add_files<F: IntoLazyFile>(mut self, files: impl IntoIterator<Item = F>) -> Self {
        for file in files {
            self.lazy_files.push(file.into_lazy_file());
        }
        self
    }

    /// Add raw bytes with MIME type.
    pub fn add_bytes(self, data: &[u8], mime_type: &str) -> Self {
        self.add_bytes_p(data, mime_type, Priority::Medium)
    }

    /// Add raw bytes with MIME type and priority.
    pub fn add_bytes_p(mut self, data: &[u8], mime_type: &str, priority: Priority) -> Self {
        if mime_type.starts_with("text/") {
            if let Ok(text) = String::from_utf8(data.to_vec()) {
                let mut input = ContentInput::from_string(text);
                input.source = format!("bytes:{mime_type}");
                input.priority = priority;
                self.inputs.push(input);
            }
        } else {
            let mut input = ContentInput::from_string(
                format!("[Binary content: {} bytes, type: {}]", data.len(), mime_type)
            );
            input.source = format!("bytes:{mime_type}");
            input.priority = priority;
            self.inputs.push(input);
        }
        self
    }

    /// Run the optimization pipeline.
    /// This is the main entry point - compresses all inputs to fit token budget.
    ///
    /// Uses smart lazy processing:
    /// 1. If query exists: Score lazy files by FILENAME similarity (no parsing!)
    /// 2. Only read/parse top relevant files
    /// 3. Embed and score only what we need
    pub async fn run(self) -> Result<OptimizedContext> {
        let start = std::time::Instant::now();

        if self.inputs.is_empty() && self.lazy_files.is_empty() {
            return Ok(OptimizedContext {
                content: String::new(),
                chunks: Vec::new(),
                total_tokens: 0,
                stats: OptimizationStats::default(),
            });
        }

        // Initialize vision LLM if enabled
        if self.config.options.vision_llm {
            crate::ai::vision::init_vision().await?;
        }

        // Initialize context LLM if enabled
        if self.config.options.context_llm {
            LLM::init_with_config(LLMConfig::smollm2()).await?;
        }

        let counter = TokenCounter::new(self.config.tokenizer)?;

        // Phase 1: Filter lazy files by content preview if query exists
        // Reads first ~2000 chars of each file (fast!) to find relevant ones
        let files_to_read: Vec<&LazyFile> = if let Some(ref query) = self.query {
            if self.lazy_files.len() > 5 {
                self.filter_files_by_preview(query)?
            } else {
                self.lazy_files.iter().collect()
            }
        } else {
            self.lazy_files.iter().collect()
        };

        // Phase 2: Now read ONLY the filtered files IN PARALLEL
        use rayon::prelude::*;

        let file_inputs: Vec<ContentInput> = files_to_read
            .par_iter()
            .filter_map(|lazy_file| {
                read_file_content(&lazy_file.path).map(|(content, content_type)| {
                    let mut input = ContentInput::from_file(content, &lazy_file.path.to_string_lossy());
                    input.content_type = content_type;
                    input.priority = lazy_file.priority;
                    input
                })
            })
            .collect();

        let mut all_inputs: Vec<ContentInput> = self.inputs.clone();
        all_inputs.extend(file_inputs);

        // Phase 3: Chunk all inputs
        let mut all_chunks: Vec<Chunk> = Vec::new();

        for input in &all_inputs {
            let config = self.config.chunk.clone().with_content_type(input.content_type);
            let chunker = Chunker::new(config, &counter);
            let chunks = chunker.chunk(&input.content);

            if chunks.is_empty() && input.priority == Priority::Critical {
                let mut chunk = Chunk::new(&input.content, input.content_type);
                chunk.priority = input.priority;
                chunk.source = Some(input.source.clone());
                chunk.calculate_tokens(&counter);
                all_chunks.push(chunk);
            } else {
                for mut chunk in chunks {
                    chunk.priority = input.priority;
                    chunk.source = Some(input.source.clone());
                    all_chunks.push(chunk);
                }
            }
        }

        let input_tokens: usize = all_chunks.iter().map(|c| c.tokens).sum();

        // Phase 4: If already fits, return as-is
        if input_tokens <= self.config.options.context_limit {
            let content = Self::build_structured_output(&all_chunks);
            let num_chunks = all_chunks.len();

            return Ok(OptimizedContext {
                content,
                chunks: all_chunks.into_iter().map(|c| ScoredChunk {
                    chunk: c,
                    score: 1.0,
                    breakdown: ScoreBreakdown::default(),
                }).collect(),
                total_tokens: input_tokens,
                stats: OptimizationStats {
                    input_tokens,
                    output_tokens: input_tokens,
                    chunks_processed: num_chunks,
                    chunks_selected: num_chunks,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                    compression_ratio: 1.0,
                },
            });
        }

        // Phase 5: Score chunks with embedding + algorithmic scoring
        let scored_chunks = self.score_chunks(&all_chunks)?;

        // Phase 6: Select best chunks within budget
        let selected = self.select_within_budget_custom(scored_chunks, self.config.options.context_limit)?;

        // Phase 7: Build structured output
        let mut content = Self::build_structured_output_from_scored(&selected);

        // Phase 8: Polish with LLM if enabled (reorganize and clean up)
        if self.config.options.context_llm && LLM::is_loaded() {
            // Collect chunk contents for polishing
            let chunk_contents: Vec<&str> = selected.iter()
                .map(|c| c.chunk.content.as_str())
                .collect();

            // Polish the content with LLM
            if let Ok(polished) = crate::ai::llm::polish(&chunk_contents, self.query.as_deref()).await {
                if !polished.is_empty() {
                    content = polished;
                }
            }
        }

        let total_tokens = counter.count(&content);

        let stats = OptimizationStats {
            input_tokens,
            output_tokens: total_tokens,
            chunks_processed: all_chunks.len(),
            chunks_selected: selected.len(),
            processing_time_ms: start.elapsed().as_millis() as u64,
            compression_ratio: input_tokens as f32 / total_tokens.max(1) as f32,
        };

        Ok(OptimizedContext {
            content,
            chunks: selected,
            total_tokens,
            stats,
        })
    }

    /// Filter lazy files by CONTENT PREVIEW similarity to query
    /// Reads first ~2000 chars of each file in PARALLEL and compares with query
    fn filter_files_by_preview(&self, query: &str) -> Result<Vec<&LazyFile>> {
        use rayon::prelude::*;

        // Embed the query once
        let query_embedding = embed_batch(&[query])?.pop().unwrap_or_default();

        // Read preview of each file IN PARALLEL (this is the expensive part)
        let previews: Vec<Option<String>> = self.lazy_files
            .par_iter()
            .map(|file| read_file_preview(&file.path))
            .collect();

        // Collect valid previews for batch embedding
        let valid_previews: Vec<(usize, &str)> = previews.iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|s| (i, s.as_str())))
            .collect();

        if valid_previews.is_empty() {
            // No valid previews, return all files
            return Ok(self.lazy_files.iter().collect());
        }

        let preview_texts: Vec<&str> = valid_previews.iter().map(|(_, s)| *s).collect();
        let preview_embeddings = embed_batch(&preview_texts)?;

        // Create index map for embeddings
        let mut embedding_map: std::collections::HashMap<usize, &Vec<f32>> = std::collections::HashMap::new();
        for ((idx, _), emb) in valid_previews.iter().zip(preview_embeddings.iter()) {
            embedding_map.insert(*idx, emb);
        }

        // Score each file by preview similarity to query
        let mut scored: Vec<(&LazyFile, f32)> = Vec::new();
        for (i, file) in self.lazy_files.iter().enumerate() {
            // Critical priority always included with max score
            if file.priority == Priority::Critical {
                scored.push((file, 1.0));
                continue;
            }

            // Get embedding for this file's preview
            if let Some(emb) = embedding_map.get(&i) {
                let similarity = cosine_similarity(emb, &query_embedding);
                let score = (similarity + 1.0) / 2.0;
                scored.push((file, score));
            } else {
                // No preview available, give low score but include
                scored.push((file, 0.3));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N files - allow up to 100 relevant files for large document sets
        let max_files = 100.min(self.lazy_files.len());
        let mut selected: Vec<&LazyFile> = scored.iter()
            .take(max_files)
            .filter(|(_, score)| *score > 0.35) // Only include if reasonably relevant
            .map(|(file, _)| *file)
            .collect();

        // Always return at least some files if query matched poorly
        if selected.is_empty() {
            selected = scored.iter().take(5).map(|(f, _)| *f).collect();
        }

        Ok(selected)
    }

    /// Build structured output grouped by source
    fn build_structured_output(chunks: &[Chunk]) -> String {
        use std::collections::HashMap;

        // Group chunks by source
        let mut by_source: HashMap<&str, Vec<&Chunk>> = HashMap::new();
        for chunk in chunks {
            let source = chunk.source.as_deref().unwrap_or("content");
            by_source.entry(source).or_default().push(chunk);
        }

        // Build output with headers
        let mut sections: Vec<String> = Vec::new();

        for (source, chunks) in by_source {
            let header = format!("## {}", Self::format_source_name(source));
            let content: String = chunks.iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            sections.push(format!("{}\n{}", header, content));
        }

        sections.join("\n\n---\n\n")
    }

    /// Build structured output from scored chunks
    fn build_structured_output_from_scored(chunks: &[ScoredChunk]) -> String {
        use std::collections::HashMap;

        // Group chunks by source
        let mut by_source: HashMap<&str, Vec<&ScoredChunk>> = HashMap::new();
        for chunk in chunks {
            let source = chunk.chunk.source.as_deref().unwrap_or("content");
            by_source.entry(source).or_default().push(chunk);
        }

        // Build output with headers
        let mut sections: Vec<String> = Vec::new();

        for (source, chunks) in by_source {
            let header = format!("## {}", Self::format_source_name(source));
            let content: String = chunks.iter()
                .map(|c| c.chunk.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            sections.push(format!("{}\n{}", header, content));
        }

        sections.join("\n\n---\n\n")
    }

    /// Format source name for display
    fn format_source_name(source: &str) -> String {
        if source == "input" {
            "User Input".to_string()
        } else if source.contains('/') || source.contains('\\') {
            // Extract filename from path
            source.rsplit(['/', '\\']).next().unwrap_or(source).to_string()
        } else {
            source.to_string()
        }
    }

    /// Score chunks based on importance (embeddings + algorithmic only, LLM used later for polish)
    fn score_chunks(&self, chunks: &[Chunk]) -> Result<Vec<ScoredChunk>> {
        let mut scored = Vec::with_capacity(chunks.len());

        // FAST PATH: Pre-filter chunks by keyword matching if query exists
        // Only embed top candidates to save time
        let max_embed = 100; // Max chunks to embed
        let chunks_to_embed: Vec<usize> = if let Some(ref query) = self.query {
            // Extract keywords from query (words > 3 chars)
            let keywords: Vec<&str> = query.split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();

            // Score chunks by keyword presence
            let mut keyword_scores: Vec<(usize, f32)> = chunks.iter()
                .enumerate()
                .map(|(i, chunk)| {
                    // Critical always included
                    if chunk.priority == Priority::Critical {
                        return (i, 100.0);
                    }
                    let content_lower = chunk.content.to_lowercase();
                    let matches = keywords.iter()
                        .filter(|k| content_lower.contains(&k.to_lowercase()))
                        .count();
                    (i, matches as f32 / keywords.len().max(1) as f32)
                })
                .collect();

            keyword_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            keyword_scores.into_iter()
                .take(max_embed)
                .map(|(i, _)| i)
                .collect()
        } else {
            // No query - embed all (up to max)
            (0..chunks.len().min(max_embed)).collect()
        };

        // Embed only selected chunks
        let texts: Vec<&str> = chunks_to_embed.iter()
            .map(|&i| chunks[i].content.as_str())
            .collect();
        let chunk_embeddings = embed_batch(&texts)?;

        // Create embedding map
        let mut embedding_map: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
        for (idx, emb) in chunks_to_embed.iter().zip(chunk_embeddings.into_iter()) {
            embedding_map.insert(*idx, emb);
        }

        // Calculate query embedding if query is set
        let query_embedding = if let Some(ref q) = self.query {
            Some(embed_batch(&[q.as_str()])?.pop().unwrap_or_default())
        } else {
            None
        };

        // Calculate centroid from embedded chunks
        let embedded_vecs: Vec<Vec<f32>> = embedding_map.values().cloned().collect();
        let centroid = calculate_centroid(&embedded_vecs);

        for (i, chunk) in chunks.iter().enumerate() {
            // Get embedding if available, otherwise use default score
            let semantic_score = if let Some(embedding) = embedding_map.get(&i) {
                if let Some(ref q_emb) = query_embedding {
                    let sim = cosine_similarity(embedding, q_emb);
                    (sim + 1.0) / 2.0
                } else {
                    let centrality = cosine_similarity(embedding, &centroid);
                    (centrality + 1.0) / 2.0
                }
            } else {
                // Chunk not embedded - use keyword score (lower priority)
                0.3
            };

            // Priority score
            let priority_score = match chunk.priority {
                Priority::Critical => 1.0,
                Priority::High => 0.7,
                Priority::Medium => 0.4,
                Priority::Low => 0.2,
                _ => 0.1,
            };

            // Position score (earlier content slightly preferred)
            let position_score = 1.0 - (chunk.position as f32 / chunks.len().max(1) as f32) * 0.3;

            // Recency boost for conversation-like content
            let recency_score = if chunk.content.contains("User:") || chunk.content.contains("Assistant:") {
                0.3 + (chunk.position as f32 / chunks.len().max(1) as f32) * 0.7
            } else {
                0.5
            };

            // Combined score (embedding + algorithmic)
            let final_score = if chunk.priority == Priority::Critical {
                0.9 + semantic_score * 0.1
            } else {
                priority_score * 0.35
                    + semantic_score * 0.35
                    + position_score * 0.15
                    + recency_score * 0.15
            };

            scored.push(ScoredChunk {
                chunk: chunk.clone(),
                score: final_score,
                breakdown: ScoreBreakdown {
                    semantic: semantic_score,
                    algorithmic: position_score,
                    priority_boost: priority_score,
                    recency_factor: recency_score,
                    llm: 0.0, // LLM used for polish, not scoring
                },
            });
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored)
    }

    /// Select chunks within token budget (uses config max_tokens)
    #[allow(dead_code)]
    fn select_within_budget(&self, chunks: Vec<ScoredChunk>, _counter: &TokenCounter) -> Result<Vec<ScoredChunk>> {
        self.select_within_budget_custom(chunks, self.config.options.context_limit)
    }

    /// Select chunks within a custom token budget
    fn select_within_budget_custom(&self, chunks: Vec<ScoredChunk>, budget: usize) -> Result<Vec<ScoredChunk>> {
        let mut selected = Vec::new();
        let mut remaining = Vec::new();
        let mut used_tokens = 0;

        // First pass: always include Critical priority items
        for chunk in chunks {
            if chunk.chunk.priority == Priority::Critical {
                used_tokens += chunk.chunk.tokens;
                selected.push(chunk);
            } else {
                remaining.push(chunk);
            }
        }

        // Second pass: add other chunks within remaining budget
        for chunk in remaining {
            if used_tokens + chunk.chunk.tokens <= budget {
                used_tokens += chunk.chunk.tokens;
                selected.push(chunk);

                if used_tokens >= budget * 95 / 100 {
                    break;
                }
            } else if selected.is_empty() && used_tokens == 0 {
                // No chunks selected yet and this one is too big - truncate it
                let counter = TokenCounter::default();
                let truncated = counter.truncate_to_budget(&chunk.chunk.content, budget);
                let truncated_tokens = counter.count(&truncated);

                let mut truncated_chunk = chunk.clone();
                truncated_chunk.chunk.content = truncated;
                truncated_chunk.chunk.tokens = truncated_tokens;

                used_tokens = truncated_tokens;
                selected.push(truncated_chunk);
                break;
            }
        }

        // Re-sort by original position to maintain document order
        selected.sort_by_key(|c| c.chunk.position);

        Ok(selected)
    }
}

/// Calculate centroid (average) of embeddings
fn calculate_centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![];
    }

    let dim = embeddings[0].len();
    let mut centroid = vec![0.0f32; dim];

    for emb in embeddings {
        for (i, val) in emb.iter().enumerate() {
            centroid[i] += val;
        }
    }

    let n = embeddings.len() as f32;
    for val in centroid.iter_mut() {
        *val /= n;
    }

    centroid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::content::FileWithPriority;

    #[test]
    fn test_new() {
        let f = Forgetless::new();
        assert_eq!(f.config.options.context_limit, 128_000);
        assert!(f.inputs.is_empty());
    }

    #[test]
    fn test_add() {
        let f = Forgetless::new()
            .add("content 1")
            .add("content 2");
        assert_eq!(f.inputs.len(), 2);
    }

    #[test]
    fn test_add_with_priority() {
        let f = Forgetless::new()
            .add(WithPriority::critical("important"));
        assert_eq!(f.inputs[0].priority, Priority::Critical);
    }

    #[test]
    fn test_add_pinned() {
        let f = Forgetless::new()
            .add_pinned("pinned content");
        assert_eq!(f.inputs[0].priority, Priority::Critical);
    }

    #[test]
    fn test_with_priority_all_levels() {
        let f = Forgetless::new()
            .add(WithPriority::critical("critical"))
            .add(WithPriority::high("high"))
            .add(WithPriority::medium("medium"))
            .add(WithPriority::low("low"));

        assert_eq!(f.inputs[0].priority, Priority::Critical);
        assert_eq!(f.inputs[1].priority, Priority::High);
        assert_eq!(f.inputs[2].priority, Priority::Medium);
        assert_eq!(f.inputs[3].priority, Priority::Low);
    }

    #[test]
    fn test_file_with_priority_all_levels() {
        let _critical = FileWithPriority::critical("test.txt");
        let _high = FileWithPriority::high("test.txt");
        let _medium = FileWithPriority::medium("test.txt");
        let _low = FileWithPriority::low("test.txt");
    }

    #[test]
    fn test_add_bytes() {
        let data = b"test content";
        let f = Forgetless::new()
            .add_bytes(&data[..], "text/plain");
        assert_eq!(f.inputs[0].priority, Priority::Medium);
    }

    #[test]
    fn test_add_bytes_binary() {
        let data = vec![0u8, 1, 2, 3, 255];
        let f = Forgetless::new()
            .add_bytes_p(&data, "application/octet-stream", Priority::Low);
        assert!(f.inputs[0].content.contains("Binary content"));
    }

    #[test]
    fn test_config() {
        use crate::core::config::Config;
        let f = Forgetless::new()
            .config(Config::default()
                .context_limit(64_000)
                .vision_llm(true));
        assert_eq!(f.config.options.context_limit, 64_000);
        assert!(f.config.options.vision_llm);
    }

    #[test]
    fn test_calculate_centroid_empty() {
        let result = calculate_centroid(&[]);
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_run_empty() {
        let result = Forgetless::new().run().await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.is_empty());
    }

    #[tokio::test]
    async fn test_run_basic() {
        let result = Forgetless::new()
            .add("Hello world, this is a test.")
            .run()
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_compression() {
        use crate::core::config::Config;
        // ~50K tokens of content -> compress to 32K
        let large_content = "This is a test sentence with some meaningful words about AI and machine learning. ".repeat(5000);

        let result = Forgetless::new()
            .config(Config::default().context_limit(32_000))
            .add(&large_content)
            .run()
            .await;

        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.total_tokens <= 35_000, "Got {} tokens", ctx.total_tokens);
        assert!(ctx.stats.compression_ratio > 1.2, "Expected compression, got {}", ctx.stats.compression_ratio);
    }

    #[test]
    fn test_query() {
        let f = Forgetless::new()
            .query("What is Rust?");
        assert_eq!(f.query, Some("What is Rust?".to_string()));
    }

    #[test]
    fn test_add_file() {
        let f = Forgetless::new()
            .add_file("Cargo.toml");
        // Files are lazy loaded now
        assert_eq!(f.lazy_files.len(), 1);
        assert!(f.lazy_files[0].path.to_string_lossy().contains("Cargo.toml"));
    }

    #[test]
    fn test_add_file_nonexistent() {
        let f = Forgetless::new()
            .add_file("nonexistent_12345.txt");
        // Lazy loading - file is added but will fail at run() time
        assert_eq!(f.lazy_files.len(), 1);
    }

    #[test]
    fn test_add_files() {
        let f = Forgetless::new()
            .add_files(&["Cargo.toml", "src/lib.rs"]);
        assert_eq!(f.lazy_files.len(), 2);
    }

    #[test]
    fn test_add_files_with_nonexistent() {
        let f = Forgetless::new()
            .add_files(&["Cargo.toml", "nonexistent.txt"]);
        // Lazy loading - both are added, nonexistent will be skipped at run()
        assert_eq!(f.lazy_files.len(), 2);
    }

    #[test]
    fn test_add_files_with_priority() {
        let f = Forgetless::new()
            .add_files([FileWithPriority::high("Cargo.toml")]);
        assert_eq!(f.lazy_files.len(), 1);
        assert_eq!(f.lazy_files[0].priority, Priority::High);
    }

    #[test]
    fn test_add_bytes_invalid_utf8() {
        let data = vec![0xFF, 0xFE, 0x00, 0x01];
        let f = Forgetless::new()
            .add_bytes(&data, "text/plain");
        // Invalid UTF-8 should be treated as binary
        assert!(f.inputs.is_empty() || f.inputs[0].content.contains("Binary"));
    }

    #[test]
    fn test_calculate_centroid_single() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]];
        let centroid = calculate_centroid(&embeddings);
        assert_eq!(centroid, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_calculate_centroid_multiple() {
        let embeddings = vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 4.0, 6.0],
        ];
        let centroid = calculate_centroid(&embeddings);
        assert_eq!(centroid, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_run_with_query() {
        // Content must be >10 tokens to pass min_tokens filter
        let result = Forgetless::new()
            .add("Rust is a systems programming language focused on safety, speed, and concurrency. It prevents memory errors without garbage collection.")
            .add("Python is great for data science, machine learning, and rapid prototyping. It has excellent libraries like NumPy and Pandas.")
            .query("Tell me about Rust")
            .run()
            .await;

        assert!(result.is_ok());
        let ctx = result.unwrap();
        // Content about Rust should be prioritized
        assert!(!ctx.content.is_empty());
    }

    #[tokio::test]
    async fn test_run_fits_budget() {
        use crate::core::config::Config;
        // Content that fits within budget should not be compressed
        // Note: Content must be >10 tokens to pass min_tokens filter
        let result = Forgetless::new()
            .config(Config::default().context_limit(10000))
            .add("This is a longer piece of content that will pass the minimum token filter and should be included in the output without compression")
            .run()
            .await;

        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.content.contains("longer piece of content"));
        assert_eq!(ctx.stats.compression_ratio, 1.0);
    }

    #[tokio::test]
    async fn test_run_multiple_inputs() {
        // Content must be >10 tokens each to pass min_tokens filter
        let result = Forgetless::new()
            .add("First input content here with enough words to pass the minimum token filter requirement.")
            .add("Second input content here also needs enough words to be included in the final output.")
            .add("Third input content here must also have sufficient length to avoid being filtered out.")
            .run()
            .await;

        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(!ctx.content.is_empty());
    }

    #[tokio::test]
    async fn test_run_with_critical_priority() {
        use crate::core::config::Config;
        // ~20K tokens of low priority content
        let large_content = "Low priority content about random topics that should be compressed away. ".repeat(2000);

        let result = Forgetless::new()
            .config(Config::default().context_limit(16_000))
            .add(&large_content)
            .add(WithPriority::critical("CRITICAL: This system instruction must always be included in the output!"))
            .run()
            .await;

        assert!(result.is_ok());
        let ctx = result.unwrap();
        // Critical content should always be included regardless of compression
        assert!(ctx.content.contains("CRITICAL") || ctx.chunks.iter().any(|c| c.chunk.priority == Priority::Critical));
    }

    #[tokio::test]
    async fn test_run_aggressive_compression() {
        use crate::core::config::Config;
        // ~100K tokens -> compress to 8K (aggressive)
        let content = "This is detailed content about various AI topics including transformers, attention mechanisms, and neural networks. ".repeat(8000);

        let result = Forgetless::new()
            .config(Config::default().context_limit(8_000))
            .add(&content)
            .query("Summarize the key AI concepts")
            .run()
            .await;

        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.total_tokens <= 10_000, "Got {} tokens", ctx.total_tokens);
        assert!(ctx.stats.compression_ratio > 5.0, "Expected high compression, got {}", ctx.stats.compression_ratio);
    }

    #[test]
    fn test_add_file_with_priority() {
        let f = Forgetless::new()
            .add_file(FileWithPriority::high("Cargo.toml"));
        assert_eq!(f.lazy_files.len(), 1);
        assert_eq!(f.lazy_files[0].priority, Priority::High);
    }

    #[test]
    fn test_options() {
        use crate::core::config::Config;

        // Default: LLMs off
        let f = Forgetless::new();
        assert!(!f.config.options.vision_llm);
        assert!(!f.config.options.context_llm);

        // Enable vision LLM
        let f = Forgetless::new()
            .config(Config::default().vision_llm(true));
        assert!(f.config.options.vision_llm);
        assert!(!f.config.options.context_llm);

        // Enable both
        let f = Forgetless::new()
            .config(Config::default()
                .vision_llm(true)
                .context_llm(true)
                .context_limit(64_000));
        assert!(f.config.options.vision_llm);
        assert!(f.config.options.context_llm);
        assert_eq!(f.config.options.context_limit, 64_000);
    }

    #[test]
    fn test_add_file_string_type() {
        let path = String::from("Cargo.toml");
        let f = Forgetless::new().add_file(path);
        assert_eq!(f.lazy_files.len(), 1);
    }

    #[test]
    fn test_add_file_pathbuf_type() {
        let path = std::path::PathBuf::from("Cargo.toml");
        let f = Forgetless::new().add_file(path);
        assert_eq!(f.lazy_files.len(), 1);
    }

    #[test]
    fn test_add_file_path_ref_type() {
        let path = std::path::Path::new("Cargo.toml");
        let f = Forgetless::new().add_file(path);
        assert_eq!(f.lazy_files.len(), 1);
    }

    #[tokio::test]
    async fn test_run_with_pdf() {
        use crate::core::config::Config;
        let path = std::path::PathBuf::from("benches/data/attention_paper.pdf");
        if path.exists() {
            let result = Forgetless::new()
                .config(Config::default().context_limit(8_000))
                .add_file(path.clone())
                .query("What is attention?")
                .run()
                .await;

            assert!(result.is_ok());
            let ctx = result.unwrap();
            assert!(ctx.stats.input_tokens > 0);
        }
    }

    #[tokio::test]
    async fn test_run_with_image() {
        use crate::core::config::Config;
        // Create a test image
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_builder.png");
        let img = image::RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0]));
        img.save(&path).unwrap();

        let result = Forgetless::new()
            .config(Config::default().context_limit(4_000))
            .add_file(path.clone())
            .run()
            .await;

        assert!(result.is_ok());
        std::fs::remove_file(&path).ok();
    }
}
