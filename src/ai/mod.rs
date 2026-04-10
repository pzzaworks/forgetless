//! AI modules - embeddings, LLM, and vision

pub mod embeddings;
pub mod llm;
pub mod vision;

pub use embeddings::{cosine_similarity, embed_batch, embed_text, CacheStats, EmbeddingCache};
pub use llm::{
    generate, generate_with_system, polish, polish_content, score_relevance, summarize, LLMConfig,
    Quantization, LLM,
};
pub use vision::{describe_image, describe_image_with_prompt, init_vision, is_vision_ready};
