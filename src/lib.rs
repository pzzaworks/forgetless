//! # Forgetless
//!
//! Context optimization for LLMs.
//! Takes massive context, outputs optimized version that fits your token budget.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use forgetless::Forgetless;
//!
//! let result = Forgetless::new(128_000)
//!     .add("system prompt + conversation + everything...")
//!     .add_file("document.pdf")
//!     .add_files(&["code.rs", "data.json"])
//!     .run()
//!     .await?;
//!
//! // Send to your LLM
//! let response = your_llm.chat(&result.content).await?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Internal modules (organized by feature)
pub mod ai;
pub mod builder;
pub mod core;
pub mod input;
pub mod processing;

// Re-exports for backward compatibility and convenience

// Core types
pub use core::config::{Config, ForgetlessConfig, ScoringConfig};
pub use core::error::{Error, Result};
pub use core::types::{
    OptimizationStats, OptimizedContext, PolishedContext, ScoreBreakdown, ScoredChunk,
};

// Builder
pub use builder::Forgetless;

// Input
pub use input::content::{
    ContentInput, FileWithPriority, IntoContent, IntoFileContent, WithPriority,
};
pub use input::file::read_file_content;

// Processing
pub use processing::chunking::{Chunk, ChunkConfig, Chunker, ContentType};
pub use processing::scoring::Priority;
pub use processing::token::{TokenCounter, TokenizerModel};

// AI
pub use ai::embeddings::{cosine_similarity, embed_batch, embed_text, EmbeddingCache};
pub use ai::llm::{LLMConfig, Quantization, LLM};
pub use ai::vision::{describe_image, describe_image_with_prompt, init_vision, is_vision_ready};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
