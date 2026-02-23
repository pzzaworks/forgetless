//! Semantic chunking for text and code
//!
//! Splits content into meaningful chunks while respecting semantic boundaries.

mod chunker;
mod config;
mod types;

pub use chunker::Chunker;
pub use config::ChunkConfig;
pub use types::{Chunk, ContentType};
