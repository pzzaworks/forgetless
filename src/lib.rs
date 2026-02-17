//! # Forgetless
//!
//! A smart context management system for LLMs - never forget what matters.
//!
//! Forgetless provides intelligent context window management for Large Language Models,
//! helping you maximize the value of every token in your context window.
//!
//! ## Features
//!
//! - **Smart Chunking**: Semantic-aware text and code chunking
//! - **Priority Retention**: Keep important information, compress the rest
//! - **Token Budget**: Precise token counting and budget management
//! - **Relevance Scoring**: Score and rank context by relevance
//! - **Conversation Memory**: Long-term memory for multi-turn conversations
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use forgetless::{ContextManager, ContextConfig};
//!
//! let config = ContextConfig::default()
//!     .with_max_tokens(8000)
//!     .with_model("gpt-4");
//!
//! let mut manager = ContextManager::new(config);
//!
//! // Add content to context
//! manager.add("User asked about Rust ownership", Priority::High);
//! manager.add("Previous discussion about variables", Priority::Medium);
//!
//! // Get optimized context within token budget
//! let context = manager.build_context()?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod chunking;
pub mod context;
pub mod error;
pub mod memory;
pub mod scoring;
pub mod token;

pub use chunking::Chunk;
pub use context::{ContextConfig, ContextManager};
pub use error::{Error, Result};
pub use memory::{ConversationMemory, Message, Role};
pub use scoring::Priority;
pub use token::TokenCounter;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
