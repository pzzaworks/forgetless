//! Core types, configuration, and error handling

pub mod config;
pub mod error;
pub mod types;

pub use config::{ForgetlessConfig, ScoringConfig};
pub use error::{Error, Result};
pub use types::{
    OptimizationStats, OptimizedContext, PolishedContext, ScoreBreakdown, ScoredChunk,
};
