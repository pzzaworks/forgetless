//! Error types for Forgetless

use thiserror::Error;

/// Result type alias for Forgetless operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in Forgetless operations
#[derive(Error, Debug)]
pub enum Error {
    /// Token budget exceeded
    #[error("Token budget exceeded: requested {requested}, available {available}")]
    TokenBudgetExceeded {
        /// Tokens requested
        requested: usize,
        /// Tokens available
        available: usize,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Chunking error
    #[error("Chunking error: {0}")]
    ChunkingError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Token counting error
    #[error("Token counting error: {0}")]
    TokenCountError(String),

    /// Memory operation error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Context building error
    #[error("Context building error: {0}")]
    ContextBuildError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget_exceeded_error() {
        let error = Error::TokenBudgetExceeded {
            requested: 1000,
            available: 500,
        };
        let msg = error.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn test_invalid_config_error() {
        let error = Error::InvalidConfig("bad value".to_string());
        assert!(error.to_string().contains("bad value"));
    }

    #[test]
    fn test_chunking_error() {
        let error = Error::ChunkingError("chunk failed".to_string());
        assert!(error.to_string().contains("chunk failed"));
    }

    #[test]
    fn test_token_count_error() {
        let error = Error::TokenCountError("count failed".to_string());
        assert!(error.to_string().contains("count failed"));
    }

    #[test]
    fn test_memory_error() {
        let error = Error::MemoryError("memory issue".to_string());
        assert!(error.to_string().contains("memory issue"));
    }

    #[test]
    fn test_context_build_error() {
        let error = Error::ContextBuildError("build failed".to_string());
        assert!(error.to_string().contains("build failed"));
    }

    #[test]
    fn test_serialization_error_from() {
        // Create an invalid JSON to trigger serde error
        let result: std::result::Result<serde_json::Value, _> = serde_json::from_str("invalid");
        let serde_err = result.unwrap_err();
        let error: Error = serde_err.into();
        assert!(matches!(error, Error::SerializationError(_)));
    }

    #[test]
    fn test_error_debug() {
        let error = Error::InvalidConfig("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("InvalidConfig"));
    }
}
