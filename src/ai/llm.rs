//! Local LLM integration for intelligent context processing
//!
//! Uses mistral.rs for fast, pure Rust LLM inference.
//!
//! # Supported Models
//!
//! - **SmolLM2-135M** - Smallest, fastest
//! - **SmolLM2-360M** - Better quality, still fast
//! - **Qwen2.5-0.5B** - Good balance (default)
//! - **Phi-3-mini** - Best quality, slower
//! - Any HuggingFace model compatible with mistral.rs
//!
//! # Example
//!
//! ```rust,ignore
//! use forgetless::llm::{LLM, LLMConfig, generate, summarize};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Default: Qwen2.5-0.5B with Q4 quantization
//!     LLM::init().await.unwrap();
//!
//!     // Or with custom config
//!     LLM::init_with_config(LLMConfig::phi3_mini()).await.unwrap();
//!
//!     // Generate text
//!     let response = generate("What is Rust?", None).await.unwrap();
//!
//!     // Summarize content
//!     let summary = summarize("Long text...", 50).await.unwrap();
//! }
//! ```

use serde::{Deserialize, Serialize};
use mistralrs::{
    IsqType, Model, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
use std::sync::{Arc, OnceLock};
use tokio::sync::Mutex;

use crate::core::error::{Error, Result};

// ============================================================================
// Configuration
// ============================================================================

/// LLM Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Model ID on HuggingFace (e.g., "HuggingFaceTB/SmolLM2-135M-Instruct")
    pub model_id: String,

    /// Quantization level
    pub quantization: Quantization,

    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative)
    pub temperature: f64,

    /// Top-p (nucleus) sampling
    pub top_p: f64,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
}

/// Quantization options for model compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    /// No quantization (full precision, largest)
    None,
    /// 4-bit quantization (smallest, fastest)
    Q4,
    /// 8-bit quantization (balanced)
    Q8,
}

impl Default for Quantization {
    fn default() -> Self {
        Self::Q4
    }
}

impl Default for LLMConfig {
    fn default() -> Self {
        // Qwen2.5-0.5B is the smallest model that reliably follows instructions
        Self::qwen_0_5b()
    }
}

impl LLMConfig {
    /// SmolLM2-135M - Smallest and fastest (135M params)
    pub fn smollm2() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-135M-Instruct".to_string(),
            quantization: Quantization::Q4,
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 256,
            repetition_penalty: 1.1,
        }
    }

    /// SmolLM2-360M - Small but smarter (360M params)
    pub fn smollm2_360m() -> Self {
        Self {
            model_id: "HuggingFaceTB/SmolLM2-360M-Instruct".to_string(),
            quantization: Quantization::Q4,
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 256,
            repetition_penalty: 1.1,
        }
    }

    /// Qwen2.5-0.5B - Good quality, still small (500M params)
    pub fn qwen_0_5b() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            quantization: Quantization::Q4,
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 256,
            repetition_penalty: 1.1,
        }
    }

    /// Phi-3-mini - Microsoft's efficient model (3.8B params)
    pub fn phi3_mini() -> Self {
        Self {
            model_id: "microsoft/Phi-3-mini-4k-instruct".to_string(),
            quantization: Quantization::Q4,
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 256,
            repetition_penalty: 1.1,
        }
    }

    /// Custom model configuration
    pub fn custom(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            quantization: Quantization::Q4,
            temperature: 0.3,
            top_p: 0.9,
            max_tokens: 256,
            repetition_penalty: 1.1,
        }
    }

    /// Set quantization level
    pub fn with_quantization(mut self, q: Quantization) -> Self {
        self.quantization = q;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
}

// ============================================================================
// LLM Provider
// ============================================================================

/// Global model instance
static GLOBAL_MODEL: OnceLock<Arc<Mutex<ModelState>>> = OnceLock::new();

struct ModelState {
    model: Model,
    config: LLMConfig,
}

/// LLM provider for text generation
pub struct LLM;

impl LLM {
    /// Initialize with default config (Qwen2.5-0.5B)
    pub async fn init() -> Result<()> {
        Self::init_with_config(LLMConfig::default()).await
    }

    /// Initialize with custom config
    pub async fn init_with_config(config: LLMConfig) -> Result<()> {
        if GLOBAL_MODEL.get().is_some() {
            tracing::warn!("LLM already initialized, skipping");
            return Ok(());
        }

        tracing::info!("Loading LLM: {} ({:?})", config.model_id, config.quantization);

        let mut builder = TextModelBuilder::new(config.model_id.clone());

        // Apply quantization
        builder = match config.quantization {
            Quantization::Q4 => builder.with_isq(IsqType::Q4_0),
            Quantization::Q8 => builder.with_isq(IsqType::Q8_0),
            Quantization::None => builder,
        };

        let model = builder
            .with_logging()
            .build()
            .await
            .map_err(|e| Error::Model(format!("Failed to load model: {e}")))?;

        let state = ModelState {
            model,
            config: config.clone(),
        };

        let _ = GLOBAL_MODEL.set(Arc::new(Mutex::new(state)));
        tracing::info!("LLM loaded successfully: {}", config.model_id);

        Ok(())
    }

    /// Check if model is loaded
    pub fn is_loaded() -> bool {
        GLOBAL_MODEL.get().is_some()
    }

    /// Get current model ID
    pub async fn model_id() -> Option<String> {
        let state = GLOBAL_MODEL.get()?;
        let guard = state.lock().await;
        Some(guard.config.model_id.clone())
    }

    fn get_state() -> Result<Arc<Mutex<ModelState>>> {
        GLOBAL_MODEL
            .get()
            .cloned()
            .ok_or_else(|| Error::Model("LLM not initialized. Call LLM::init() first.".into()))
    }
}

// ============================================================================
// Generation Functions
// ============================================================================

/// Default max tokens if not specified (prevent runaway generation)
const DEFAULT_MAX_TOKENS: usize = 256;

/// Generate text from a prompt
pub async fn generate(prompt: &str, max_tokens: Option<usize>) -> Result<String> {
    let state = LLM::get_state()?;
    let guard = state.lock().await;

    let limit = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, prompt);

    let request: RequestBuilder = messages.into();
    let request = request
        .set_sampler_max_len(limit)
        .set_sampler_frequency_penalty(1.2)
        .set_sampler_presence_penalty(0.6)
        .set_sampler_temperature(0.7);

    let response = guard
        .model
        .send_chat_request(request)
        .await
        .map_err(|e| Error::Model(format!("Generation error: {e}")))?;

    let content = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .map(|s| s.to_string())
        .unwrap_or_default();

    Ok(content)
}

/// Generate with system prompt
pub async fn generate_with_system(system: &str, user: &str, max_tokens: Option<usize>) -> Result<String> {
    let state = LLM::get_state()?;
    let guard = state.lock().await;

    let limit = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let messages = TextMessages::new()
        .add_message(TextMessageRole::System, system)
        .add_message(TextMessageRole::User, user);

    let request: RequestBuilder = messages.into();
    let request = request
        .set_sampler_max_len(limit)
        .set_sampler_frequency_penalty(1.2)
        .set_sampler_presence_penalty(0.6)
        .set_sampler_temperature(0.7);

    let response = guard
        .model
        .send_chat_request(request)
        .await
        .map_err(|e| Error::Model(format!("Generation error: {e}")))?;

    let content = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .map(|s| s.to_string())
        .unwrap_or_default();

    Ok(content)
}

/// Summarize content to target length
pub async fn summarize(content: &str, target_words: usize) -> Result<String> {
    let system = "You are a concise summarizer. Output only the summary, nothing else.";
    let user = format!(
        "Summarize in about {target_words} words:\n\n{content}"
    );

    // Words to tokens ratio is ~1.3, add buffer
    let max_tokens = (target_words as f32 * 1.5) as usize + 20;
    generate_with_system(system, &user, Some(max_tokens)).await
}

/// Polish and organize content chunks
pub async fn polish(chunks: &[&str], query: Option<&str>) -> Result<String> {
    let content = chunks.join("\n\n---\n\n");

    let system = "You organize and clean up text. Remove redundancy, improve flow. Output only the cleaned text.";

    let user = match query {
        Some(q) => format!(
            "Clean up this content for someone asking: \"{q}\"\n\nContent:\n{content}"
        ),
        None => format!("Clean up this content:\n\n{content}"),
    };

    // Use reasonable limit based on input size, max 512
    let max_tokens = (content.split_whitespace().count() * 2).min(512);
    generate_with_system(system, &user, Some(max_tokens)).await
}

/// Polish content - reorganize and clean up text WITHOUT adding new information
/// This is used after optimization to make the output more readable.
/// CRITICAL: The LLM only reorganizes - it does NOT add any new facts or information.
pub async fn polish_content(content: &str) -> Result<String> {
    let system = r#"You are a text organizer. Your ONLY job is to clean up and reorganize the given text.

CRITICAL RULES:
1. Output ONLY content from the input - do NOT add any new information
2. Remove redundancy and duplicates
3. Improve flow and organization
4. Keep all facts exactly as stated in the input
5. Do NOT generate answers, explanations, or new content
6. If the text mentions a question, do NOT answer it - just include the question

Output the cleaned, organized version of the text only."#;

    let user = format!("Organize this text:\n\n{content}");

    // Use ~80% of input size as output limit
    let max_tokens = (content.split_whitespace().count() as f32 * 1.3 * 0.8) as usize;
    let max_tokens = max_tokens.clamp(100, 1024);

    generate_with_system(system, &user, Some(max_tokens)).await
}

/// Score content relevance to a query (returns 0.0-1.0)
pub async fn score_relevance(content: &str, query: &str) -> Result<f32> {
    let system = "Output ONLY a number. No explanation.";
    let user = format!(
        "Relevance score (0.0=unrelated, 1.0=perfect match):\nQ: {}\nC: {}",
        query,
        &content[..content.len().min(200)]
    );

    let response = generate_with_system(system, &user, Some(5)).await?;

    // Extract first number from response
    let score = response
        .split_whitespace()
        .find_map(|word| {
            word.trim_matches(|c: char| !c.is_numeric() && c != '.')
                .parse::<f32>()
                .ok()
        })
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);

    Ok(score)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_llm_init() {
        let result = LLM::init_with_config(LLMConfig::smollm2()).await;
        assert!(result.is_ok());
        assert!(LLM::is_loaded());
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_generate() {
        LLM::init_with_config(LLMConfig::smollm2()).await.unwrap();

        let response = generate("What is 2+2?", Some(20)).await;
        assert!(response.is_ok());
        let text = response.unwrap();
        assert!(!text.is_empty());
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_summarize() {
        LLM::init_with_config(LLMConfig::smollm2()).await.unwrap();

        let long_text = "Machine learning is a subset of artificial intelligence. \
                        It involves training algorithms on data to make predictions. \
                        Deep learning uses neural networks with many layers. \
                        These techniques power modern AI applications. \
                        Machine learning models learn patterns from data and use them \
                        to make decisions without being explicitly programmed.";

        let summary = summarize(long_text, 50).await;
        assert!(summary.is_ok());
        let text = summary.unwrap();
        assert!(!text.is_empty(), "Summary should not be empty");
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_score_relevance() {
        LLM::init_with_config(LLMConfig::smollm2()).await.unwrap();

        let content = "Rust is a systems programming language focused on safety.";
        let query = "What is Rust?";

        let score = score_relevance(content, query).await;
        assert!(score.is_ok());
        let s = score.unwrap();
        assert!(s >= 0.0 && s <= 1.0);
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_polish_content() {
        LLM::init_with_config(LLMConfig::smollm2()).await.unwrap();

        let messy = "The cat sat. The cat sat on mat. Cat was sitting. Mat was soft.";

        let polished = polish_content(messy).await;
        assert!(polished.is_ok());
        let text = polished.unwrap();
        assert!(!text.is_empty());
    }

    #[test]
    fn test_config_presets() {
        let smol = LLMConfig::smollm2();
        assert!(smol.model_id.contains("SmolLM2"));

        let qwen = LLMConfig::default();
        assert!(qwen.model_id.contains("Qwen"));
    }
}
