//! Vision module - SmolVLM for image understanding (OPTIONAL)
//!
//! **WARNING: Slow! Model load takes 2+ minutes. Use only if you need it.**
//!
//! Uses SmolVLM-256M for local image captioning.
//! Default behavior (without init_vision): fast metadata only.
//!
//! # Example
//!
//! ```rust,ignore
//! use forgetless::ai::vision::{describe_image, init_vision};
//!
//! // OPTIONAL: Initialize vision LLM (slow, 2+ min first run)
//! // Skip this for fast metadata-only mode
//! init_vision().await?;
//!
//! // With init: LLM description
//! // Without init: returns Err (use get_image_metadata instead)
//! let caption = describe_image(&image_bytes).await?;
//! ```

use std::sync::OnceLock;
use tokio::sync::Mutex;
use image::DynamicImage;
use mistralrs::{
    IsqType, Model, RequestBuilder, TextMessageRole, VisionMessages, VisionModelBuilder,
    ChatCompletionResponse,
};

use crate::core::error::{Error, Result};

// ============================================================================
// Global Vision Model
// ============================================================================

static VISION_MODEL: OnceLock<Mutex<Model>> = OnceLock::new();

/// Initialize the vision model (SmolVLM-256M)
pub async fn init_vision() -> Result<()> {
    if VISION_MODEL.get().is_some() {
        return Ok(());
    }

    let model = VisionModelBuilder::new("HuggingFaceTB/SmolVLM-256M-Instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await
        .map_err(|e| Error::ContextBuildError(format!("Failed to load SmolVLM: {e}")))?;

    let _ = VISION_MODEL.set(Mutex::new(model));
    Ok(())
}

/// Check if vision model is initialized
pub fn is_vision_ready() -> bool {
    VISION_MODEL.get().is_some()
}

/// Describe an image using SmolVLM
///
/// Returns a short description of the image content.
pub async fn describe_image(image_bytes: &[u8]) -> Result<String> {
    // Load image from bytes
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| Error::ContextBuildError(format!("Failed to load image: {e}")))?;

    describe_dynamic_image(img, "Describe this image briefly and concisely. Focus on the main content, objects, text, or diagram elements visible.").await
}

/// Describe an image with a specific question/prompt
pub async fn describe_image_with_prompt(image_bytes: &[u8], prompt: &str) -> Result<String> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| Error::ContextBuildError(format!("Failed to load image: {e}")))?;

    describe_dynamic_image(img, prompt).await
}

/// Resize image to fixed size for SmolVLM compatibility
/// Using 224x224 (ViT standard) to avoid mistralrs index bugs
fn resize_if_needed(img: DynamicImage) -> DynamicImage {
    const TARGET: u32 = 224;
    img.resize_exact(TARGET, TARGET, image::imageops::FilterType::Triangle)
}

/// Describe a DynamicImage
async fn describe_dynamic_image(img: DynamicImage, prompt: &str) -> Result<String> {
    let model = VISION_MODEL.get()
        .ok_or_else(|| Error::ContextBuildError("Vision model not initialized. Call init_vision() first.".into()))?;

    let guard = model.lock().await;

    // Resize to avoid index out of bounds in SmolVLM
    let img = resize_if_needed(img);

    // Create image message
    let messages = VisionMessages::new()
        .add_image_message(
            TextMessageRole::User,
            prompt,
            vec![img],
            &guard,
        )
        .map_err(|e| Error::ContextBuildError(format!("Failed to create image message: {e}")))?;

    // Build sampling params with strict max_len
    let request = RequestBuilder::from(messages)
        .set_sampler_max_len(64)
        .set_sampler_temperature(0.1)
        .set_sampler_topp(0.9);

    let response = guard.send_chat_request(request)
        .await
        .map_err(|e| Error::ContextBuildError(format!("Vision inference failed: {e}")))?;

    extract_response_text(response)
}

/// Extract text from response
fn extract_response_text(response: ChatCompletionResponse) -> Result<String> {
    Ok(response.choices.first()
        .map(|c| c.message.content.clone().unwrap_or_default())
        .unwrap_or_default())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_vision_init() {
        init_vision().await.unwrap();
        assert!(is_vision_ready());
    }

    #[tokio::test]
    #[ignore] // Requires model download
    async fn test_describe_image() {
        use std::io::Cursor;
        use image::{ImageBuffer, Rgb, ImageFormat};

        init_vision().await.unwrap();

        // Create a simple 32x32 red square image
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(32, 32, |_, _| {
            Rgb([255, 0, 0])
        });

        let mut png_data: Vec<u8> = Vec::new();
        img.write_to(&mut Cursor::new(&mut png_data), ImageFormat::Png).unwrap();

        let description = describe_image(&png_data).await.unwrap();
        assert!(!description.is_empty());
    }
}
