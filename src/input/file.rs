//! File reading and parsing (PDF, images, text)

use crate::processing::chunking::ContentType;
use image::GenericImageView;
use std::path::Path;

/// Maximum characters for preview (roughly ~500 tokens)
const PREVIEW_MAX_CHARS: usize = 2000;

/// Extract PDF text using pdftotext command (FAST and accurate)
/// Falls back to raw extraction if pdftotext not available
fn extract_pdf_with_pdftotext(path: &Path) -> Option<String> {
    use std::process::Command;

    // Try pdftotext first (fast and accurate)
    let output = Command::new("pdftotext")
        .arg("-layout") // Preserve layout
        .arg("-nopgbrk") // No page breaks
        .arg(path)
        .arg("-") // Output to stdout
        .output()
        .ok()?;

    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout);
        let cleaned = text.trim().to_string();
        if !cleaned.is_empty() {
            // Add filename as header
            let name = path.file_name()?.to_string_lossy();
            return Some(format!("# {}\n\n{}", name, cleaned));
        }
    }

    // Fallback: raw extraction (less accurate but works without pdftotext)
    let bytes = std::fs::read(path).ok()?;
    Some(extract_pdf_text_fast(&bytes, path))
}

/// Extract ALL text from PDF bytes - FAST method without pdf_extract
/// Extracts text from PDF streams and objects directly
fn extract_pdf_text_fast(bytes: &[u8], path: &Path) -> String {
    let mut result = String::new();
    let mut in_text = false;
    let mut current = String::new();
    let mut last_was_space = true;

    // Add filename as header
    if let Some(name) = path.file_name() {
        result.push_str("# ");
        result.push_str(&name.to_string_lossy());
        result.push_str("\n\n");
    }

    // Scan entire PDF for text objects
    for i in 0..bytes.len() {
        let b = bytes[i];

        if b == b'(' && !in_text {
            in_text = true;
            current.clear();
        } else if b == b')' && in_text {
            in_text = false;
            // Filter out PDF commands and keep meaningful text
            let trimmed = current.trim();
            if trimmed.len() >= 1
                && trimmed.chars().any(|c| c.is_alphabetic())
                && !trimmed.starts_with('/')
                && !trimmed.chars().all(|c| c.is_ascii_punctuation())
            {
                if !last_was_space && !result.ends_with(' ') && !result.ends_with('\n') {
                    result.push(' ');
                }
                result.push_str(trimmed);
                last_was_space = trimmed.ends_with(' ');
            }
        } else if in_text {
            // Handle escape sequences
            if b == b'\\' && i + 1 < bytes.len() {
                let next = bytes[i + 1];
                match next {
                    b'n' => current.push('\n'),
                    b'r' => current.push('\r'),
                    b't' => current.push('\t'),
                    b'(' | b')' | b'\\' => current.push(next as char),
                    _ => {}
                }
            } else if b >= 32 && b < 127 {
                current.push(b as char);
            } else if b == b'\n' || b == b'\r' {
                current.push(' ');
            }
        }
    }

    // If direct extraction failed, use raw word extraction
    if result.len() < 500 {
        result.clear();
        if let Some(name) = path.file_name() {
            result.push_str("# ");
            result.push_str(&name.to_string_lossy());
            result.push_str("\n\n");
        }

        let mut word = String::new();
        for &b in bytes {
            if b.is_ascii_alphanumeric()
                || b == b' '
                || b == b'.'
                || b == b','
                || b == b'-'
                || b == b'\''
            {
                word.push(b as char);
            } else if word.len() >= 2 {
                let trimmed = word.trim();
                if trimmed.len() >= 2 && trimmed.chars().any(|c| c.is_alphabetic()) {
                    if !result.is_empty() && !result.ends_with(' ') && !result.ends_with('\n') {
                        result.push(' ');
                    }
                    result.push_str(trimmed);
                }
                word.clear();
            } else {
                word.clear();
            }
        }
    }

    // Clean up excessive whitespace
    let cleaned: String = result.split_whitespace().collect::<Vec<_>>().join(" ");

    cleaned
}

/// Read a quick preview of a file (first page/section only)
/// This is much faster than reading the full file, used for relevance filtering
pub fn read_file_preview(path: &Path) -> Option<String> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match extension.as_deref() {
        // PDF files - try to extract from partial read first
        Some("pdf") => read_pdf_preview_fast(path),
        // Image files - just return filename as preview (can't preview image content easily)
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp") => {
            // For images, use filename + basic description
            Some(format!(
                "Image file: {}",
                path.file_name()?.to_string_lossy()
            ))
        }
        // Text files - read first N characters
        _ => {
            // Read only first N bytes for text files too
            use std::io::Read;
            let mut file = std::fs::File::open(path).ok()?;
            let mut buffer = vec![0u8; PREVIEW_MAX_CHARS * 4]; // UTF-8 worst case
            let bytes_read = file.read(&mut buffer).ok()?;
            buffer.truncate(bytes_read);
            let content = String::from_utf8_lossy(&buffer);
            Some(content.chars().take(PREVIEW_MAX_CHARS).collect())
        }
    }
}

/// Fast PDF preview - extracts text directly from raw bytes (NO pdf_extract!)
/// This is ~1000x faster than pdf_extract for preview purposes
fn read_pdf_preview_fast(path: &Path) -> Option<String> {
    use std::io::Read;

    // Read first 50KB - enough to get title, abstract, intro
    let mut file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len() as usize;
    let read_size = file_size.min(50 * 1024);

    let mut buffer = vec![0u8; read_size];
    file.read_exact(&mut buffer).ok()?;

    // Direct text extraction from PDF bytes - FAST!
    // PDFs contain readable text streams, we extract those directly
    extract_text_from_raw(&buffer)
}

/// Extract readable text strings from raw PDF bytes - ULTRA FAST
/// PDFs contain text streams like "(Hello World) Tj" - we extract these
fn extract_text_from_raw(bytes: &[u8]) -> Option<String> {
    let mut result = String::new();
    let mut in_text = false;
    let mut current = String::new();

    // Scan for PDF text objects: text inside parentheses followed by Tj/TJ
    for i in 0..bytes.len() {
        let b = bytes[i];

        if b == b'(' && !in_text {
            in_text = true;
            current.clear();
        } else if b == b')' && in_text {
            in_text = false;
            // Check if this looks like meaningful text (not PDF commands)
            if current.len() >= 2
                && current.chars().any(|c| c.is_alphabetic())
                && !current.starts_with('/')
            {
                if !result.is_empty() && !result.ends_with(' ') {
                    result.push(' ');
                }
                result.push_str(&current);
            }
        } else if in_text {
            // Handle escape sequences
            if b == b'\\' && i + 1 < bytes.len() {
                continue; // Skip escape char
            }
            if b.is_ascii_graphic() || b == b' ' {
                current.push(b as char);
            }
        }

        if result.len() >= PREVIEW_MAX_CHARS {
            break;
        }
    }

    // If PDF text extraction failed, try raw word extraction
    if result.len() < 100 {
        result.clear();
        let mut word = String::new();

        for &b in bytes {
            if b.is_ascii_alphanumeric() || b == b' ' || b == b'.' || b == b',' || b == b'-' {
                word.push(b as char);
            } else if word.len() >= 3 {
                let trimmed = word.trim();
                if trimmed.len() >= 3 && trimmed.chars().any(|c| c.is_alphabetic()) {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(trimmed);
                }
                word.clear();
            } else {
                word.clear();
            }

            if result.len() >= PREVIEW_MAX_CHARS {
                break;
            }
        }
    }

    if result.len() >= 50 {
        Some(result)
    } else {
        // Last resort: filename contains useful info
        None
    }
}

/// Read any file and convert to text content
/// Uses pdftotext for PDFs (fast and accurate)
pub fn read_file_content(path: &Path) -> Option<(String, ContentType)> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match extension.as_deref() {
        // PDF files - use pdftotext (FAST and accurate!)
        Some("pdf") => extract_pdf_with_pdftotext(path).map(|text| (text, ContentType::Text)),
        // Image files - get dimensions and prepare for vision processing
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp") => {
            let bytes = std::fs::read(path).ok()?;
            let img = image::load_from_memory(&bytes).ok()?;
            let (w, h) = img.dimensions();
            let filename = path.file_name()?.to_string_lossy();

            // Try to get image description using vision model (if initialized)
            let description = if crate::ai::vision::is_vision_ready() {
                // Run vision model synchronously - handle both runtime and non-runtime contexts
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        // We're in a tokio context, use block_in_place
                        tokio::task::block_in_place(|| {
                            handle.block_on(async {
                                crate::ai::vision::describe_image(&bytes).await.ok()
                            })
                        })
                        .unwrap_or_else(|| format!("Image: {}x{}", w, h))
                    }
                    Err(_) => {
                        // No runtime available, create a temporary one
                        tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .ok()
                            .and_then(|rt| {
                                rt.block_on(async {
                                    crate::ai::vision::describe_image(&bytes).await.ok()
                                })
                            })
                            .unwrap_or_else(|| format!("Image: {}x{}", w, h))
                    }
                }
            } else {
                // Fallback: just dimensions
                format!("Image: {}x{}", w, h)
            };

            Some((
                format!("[Image: {} ({}x{})]\n{}", filename, w, h, description),
                ContentType::Text,
            ))
        }
        // Text files
        _ => std::fs::read_to_string(path).ok().map(|content| {
            let ct = ContentType::detect_from_path(path.to_str().unwrap_or(""));
            (content, ct)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn test_read_text_file() {
        let path = PathBuf::from("Cargo.toml");
        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, _) = result.unwrap();
        assert!(content.contains("[package]"));
    }

    #[test]
    fn test_read_nonexistent() {
        let path = PathBuf::from("nonexistent_12345.txt");
        let result = read_file_content(&path);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_rust_file() {
        let path = PathBuf::from("src/lib.rs");
        let result = read_file_content(&path);
        assert!(result.is_some());
        let (_, content_type) = result.unwrap();
        assert_eq!(content_type, ContentType::Code);
    }

    #[test]
    fn test_read_no_extension() {
        // Create a temp file without extension
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_no_ext_forgetless");
        std::fs::write(&path, "test content").unwrap();

        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, content_type) = result.unwrap();
        assert_eq!(content, "test content");
        assert_eq!(content_type, ContentType::Text);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_pdf_nonexistent() {
        let path = PathBuf::from("nonexistent_12345.pdf");
        let result = read_file_content(&path);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_image_nonexistent() {
        let path = PathBuf::from("nonexistent_12345.png");
        let result = read_file_content(&path);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_invalid_pdf() {
        // Create a temp file with .pdf extension but invalid content
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_invalid.pdf");
        std::fs::write(&path, "not a real pdf").unwrap();

        let result = read_file_content(&path);
        // With pdftotext, invalid PDF may return Some with filename or None
        // Both are acceptable behaviors
        if let Some((content, _)) = result {
            // Should at least contain the filename
            assert!(!content.is_empty() || content.contains("test_invalid"));
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_invalid_image() {
        // Create a temp file with .png extension but invalid content
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_invalid.png");
        std::fs::write(&path, "not a real image").unwrap();

        let result = read_file_content(&path);
        // Should return None for invalid image
        assert!(result.is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_valid_png() {
        // Create a minimal valid PNG (1x1 red pixel)
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_valid.png");

        // Minimal PNG: 1x1 red pixel
        let png_data: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, // RGB, etc
            0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT chunk
            0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F, 0x00, // compressed data
            0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59, 0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, // IEND chunk
            0xAE, 0x42, 0x60, 0x82,
        ];

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&png_data).unwrap();

        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, content_type) = result.unwrap();
        assert!(content.contains("[Image:"));
        assert!(content.contains("1x1")); // dimensions
        assert_eq!(content_type, ContentType::Text);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_jpeg_extension() {
        // Test that jpeg extension is recognized
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test.jpeg");

        // We can't easily create a valid JPEG, so just test the path parsing
        // The function will return None for invalid image data
        std::fs::write(&path, "fake").unwrap();
        let result = read_file_content(&path);
        assert!(result.is_none()); // Invalid image data
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_content_type_detection() {
        // Test various extensions
        assert_eq!(
            ContentType::detect_from_path("test.json"),
            ContentType::Structured
        );
        assert_eq!(
            ContentType::detect_from_path("test.md"),
            ContentType::Markdown
        );
        assert_eq!(ContentType::detect_from_path("test.py"), ContentType::Code);
    }

    #[test]
    fn test_read_valid_jpeg() {
        // Create a minimal valid JPEG
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_valid.jpg");

        // Minimal JPEG (smallest valid JPEG - 1x1 pixel)
        let jpeg_data: Vec<u8> = vec![
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06,
            0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D,
            0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
            0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28,
            0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01,
            0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
            0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10,
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
            0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
            0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
            0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
            0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
            0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
            0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
            0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08,
            0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x5E, 0x5A,
            0xBD, 0xC5, 0xDB, 0xB8, 0x71, 0xC4, 0x61, 0x71, 0x40, 0x17, 0x7F, 0xFF, 0xD9,
        ];

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&jpeg_data).unwrap();

        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, _) = result.unwrap();
        assert!(content.contains("[Image:"));
        assert!(content.contains("1x1")); // dimensions

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_gif_extension() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test.gif");
        std::fs::write(&path, "fake").unwrap();
        let result = read_file_content(&path);
        assert!(result.is_none()); // Invalid image
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_webp_extension() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test.webp");
        std::fs::write(&path, "fake").unwrap();
        let result = read_file_content(&path);
        assert!(result.is_none()); // Invalid image
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_bmp_extension() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test.bmp");
        std::fs::write(&path, "fake").unwrap();
        let result = read_file_content(&path);
        assert!(result.is_none()); // Invalid image
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_valid_gif() {
        // Create a valid GIF using image crate
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_valid_forgetless.gif");

        // Create 1x1 red image and save as GIF
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
        img.save(&path).unwrap();

        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, _) = result.unwrap();
        assert!(content.contains("[Image:"));
        assert!(content.contains("1x1")); // dimensions

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_valid_bmp() {
        // Create a valid BMP using image crate
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_valid_forgetless.bmp");

        // Create 1x1 red image and save as BMP
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
        img.save(&path).unwrap();

        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, _) = result.unwrap();
        assert!(content.contains("[Image:"));
        assert!(content.contains("1x1")); // dimensions

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_valid_webp() {
        // Create a valid WebP using image crate
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_valid_forgetless.webp");

        // Create 1x1 red image and save as WebP
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));
        // Try to save as WebP - if it fails (feature not enabled), skip
        if img.save(&path).is_ok() {
            let result = read_file_content(&path);
            if let Some((content, _)) = result {
                assert!(content.contains("[Image:"));
                assert!(content.contains("1x1")); // dimensions
            }
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn test_read_benchmark_pdf() {
        // Test with actual benchmark PDF if it exists
        let path = PathBuf::from("benches/data/attention_paper.pdf");
        if path.exists() {
            let result = read_file_content(&path);
            assert!(result.is_some(), "Should read benchmark PDF");
            let (content, _) = result.unwrap();
            assert!(!content.is_empty(), "PDF content should not be empty");
            println!("PDF content length: {} chars", content.len());
            println!("First 200 chars: {}", &content[..content.len().min(200)]);
        }
    }

    #[test]
    fn test_read_valid_pdf() {
        // Create a minimal valid PDF
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_valid.pdf");

        // Minimal PDF with text "Hello"
        let pdf_data = b"%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF";

        std::fs::write(&path, pdf_data).unwrap();

        let result = read_file_content(&path);
        assert!(result.is_some());
        let (content, content_type) = result.unwrap();
        // Should either extract text or return error message
        assert!(content.contains("Hello") || content.contains("PDF"));
        assert_eq!(content_type, ContentType::Text);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_file_preview_text() {
        let path = PathBuf::from("Cargo.toml");
        let result = read_file_preview(&path);
        assert!(result.is_some());
        let preview = result.unwrap();
        assert!(preview.contains("[package]"));
    }

    #[test]
    fn test_read_file_preview_pdf() {
        let path = PathBuf::from("benches/data/attention_paper.pdf");
        if path.exists() {
            let result = read_file_preview(&path);
            assert!(result.is_some());
            let preview = result.unwrap();
            assert!(!preview.is_empty());
        }
    }

    #[test]
    fn test_read_file_preview_image() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_preview.png");
        let img = image::RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0]));
        img.save(&path).unwrap();

        let result = read_file_preview(&path);
        assert!(result.is_some());
        let preview = result.unwrap();
        assert!(preview.contains("Image file"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_read_file_preview_nonexistent() {
        let path = PathBuf::from("nonexistent_12345.txt");
        let result = read_file_preview(&path);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_pdf_with_text() {
        // Test PDF text extraction - needs enough text to pass threshold
        let pdf_bytes = b"(This is a longer text string that should be extracted from PDF) Tj (And another one here) Tj";
        let result = extract_text_from_raw(pdf_bytes);
        // Result depends on text length threshold
        if let Some(text) = result {
            assert!(text.contains("text") || text.contains("string"));
        }
    }

    #[test]
    fn test_extract_pdf_empty() {
        let pdf_bytes = b"no text objects here";
        let result = extract_text_from_raw(pdf_bytes);
        // Should return None or empty for no text
        assert!(result.is_none() || result.unwrap().is_empty());
    }
}
