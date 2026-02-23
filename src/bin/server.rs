//! Forgetless HTTP Server
//!
//! Run with: cargo run --features server --bin forgetless-server
//!
//! POST / - Optimize context (multipart/form-data)
//! GET /health - Health check

use axum::{
    extract::{DefaultBodyLimit, Multipart},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use forgetless::{Config, FileWithPriority, Forgetless, WithPriority};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::fs;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

/// Response from context optimization
#[derive(Debug, Serialize)]
struct OptimizeResponse {
    content: String,
    total_tokens: usize,
    compression_ratio: f32,
    stats: Stats,
}

#[derive(Debug, Serialize)]
struct Stats {
    input_tokens: usize,
    output_tokens: usize,
    chunks_processed: usize,
    chunks_selected: usize,
    processing_time_ms: u64,
}

/// Error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Metadata for the request (sent as JSON in 'metadata' field)
#[derive(Debug, Deserialize, Default)]
struct RequestMetadata {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    contents: Vec<ContentItem>,
}

#[derive(Debug, Deserialize)]
struct ContentItem {
    content: String,
    #[serde(default = "default_priority")]
    priority: String,
}

fn default_max_tokens() -> usize {
    128_000
}

fn default_priority() -> String {
    "medium".to_string()
}

/// Temporary file tracker for cleanup
struct TempFiles {
    dir: PathBuf,
    files: Vec<PathBuf>,
}

impl TempFiles {
    async fn new() -> std::io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("forgetless-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).await?;
        Ok(Self { dir, files: Vec::new() })
    }

    async fn save(&mut self, name: &str, data: &[u8], priority: &str) -> std::io::Result<(PathBuf, String)> {
        let path = self.dir.join(name);
        fs::write(&path, data).await?;
        self.files.push(path.clone());
        Ok((path, priority.to_string()))
    }

    async fn cleanup(self) {
        // Remove all temp files and directory
        for file in &self.files {
            let _ = fs::remove_file(file).await;
        }
        let _ = fs::remove_dir(&self.dir).await;
    }
}

/// Health check endpoint
async fn health() -> &'static str {
    "ok"
}

/// Main optimization endpoint with multipart file upload
///
/// Form fields:
/// - metadata: JSON with {max_tokens, query, contents: [{content, priority}]}
/// - files: File uploads (field name format: "file" or "file:priority")
async fn optimize(mut multipart: Multipart) -> impl IntoResponse {
    let start = std::time::Instant::now();

    // Create temp directory for uploaded files
    let mut temp_files = match TempFiles::new().await {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: format!("Failed to create temp dir: {}", e) }),
            ).into_response();
        }
    };

    let mut metadata = RequestMetadata::default();
    let mut file_items: Vec<(PathBuf, String)> = Vec::new();

    // Process multipart fields
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        let filename = field.file_name().map(|s| s.to_string());

        if name == "metadata" {
            if let Ok(text) = field.text().await {
                if let Ok(parsed) = serde_json::from_str::<RequestMetadata>(&text) {
                    metadata = parsed;
                }
            }
        } else if name.starts_with("file") {
            let priority = name.split(':').nth(1).unwrap_or("medium").to_string();

            if let Some(fname) = filename {
                if let Ok(data) = field.bytes().await {
                    match temp_files.save(&fname, &data, &priority).await {
                        Ok(item) => file_items.push(item),
                        Err(e) => {
                            temp_files.cleanup().await;
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(ErrorResponse { error: format!("Failed to save file: {}", e) }),
                            ).into_response();
                        }
                    }
                }
            }
        }
    }

    // Build forgetless instance
    let mut builder = Forgetless::new()
        .config(Config::default().context_limit(metadata.max_tokens));

    // Add query if provided
    if let Some(query) = metadata.query {
        builder = builder.query(query);
    }

    // Add text contents with priorities
    for item in metadata.contents {
        builder = match item.priority.to_lowercase().as_str() {
            "critical" => builder.add(WithPriority::critical(item.content)),
            "high" => builder.add(WithPriority::high(item.content)),
            "low" => builder.add(WithPriority::low(item.content)),
            _ => builder.add(item.content),
        };
    }

    // Add uploaded files with priorities
    for (path, priority) in file_items {
        let path_str = path.to_string_lossy().to_string();
        builder = match priority.to_lowercase().as_str() {
            "critical" => builder.add_file(FileWithPriority::critical(path_str.as_str())),
            "high" => builder.add_file(FileWithPriority::high(path_str.as_str())),
            "low" => builder.add_file(FileWithPriority::low(path_str.as_str())),
            _ => builder.add_file(path_str),
        };
    }

    // Run optimization
    let result = builder.run().await;

    // Cleanup temp files regardless of result
    temp_files.cleanup().await;

    match result {
        Ok(result) => {
            let compression_ratio = result.compression_ratio();
            let response = OptimizeResponse {
                content: result.content,
                total_tokens: result.total_tokens,
                compression_ratio,
                stats: Stats {
                    input_tokens: result.stats.input_tokens,
                    output_tokens: result.stats.output_tokens,
                    chunks_processed: result.stats.chunks_processed,
                    chunks_selected: result.stats.chunks_selected,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                },
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            let error = ErrorResponse { error: e.to_string() };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router with increased body limit for file uploads (50MB)
    let app = Router::new()
        .route("/", post(optimize))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(cors);

    // Server address
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("forgetless server listening on http://{}", addr);

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
