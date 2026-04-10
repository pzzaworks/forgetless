//! Vision LLM Benchmarks
//!
//! Tests image understanding with Vision LLM (SmolVLM).
//! Requires image files in benches/data/
//!
//! Run with: cargo bench --bench vision --features metal
//! (or --features cuda on Linux/Windows)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forgetless::{Config, FileWithPriority, Forgetless, WithPriority};
use std::fs;
use std::path::Path;
use std::time::Instant;

const DATA_DIR: &str = "benches/data";

fn get_image_files() -> Vec<std::path::PathBuf> {
    let path = Path::new(DATA_DIR);
    if !path.exists() {
        return vec![];
    }

    let mut images = Vec::new();
    let extensions = ["png", "jpg", "jpeg", "webp"];

    for entry in fs::read_dir(path).into_iter().flatten().flatten() {
        let p = entry.path();
        if let Some(ext) = p.extension() {
            if extensions.iter().any(|e| ext.eq_ignore_ascii_case(e)) {
                images.push(p);
            }
        }
    }

    images
}

// =============================================================================
// Vision LLM Benchmarks
// =============================================================================

fn bench_vision(c: &mut Criterion) {
    let images = get_image_files();

    if images.is_empty() {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  No images found in benches/data/                                ║");
        println!("║  Place PNG/JPG files there to run vision benchmarks.             ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!("\n");
        return;
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  VISION LLM BENCHMARK: {} images                                  ║",
        images.len()
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let mut group = c.benchmark_group("vision");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(3));

    // Single image with Vision LLM
    group.bench_function("single_image", |b| {
        let img = images[0].to_string_lossy().to_string();

        b.iter(|| {
            rt.block_on(async {
                let result = Forgetless::new()
                    .config(Config::default().context_limit(16_000).vision_llm(true))
                    .add(WithPriority::critical("Analyze this diagram."))
                    .add_file(FileWithPriority::high(&img))
                    .query("What does this image show?")
                    .run()
                    .await
                    .unwrap();
                black_box((result.total_tokens, result.stats.chunks_selected))
            })
        });
    });

    // Multiple images
    if images.len() >= 2 {
        group.bench_function("two_images", |b| {
            let imgs: Vec<_> = images
                .iter()
                .take(2)
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            b.iter(|| {
                rt.block_on(async {
                    let mut builder = Forgetless::new()
                        .config(Config::default().context_limit(16_000).vision_llm(true))
                        .add(WithPriority::critical("Compare these diagrams."))
                        .query("What do these images show?");

                    for img in &imgs {
                        builder = builder.add_file(FileWithPriority::high(img.as_str()));
                    }

                    let result = builder.run().await.unwrap();
                    black_box((result.total_tokens, result.stats.chunks_selected))
                })
            });
        });
    }

    // All images
    if images.len() >= 3 {
        group.bench_function("all_images", |b| {
            let imgs: Vec<_> = images
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            b.iter(|| {
                rt.block_on(async {
                    let mut builder = Forgetless::new()
                        .config(Config::default().context_limit(32_000).vision_llm(true))
                        .add(WithPriority::critical("Analyze all diagrams."))
                        .query("Describe each image and their relationships");

                    for img in &imgs {
                        builder = builder.add_file(FileWithPriority::high(img.as_str()));
                    }

                    let result = builder.run().await.unwrap();
                    black_box((result.total_tokens, result.stats.chunks_selected))
                })
            });
        });
    }

    group.finish();
}

// =============================================================================
// Detailed Vision Stats
// =============================================================================

fn bench_vision_detailed(c: &mut Criterion) {
    let images = get_image_files();
    if images.is_empty() {
        return;
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  DETAILED VISION BENCHMARK                                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    for img in &images {
        println!(
            "  - {}",
            img.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    let start = Instant::now();

    let result = rt.block_on(async {
        let mut builder = Forgetless::new()
            .config(Config::default().context_limit(16_000).vision_llm(true))
            .add(WithPriority::critical(
                "You are an expert at analyzing technical diagrams.",
            ))
            .query("Describe what each diagram shows and explain the key concepts");

        for img in &images {
            builder = builder.add_file(FileWithPriority::high(img.to_string_lossy().as_ref()));
        }

        builder.run().await.unwrap()
    });

    let elapsed = start.elapsed();

    println!("\n  Results:");
    println!("    Images:      {:>10}", images.len());
    println!("    Output:      {:>10} tokens", result.stats.output_tokens);
    println!("    Chunks:      {:>10}", result.stats.chunks_selected);
    println!("    Time:        {:>10.2}s", elapsed.as_secs_f64());
    println!("\n  Content preview:");
    println!(
        "    {}",
        &result.content.chars().take(200).collect::<String>()
    );
    println!("    ...");
    println!("\n");

    // Skip criterion (just show stats)
    let mut group = c.benchmark_group("vision_detailed");
    group.sample_size(10);
    group.finish();
}

// =============================================================================
// Mixed: Images + PDFs + Text
// =============================================================================

fn bench_mixed(c: &mut Criterion) {
    let images = get_image_files();

    let pdfs: Vec<_> = fs::read_dir(DATA_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().map(|x| x == "pdf").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    if images.is_empty() || pdfs.is_empty() {
        println!("\n  Skipping mixed benchmark (need both images and PDFs)\n");
        return;
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  MIXED BENCHMARK: {} images + {} PDFs                              ║",
        images.len(),
        pdfs.len()
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let mut group = c.benchmark_group("mixed");
    group.sample_size(10);

    group.bench_function("images_pdfs_text", |b| {
        let imgs: Vec<_> = images
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let pdf_paths: Vec<_> = pdfs
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let conversation =
            "User: Explain transformers.\nAssistant: Transformers use self-attention...";

        b.iter(|| {
            rt.block_on(async {
                let mut builder = Forgetless::new()
                    .config(Config::default().context_limit(16_000).vision_llm(true))
                    .add(WithPriority::critical("You are an AI research expert."))
                    .add(WithPriority::high(conversation))
                    .query("Explain the transformer architecture using the diagrams and papers");

                for img in &imgs {
                    builder = builder.add_file(FileWithPriority::high(img.as_str()));
                }
                for pdf in &pdf_paths {
                    builder = builder.add_file(pdf.as_str());
                }

                let result = builder.run().await.unwrap();
                black_box((result.total_tokens, result.stats.compression_ratio))
            })
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_vision_detailed, bench_vision, bench_mixed
}

criterion_main!(benches);
