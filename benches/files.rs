//! File Benchmarks - Requires PDF files in benches/data/
//!
//! Tests real-world scenarios with PDF documents.
//!
//! Setup:
//!   1. Place your PDF files in benches/data/
//!   2. Run: cargo bench --bench files
//!
//! If no PDFs found, benchmarks will be skipped.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forgetless::{Config, FileWithPriority, Forgetless, WithPriority};
use std::fs;
use std::path::Path;
use std::time::Instant;

const DATA_DIR: &str = "benches/data";

fn get_pdf_files() -> Vec<std::path::PathBuf> {
    let path = Path::new(DATA_DIR);
    if !path.exists() {
        return vec![];
    }

    let mut pdfs = Vec::new();

    // Check main directory and subdirectories
    for entry in fs::read_dir(path).into_iter().flatten().flatten() {
        let p = entry.path();
        if p.extension().map(|x| x == "pdf").unwrap_or(false) {
            pdfs.push(p);
        } else if p.is_dir() {
            // Check subdirectories
            if let Ok(entries) = fs::read_dir(&p) {
                for sub in entries.flatten() {
                    if sub.path().extension().map(|x| x == "pdf").unwrap_or(false) {
                        pdfs.push(sub.path());
                    }
                }
            }
        }
    }

    pdfs
}

// =============================================================================
// PDF Benchmarks
// =============================================================================

fn bench_pdfs(c: &mut Criterion) {
    let pdfs = get_pdf_files();

    if pdfs.is_empty() {
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  No PDFs found in benches/data/                                  ║");
        println!("║  Place your PDF files there to run file benchmarks.              ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!("\n");
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Found {} PDF files for benchmarking                              ║",
        pdfs.len()
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let mut group = c.benchmark_group("pdfs");
    group.sample_size(10);

    // Single PDF
    if !pdfs.is_empty() {
        group.bench_function("single_to_32K", |b| {
            let pdf = pdfs[0].to_string_lossy().to_string();

            b.iter(|| {
                rt.block_on(async {
                    let result = Forgetless::new()
                        .config(Config::default().context_limit(32_000))
                        .add_file(pdf.as_str())
                        .query("Summarize this document")
                        .run()
                        .await
                        .unwrap();
                    black_box((result.total_tokens, result.stats.compression_ratio))
                })
            });
        });
    }

    // Multiple PDFs
    if pdfs.len() >= 5 {
        group.bench_function("5_pdfs_to_64K", |b| {
            let files: Vec<_> = pdfs
                .iter()
                .take(5)
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            b.iter(|| {
                rt.block_on(async {
                    let mut builder = Forgetless::new()
                        .config(Config::default().context_limit(64_000))
                        .query("Compare the approaches");

                    for f in &files {
                        builder = builder.add_file(f.as_str());
                    }

                    let result = builder.run().await.unwrap();
                    black_box((result.total_tokens, result.stats.compression_ratio))
                })
            });
        });
    }

    // All PDFs
    if pdfs.len() >= 10 {
        group.bench_function("all_to_128K", |b| {
            let files: Vec<_> = pdfs
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            b.iter(|| {
                rt.block_on(async {
                    let mut builder = Forgetless::new()
                        .config(Config::default().context_limit(128_000))
                        .query("Key findings across all documents");

                    for f in &files {
                        builder = builder.add_file(f.as_str());
                    }

                    let result = builder.run().await.unwrap();
                    black_box((result.total_tokens, result.stats.compression_ratio))
                })
            });
        });
    }

    group.finish();
}

// =============================================================================
// Detailed Stats (Single Run)
// =============================================================================

fn bench_detailed(c: &mut Criterion) {
    let pdfs = get_pdf_files();
    if pdfs.is_empty() {
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  DETAILED BENCHMARK: {} PDFs                                      ║",
        pdfs.len()
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let start = Instant::now();

    let result = rt.block_on(async {
        let mut builder = Forgetless::new()
            .config(Config::default().context_limit(8_000)) // Low limit to force compression
            .add(WithPriority::critical(
                "You are an expert research assistant.",
            ))
            .query("Summarize and compare the key findings");

        for pdf in &pdfs {
            builder = builder.add_file(pdf.to_string_lossy().as_ref());
        }

        builder.run().await.unwrap()
    });

    let elapsed = start.elapsed();

    println!("\n  Results:");
    println!(
        "    Input:       {:>10} tokens ({:.2}M)",
        result.stats.input_tokens,
        result.stats.input_tokens as f64 / 1_000_000.0
    );
    println!(
        "    Output:      {:>10} tokens ({:.1}K)",
        result.stats.output_tokens,
        result.stats.output_tokens as f64 / 1_000.0
    );
    println!("    Compression: {:>10.1}x", result.stats.compression_ratio);
    println!(
        "    Chunks:      {:>10} -> {}",
        result.stats.chunks_processed, result.stats.chunks_selected
    );
    println!("    Time:        {:>10.2}s", elapsed.as_secs_f64());
    println!(
        "    Throughput:  {:>10.0} tokens/sec",
        result.stats.input_tokens as f64 / elapsed.as_secs_f64()
    );
    println!("\n");

    // Skip criterion benchmark (just stats)
    let mut group = c.benchmark_group("detailed");
    group.sample_size(10);
    group.finish();
}

// =============================================================================
// Priority with Files
// =============================================================================

fn bench_priorities(c: &mut Criterion) {
    let pdfs = get_pdf_files();
    if pdfs.len() < 3 {
        return;
    }

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("file_priorities");
    group.sample_size(10);

    group.bench_function("mixed_priorities", |b| {
        let files: Vec<_> = pdfs
            .iter()
            .take(3)
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        b.iter(|| {
            rt.block_on(async {
                let result = Forgetless::new()
                    .config(Config::default().context_limit(32_000))
                    .add(WithPriority::critical("Focus on innovations."))
                    .add_file(FileWithPriority::high(&files[0]))
                    .add_file(FileWithPriority::medium(&files[1]))
                    .add_file(FileWithPriority::low(&files[2]))
                    .query("What are the key innovations?")
                    .run()
                    .await
                    .unwrap();
                black_box((result.total_tokens, result.stats.chunks_selected))
            })
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_detailed, bench_pdfs, bench_priorities
}

criterion_main!(benches);
