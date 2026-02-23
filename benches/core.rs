//! Core Benchmarks - No external files required
//!
//! Tests text processing, chunking, and compression with generated content.
//! Run with: cargo bench --bench core

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forgetless::{Config, Forgetless, WithPriority};

// =============================================================================
// Data Generators
// =============================================================================

fn generate_conversation(messages: usize) -> String {
    let mut conv = String::new();
    let topics = [
        "transformer architectures and self-attention mechanisms",
        "diffusion models for image generation",
        "reinforcement learning from human feedback",
        "large language model scaling laws",
    ];

    for i in 0..messages {
        let topic = topics[i % topics.len()];
        conv.push_str(&format!(
            "[User {}]: Can you explain {} in detail?\n\n\
             [Assistant {}]: {} is a fundamental concept in modern AI. \
             The underlying mechanism relies on mathematical transformations. \
             Practical applications include NLP, computer vision, and multimodal systems. \
             Recent advances focus on scaling and efficiency.\n\n",
            i + 1, topic, i + 1, topic
        ));
    }
    conv
}

fn generate_document(thousands_of_tokens: usize) -> String {
    let content = r#"
# Research Document

## Introduction
This document explores fundamental concepts in artificial intelligence systems.
The transformer architecture has revolutionized sequence modeling tasks.
Self-attention mechanisms enable models to capture long-range dependencies.

## Methodology
Our approach combines multiple state-of-the-art techniques for optimal performance.
Sparse attention patterns reduce computational complexity from O(n²) to O(n log n).
Mixed precision training accelerates throughput while maintaining numerical stability.

## Results
| Model | Parameters | Accuracy | Latency |
|-------|------------|----------|---------|
| Base  | 125M       | 78.3%    | 45ms    |
| Large | 350M       | 82.1%    | 112ms   |
| XL    | 1.3B       | 85.7%    | 287ms   |

## Discussion
Scaling behavior follows predictable power laws.
Efficiency gains through architectural innovations remain crucial.
Future directions include novel attention mechanisms and emergent capabilities.

"#;
    content.repeat(thousands_of_tokens)
}

// =============================================================================
// Large Scale Benchmarks
// =============================================================================

fn bench_large_scale(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10);

    // 500K tokens → 128K
    group.bench_function("500K_to_128K", |b| {
        let doc = generate_document(500);
        let conv = generate_conversation(50);

        b.iter(|| {
            rt.block_on(async {
                let result = Forgetless::new()
                    .config(Config::default().context_limit(128_000))
                    .add(WithPriority::critical("You are an AI assistant."))
                    .add(WithPriority::high(&conv))
                    .add(&doc)
                    .query("Summarize the key findings")
                    .run()
                    .await
                    .unwrap();
                black_box((result.total_tokens, result.stats.compression_ratio))
            })
        });
    });

    // 1M tokens → 128K
    group.bench_function("1M_to_128K", |b| {
        let doc = generate_document(1000);
        let conv = generate_conversation(100);

        b.iter(|| {
            rt.block_on(async {
                let result = Forgetless::new()
                    .config(Config::default().context_limit(128_000))
                    .add(WithPriority::critical("You are an AI assistant."))
                    .add(WithPriority::high(&conv))
                    .add(&doc)
                    .query("What are the main contributions?")
                    .run()
                    .await
                    .unwrap();
                black_box((result.total_tokens, result.stats.compression_ratio))
            })
        });
    });

    group.finish();
}

// =============================================================================
// Compression Benchmarks
// =============================================================================

fn bench_compression(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("compression");
    group.sample_size(10);

    let doc = generate_document(500);

    for target in [128_000, 64_000, 32_000, 16_000] {
        group.bench_function(BenchmarkId::new("500K_to", target), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let result = Forgetless::new()
                        .config(Config::default().context_limit(target))
                        .add(&doc)
                        .query("Summarize")
                        .run()
                        .await
                        .unwrap();
                    black_box((result.total_tokens, result.stats.compression_ratio))
                })
            });
        });
    }

    group.finish();
}

// =============================================================================
// Priority Benchmarks
// =============================================================================

fn bench_priorities(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("priorities");
    group.sample_size(10);

    let doc = generate_document(200);
    let conv = generate_conversation(50);

    group.bench_function("mixed", |b| {
        b.iter(|| {
            rt.block_on(async {
                let result = Forgetless::new()
                    .config(Config::default().context_limit(32_000))
                    .add(WithPriority::critical("SYSTEM: You are an AI."))
                    .add(WithPriority::high(&conv))
                    .add(WithPriority::low(&doc))
                    .query("Focus on important parts")
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
    targets = bench_large_scale, bench_compression, bench_priorities
}

criterion_main!(benches);
