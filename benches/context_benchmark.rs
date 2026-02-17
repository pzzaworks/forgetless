use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forgetless::{ContextConfig, ContextManager, Priority};

fn benchmark_token_counting(c: &mut Criterion) {
    let config = ContextConfig::default();
    let manager = ContextManager::new(config).unwrap();
    let counter = manager.token_counter();

    let short_text = "Hello, world!";
    let medium_text = "This is a medium length text that contains several sentences. ".repeat(10);
    let long_text = "This is a long text for benchmarking token counting performance. ".repeat(100);

    c.bench_function("count_short", |b| {
        b.iter(|| counter.count(black_box(short_text)))
    });

    c.bench_function("count_medium", |b| {
        b.iter(|| counter.count(black_box(&medium_text)))
    });

    c.bench_function("count_long", |b| {
        b.iter(|| counter.count(black_box(&long_text)))
    });
}

fn benchmark_context_building(c: &mut Criterion) {
    let config = ContextConfig::default().with_max_tokens(8000);
    let mut manager = ContextManager::new(config).unwrap();

    manager.set_system("You are a helpful assistant.");
    for i in 0..20 {
        manager.add_user(format!("User message {}", i)).unwrap();
        manager.add_assistant(format!("Assistant response {}", i)).unwrap();
    }

    for i in 0..10 {
        manager.add(
            format!("item_{}", i),
            format!("Context item {} with some content", i),
            Priority::Medium,
        );
    }

    c.bench_function("build_context", |b| {
        b.iter(|| manager.build())
    });
}

criterion_group!(benches, benchmark_token_counting, benchmark_context_building);
criterion_main!(benches);
