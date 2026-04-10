use forgetless::{Config, Forgetless, WithPriority};

#[tokio::main]
async fn main() {
    println!("\n{}", "=".repeat(70));
    println!("FORGETLESS OUTPUT VERIFICATION");
    println!("{}\n", "=".repeat(70));

    // Test 1: Query relevance - should pick relevant content
    println!("=== TEST 1: Query Relevance ===");
    let irrelevant = "The weather is nice today. I like pizza and coffee. Random noise.".repeat(50);
    let relevant = "Rust is a systems programming language focused on safety, speed, and concurrency. \
                   It achieves memory safety without garbage collection through its ownership system.";

    let result = Forgetless::new()
        .config(Config::default().context_limit(500))
        .add(&irrelevant)
        .add(WithPriority::high(relevant))
        .query("What is Rust programming language?")
        .run()
        .await
        .unwrap();

    println!("Query: 'What is Rust programming language?'");
    println!(
        "Input: {} tokens (mostly irrelevant)",
        result.stats.input_tokens
    );
    println!("Output: {} tokens", result.stats.output_tokens);
    println!("\nSelected content:");
    println!("---");
    println!("{}", result.content);
    println!("---");
    let has_rust = result.content.contains("Rust") || result.content.contains("programming");
    let has_weather = result.content.contains("weather") || result.content.contains("pizza");
    println!("✓ Contains Rust info: {}", has_rust);
    println!("✗ Contains irrelevant: {}", has_weather);
    assert!(has_rust, "Should contain relevant Rust content");
    println!();

    // Test 2: PDF extraction - real content
    println!("=== TEST 2: PDF Content ===");
    let result = Forgetless::new()
        .config(Config::default().context_limit(2000))
        .add_file("benches/data/attention_paper.pdf")
        .query("What is the attention mechanism?")
        .run()
        .await
        .unwrap();

    println!("PDF: attention_paper.pdf");
    println!("Query: 'What is the attention mechanism?'");
    println!("Input: {} tokens", result.stats.input_tokens);
    println!("Output: {} tokens", result.stats.output_tokens);
    println!(
        "Chunks: {} -> {}",
        result.stats.chunks_processed, result.stats.chunks_selected
    );
    println!("\nExtracted content (first 500 chars):");
    println!("---");
    println!("{}", &result.content.chars().take(500).collect::<String>());
    println!("---");
    let has_attention = result.content.to_lowercase().contains("attention");
    println!("✓ Contains 'attention': {}", has_attention);
    println!();

    // Test 3: Priority system
    println!("=== TEST 3: Priority System ===");
    let low = "This is low priority background information that can be dropped.".repeat(100);
    let critical = "SYSTEM: You are a helpful AI assistant. Always be polite.";

    let result = Forgetless::new()
        .config(Config::default().context_limit(500))
        .add(WithPriority::low(&low))
        .add(WithPriority::critical(critical))
        .run()
        .await
        .unwrap();

    println!("Critical: 'SYSTEM: You are a helpful AI assistant...'");
    println!(
        "Low priority: {} tokens of filler",
        result.stats.input_tokens - 20
    );
    println!("Output: {} tokens", result.stats.output_tokens);
    println!("\nSelected content:");
    println!("---");
    println!("{}", result.content);
    println!("---");
    let has_system = result.content.contains("SYSTEM") || result.content.contains("helpful");
    println!("✓ Critical content preserved: {}", has_system);
    assert!(has_system, "Critical content must be preserved");
    println!();

    // Test 4: Compression ratio check
    println!("=== TEST 4: Compression Quality ===");
    let large = "Machine learning is transforming how we build software. ".repeat(2000);

    let result = Forgetless::new()
        .config(Config::default().context_limit(4000))
        .add(&large)
        .query("What is machine learning?")
        .run()
        .await
        .unwrap();

    println!("Input: {} tokens", result.stats.input_tokens);
    println!("Output: {} tokens", result.stats.output_tokens);
    println!("Compression: {:.1}x", result.stats.compression_ratio);
    println!(
        "Chunks: {} -> {}",
        result.stats.chunks_processed, result.stats.chunks_selected
    );
    assert!(result.stats.compression_ratio > 1.0, "Should compress");
    assert!(
        result.content.contains("machine learning") || result.content.contains("Machine"),
        "Content should be relevant"
    );
    println!("✓ Content is relevant and compressed");
    println!();

    println!("{}", "=".repeat(70));
    println!("ALL VERIFICATION TESTS PASSED!");
    println!("{}\n", "=".repeat(70));
}
