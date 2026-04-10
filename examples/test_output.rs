use forgetless::{Config, Forgetless, WithPriority};

#[tokio::main]
async fn main() {
    // Test 1: Text compression with mixed content
    println!("=== TEST 1: Mixed Content with Query ===");
    let system = "You are a helpful assistant.";
    let irrelevant = "The weather today is sunny. I like pizza. Random noise text.".repeat(20);
    let relevant = "Transformers use self-attention mechanisms to process sequences. \
                   The key innovation is parallel processing of all positions. \
                   Multi-head attention allows the model to focus on different aspects.";

    let result = Forgetless::new()
        .config(Config::default().context_limit(200))
        .add(WithPriority::critical(system))
        .add(&irrelevant)
        .add(WithPriority::high(relevant))
        .query("How do transformers work?")
        .run()
        .await
        .unwrap();

    println!("Input tokens: {}", result.stats.input_tokens);
    println!("Output tokens: {}", result.stats.output_tokens);
    println!("Compression: {:.1}x", result.stats.compression_ratio);
    println!(
        "Chunks selected: {} / {}",
        result.stats.chunks_selected, result.stats.chunks_processed
    );
    println!("\nContent:\n{}\n", &result.content);

    // Test 2: PDF with specific query
    println!("=== TEST 2: PDF with Query ===");
    let result = Forgetless::new()
        .config(Config::default().context_limit(4000))
        .add_file("benches/data/attention_paper.pdf")
        .query("What is the main contribution of this paper?")
        .run()
        .await
        .unwrap();

    println!("Input tokens: {}", result.stats.input_tokens);
    println!("Output tokens: {}", result.stats.output_tokens);
    println!(
        "Chunks: {} -> {}",
        result.stats.chunks_processed, result.stats.chunks_selected
    );
    println!(
        "\nContent (first 500 chars):\n{}\n",
        &result.content.chars().take(500).collect::<String>()
    );

    // Test 3: Multiple PDFs compression
    println!("=== TEST 3: Two PDFs Compressed ===");
    let result = Forgetless::new()
        .config(Config::default().context_limit(3000))
        .add_file("benches/data/attention_paper.pdf")
        .add_file("benches/data/vit_paper.pdf")
        .query("Compare the architectures")
        .run()
        .await
        .unwrap();

    println!("Input tokens: {}", result.stats.input_tokens);
    println!("Output tokens: {}", result.stats.output_tokens);
    println!("Compression: {:.1}x", result.stats.compression_ratio);
    println!(
        "Chunks: {} -> {}",
        result.stats.chunks_processed, result.stats.chunks_selected
    );
}
