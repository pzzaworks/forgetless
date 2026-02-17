//! Basic usage example for Forgetless
//!
//! Run with: cargo run --example basic_usage

use forgetless::{ContextConfig, ContextManager, Priority};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a context manager with default config
    let config = ContextConfig::default()
        .with_max_tokens(8000)
        .with_model("gpt-4")
        .with_reserved_tokens(1000);

    let mut manager = ContextManager::new(config)?;

    // Set system prompt
    manager.set_system(
        "You are a helpful AI assistant that helps users with programming questions.",
    );

    // Simulate a conversation
    manager.add_user("What is Rust's ownership model?")?;
    manager.add_assistant(
        "Rust's ownership model is a set of rules that govern how memory is managed. \
         Each value has a single owner, and when the owner goes out of scope, the value is dropped.",
    )?;

    manager.add_user("Can you explain borrowing?")?;
    manager.add_assistant(
        "Borrowing allows you to reference a value without taking ownership. \
         There are two types: immutable borrows (&T) and mutable borrows (&mut T).",
    )?;

    // Add some context items with different priorities
    manager.add(
        "rust_docs",
        "Rust documentation excerpt about memory safety...",
        Priority::High,
    );

    manager.add(
        "code_example",
        r#"
fn main() {
    let s1 = String::from("hello");
    let s2 = &s1; // Borrow
    println!("{}", s2);
}
"#,
        Priority::Medium,
    );

    manager.add(
        "background_info",
        "Some background information about systems programming...",
        Priority::Low,
    );

    // Build the optimized context
    let context = manager.build()?;

    // Print statistics
    println!("=== Forgetless Context Statistics ===");
    println!("Total tokens used: {}", context.total_tokens);
    println!("Available tokens: {}", context.available_tokens);
    println!("Remaining for response: {}", context.remaining_tokens());
    println!("Messages included: {}", context.messages.len());
    println!("Context items included: {}", context.items.len());
    println!("Items excluded: {}", context.excluded_count);
    println!();

    // Print the built context
    println!("=== Built Context ===");
    if let Some(ref sys) = context.system {
        println!("[System] {}", sys);
        println!();
    }

    for msg in &context.messages {
        println!("[{:?}] {}", msg.role, msg.content);
        println!();
    }

    for item in &context.items {
        println!("[Context: {}] (priority: {:?})", item.id, item.score.priority);
        println!("{}", item.content);
        println!();
    }

    Ok(())
}
