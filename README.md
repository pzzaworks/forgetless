<p align="center">
  <h1 align="center">Forgetless</h1>
  <p align="center">
    <strong>Smart context management for LLMs — never forget what matters.</strong>
  </p>
</p>

<p align="center">
  <a href="https://crates.io/crates/forgetless"><img src="https://img.shields.io/crates/v/forgetless.svg" alt="Crates.io"></a>
  <a href="https://docs.rs/forgetless"><img src="https://docs.rs/forgetless/badge.svg" alt="Documentation"></a>
  <a href="https://github.com/forgetless/forgetless/actions"><img src="https://github.com/forgetless/forgetless/workflows/CI/badge.svg" alt="CI Status"></a>
  <a href="https://codecov.io/gh/forgetless/forgetless"><img src="https://codecov.io/gh/forgetless/forgetless/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

High-performance Rust library for intelligent context window management in Large Language Models. Maximize the value of every token through smart prioritization, semantic chunking, and conversation memory.

## Features

- **Smart Chunking** — Semantic-aware text and code chunking that respects natural boundaries
- **Priority-based Retention** — Keep important information, compress or drop the rest
- **Token Budget Management** — Precise token counting compatible with OpenAI and Anthropic models
- **Relevance Scoring** — Score and rank context items by recency, semantic relevance, and priority
- **Conversation Memory** — Long-term memory with automatic compression for multi-turn conversations
- **Zero-copy Design** — Efficient memory usage for high-throughput applications

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
forgetless = "0.1"
```

## Quick Start

```rust
use forgetless::{ContextManager, ContextConfig, Priority};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a context manager
    let config = ContextConfig::default()
        .with_max_tokens(8000)
        .with_model("gpt-4");

    let mut manager = ContextManager::new(config)?;

    // Set system prompt
    manager.set_system("You are a helpful assistant.");

    // Add conversation
    manager.add_user("What is Rust?")?;
    manager.add_assistant("Rust is a systems programming language...")?;

    // Add context with priorities
    manager.add("docs", "Important documentation...", Priority::High);
    manager.add("background", "Additional context...", Priority::Low);

    // Build optimized context within token budget
    let context = manager.build()?;

    println!("Tokens used: {}/{}", context.total_tokens, context.available_tokens);
    println!("Items included: {}", context.items.len());
    println!("Items excluded: {}", context.excluded_count);

    Ok(())
}
```

## Priority Levels

| Priority | Score | Use Case |
|----------|-------|----------|
| `Critical` | 100 | System prompts, essential instructions |
| `High` | 75 | Recent user queries, important context |
| `Medium` | 50 | Relevant background information |
| `Low` | 25 | Nice-to-have context |
| `Minimal` | 10 | Can be dropped first |

## Supported Models

| Model | Tokenizer |
|-------|-----------|
| GPT-4 / GPT-4 Turbo / GPT-4o | `cl100k_base` |
| GPT-3.5 Turbo | `cl100k_base` |
| Claude 3 / Claude 3.5 | Approximate (`cl100k_base`) |
| Custom | Configurable chars-per-token ratio |

## Architecture

```
forgetless/
├── context/    # Main context manager — orchestrates everything
├── memory/     # Conversation memory & long-term storage
├── chunking/   # Semantic text and code chunking
├── scoring/    # Priority and relevance scoring
└── token/      # Token counting (tiktoken-compatible)
```

## Advanced Usage

### Document Chunking

```rust
use forgetless::{ContextManager, ContextConfig, ContentType};

let mut manager = ContextManager::new(ContextConfig::default())?;

// Add a large document - automatically chunked
manager.add_document("readme", &readme_content, ContentType::Markdown);

// Add code with syntax-aware chunking
manager.add_document("main.rs", &code_content, ContentType::Code);
```

### Pinned Items

```rust
// Pin critical context that must always be included
manager.add("api_key_format", "API keys must start with 'sk-'", Priority::High);
manager.pin_item("api_key_format");
```

### Conversation Memory

```rust
use forgetless::{ConversationMemory, MemoryConfig, Role};

let mut memory = ConversationMemory::new(
    MemoryConfig {
        max_messages: 100,
        max_tokens: 8000,
        auto_summarize: true,
        ..Default::default()
    },
    token_counter,
);

memory.set_system_prompt("You are helpful.");
memory.add_user("Hello!")?;
memory.add_assistant("Hi there!")?;

// Get messages within token budget
let messages = memory.get_messages_within_budget(4000);
```

## Use Cases

- **Chatbots** — Manage conversation history within token limits
- **RAG Systems** — Optimize retrieved context for maximum relevance
- **Code Assistants** — Smart code context with syntax-aware chunking
- **Document Q&A** — Efficient document chunking and prioritization
- **Agent Systems** — Memory management for autonomous agents

## Performance

- Sub-microsecond token counting for cached content
- Zero-copy chunking where possible
- Efficient priority queue for context selection
- ~99% test coverage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Project Link: [https://github.com/forgetless/forgetless](https://github.com/forgetless/forgetless)
