# Forgetless

Smart context management for LLMs - never forget what matters.

[![Crates.io](https://img.shields.io/crates/v/forgetless.svg)](https://crates.io/crates/forgetless)
[![Documentation](https://docs.rs/forgetless/badge.svg)](https://docs.rs/forgetless)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Rust library for intelligent context window management in Large Language Models. Maximize the value of every token through smart prioritization, semantic chunking, embedding-based retrieval, and cognitive-inspired agent memory.

## Features

- **Smart Chunking** - Semantic-aware text and code chunking that respects natural boundaries
- **Priority-based Retention** - Keep important information, compress or drop the rest
- **Token Budget Management** - Precise token counting for all major LLM providers
- **Embedding Support** - Semantic similarity scoring with cosine similarity
- **Agent Memory** - Cognitive-inspired memory architecture (Working/Episodic/Semantic)
- **Multi-modal** - Image token counting for vision models
- **Relevance Scoring** - Score and rank context items by recency, semantic relevance, and priority
- **Conversation Memory** - Long-term memory with automatic compression for multi-turn conversations

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
        .with_max_tokens(200_000)
        .with_model("claude-sonnet-4.5");

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

## Supported Models (February 2026)

| Provider | Models | Context |
|----------|--------|---------|
| **OpenAI** | GPT-5.3 Codex, GPT-5.2, GPT-4o, gpt-oss-120b | Up to 400K |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 | Up to 200K |
| **Google** | Gemini 3 Pro/Flash, Gemini 2.5 | Up to 1M |
| **xAI** | Grok 4.1 Fast | 128K |
| **DeepSeek** | DeepSeek V3.2 | 256K |
| **Alibaba** | Qwen3-Coder-480B | 128K |
| **Meta** | Llama 4 | 128K |
| **Mistral** | Mistral Large 2 | 128K |
| **Custom** | Configurable chars-per-token ratio | Any |

## Architecture

```
forgetless/
├── context/    # Main context manager - orchestrates everything
├── memory/     # Conversation memory and long-term storage
├── agent/      # Agent memory patterns (Working/Episodic/Semantic)
├── embedding/  # Semantic similarity and embedding support
├── chunking/   # Semantic text and code chunking
├── scoring/    # Priority and relevance scoring
└── token/      # Token counting with multi-modal support
```

## Advanced Usage

### Agent Memory

```rust
use forgetless::{AgentMemory, AgentMemoryConfig, MemoryEntry, MemoryType, Priority};

let config = AgentMemoryConfig {
    working_memory_tokens: 8000,
    working_memory_size: 20,
    episodic_memory_size: 1000,
    semantic_memory_size: 500,
    consolidation_threshold: 0.5,
    auto_consolidate: true,
};

let mut memory = AgentMemory::new(config, token_counter);

// Add to working memory (current task context)
memory.add_working(MemoryEntry::new("task1", "Current task details", MemoryType::Working));

// Add to semantic memory (long-term knowledge)
memory.add_semantic(
    MemoryEntry::new("fact1", "The sky is blue", MemoryType::Semantic)
        .with_priority(Priority::High)
);

// Search across memory types
let results = memory.search("sky", &[MemoryType::Working, MemoryType::Semantic]);
```

### Embedding-based Retrieval

```rust
use forgetless::{SemanticScorer, EmbeddedItem, EmbeddingModel};

// Create scorer with query embedding
let scorer = SemanticScorer::new()
    .with_query(query_embedding)
    .with_threshold(0.7);

// Score and rank items
let ranked = scorer.rank_by_similarity(&embedded_items);

// Filter by threshold
let relevant = scorer.filter_and_rank(&embedded_items);
```

### Multi-modal Token Counting

```rust
use forgetless::{TokenCounter, TokenizerModel, ImageDimensions, ImageDetail};

let counter = TokenCounter::new(TokenizerModel::Gpt4o)?;

// Count image tokens
let dims = ImageDimensions { width: 1024, height: 768 };
let image_tokens = counter.count_image(dims, ImageDetail::High);

// Count multiple images
let images = vec![
    (ImageDimensions { width: 512, height: 512 }, ImageDetail::Low),
    (ImageDimensions { width: 2048, height: 1536 }, ImageDetail::High),
];
let total = counter.count_images(&images);
```

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

- **Chatbots** - Manage conversation history within token limits
- **RAG Systems** - Optimize retrieved context for maximum relevance
- **Code Assistants** - Smart code context with syntax-aware chunking
- **Document Q&A** - Efficient document chunking and prioritization
- **Agent Systems** - Cognitive-inspired memory for autonomous agents
- **Multi-modal Apps** - Vision model token optimization

## Performance

- Sub-microsecond token counting for cached content
- Zero-copy chunking where possible
- Efficient priority queue for context selection
- 144 tests with ~99% coverage

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

Project Link: [https://github.com/pzzaworks/forgetless](https://github.com/pzzaworks/forgetless)
