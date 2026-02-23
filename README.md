<p align="center">
  <img src="assets/forgetless.png" alt="Forgetless" width="200" />
</p>

<p align="center">
  Smart context management for LLMs - never forget what matters.
</p>

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```toml
[dependencies]
forgetless = { git = "https://github.com/pzzaworks/forgetless" }
```

## Basic Usage

```rust
use forgetless::{Forgetless, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = Forgetless::new()
        .config(Config::default().context_limit(128_000))
        .add(&large_content)
        .run()
        .await?;

    println!("{}", result.content);
    Ok(())
}
```

## Core Features

- **Smart Compression**: Intelligent content prioritization and compression to fit any token budget
- **Multi-Format Support**: PDF, images, text, code - all handled seamlessly with automatic extraction
- **Priority System**: Critical/High/Medium/Low/Minimal priority levels for fine-grained control
- **Query-Based Filtering**: LLM-powered relevance scoring to keep only what matters
- **Vision Processing**: Image analysis and description with GPU acceleration (Metal/CUDA)
- **Semantic Chunking**: Syntax-aware chunking for code and semantic boundaries for text
- **Embedding Support**: Vector-based similarity scoring for semantic retrieval
- **Agent Memory**: Cognitive-inspired memory architecture (Working/Episodic/Semantic)
- **HTTP Server**: REST API for language-agnostic integration

## Documentation

For detailed documentation and examples, visit:
- [Getting Started](https://forgetless.org/getting-started)
- [API Reference](https://forgetless.org/api)

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
