"use client";

import { useState } from "react";
import Link from "next/link";

const navigation = [
  { name: "Getting Started", href: "#getting-started" },
  { name: "Installation", href: "#installation" },
  { name: "Quick Start", href: "#quick-start" },
  { name: "Core Concepts", href: "#core-concepts" },
  { name: "Context Manager", href: "#context-manager" },
  { name: "Agent Memory", href: "#agent-memory" },
  { name: "Embeddings", href: "#embeddings" },
  { name: "Multi-modal", href: "#multi-modal" },
  { name: "Supported Models", href: "#supported-models" },
  { name: "API Reference", href: "#api-reference" },
];

function CodeBlock({ children, language = "rust" }: { children: string; language?: string }) {
  return (
    <div className="relative group">
      <div className="absolute right-3 top-3 text-zinc-600 text-xs">{language}</div>
      <pre className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 overflow-x-auto">
        <code className="text-sm text-zinc-300 leading-relaxed">{children}</code>
      </pre>
    </div>
  );
}

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <span className="text-zinc-100 text-lg font-semibold tracking-tight">forgetless</span>
              <span className="text-zinc-600 text-sm">v0.1.0</span>
            </div>
            <div className="flex items-center gap-6">
              <a
                href="https://github.com/pzzaworks/forgetless"
                target="_blank"
                rel="noopener noreferrer"
                className="text-zinc-400 hover:text-zinc-100 transition-colors text-sm"
              >
                GitHub
              </a>
              <a
                href="https://crates.io/crates/forgetless"
                target="_blank"
                rel="noopener noreferrer"
                className="text-zinc-400 hover:text-zinc-100 transition-colors text-sm"
              >
                crates.io
              </a>
            </div>
          </div>
        </div>
      </header>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside className="hidden lg:block fixed left-0 top-16 bottom-0 w-64 border-r border-zinc-800 overflow-y-auto">
          <nav className="p-6 space-y-1">
            {navigation.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className="block px-3 py-2 text-sm text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 rounded-md transition-colors"
              >
                {item.name}
              </a>
            ))}
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 lg:ml-64 min-h-screen">
          <div className="max-w-4xl mx-auto px-6 py-12 lg:px-12">

            {/* Hero */}
            <section className="mb-16">
              <h1 className="text-4xl font-bold text-zinc-100 mb-4 tracking-tight">
                forgetless
              </h1>
              <p className="text-xl text-zinc-400 mb-8 leading-relaxed">
                Smart context management for LLMs — never forget what matters.
              </p>
              <div className="flex flex-wrap gap-3">
                <span className="px-3 py-1 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-400">
                  Rust
                </span>
                <span className="px-3 py-1 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-400">
                  LLM
                </span>
                <span className="px-3 py-1 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-400">
                  Context Management
                </span>
                <span className="px-3 py-1 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-400">
                  Agent Memory
                </span>
              </div>
            </section>

            {/* Getting Started */}
            <section id="getting-started" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Getting Started</h2>
              <p className="text-zinc-400 mb-6 leading-relaxed">
                Forgetless is a high-performance Rust library for intelligent context window management
                in Large Language Models. It helps you maximize the value of every token through smart
                prioritization, semantic chunking, embedding-based retrieval, and cognitive-inspired agent memory.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <h3 className="text-zinc-100 font-medium mb-2">Smart Chunking</h3>
                  <p className="text-zinc-500 text-sm">Semantic-aware text and code chunking</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <h3 className="text-zinc-100 font-medium mb-2">Priority Retention</h3>
                  <p className="text-zinc-500 text-sm">Keep important information, compress the rest</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <h3 className="text-zinc-100 font-medium mb-2">Agent Memory</h3>
                  <p className="text-zinc-500 text-sm">Working, Episodic, and Semantic memory</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <h3 className="text-zinc-100 font-medium mb-2">Multi-modal</h3>
                  <p className="text-zinc-500 text-sm">Image token counting for vision models</p>
                </div>
              </div>
            </section>

            {/* Installation */}
            <section id="installation" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Installation</h2>
              <p className="text-zinc-400 mb-4">Add forgetless to your Cargo.toml:</p>
              <CodeBlock language="toml">{`[dependencies]
forgetless = "0.1"`}</CodeBlock>
            </section>

            {/* Quick Start */}
            <section id="quick-start" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Quick Start</h2>
              <p className="text-zinc-400 mb-4">Create a context manager and start adding content:</p>
              <CodeBlock>{`use forgetless::{ContextManager, ContextConfig, Priority};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Build optimized context
    let context = manager.build()?;

    println!("Tokens: {}/{}", context.total_tokens, context.available_tokens);

    Ok(())
}`}</CodeBlock>
            </section>

            {/* Core Concepts */}
            <section id="core-concepts" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Core Concepts</h2>

              <h3 className="text-lg font-medium text-zinc-200 mt-6 mb-3">Priority Levels</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      <th className="text-left py-3 px-4 text-zinc-300 font-medium">Priority</th>
                      <th className="text-left py-3 px-4 text-zinc-300 font-medium">Score</th>
                      <th className="text-left py-3 px-4 text-zinc-300 font-medium">Use Case</th>
                    </tr>
                  </thead>
                  <tbody className="text-zinc-400">
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4"><code className="text-zinc-300">Critical</code></td>
                      <td className="py-3 px-4">100</td>
                      <td className="py-3 px-4">System prompts, essential instructions</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4"><code className="text-zinc-300">High</code></td>
                      <td className="py-3 px-4">75</td>
                      <td className="py-3 px-4">Recent user queries, important context</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4"><code className="text-zinc-300">Medium</code></td>
                      <td className="py-3 px-4">50</td>
                      <td className="py-3 px-4">Relevant background information</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4"><code className="text-zinc-300">Low</code></td>
                      <td className="py-3 px-4">25</td>
                      <td className="py-3 px-4">Nice-to-have context</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4"><code className="text-zinc-300">Minimal</code></td>
                      <td className="py-3 px-4">10</td>
                      <td className="py-3 px-4">Can be dropped first</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Context Manager */}
            <section id="context-manager" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Context Manager</h2>
              <p className="text-zinc-400 mb-4">
                The ContextManager is the main orchestrator that brings together chunking,
                scoring, and memory management.
              </p>
              <CodeBlock>{`// Pin critical context
manager.add("api_format", "API keys must start with 'sk-'", Priority::High);
manager.pin_item("api_format");

// Add documents with automatic chunking
manager.add_document("readme", &content, ContentType::Markdown);
manager.add_document("code", &source, ContentType::Code);

// Build and check budget
let context = manager.build()?;
println!("Excluded: {}", context.excluded_count);`}</CodeBlock>
            </section>

            {/* Agent Memory */}
            <section id="agent-memory" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Agent Memory</h2>
              <p className="text-zinc-400 mb-4">
                Cognitive-inspired memory architecture with three memory types:
              </p>
              <ul className="list-disc list-inside text-zinc-400 mb-6 space-y-2">
                <li><strong className="text-zinc-300">Working Memory</strong> — Current task context, limited capacity</li>
                <li><strong className="text-zinc-300">Episodic Memory</strong> — Past interactions and events</li>
                <li><strong className="text-zinc-300">Semantic Memory</strong> — Long-term knowledge and facts</li>
              </ul>
              <CodeBlock>{`use forgetless::{AgentMemory, AgentMemoryConfig, MemoryEntry, MemoryType};

let config = AgentMemoryConfig {
    working_memory_tokens: 8000,
    working_memory_size: 20,
    episodic_memory_size: 1000,
    semantic_memory_size: 500,
    consolidation_threshold: 0.5,
    auto_consolidate: true,
};

let mut memory = AgentMemory::new(config, token_counter);

// Add to working memory
memory.add_working(MemoryEntry::new("task1", "Current task", MemoryType::Working));

// Add to semantic memory
memory.add_semantic(
    MemoryEntry::new("fact1", "The sky is blue", MemoryType::Semantic)
        .with_priority(Priority::High)
);

// Search across memory types
let results = memory.search("sky", &[MemoryType::Working, MemoryType::Semantic]);`}</CodeBlock>
            </section>

            {/* Embeddings */}
            <section id="embeddings" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Embeddings</h2>
              <p className="text-zinc-400 mb-4">
                Semantic similarity scoring with cosine similarity for intelligent retrieval.
              </p>
              <CodeBlock>{`use forgetless::{SemanticScorer, EmbeddedItem};

let scorer = SemanticScorer::new()
    .with_query(query_embedding)
    .with_threshold(0.7);

// Rank items by similarity
let ranked = scorer.rank_by_similarity(&embedded_items);

// Filter and rank above threshold
let relevant = scorer.filter_and_rank(&embedded_items);`}</CodeBlock>
            </section>

            {/* Multi-modal */}
            <section id="multi-modal" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Multi-modal</h2>
              <p className="text-zinc-400 mb-4">
                Image token counting for vision models based on OpenAI&apos;s calculation.
              </p>
              <CodeBlock>{`use forgetless::{TokenCounter, TokenizerModel, ImageDimensions, ImageDetail};

let counter = TokenCounter::new(TokenizerModel::Gpt4o)?;

let dims = ImageDimensions { width: 1024, height: 768 };
let tokens = counter.count_image(dims, ImageDetail::High);

// Count multiple images
let images = vec![
    (ImageDimensions { width: 512, height: 512 }, ImageDetail::Low),
    (ImageDimensions { width: 2048, height: 1536 }, ImageDetail::High),
];
let total = counter.count_images(&images);`}</CodeBlock>
            </section>

            {/* Supported Models */}
            <section id="supported-models" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">Supported Models</h2>
              <p className="text-zinc-400 mb-4">February 2026 model support:</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      <th className="text-left py-3 px-4 text-zinc-300 font-medium">Provider</th>
                      <th className="text-left py-3 px-4 text-zinc-300 font-medium">Models</th>
                      <th className="text-left py-3 px-4 text-zinc-300 font-medium">Context</th>
                    </tr>
                  </thead>
                  <tbody className="text-zinc-400">
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">OpenAI</td>
                      <td className="py-3 px-4">GPT-5.3 Codex, GPT-5.2, GPT-4o</td>
                      <td className="py-3 px-4">400K</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">Anthropic</td>
                      <td className="py-3 px-4">Claude Opus 4.6, Sonnet 4.5, Haiku 4.5</td>
                      <td className="py-3 px-4">200K</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">Google</td>
                      <td className="py-3 px-4">Gemini 3 Pro/Flash, Gemini 2.5</td>
                      <td className="py-3 px-4">1M</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">xAI</td>
                      <td className="py-3 px-4">Grok 4.1 Fast</td>
                      <td className="py-3 px-4">128K</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">DeepSeek</td>
                      <td className="py-3 px-4">DeepSeek V3.2</td>
                      <td className="py-3 px-4">256K</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">Alibaba</td>
                      <td className="py-3 px-4">Qwen3-Coder-480B</td>
                      <td className="py-3 px-4">128K</td>
                    </tr>
                    <tr className="border-b border-zinc-800/50">
                      <td className="py-3 px-4 text-zinc-300">Meta</td>
                      <td className="py-3 px-4">Llama 4</td>
                      <td className="py-3 px-4">128K</td>
                    </tr>
                    <tr>
                      <td className="py-3 px-4 text-zinc-300">Mistral</td>
                      <td className="py-3 px-4">Mistral Large 2</td>
                      <td className="py-3 px-4">128K</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* API Reference */}
            <section id="api-reference" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-semibold text-zinc-100 mb-4">API Reference</h2>
              <p className="text-zinc-400 mb-4">
                Full API documentation is available on{" "}
                <a
                  href="https://docs.rs/forgetless"
                  className="text-zinc-300 underline underline-offset-4 hover:text-zinc-100"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  docs.rs/forgetless
                </a>
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <code className="text-zinc-300 text-sm">ContextManager</code>
                  <p className="text-zinc-500 text-xs mt-2">Main context orchestrator</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <code className="text-zinc-300 text-sm">AgentMemory</code>
                  <p className="text-zinc-500 text-xs mt-2">Cognitive memory system</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <code className="text-zinc-300 text-sm">SemanticScorer</code>
                  <p className="text-zinc-500 text-xs mt-2">Embedding similarity</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <code className="text-zinc-300 text-sm">TokenCounter</code>
                  <p className="text-zinc-500 text-xs mt-2">Multi-modal token counting</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <code className="text-zinc-300 text-sm">ConversationMemory</code>
                  <p className="text-zinc-500 text-xs mt-2">Conversation management</p>
                </div>
                <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
                  <code className="text-zinc-300 text-sm">Chunker</code>
                  <p className="text-zinc-500 text-xs mt-2">Semantic text chunking</p>
                </div>
              </div>
            </section>

            {/* Footer */}
            <footer className="pt-8 border-t border-zinc-800">
              <div className="flex flex-col sm:flex-row justify-between items-center gap-4 text-sm text-zinc-500">
                <p>MIT License — 2026 Berke</p>
                <div className="flex gap-6">
                  <a
                    href="https://github.com/pzzaworks/forgetless"
                    className="hover:text-zinc-300 transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GitHub
                  </a>
                  <a
                    href="https://crates.io/crates/forgetless"
                    className="hover:text-zinc-300 transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    crates.io
                  </a>
                </div>
              </div>
            </footer>

          </div>
        </main>
      </div>
    </div>
  );
}
