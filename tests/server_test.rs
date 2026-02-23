//! Server API Integration Tests
//!
//! Run with: cargo test --features server --test server_test

#![cfg(feature = "server")]

use std::process::{Child, Command, Stdio};
use std::time::Duration;
use std::thread;

struct ServerGuard {
    process: Child,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.process.kill();
    }
}

fn start_server() -> Option<ServerGuard> {
    let child = Command::new("cargo")
        .args(["run", "--features", "server", "--release", "--bin", "forgetless-server"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    // Wait for server to start
    thread::sleep(Duration::from_secs(5));

    Some(ServerGuard { process: child })
}

fn check_server_running() -> bool {
    reqwest::blocking::get("http://localhost:8080/health")
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[test]
#[ignore] // Requires server to be running
fn test_health_endpoint() {
    if !check_server_running() {
        eprintln!("Server not running, skipping test");
        return;
    }

    let response = reqwest::blocking::get("http://localhost:8080/health").unwrap();
    assert!(response.status().is_success());
    assert_eq!(response.text().unwrap(), "ok");
}

#[test]
#[ignore] // Requires server to be running
fn test_simple_text_compression() {
    if !check_server_running() {
        eprintln!("Server not running, skipping test");
        return;
    }

    let client = reqwest::blocking::Client::new();
    let form = reqwest::blocking::multipart::Form::new()
        .text("metadata", r#"{"max_tokens": 1000, "contents": [{"content": "Hello world, this is a test message that should be processed.", "priority": "high"}]}"#);

    let response = client
        .post("http://localhost:8080/")
        .multipart(form)
        .send()
        .unwrap();

    assert!(response.status().is_success());

    let json: serde_json::Value = response.json().unwrap();
    assert!(json.get("content").is_some());
    assert!(json.get("stats").is_some());
}
