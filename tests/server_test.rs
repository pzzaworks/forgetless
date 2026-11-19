//! Server API integration tests.
//!
//! Each test starts the real `forgetless-server` binary on a free local port and talks to it
//! over plain HTTP, so nothing has to be running beforehand.
//!
//! Run with: cargo test --features server --test server_test

#![cfg(feature = "server")]

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);

/// A running server, killed when the test ends (pass or panic).
struct Server {
    process: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|addr| addr.port())
        .expect("no free local port")
}

fn start_server() -> Server {
    let port = free_port();
    let process = Command::new(env!("CARGO_BIN_EXE_forgetless-server"))
        .env("FORGETLESS_PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start forgetless-server");
    let server = Server { process, port };

    // Ready means /health answers, not that some fixed amount of time has passed.
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        if matches!(request(port, "GET", "/health", None), Ok((200, _))) {
            return server;
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("forgetless-server did not become healthy on port {port}");
}

/// Minimal HTTP/1.1 client: one request per connection, response read until close.
fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<(&str, &[u8])>,
) -> io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(RESPONSE_TIMEOUT))?;

    let mut head =
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    if let Some((content_type, bytes)) = body {
        head.push_str(&format!(
            "Content-Type: {content_type}\r\nContent-Length: {}\r\n",
            bytes.len()
        ));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if let Some((_, bytes)) = body {
        stream.write_all(bytes)?;
    }

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw);
    let (status_and_headers, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| io::Error::other("response has no header terminator"))?;
    let status = status_and_headers
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| io::Error::other("response has no status code"))?;
    Ok((status, body.to_string()))
}

fn multipart_text_field(boundary: &str, name: &str, value: &str) -> Vec<u8> {
    format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n--{boundary}--\r\n"
    )
    .into_bytes()
}

#[test]
fn test_health_endpoint() {
    let server = start_server();

    let (status, body) = request(server.port, "GET", "/health", None).unwrap();
    assert_eq!(status, 200);
    assert_eq!(body, "ok");
}

#[test]
fn test_simple_text_compression() {
    let server = start_server();
    let boundary = "forgetless-test-boundary";
    let body = multipart_text_field(
        boundary,
        "metadata",
        r#"{"max_tokens": 1000, "contents": [{"content": "Hello world, this is a test message that should be processed.", "priority": "high"}]}"#,
    );
    let content_type = format!("multipart/form-data; boundary={boundary}");

    let (status, body) = request(server.port, "POST", "/", Some((&content_type, &body))).unwrap();

    assert_eq!(status, 200, "unexpected response: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("content").is_some());
    assert!(json.get("stats").is_some());
}
