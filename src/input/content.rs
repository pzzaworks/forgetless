//! Input types and traits for content handling

use crate::processing::chunking::ContentType;
use crate::processing::scoring::Priority;
use crate::input::file::read_file_content;
use std::path::Path;

/// Content with priority attached - use for .add()
#[derive(Debug, Clone)]
pub struct WithPriority {
    pub(crate) content: String,
    pub(crate) priority: Priority,
}

impl WithPriority {
    /// Create content with Critical priority
    pub fn critical(content: impl Into<String>) -> Self {
        Self { content: content.into(), priority: Priority::Critical }
    }
    /// Create content with High priority
    pub fn high(content: impl Into<String>) -> Self {
        Self { content: content.into(), priority: Priority::High }
    }
    /// Create content with Medium priority
    pub fn medium(content: impl Into<String>) -> Self {
        Self { content: content.into(), priority: Priority::Medium }
    }
    /// Create content with Low priority
    pub fn low(content: impl Into<String>) -> Self {
        Self { content: content.into(), priority: Priority::Low }
    }
}

/// File path with priority attached
#[derive(Debug, Clone)]
pub struct FileWithPriority<P: AsRef<Path>> {
    pub(crate) path: P,
    pub(crate) priority: Priority,
}

impl<P: AsRef<Path>> FileWithPriority<P> {
    /// Create file with Critical priority
    pub fn critical(path: P) -> Self {
        Self { path, priority: Priority::Critical }
    }
    /// Create file with High priority
    pub fn high(path: P) -> Self {
        Self { path, priority: Priority::High }
    }
    /// Create file with Medium priority
    pub fn medium(path: P) -> Self {
        Self { path, priority: Priority::Medium }
    }
    /// Create file with Low priority
    pub fn low(path: P) -> Self {
        Self { path, priority: Priority::Low }
    }
    /// Get the path
    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }
    /// Get the priority
    pub fn priority(&self) -> Priority {
        self.priority
    }
}

/// Input content item - can be string, file, or bytes
#[derive(Debug, Clone)]
pub struct ContentInput {
    /// The actual content
    pub content: String,
    /// Source identifier (filename, "string", etc.)
    pub source: String,
    /// Content type hint
    pub content_type: ContentType,
    /// Priority level
    pub priority: Priority,
}

impl ContentInput {
    /// Create from string
    pub fn from_string(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            source: "input".to_string(),
            content_type: ContentType::Text,
            priority: Priority::Medium,
        }
    }

    /// Create from file contents
    pub fn from_file(content: String, path: &str) -> Self {
        let content_type = ContentType::detect_from_path(path);
        Self {
            content,
            source: path.to_string(),
            content_type,
            priority: Priority::Medium,
        }
    }
}

/// Trait for content that can be added to Forgetless
pub trait IntoContent {
    /// Convert to ContentInput
    fn into_content_input(self) -> ContentInput;
}

impl IntoContent for String {
    fn into_content_input(self) -> ContentInput {
        ContentInput::from_string(self)
    }
}

impl IntoContent for &str {
    fn into_content_input(self) -> ContentInput {
        ContentInput::from_string(self)
    }
}

impl IntoContent for &String {
    fn into_content_input(self) -> ContentInput {
        ContentInput::from_string(self.clone())
    }
}

impl IntoContent for WithPriority {
    fn into_content_input(self) -> ContentInput {
        let mut input = ContentInput::from_string(self.content);
        input.priority = self.priority;
        input
    }
}

/// Trait for file content that can be added
pub trait IntoFileContent {
    /// Convert to ContentInput by reading file
    fn into_file_input(self) -> Option<ContentInput>;
}

impl<P: AsRef<Path>> IntoFileContent for P {
    fn into_file_input(self) -> Option<ContentInput> {
        let path = self.as_ref();
        read_file_content(path).map(|(content, content_type)| {
            let mut input = ContentInput::from_file(content, &path.to_string_lossy());
            input.content_type = content_type;
            input
        })
    }
}

impl<P: AsRef<Path>> IntoFileContent for FileWithPriority<P> {
    fn into_file_input(self) -> Option<ContentInput> {
        let path = self.path.as_ref();
        read_file_content(path).map(|(content, content_type)| {
            let mut input = ContentInput::from_file(content, &path.to_string_lossy());
            input.content_type = content_type;
            input.priority = self.priority;
            input
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_with_priority_critical() {
        let wp = WithPriority::critical("test");
        assert_eq!(wp.content, "test");
        assert_eq!(wp.priority, Priority::Critical);
    }

    #[test]
    fn test_with_priority_high() {
        let wp = WithPriority::high("test");
        assert_eq!(wp.priority, Priority::High);
    }

    #[test]
    fn test_with_priority_medium() {
        let wp = WithPriority::medium("test");
        assert_eq!(wp.priority, Priority::Medium);
    }

    #[test]
    fn test_with_priority_low() {
        let wp = WithPriority::low("test");
        assert_eq!(wp.priority, Priority::Low);
    }

    #[test]
    fn test_file_with_priority_critical() {
        let fp = FileWithPriority::critical("test.txt");
        assert_eq!(fp.path, "test.txt");
        assert_eq!(fp.priority, Priority::Critical);
    }

    #[test]
    fn test_file_with_priority_high() {
        let fp = FileWithPriority::high("test.txt");
        assert_eq!(fp.priority, Priority::High);
    }

    #[test]
    fn test_file_with_priority_medium() {
        let fp = FileWithPriority::medium("test.txt");
        assert_eq!(fp.priority, Priority::Medium);
    }

    #[test]
    fn test_file_with_priority_low() {
        let fp = FileWithPriority::low("test.txt");
        assert_eq!(fp.priority, Priority::Low);
    }

    #[test]
    fn test_content_input_from_string() {
        let input = ContentInput::from_string("hello");
        assert_eq!(input.content, "hello");
        assert_eq!(input.source, "input");
        assert_eq!(input.content_type, ContentType::Text);
        assert_eq!(input.priority, Priority::Medium);
    }

    #[test]
    fn test_content_input_from_file() {
        let input = ContentInput::from_file("content".to_string(), "test.rs");
        assert_eq!(input.content, "content");
        assert_eq!(input.source, "test.rs");
        assert_eq!(input.content_type, ContentType::Code);
    }

    #[test]
    fn test_into_content_string() {
        let input: ContentInput = String::from("test").into_content_input();
        assert_eq!(input.content, "test");
    }

    #[test]
    fn test_into_content_str() {
        let input: ContentInput = "test".into_content_input();
        assert_eq!(input.content, "test");
    }

    #[test]
    fn test_into_content_string_ref() {
        let s = String::from("test");
        let input: ContentInput = (&s).into_content_input();
        assert_eq!(input.content, "test");
    }

    #[test]
    fn test_into_content_with_priority() {
        let wp = WithPriority::high("important");
        let input: ContentInput = wp.into_content_input();
        assert_eq!(input.content, "important");
        assert_eq!(input.priority, Priority::High);
    }

    #[test]
    fn test_into_file_content_path() {
        let path = PathBuf::from("Cargo.toml");
        let input = path.into_file_input();
        assert!(input.is_some());
        let input = input.unwrap();
        assert!(input.content.contains("[package]"));
    }

    #[test]
    fn test_into_file_content_nonexistent() {
        let path = PathBuf::from("nonexistent_file_12345.txt");
        let input = path.into_file_input();
        assert!(input.is_none());
    }

    #[test]
    fn test_into_file_content_with_priority() {
        let fp = FileWithPriority::critical("Cargo.toml");
        let input = fp.into_file_input();
        assert!(input.is_some());
        let input = input.unwrap();
        assert_eq!(input.priority, Priority::Critical);
    }

    #[test]
    fn test_into_file_content_with_priority_nonexistent() {
        let fp = FileWithPriority::high("nonexistent_12345.txt");
        let input = fp.into_file_input();
        assert!(input.is_none());
    }
}
