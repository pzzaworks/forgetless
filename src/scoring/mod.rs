//! Relevance scoring and priority management
//!
//! Provides mechanisms to score and prioritize context items.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Priority level for context items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Critical information that must always be included
    Critical,
    /// High priority - recent or highly relevant
    High,
    /// Medium priority - useful context
    Medium,
    /// Low priority - background information
    Low,
    /// Minimal priority - can be dropped first
    Minimal,
}

impl Priority {
    /// Convert priority to numeric score (higher = more important)
    pub fn score(&self) -> u32 {
        match self {
            Priority::Critical => 100,
            Priority::High => 75,
            Priority::Medium => 50,
            Priority::Low => 25,
            Priority::Minimal => 10,
        }
    }

    /// Check if this priority is at least as important as another
    pub fn at_least(&self, other: Priority) -> bool {
        self.score() >= other.score()
    }
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Medium
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score().cmp(&other.score())
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Relevance score combining multiple factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceScore {
    /// Base priority
    pub priority: Priority,
    /// Recency score (0.0 - 1.0, higher = more recent)
    pub recency: f32,
    /// Semantic relevance (0.0 - 1.0)
    pub semantic: f32,
    /// Reference count (how often this is referenced)
    pub references: u32,
    /// Custom boost factor
    pub boost: f32,
}

impl RelevanceScore {
    /// Create a new relevance score
    pub fn new(priority: Priority) -> Self {
        Self {
            priority,
            recency: 1.0,
            semantic: 0.5,
            references: 0,
            boost: 1.0,
        }
    }

    /// Set recency score
    pub fn with_recency(mut self, recency: f32) -> Self {
        self.recency = recency.clamp(0.0, 1.0);
        self
    }

    /// Set semantic relevance
    pub fn with_semantic(mut self, semantic: f32) -> Self {
        self.semantic = semantic.clamp(0.0, 1.0);
        self
    }

    /// Add a reference
    pub fn add_reference(&mut self) {
        self.references = self.references.saturating_add(1);
    }

    /// Set boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost.max(0.0);
        self
    }

    /// Calculate final score
    pub fn final_score(&self) -> f32 {
        let base = self.priority.score() as f32;
        let recency_factor = 1.0 + (self.recency * 0.5);
        let semantic_factor = 1.0 + (self.semantic * 0.3);
        let reference_factor = 1.0 + (self.references as f32 * 0.1).min(0.5);

        base * recency_factor * semantic_factor * reference_factor * self.boost
    }
}

impl Default for RelevanceScore {
    fn default() -> Self {
        Self::new(Priority::Medium)
    }
}

impl Ord for RelevanceScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.final_score()
            .partial_cmp(&other.final_score())
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for RelevanceScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RelevanceScore {
    fn eq(&self, other: &Self) -> bool {
        self.final_score() == other.final_score()
    }
}

impl Eq for RelevanceScore {}

/// Decay function for recency scoring
pub struct RecencyDecay {
    /// Half-life in number of turns/items
    half_life: f32,
}

impl RecencyDecay {
    /// Create a new recency decay calculator
    pub fn new(half_life: f32) -> Self {
        Self {
            half_life: half_life.max(1.0),
        }
    }

    /// Calculate decay factor for a given age
    pub fn decay(&self, age: usize) -> f32 {
        0.5_f32.powf(age as f32 / self.half_life)
    }

    /// Update recency scores for a list of items by their index
    pub fn apply_decay(&self, scores: &mut [RelevanceScore]) {
        let len = scores.len();
        for (i, score) in scores.iter_mut().enumerate() {
            let age = len - i - 1;
            score.recency = self.decay(age);
        }
    }
}

impl Default for RecencyDecay {
    fn default() -> Self {
        Self::new(10.0) // 10 turns half-life
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
        assert!(Priority::Low > Priority::Minimal);
    }

    #[test]
    fn test_priority_scores() {
        assert_eq!(Priority::Critical.score(), 100);
        assert_eq!(Priority::High.score(), 75);
        assert_eq!(Priority::Medium.score(), 50);
        assert_eq!(Priority::Low.score(), 25);
        assert_eq!(Priority::Minimal.score(), 10);
    }

    #[test]
    fn test_priority_at_least() {
        assert!(Priority::Critical.at_least(Priority::Critical));
        assert!(Priority::Critical.at_least(Priority::High));
        assert!(Priority::Critical.at_least(Priority::Minimal));
        assert!(Priority::High.at_least(Priority::Medium));
        assert!(!Priority::Low.at_least(Priority::High));
        assert!(!Priority::Minimal.at_least(Priority::Low));
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Medium);
    }

    #[test]
    fn test_priority_partial_ord() {
        assert!(Priority::Critical.partial_cmp(&Priority::High) == Some(Ordering::Greater));
        assert!(Priority::Low.partial_cmp(&Priority::Low) == Some(Ordering::Equal));
        assert!(Priority::Minimal.partial_cmp(&Priority::Medium) == Some(Ordering::Less));
    }

    #[test]
    fn test_relevance_score_new() {
        let score = RelevanceScore::new(Priority::High);
        assert_eq!(score.priority, Priority::High);
        assert_eq!(score.recency, 1.0);
        assert_eq!(score.semantic, 0.5);
        assert_eq!(score.references, 0);
        assert_eq!(score.boost, 1.0);
    }

    #[test]
    fn test_relevance_score_default() {
        let score = RelevanceScore::default();
        assert_eq!(score.priority, Priority::Medium);
    }

    #[test]
    fn test_relevance_score_with_recency() {
        let score = RelevanceScore::new(Priority::High).with_recency(0.5);
        assert_eq!(score.recency, 0.5);

        // Test clamping
        let score_high = RelevanceScore::new(Priority::High).with_recency(2.0);
        assert_eq!(score_high.recency, 1.0);

        let score_low = RelevanceScore::new(Priority::High).with_recency(-1.0);
        assert_eq!(score_low.recency, 0.0);
    }

    #[test]
    fn test_relevance_score_with_semantic() {
        let score = RelevanceScore::new(Priority::High).with_semantic(0.8);
        assert_eq!(score.semantic, 0.8);

        // Test clamping
        let score_high = RelevanceScore::new(Priority::High).with_semantic(1.5);
        assert_eq!(score_high.semantic, 1.0);

        let score_low = RelevanceScore::new(Priority::High).with_semantic(-0.5);
        assert_eq!(score_low.semantic, 0.0);
    }

    #[test]
    fn test_relevance_score_add_reference() {
        let mut score = RelevanceScore::new(Priority::High);
        assert_eq!(score.references, 0);
        score.add_reference();
        assert_eq!(score.references, 1);
        score.add_reference();
        assert_eq!(score.references, 2);
    }

    #[test]
    fn test_relevance_score_with_boost() {
        let score = RelevanceScore::new(Priority::High).with_boost(2.0);
        assert_eq!(score.boost, 2.0);

        // Test negative clamping
        let score_neg = RelevanceScore::new(Priority::High).with_boost(-1.0);
        assert_eq!(score_neg.boost, 0.0);
    }

    #[test]
    fn test_relevance_score_final_score() {
        let score = RelevanceScore::new(Priority::High)
            .with_recency(1.0)
            .with_semantic(0.8);

        assert!(score.final_score() > Priority::High.score() as f32);
    }

    #[test]
    fn test_relevance_score_ordering() {
        let low = RelevanceScore::new(Priority::Low);
        let high = RelevanceScore::new(Priority::High);
        let critical = RelevanceScore::new(Priority::Critical);

        assert!(critical > high);
        assert!(high > low);
        assert!(low < high);
    }

    #[test]
    fn test_relevance_score_equality() {
        let score1 = RelevanceScore::new(Priority::High);
        let score2 = RelevanceScore::new(Priority::High);
        assert_eq!(score1, score2);

        let score3 = RelevanceScore::new(Priority::Low);
        assert_ne!(score1, score3);
    }

    #[test]
    fn test_relevance_score_partial_cmp() {
        let low = RelevanceScore::new(Priority::Low);
        let high = RelevanceScore::new(Priority::High);
        assert_eq!(low.partial_cmp(&high), Some(Ordering::Less));
        assert_eq!(high.partial_cmp(&low), Some(Ordering::Greater));
        assert_eq!(low.partial_cmp(&low), Some(Ordering::Equal));
    }

    #[test]
    fn test_recency_decay() {
        let decay = RecencyDecay::new(10.0);
        assert!((decay.decay(0) - 1.0).abs() < 0.01);
        assert!((decay.decay(10) - 0.5).abs() < 0.01);
        assert!(decay.decay(20) < 0.3);
    }

    #[test]
    fn test_recency_decay_default() {
        let decay = RecencyDecay::default();
        // Default half-life is 10.0
        assert!((decay.decay(10) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_recency_decay_clamping() {
        // Half-life less than 1.0 should be clamped to 1.0
        let decay = RecencyDecay::new(0.5);
        // With half_life = 1.0, decay(1) should be 0.5
        assert!((decay.decay(1) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_recency_decay_apply() {
        let decay = RecencyDecay::new(10.0);
        let mut scores = vec![
            RelevanceScore::new(Priority::High),
            RelevanceScore::new(Priority::High),
            RelevanceScore::new(Priority::High),
        ];

        decay.apply_decay(&mut scores);

        // First item (oldest) should have lowest recency
        // Last item (newest) should have highest recency
        assert!(scores[0].recency < scores[2].recency);
        assert!((scores[2].recency - 1.0).abs() < 0.01); // Most recent
    }

    #[test]
    fn test_boost_affects_final_score() {
        let base = RelevanceScore::new(Priority::High);
        let boosted = RelevanceScore::new(Priority::High).with_boost(2.0);

        assert!(boosted.final_score() > base.final_score());
        // With boost=2.0, score should be approximately double
        let ratio = boosted.final_score() / base.final_score();
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_references_affect_final_score() {
        let mut with_refs = RelevanceScore::new(Priority::High);
        let without_refs = RelevanceScore::new(Priority::High);

        with_refs.add_reference();
        with_refs.add_reference();
        with_refs.add_reference();

        assert!(with_refs.final_score() > without_refs.final_score());
    }
}
