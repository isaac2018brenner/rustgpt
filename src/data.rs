//! Character-level tokenizer and random batch sampling.

use crate::rng::Rng;
use std::collections::HashMap;

pub struct Tokenizer {
    chars: Vec<char>,
    index: HashMap<char, usize>,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        Self::from_chars(chars)
    }

    pub fn from_chars(chars: Vec<char>) -> Self {
        let index = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Tokenizer { chars, index }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// Encodes `text`, silently dropping characters outside the vocabulary.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.index.get(&c).copied()).collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&t| self.chars[t]).collect()
    }
}

/// Samples `b` random windows of length `t`; targets are inputs shifted by one.
pub fn get_batch(tokens: &[usize], b: usize, t: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    assert!(tokens.len() > t + 1, "dataset is shorter than the block size");
    let mut x = Vec::with_capacity(b * t);
    let mut y = Vec::with_capacity(b * t);
    for _ in 0..b {
        let start = rng.below(tokens.len() - t - 1);
        x.extend_from_slice(&tokens[start..start + t]);
        y.extend_from_slice(&tokens[start + 1..start + t + 1]);
    }
    (x, y)
}
