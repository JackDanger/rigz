//! Markov Chain Symbol Prediction
//!
//! Uses P(symbol[i+1] | symbol[i]) to predict and speculatively decode
//! the next symbol when decoding literals.
//!
//! ## Concept
//!
//! In text data, certain character pairs are very common:
//! - After 'e': ' ' (20%), 's' (15%), 'r' (12%), 'd' (10%)
//! - After 't': 'h' (25%), ' ' (20%), 'i' (10%)
//! - After ' ': 't' (15%), 'a' (10%), 'i' (8%)
//!
//! By predicting the next symbol, we can speculatively check if the
//! prediction is correct without fully decoding, potentially saving
//! a table lookup.
//!
//! ## Implementation
//!
//! For each byte value (0-255), we store:
//! - The most likely next byte
//! - A confidence score (0-255 representing probability * 255)

#![allow(dead_code)]

/// Markov predictor trained on English text patterns
#[derive(Clone)]
pub struct MarkovPredictor {
    /// For each byte, the most likely following byte
    prediction: [u8; 256],
    /// Confidence level for each prediction (0-255)
    confidence: [u8; 256],
}

impl Default for MarkovPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkovPredictor {
    /// Create a new predictor with default English text patterns
    pub fn new() -> Self {
        let mut predictor = Self {
            prediction: [0; 256],
            confidence: [0; 256],
        };
        predictor.init_english_patterns();
        predictor
    }

    /// Initialize with common English text patterns
    fn init_english_patterns(&mut self) {
        // Most common character transitions in English text
        // Format: (current, next, confidence)
        let patterns: &[(u8, u8, u8)] = &[
            // After space: common word starters
            (b' ', b't', 64), // " t" - the, to, that
            (b' ', b'a', 51), // " a" - a, and, at
            (b' ', b'i', 38), // " i" - in, is, it
            (b' ', b's', 38), // " s" - so, such
            (b' ', b'o', 38), // " o" - of, on, or
            // After 'e': common endings
            (b'e', b' ', 51), // "e " - end of word
            (b'e', b's', 38), // "es" - plural
            (b'e', b'r', 31), // "er" - comparative
            (b'e', b'd', 26), // "ed" - past tense
            (b'e', b'n', 26), // "en" - taken, etc
            // After 't': common patterns
            (b't', b'h', 64), // "th" - the, that, this
            (b't', b' ', 51), // "t " - end of word
            (b't', b'i', 26), // "ti" - tion, time
            (b't', b'o', 26), // "to" - to, together
            // After 'h': common patterns
            (b'h', b'e', 77), // "he" - he, the, them
            (b'h', b'a', 38), // "ha" - have, had
            (b'h', b'i', 26), // "hi" - this, his
            // After 'a': common patterns
            (b'a', b'n', 51), // "an" - and, an
            (b'a', b't', 38), // "at" - at, that
            (b'a', b'l', 26), // "al" - all, also
            (b'a', b's', 26), // "as" - as, was
            // After 'n': common patterns
            (b'n', b' ', 51), // "n " - end of word
            (b'n', b'd', 38), // "nd" - and, end
            (b'n', b'g', 26), // "ng" - ing
            (b'n', b't', 26), // "nt" - not, int
            // After 'i': common patterns
            (b'i', b'n', 51), // "in" - in, ing
            (b'i', b's', 38), // "is" - is, this
            (b'i', b't', 26), // "it" - it, with
            (b'i', b'o', 26), // "io" - tion
            // After 'o': common patterns
            (b'o', b'n', 51), // "on" - on, one
            (b'o', b'r', 38), // "or" - or, for
            (b'o', b'f', 26), // "of" - of
            (b'o', b'u', 26), // "ou" - you, out
            // After 's': common patterns
            (b's', b' ', 51), // "s " - end of word
            (b's', b't', 38), // "st" - st
            (b's', b'e', 26), // "se" - se
            (b's', b'i', 26), // "si" - si
            // After 'r': common patterns
            (b'r', b'e', 51), // "re" - re
            (b'r', b' ', 38), // "r " - end
            (b'r', b'o', 26), // "ro" - ro
            (b'r', b'i', 26), // "ri" - ri
            // After 'd': common patterns
            (b'd', b' ', 64), // "d " - end
            (b'd', b'e', 26), // "de" - de
            (b'd', b'i', 26), // "di" - di
            // After 'l': common patterns
            (b'l', b'l', 38), // "ll" - all, will
            (b'l', b'e', 38), // "le" - le
            (b'l', b' ', 26), // "l " - end
            (b'l', b'y', 26), // "ly" - adverb
            // Newline patterns
            (b'\n', b' ', 64),
            (b'\n', b'\t', 26),
            // After lowercase letters, predict space or common followers
            (b'f', b' ', 38),
            (b'f', b'o', 51), // "fo" - for
            (b'y', b' ', 64), // end of word
            (b'w', b'a', 51), // "wa" - was, want
            (b'w', b'i', 38), // "wi" - with, will
            (b'w', b'h', 26), // "wh" - wh words
            // Punctuation
            (b'.', b' ', 89),
            (b',', b' ', 89),
            (b':', b' ', 77),
            (b';', b' ', 77),
            (b'!', b' ', 77),
            (b'?', b' ', 77),
            // Quotes
            (b'"', b' ', 38),
            (b'\'', b's', 38), // possessive
            (b'\'', b't', 38), // contractions
        ];

        for &(current, next, conf) in patterns {
            // Only update if this is a higher confidence prediction
            if conf > self.confidence[current as usize] {
                self.prediction[current as usize] = next;
                self.confidence[current as usize] = conf;
            }
        }
    }

    /// Train on a corpus of data
    pub fn train(&mut self, data: &[u8]) {
        if data.len() < 2 {
            return;
        }

        // Count transitions
        let mut counts = [[0u32; 256]; 256];
        for window in data.windows(2) {
            counts[window[0] as usize][window[1] as usize] += 1;
        }

        // Find most common next byte for each byte
        for (i, row) in counts.iter().enumerate() {
            let (max_j, max_count) = row.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();

            if *max_count > 0 {
                let total: u32 = row.iter().sum();
                self.prediction[i] = max_j as u8;
                self.confidence[i] = ((max_count * 255) / total.max(1)) as u8;
            }
        }
    }

    /// Get the predicted next byte after the given byte
    #[inline(always)]
    pub fn predict(&self, current: u8) -> (u8, u8) {
        (
            self.prediction[current as usize],
            self.confidence[current as usize],
        )
    }

    /// Check if prediction confidence exceeds threshold
    #[inline(always)]
    pub fn is_confident(&self, current: u8, threshold: u8) -> bool {
        self.confidence[current as usize] >= threshold
    }
}

/// Statistics about prediction accuracy
#[derive(Default, Debug)]
pub struct PredictionStats {
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub high_confidence_predictions: usize,
    pub high_confidence_correct: usize,
}

impl PredictionStats {
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions > 0 {
            self.correct_predictions as f64 / self.total_predictions as f64
        } else {
            0.0
        }
    }

    pub fn high_confidence_accuracy(&self) -> f64 {
        if self.high_confidence_predictions > 0 {
            self.high_confidence_correct as f64 / self.high_confidence_predictions as f64
        } else {
            0.0
        }
    }
}

/// Test prediction accuracy on data
pub fn test_prediction_accuracy(
    predictor: &MarkovPredictor,
    data: &[u8],
    threshold: u8,
) -> PredictionStats {
    let mut stats = PredictionStats::default();

    if data.len() < 2 {
        return stats;
    }

    for window in data.windows(2) {
        let current = window[0];
        let actual_next = window[1];
        let (predicted, confidence) = predictor.predict(current);

        stats.total_predictions += 1;
        if predicted == actual_next {
            stats.correct_predictions += 1;
        }

        if confidence >= threshold {
            stats.high_confidence_predictions += 1;
            if predicted == actual_next {
                stats.high_confidence_correct += 1;
            }
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn test_markov_predictor() {
        let predictor = MarkovPredictor::new();

        // Test some common patterns
        let (pred, conf) = predictor.predict(b' ');
        eprintln!(
            "[MARKOV] After ' ': predict '{}' (conf={})",
            pred as char, conf
        );

        let (pred, conf) = predictor.predict(b'e');
        eprintln!(
            "[MARKOV] After 'e': predict '{}' (conf={})",
            pred as char, conf
        );

        let (pred, conf) = predictor.predict(b't');
        eprintln!(
            "[MARKOV] After 't': predict '{}' (conf={})",
            pred as char, conf
        );
    }

    #[test]
    fn test_prediction_accuracy_english() {
        let predictor = MarkovPredictor::new();

        // Test on some English text
        let text = b"The quick brown fox jumps over the lazy dog. This is a test of the emergency broadcast system.";

        let stats = test_prediction_accuracy(&predictor, text, 50);
        eprintln!("\n[MARKOV] English Text Accuracy:");
        eprintln!(
            "[MARKOV]   Overall: {:.1}% ({}/{})",
            stats.accuracy() * 100.0,
            stats.correct_predictions,
            stats.total_predictions
        );
        eprintln!(
            "[MARKOV]   High-confidence (>50): {:.1}% ({}/{})",
            stats.high_confidence_accuracy() * 100.0,
            stats.high_confidence_correct,
            stats.high_confidence_predictions
        );
    }

    #[test]
    fn test_trained_predictor() {
        // Train on a larger corpus
        let corpus = include_bytes!("../Cargo.toml"); // Use Cargo.toml as sample text

        let mut predictor = MarkovPredictor::new();
        predictor.train(corpus);

        let stats = test_prediction_accuracy(&predictor, corpus, 50);
        eprintln!("\n[MARKOV] Trained on Cargo.toml:");
        eprintln!(
            "[MARKOV]   Overall: {:.1}% ({}/{})",
            stats.accuracy() * 100.0,
            stats.correct_predictions,
            stats.total_predictions
        );
        eprintln!(
            "[MARKOV]   High-confidence (>50): {:.1}% ({}/{})",
            stats.high_confidence_accuracy() * 100.0,
            stats.high_confidence_correct,
            stats.high_confidence_predictions
        );
    }

    #[test]
    fn bench_markov_predict() {
        let predictor = MarkovPredictor::new();
        let iterations = 100_000_000;

        let start = std::time::Instant::now();
        let mut total_conf = 0u64;
        for i in 0..iterations {
            let byte = (i % 256) as u8;
            let (_, conf) = black_box(&predictor).predict(black_box(byte));
            total_conf = total_conf.wrapping_add(conf as u64);
        }
        black_box(total_conf);
        let elapsed = start.elapsed();

        let predictions_per_sec = iterations as f64 / elapsed.as_secs_f64();
        eprintln!("\n[BENCH] Markov Predict Speed:");
        eprintln!(
            "[BENCH]   {:.2} M predictions/sec",
            predictions_per_sec / 1_000_000.0
        );
        eprintln!("[BENCH]   Total confidence: {}", total_conf);
    }

    #[test]
    fn bench_markov_overhead() {
        use crate::libdeflate_entry::LitLenTable;

        // Build fixed Huffman table
        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256] = 7;
        litlen_lens[257..280].fill(7);
        litlen_lens[280..288].fill(8);

        let table = LitLenTable::build(&litlen_lens).unwrap();
        let predictor = MarkovPredictor::new();

        let iterations = 10_000_000;
        let test_patterns: Vec<u64> = (0..1000).map(|i| i * 7919 % 2048).collect();

        // Benchmark baseline (just table lookup)
        let start = std::time::Instant::now();
        let mut baseline_bits = 0u64;
        for _ in 0..iterations / 1000 {
            for &pattern in &test_patterns {
                let entry = black_box(&table).lookup(black_box(pattern));
                baseline_bits = baseline_bits.wrapping_add(entry.total_bits() as u64);
            }
        }
        black_box(baseline_bits);
        let baseline_elapsed = start.elapsed();
        let baseline_rate = iterations as f64 / baseline_elapsed.as_secs_f64() / 1_000_000.0;

        // Benchmark with prediction check
        let start = std::time::Instant::now();
        let mut predict_bits = 0u64;
        let mut last_byte = 0u8;
        for _ in 0..iterations / 1000 {
            for &pattern in &test_patterns {
                let entry = black_box(&table).lookup(black_box(pattern));
                let bits = entry.total_bits();
                predict_bits = predict_bits.wrapping_add(bits as u64);

                // Simulate prediction check overhead
                let (predicted, confidence) = black_box(&predictor).predict(black_box(last_byte));
                if confidence > 50 {
                    // Would do speculative decode here
                    last_byte = predicted;
                } else {
                    last_byte = (pattern & 0xFF) as u8;
                }
            }
        }
        black_box(predict_bits);
        let predict_elapsed = start.elapsed();
        let predict_rate = iterations as f64 / predict_elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!("\n[BENCH] Markov Overhead Analysis:");
        eprintln!(
            "[BENCH]   Baseline:         {:.2} M decodes/sec",
            baseline_rate
        );
        eprintln!(
            "[BENCH]   With prediction:  {:.2} M decodes/sec",
            predict_rate
        );
        eprintln!(
            "[BENCH]   Overhead:         {:.1}%",
            (1.0 - predict_rate / baseline_rate) * 100.0
        );
        eprintln!(
            "[BENCH]   Ratio:            {:.1}%",
            predict_rate / baseline_rate * 100.0
        );
    }
}
