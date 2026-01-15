#![allow(dead_code, unused_imports, unused_variables)]

use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::hardware_analysis::SystemProfile;
use crate::adaptive_compression::{AdaptiveAlgorithm, ContentCharacteristics};
use crate::simd_compression::SimdCompressionAccelerator;

/// Advanced dictionary pre-computation for compression
pub struct DictionaryEngine {
    global_dictionary: Arc<RwLock<CompressionDictionary>>,
    content_dictionaries: Arc<Mutex<HashMap<u64, CompressionDictionary>>>,
    learning_enabled: bool,
    max_dictionary_size: usize,
}

#[derive(Debug, Clone)]
pub struct CompressionDictionary {
    pub patterns: Vec<DictionaryPattern>,
    pub pattern_frequencies: HashMap<Vec<u8>, u32>,
    pub last_updated: Instant,
    pub effectiveness_score: f64,
}

#[derive(Debug, Clone)]
pub struct DictionaryPattern {
    pub data: Vec<u8>,
    pub frequency: u32,
    pub compression_gain: f64,
    pub positions: Vec<usize>,
}

impl DictionaryEngine {
    pub fn new(max_dictionary_size: usize) -> Self {
        Self {
            global_dictionary: Arc::new(RwLock::new(CompressionDictionary::new())),
            content_dictionaries: Arc::new(Mutex::new(HashMap::new())),
            learning_enabled: true,
            max_dictionary_size,
        }
    }
    
    pub fn analyze_and_build_dictionary(&self, data: &[u8], content_hash: u64) -> CompressionDictionary {
        // Extract patterns using suffix array approach
        let patterns = self.extract_patterns(data);
        
        // Score patterns by frequency and potential compression gain
        let scored_patterns = self.score_patterns(&patterns, data);
        
        // Build optimized dictionary
        let mut dictionary = CompressionDictionary::new();
        dictionary.patterns = scored_patterns.into_iter()
            .take(self.max_dictionary_size)
            .collect();
        
        // Update global and content-specific dictionaries
        if self.learning_enabled {
            self.update_dictionaries(&dictionary, content_hash);
        }
        
        dictionary
    }
    
    fn extract_patterns(&self, data: &[u8]) -> Vec<Vec<u8>> {
        let mut patterns = Vec::new();
        let min_pattern_length = 4;
        let max_pattern_length = 256;
        
        // Use a sliding window approach to find repeated patterns
        for window_size in min_pattern_length..=max_pattern_length.min(data.len()) {
            let mut pattern_counts: HashMap<Vec<u8>, u32> = HashMap::new();
            
            for i in 0..=data.len().saturating_sub(window_size) {
                let pattern = data[i..i + window_size].to_vec();
                *pattern_counts.entry(pattern).or_insert(0) += 1;
            }
            
            // Only keep patterns that appear multiple times
            for (pattern, count) in pattern_counts {
                if count >= 2 && pattern.len() >= min_pattern_length {
                    patterns.push(pattern);
                }
            }
            
            // Stop early if we have enough patterns
            if patterns.len() > self.max_dictionary_size * 10 {
                break;
            }
        }
        
        patterns
    }
    
    fn score_patterns(&self, patterns: &[Vec<u8>], data: &[u8]) -> Vec<DictionaryPattern> {
        let mut scored_patterns = Vec::new();
        
        for pattern in patterns {
            let frequency = self.count_pattern_occurrences(data, pattern);
            if frequency < 2 {
                continue;
            }
            
            // Calculate compression gain: savings from replacing pattern with reference
            let pattern_size = pattern.len();
            let reference_size = 3; // Typical LZ77 reference size (distance + length)
            let savings_per_occurrence = pattern_size.saturating_sub(reference_size);
            let total_savings = savings_per_occurrence * frequency as usize;
            let compression_gain = total_savings as f64 / data.len() as f64;
            
            // Find all positions of this pattern
            let positions = self.find_pattern_positions(data, pattern);
            
            scored_patterns.push(DictionaryPattern {
                data: pattern.clone(),
                frequency,
                compression_gain,
                positions,
            });
        }
        
        // Sort by compression gain (descending)
        scored_patterns.sort_by(|a, b| b.compression_gain.partial_cmp(&a.compression_gain).unwrap());
        
        scored_patterns
    }
    
    fn count_pattern_occurrences(&self, data: &[u8], pattern: &[u8]) -> u32 {
        if pattern.len() > data.len() {
            return 0;
        }
        
        let mut count = 0;
        for i in 0..=data.len() - pattern.len() {
            if &data[i..i + pattern.len()] == pattern {
                count += 1;
            }
        }
        count
    }
    
    fn find_pattern_positions(&self, data: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        
        if pattern.len() > data.len() {
            return positions;
        }
        
        for i in 0..=data.len() - pattern.len() {
            if &data[i..i + pattern.len()] == pattern {
                positions.push(i);
            }
        }
        positions
    }
    
    fn update_dictionaries(&self, dictionary: &CompressionDictionary, content_hash: u64) {
        // Update content-specific dictionary
        if let Ok(mut content_dicts) = self.content_dictionaries.lock() {
            content_dicts.insert(content_hash, dictionary.clone());
            
            // Limit cache size
            if content_dicts.len() > 1000 {
                // Remove oldest entries (simple LRU)
                let oldest_hash = content_dicts.keys().next().copied();
                if let Some(hash) = oldest_hash {
                    content_dicts.remove(&hash);
                }
            }
        }
        
        // Update global dictionary with most effective patterns
        if let Ok(mut global_dict) = self.global_dictionary.write() {
            for pattern in &dictionary.patterns {
                if pattern.compression_gain > 0.01 { // Only high-value patterns
                    global_dict.patterns.push(pattern.clone());
                }
            }
            
            // Limit global dictionary size
            if global_dict.patterns.len() > self.max_dictionary_size {
                global_dict.patterns.sort_by(|a, b| b.compression_gain.partial_cmp(&a.compression_gain).unwrap());
                global_dict.patterns.truncate(self.max_dictionary_size);
            }
            
            global_dict.last_updated = Instant::now();
        }
    }
    
    pub fn get_dictionary_for_content(&self, content_hash: u64) -> Option<CompressionDictionary> {
        if let Ok(content_dicts) = self.content_dictionaries.lock() {
            content_dicts.get(&content_hash).cloned()
        } else {
            None
        }
    }
    
    pub fn get_global_dictionary(&self) -> CompressionDictionary {
        if let Ok(global_dict) = self.global_dictionary.read() {
            global_dict.clone()
        } else {
            CompressionDictionary::new()
        }
    }
}

impl CompressionDictionary {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            pattern_frequencies: HashMap::new(),
            last_updated: Instant::now(),
            effectiveness_score: 0.0,
        }
    }
    
    pub fn estimate_compression_improvement(&self, data: &[u8]) -> f64 {
        let mut total_savings = 0;
        
        for pattern in &self.patterns {
            let occurrences = count_occurrences(data, &pattern.data);
            if occurrences > 1 {
                let pattern_size = pattern.data.len();
                let reference_size = 3; // LZ77 reference
                let savings_per_occurrence = pattern_size.saturating_sub(reference_size);
                total_savings += savings_per_occurrence * occurrences;
            }
        }
        
        total_savings as f64 / data.len() as f64
    }
}

fn count_occurrences(data: &[u8], pattern: &[u8]) -> usize {
    if pattern.len() > data.len() {
        return 0;
    }
    
    let mut count = 0;
    for i in 0..=data.len() - pattern.len() {
        if &data[i..i + pattern.len()] == pattern {
            count += 1;
        }
    }
    count
}

/// Machine Learning-based parameter optimization
pub struct MLOptimizer {
    parameter_history: Arc<Mutex<Vec<OptimizationRecord>>>,
    feature_weights: Arc<RwLock<HashMap<String, f64>>>,
    model_accuracy: f64,
    system_profile: SystemProfile,
}

#[derive(Debug, Clone)]
pub struct OptimizationRecord {
    pub features: FeatureVector,
    pub parameters: CompressionParameters,
    pub performance: PerformanceResult,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub file_size: f64,
    pub entropy: f64,
    pub repetition_factor: f64,
    pub binary_ratio: f64,
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub target_compression_level: f64,
}

#[derive(Debug, Clone)]
pub struct CompressionParameters {
    pub chunk_size: usize,
    pub thread_count: usize,
    pub window_size: usize,
    pub algorithm: AdaptiveAlgorithm,
    pub use_dictionary: bool,
    pub simd_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub compression_time: Duration,
    pub compression_ratio: f64,
    pub throughput_mbps: f64,
    pub efficiency_score: f64,
}

impl MLOptimizer {
    pub fn new(system_profile: SystemProfile) -> Self {
        Self {
            parameter_history: Arc::new(Mutex::new(Vec::new())),
            feature_weights: Arc::new(RwLock::new(Self::initialize_weights())),
            model_accuracy: 0.5, // Start with low confidence
            system_profile,
        }
    }
    
    fn initialize_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("file_size".to_string(), 0.3);
        weights.insert("entropy".to_string(), 0.25);
        weights.insert("repetition_factor".to_string(), 0.2);
        weights.insert("binary_ratio".to_string(), 0.1);
        weights.insert("cpu_cores".to_string(), 0.1);
        weights.insert("memory_gb".to_string(), 0.05);
        weights
    }
    
    pub fn extract_features(&self, data: &[u8], characteristics: &ContentCharacteristics) -> FeatureVector {
        FeatureVector {
            file_size: (data.len() as f64).log10(),
            entropy: characteristics.entropy,
            repetition_factor: characteristics.repetition_factor,
            binary_ratio: characteristics.binary_ratio,
            cpu_cores: self.system_profile.cpu.logical_cores as f64,
            memory_gb: (self.system_profile.memory.total_memory as f64) / (1024.0 * 1024.0 * 1024.0),
            target_compression_level: 6.0, // Default, will be overridden
        }
    }
    
    pub fn predict_optimal_parameters(&self, features: &FeatureVector) -> CompressionParameters {
        if self.model_accuracy < 0.7 {
            // Use heuristics if model isn't confident enough
            return self.heuristic_parameters(features);
        }
        
        // Simple linear model for now (in production, would use more sophisticated ML)
        let weights = self.feature_weights.read().unwrap();
        
        // Predict optimal chunk size
        let chunk_score = features.file_size * weights.get("file_size").unwrap_or(&0.3)
            + features.entropy * weights.get("entropy").unwrap_or(&0.25);
        
        let chunk_size = match chunk_score {
            score if score < 2.0 => 32 * 1024,    // Small files
            score if score < 4.0 => 128 * 1024,   // Medium files
            score if score < 6.0 => 512 * 1024,   // Large files
            _ => 2 * 1024 * 1024,                  // Very large files
        };
        
        // Predict optimal thread count
        let thread_score = features.cpu_cores * 0.8 + features.file_size * 0.2;
        let thread_count = (thread_score as usize).min(self.system_profile.cpu.logical_cores).max(1);
        
        // Predict algorithm
        let algorithm = if features.entropy > 0.8 {
            AdaptiveAlgorithm::Lz4  // High entropy = less compressible
        } else if features.repetition_factor > 0.7 {
            AdaptiveAlgorithm::Zstd // High repetition = good for advanced compression
        } else {
            AdaptiveAlgorithm::BalancedGzip
        };
        
        CompressionParameters {
            chunk_size,
            thread_count,
            window_size: if features.repetition_factor > 0.5 { 32768 } else { 16384 },
            algorithm,
            use_dictionary: features.repetition_factor > 0.4,
            simd_enabled: self.system_profile.cpu.has_avx2 || self.system_profile.cpu.has_neon,
        }
    }
    
    fn heuristic_parameters(&self, features: &FeatureVector) -> CompressionParameters {
        // Conservative heuristic-based parameters
        let file_size = (10.0_f64).powf(features.file_size) as usize;
        
        CompressionParameters {
            chunk_size: match file_size {
                size if size < 64 * 1024 => 16 * 1024,
                size if size < 1024 * 1024 => 64 * 1024,
                size if size < 10 * 1024 * 1024 => 256 * 1024,
                _ => 1024 * 1024,
            },
            thread_count: (self.system_profile.cpu.physical_cores / 2).max(1),
            window_size: 16384,
            algorithm: AdaptiveAlgorithm::BalancedGzip,
            use_dictionary: features.repetition_factor > 0.5,
            simd_enabled: true,
        }
    }
    
    pub fn record_performance(&self, features: FeatureVector, parameters: CompressionParameters, performance: PerformanceResult) {
        let record = OptimizationRecord {
            features,
            parameters,
            performance,
            timestamp: Instant::now(),
        };
        
        if let Ok(mut history) = self.parameter_history.lock() {
            history.push(record);
            
            // Limit history size
            if history.len() > 10000 {
                history.remove(0);
            }
            
            // Update model periodically
            if history.len() % 100 == 0 {
                self.update_model(&history);
            }
        }
    }
    
    fn update_model(&self, history: &[OptimizationRecord]) {
        if history.len() < 50 {
            return; // Need more data
        }
        
        // Simple weight update based on correlation with performance
        let mut new_weights = HashMap::new();
        
        // Calculate correlation between features and efficiency scores
        let efficiency_scores: Vec<f64> = history.iter().map(|r| r.performance.efficiency_score).collect();
        let avg_efficiency = efficiency_scores.iter().sum::<f64>() / efficiency_scores.len() as f64;
        
        // File size correlation
        let file_size_correlation = self.calculate_correlation(
            &history.iter().map(|r| r.features.file_size).collect::<Vec<_>>(),
            &efficiency_scores,
        );
        new_weights.insert("file_size".to_string(), file_size_correlation.abs());
        
        // Entropy correlation
        let entropy_correlation = self.calculate_correlation(
            &history.iter().map(|r| r.features.entropy).collect::<Vec<_>>(),
            &efficiency_scores,
        );
        new_weights.insert("entropy".to_string(), entropy_correlation.abs());
        
        // Update weights
        if let Ok(mut weights) = self.feature_weights.write() {
            *weights = new_weights;
        }
        
        // Update model accuracy based on recent predictions
        // (In a real implementation, this would be more sophisticated)
        let recent_records = &history[history.len().saturating_sub(20)..];
        let accuracy = self.evaluate_predictions(recent_records);
        // Update model_accuracy (would need mutable access in real implementation)
    }
    
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn evaluate_predictions(&self, records: &[OptimizationRecord]) -> f64 {
        if records.is_empty() {
            return 0.5;
        }
        
        // Simple accuracy: percentage of predictions that resulted in above-average performance
        let avg_efficiency: f64 = records.iter().map(|r| r.performance.efficiency_score).sum::<f64>() / records.len() as f64;
        let above_average = records.iter().filter(|r| r.performance.efficiency_score > avg_efficiency).count();
        
        above_average as f64 / records.len() as f64
    }
}

/// Advanced compression coordinator that combines all optimization techniques
pub struct AdvancedCompressionCoordinator {
    dictionary_engine: DictionaryEngine,
    ml_optimizer: MLOptimizer,
    simd_accelerator: SimdCompressionAccelerator,
    system_profile: SystemProfile,
}

impl AdvancedCompressionCoordinator {
    pub fn new(system_profile: SystemProfile) -> Self {
        Self {
            dictionary_engine: DictionaryEngine::new(512), // Max 512 dictionary patterns
            ml_optimizer: MLOptimizer::new(system_profile.clone()),
            simd_accelerator: SimdCompressionAccelerator::new(&system_profile),
            system_profile,
        }
    }
    
    pub fn optimize_compression(&self, data: &[u8], characteristics: &ContentCharacteristics, target_level: u8) -> AdvancedCompressionPlan {
        // Extract ML features
        let mut features = self.ml_optimizer.extract_features(data, characteristics);
        features.target_compression_level = target_level as f64;
        
        // Get ML-predicted parameters
        let base_parameters = self.ml_optimizer.predict_optimal_parameters(&features);
        
        // Build content-specific dictionary
        let content_hash = self.calculate_content_hash(data);
        let dictionary = self.dictionary_engine.analyze_and_build_dictionary(data, content_hash);
        
        // Estimate compression improvement from dictionary
        let dictionary_improvement = dictionary.estimate_compression_improvement(data);
        
        // Adjust parameters based on dictionary effectiveness
        let adjusted_parameters = if dictionary_improvement > 0.05 {
            CompressionParameters {
                use_dictionary: true,
                window_size: base_parameters.window_size * 2, // Larger window for dictionary
                ..base_parameters
            }
        } else {
            CompressionParameters {
                use_dictionary: false,
                ..base_parameters
            }
        };
        
        AdvancedCompressionPlan {
            parameters: adjusted_parameters,
            dictionary: if dictionary_improvement > 0.05 { Some(dictionary) } else { None },
            expected_improvement: dictionary_improvement,
            confidence_score: self.ml_optimizer.model_accuracy,
            features,
        }
    }
    
    fn calculate_content_hash(&self, data: &[u8]) -> u64 {
        // Simple hash function for content identification
        let mut hash = 0u64;
        for (i, &byte) in data.iter().enumerate().take(1024) { // Sample first 1KB
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64).wrapping_add(i as u64);
        }
        hash
    }
    
    pub fn record_compression_result(&self, plan: &AdvancedCompressionPlan, performance: PerformanceResult) {
        self.ml_optimizer.record_performance(plan.features.clone(), plan.parameters.clone(), performance);
    }
    
    pub fn get_performance_insights(&self) -> PerformanceInsights {
        if let Ok(history) = self.ml_optimizer.parameter_history.lock() {
            if history.is_empty() {
                return PerformanceInsights::default();
            }
            
            let recent_records = &history[history.len().saturating_sub(100)..];
            
            let avg_efficiency: f64 = recent_records.iter().map(|r| r.performance.efficiency_score).sum::<f64>() / recent_records.len() as f64;
            let avg_throughput: f64 = recent_records.iter().map(|r| r.performance.throughput_mbps).sum::<f64>() / recent_records.len() as f64;
            let avg_compression_ratio: f64 = recent_records.iter().map(|r| r.performance.compression_ratio).sum::<f64>() / recent_records.len() as f64;
            
            // Find best performing configuration
            let best_record = recent_records.iter().max_by(|a, b| a.performance.efficiency_score.partial_cmp(&b.performance.efficiency_score).unwrap());
            
            PerformanceInsights {
                average_efficiency: avg_efficiency,
                average_throughput: avg_throughput,
                average_compression_ratio: avg_compression_ratio,
                best_parameters: best_record.map(|r| r.parameters.clone()),
                model_confidence: self.ml_optimizer.model_accuracy,
                total_optimizations: history.len(),
            }
        } else {
            PerformanceInsights::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdvancedCompressionPlan {
    pub parameters: CompressionParameters,
    pub dictionary: Option<CompressionDictionary>,
    pub expected_improvement: f64,
    pub confidence_score: f64,
    pub features: FeatureVector,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceInsights {
    pub average_efficiency: f64,
    pub average_throughput: f64,
    pub average_compression_ratio: f64,
    pub best_parameters: Option<CompressionParameters>,
    pub model_confidence: f64,
    pub total_optimizations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_compression::ContentCharacteristics;
    
    #[test]
    fn test_dictionary_engine() {
        let engine = DictionaryEngine::new(100);
        let test_data = b"Hello world! Hello world! This is a test. This is a test.";
        
        let dictionary = engine.analyze_and_build_dictionary(test_data, 12345);
        
        assert!(!dictionary.patterns.is_empty(), "Dictionary should contain patterns");
        
        // Check that patterns are meaningful
        let has_hello_world = dictionary.patterns.iter().any(|p| p.data == b"Hello world!");
        let has_this_is = dictionary.patterns.iter().any(|p| p.data.starts_with(b"This is"));
        
        assert!(has_hello_world || has_this_is, "Dictionary should contain repeated patterns");
        
        // Test compression improvement estimation
        let improvement = dictionary.estimate_compression_improvement(test_data);
        assert!(improvement > 0.0, "Should estimate some compression improvement");
    }
    
    #[test]
    fn test_ml_optimizer() {
        let system_profile = SystemProfile::detect();
        let optimizer = MLOptimizer::new(system_profile);
        
        let characteristics = ContentCharacteristics {
            entropy: 0.5,
            repetition_factor: 0.7,
            binary_ratio: 0.2,
            compressibility_score: 0.8,
        };
        
        let test_data = vec![0xAB; 10000];
        let features = optimizer.extract_features(&test_data, &characteristics);
        
        assert!(features.file_size > 0.0, "Should extract valid file size feature");
        assert_eq!(features.entropy, 0.5, "Should preserve entropy from characteristics");
        
        let parameters = optimizer.predict_optimal_parameters(&features);
        
        assert!(parameters.chunk_size > 0, "Should predict valid chunk size");
        assert!(parameters.thread_count > 0, "Should predict valid thread count");
        assert!(parameters.window_size > 0, "Should predict valid window size");
    }
    
    #[test]
    fn test_advanced_coordinator() {
        let system_profile = SystemProfile::detect();
        let coordinator = AdvancedCompressionCoordinator::new(system_profile);
        
        let test_data = b"Repeated pattern! Repeated pattern! Different text here. Different text here.";
        let characteristics = ContentCharacteristics {
            entropy: 0.6,
            repetition_factor: 0.8,
            binary_ratio: 0.0,
            compressibility_score: 0.9,
        };
        
        let plan = coordinator.optimize_compression(test_data, &characteristics, 6);
        
        assert!(plan.parameters.chunk_size > 0, "Plan should have valid chunk size");
        assert!(plan.confidence_score >= 0.0 && plan.confidence_score <= 1.0, "Confidence should be valid probability");
        
        // High repetition should suggest using dictionary
        if plan.expected_improvement > 0.05 {
            assert!(plan.dictionary.is_some(), "High repetition should suggest dictionary use");
        }
        
        // Simulate compression result
        let performance = PerformanceResult {
            compression_time: Duration::from_millis(100),
            compression_ratio: 0.6,
            throughput_mbps: 50.0,
            efficiency_score: 0.8,
        };
        
        coordinator.record_compression_result(&plan, performance);
        
        let insights = coordinator.get_performance_insights();
        assert!(insights.total_optimizations > 0, "Should record optimization attempts");
    }
    
    #[test]
    fn test_pattern_extraction() {
        let engine = DictionaryEngine::new(50);
        let data = b"abcabc123123defdefghighighi";
        
        let patterns = engine.extract_patterns(data);
        
        // Should find repeated patterns
        assert!(patterns.iter().any(|p| p == b"abc"), "Should find 'abc' pattern");
        assert!(patterns.iter().any(|p| p == b"123"), "Should find '123' pattern");
        assert!(patterns.iter().any(|p| p == b"def"), "Should find 'def' pattern");
        assert!(patterns.iter().any(|p| p == b"ghi"), "Should find 'ghi' pattern");
    }
    
    #[test]
    fn test_correlation_calculation() {
        let system_profile = SystemProfile::detect();
        let optimizer = MLOptimizer::new(system_profile);
        
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = optimizer.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 0.001, "Should detect perfect positive correlation");
        
        // No correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let correlation = optimizer.calculate_correlation(&x, &y);
        assert!(correlation.abs() < 0.001, "Should detect no correlation");
    }
}