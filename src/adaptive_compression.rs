use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::Duration;

use crate::hardware_analysis::{SystemProfile, CompressionStrategy};
use crate::profiler::AdvancedProfiler;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveAlgorithm {
    FastGzip,     // Level 1-3: Speed optimized
    BalancedGzip, // Level 4-6: Balance speed/compression
    MaxGzip,      // Level 7-9: Compression optimized
    Zstd,         // Modern alternative
    Lz4,          // Ultra-fast for specific workloads
}

#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    pub algorithm: AdaptiveAlgorithm,
    pub input_size: usize,
    pub output_size: usize,
    pub compression_time: Duration,
    pub compression_ratio: f64,
    pub throughput_mbps: f64,
    pub efficiency_score: f64, // Combines ratio and speed
}

#[derive(Debug, Clone)]
pub struct ContentCharacteristics {
    pub entropy: f64,           // Measure of randomness (0.0 = very compressible, 1.0 = random)
    pub repetition_factor: f64, // Amount of repeated patterns
    pub binary_ratio: f64,      // Ratio of binary vs text content
    pub compressibility_score: f64, // Overall compressibility estimate
}

pub struct AdaptiveCompressionEngine {
    system_profile: SystemProfile,
    profiler: Option<Arc<AdvancedProfiler>>,
    performance_history: Arc<Mutex<HashMap<(AdaptiveAlgorithm, usize), CompressionMetrics>>>,
    learning_enabled: bool,
}

impl AdaptiveCompressionEngine {
    pub fn new(system_profile: SystemProfile) -> Self {
        Self {
            system_profile,
            profiler: None,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            learning_enabled: true,
        }
    }
    
    pub fn with_profiler(mut self, profiler: Arc<AdvancedProfiler>) -> Self {
        self.profiler = Some(profiler);
        self
    }
    
    pub fn analyze_content(&self, data: &[u8]) -> ContentCharacteristics {
        let entropy = calculate_entropy(data);
        let repetition_factor = calculate_repetition_factor(data);
        let binary_ratio = calculate_binary_ratio(data);
        let compressibility_score = estimate_compressibility(entropy, repetition_factor, binary_ratio);
        
        ContentCharacteristics {
            entropy,
            repetition_factor,
            binary_ratio,
            compressibility_score,
        }
    }
    
    pub fn select_optimal_algorithm(
        &self,
        content_characteristics: &ContentCharacteristics,
        file_size: usize,
        compression_level: u8,
        target_metric: OptimizationTarget,
    ) -> AdaptiveAlgorithm {
        // First, try to use learned performance data
        if self.learning_enabled {
            if let Some(learned_choice) = self.query_performance_history(
                content_characteristics,
                file_size,
                target_metric,
            ) {
                return learned_choice;
            }
        }
        
        // Fallback to heuristic-based selection
        self.heuristic_algorithm_selection(content_characteristics, file_size, compression_level, target_metric)
    }
    
    fn heuristic_algorithm_selection(
        &self,
        characteristics: &ContentCharacteristics,
        file_size: usize,
        compression_level: u8,
        target_metric: OptimizationTarget,
    ) -> AdaptiveAlgorithm {
        match target_metric {
            OptimizationTarget::Speed => {
                if characteristics.compressibility_score < 0.3 {
                    // Highly compressible content
                    AdaptiveAlgorithm::Lz4
                } else if file_size < 1024 * 1024 {
                    // Small files
                    AdaptiveAlgorithm::FastGzip
                } else {
                    AdaptiveAlgorithm::Lz4
                }
            },
            OptimizationTarget::Compression => {
                if characteristics.entropy > 0.8 {
                    // High entropy (random-like) data - use fast algorithm
                    AdaptiveAlgorithm::FastGzip
                } else if characteristics.repetition_factor > 0.7 {
                    // Highly repetitive - benefit from advanced compression
                    AdaptiveAlgorithm::Zstd
                } else {
                    match compression_level {
                        1..=3 => AdaptiveAlgorithm::FastGzip,
                        4..=6 => AdaptiveAlgorithm::BalancedGzip,
                        7..=9 => AdaptiveAlgorithm::MaxGzip,
                        _ => AdaptiveAlgorithm::BalancedGzip,
                    }
                }
            },
            OptimizationTarget::Balanced => {
                if file_size < 64 * 1024 {
                    // Small files: prefer speed
                    AdaptiveAlgorithm::FastGzip
                } else if characteristics.compressibility_score < 0.4 {
                    // Highly compressible: worth the compression cost
                    AdaptiveAlgorithm::Zstd
                } else {
                    AdaptiveAlgorithm::BalancedGzip
                }
            },
        }
    }
    
    pub fn record_performance(&self, metrics: CompressionMetrics) {
        if !self.learning_enabled {
            return;
        }
        
        if let Ok(mut history) = self.performance_history.lock() {
            let key = (metrics.algorithm, size_bucket(metrics.input_size));
            
            // Store the best performing configuration for each algorithm/size combination
            let should_update = history.get(&key)
                .map(|existing| metrics.efficiency_score > existing.efficiency_score)
                .unwrap_or(true);
            
            if should_update {
                history.insert(key, metrics);
            }
        }
    }
    
    fn query_performance_history(
        &self,
        _characteristics: &ContentCharacteristics,
        file_size: usize,
        target_metric: OptimizationTarget,
    ) -> Option<AdaptiveAlgorithm> {
        let history = self.performance_history.lock().ok()?;
        let size_bucket = size_bucket(file_size);
        
        // Find the best algorithm for this size bucket and optimization target
        let mut best_algorithm = None;
        let mut best_score = 0.0;
        
        for algorithm in [
            AdaptiveAlgorithm::FastGzip,
            AdaptiveAlgorithm::BalancedGzip,
            AdaptiveAlgorithm::MaxGzip,
            AdaptiveAlgorithm::Zstd,
            AdaptiveAlgorithm::Lz4,
        ] {
            if let Some(metrics) = history.get(&(algorithm, size_bucket)) {
                let score = match target_metric {
                    OptimizationTarget::Speed => metrics.throughput_mbps,
                    OptimizationTarget::Compression => 1.0 / metrics.compression_ratio, // Lower ratio is better
                    OptimizationTarget::Balanced => metrics.efficiency_score,
                };
                
                if score > best_score {
                    best_score = score;
                    best_algorithm = Some(algorithm);
                }
            }
        }
        
        best_algorithm
    }
    
    pub fn get_compression_strategy(&self, algorithm: AdaptiveAlgorithm, file_size: usize) -> CompressionStrategy {
        let base_strategy = self.system_profile.recommend_compression_strategy(file_size, 6);
        
        // Adjust strategy based on algorithm characteristics
        match algorithm {
            AdaptiveAlgorithm::Lz4 => CompressionStrategy {
                thread_count: (base_strategy.thread_count * 2).min(self.system_profile.cpu.logical_cores),
                chunk_size: base_strategy.chunk_size * 2, // Larger chunks for fast algorithm
                buffer_size: base_strategy.buffer_size,
                use_simd: self.system_profile.cpu.has_avx2 || self.system_profile.cpu.has_neon,
                use_prefetch: file_size > 10 * 1024 * 1024,
                use_huge_pages: base_strategy.use_huge_pages,
            },
            AdaptiveAlgorithm::FastGzip => base_strategy,
            AdaptiveAlgorithm::BalancedGzip => CompressionStrategy {
                chunk_size: base_strategy.chunk_size / 2, // Smaller chunks for better compression
                ..base_strategy
            },
            AdaptiveAlgorithm::MaxGzip => CompressionStrategy {
                thread_count: base_strategy.thread_count,
                chunk_size: base_strategy.chunk_size / 4, // Much smaller chunks
                use_prefetch: true, // Always use prefetch for max compression
                ..base_strategy
            },
            AdaptiveAlgorithm::Zstd => CompressionStrategy {
                chunk_size: base_strategy.chunk_size, // Zstd handles chunks well
                use_simd: true, // Zstd benefits from SIMD
                ..base_strategy
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationTarget {
    Speed,       // Minimize compression time
    Compression, // Maximize compression ratio
    Balanced,    // Balance speed and compression
}

// Helper functions for content analysis
fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }
    
    let len = data.len() as f64;
    let mut entropy = 0.0;
    
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }
    
    entropy / 8.0 // Normalize to 0-1 range
}

fn calculate_repetition_factor(data: &[u8]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    
    let mut repetitive_bytes = 0;
    let window_size = 4.min(data.len());
    
    for i in 0..data.len() - window_size {
        let window = &data[i..i + window_size];
        // Look for this pattern in the next 64 bytes
        let search_end = (i + window_size + 64).min(data.len() - window_size);
        
        for j in i + window_size..search_end {
            if data[j..j + window_size] == *window {
                repetitive_bytes += window_size;
                break;
            }
        }
    }
    
    repetitive_bytes as f64 / data.len() as f64
}

fn calculate_binary_ratio(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let binary_bytes = data.iter()
        .filter(|&&b| b < 32 && b != b'\t' && b != b'\n' && b != b'\r')
        .count();
    
    binary_bytes as f64 / data.len() as f64
}

fn estimate_compressibility(entropy: f64, repetition_factor: f64, _binary_ratio: f64) -> f64 {
    // Simple heuristic: low entropy + high repetition = high compressibility
    let entropy_contribution = 1.0 - entropy;
    let repetition_contribution = repetition_factor;
    
    (entropy_contribution * 0.6 + repetition_contribution * 0.4).clamp(0.0, 1.0)
}

fn size_bucket(size: usize) -> usize {
    match size {
        0..=1024 => 0,
        1025..=8192 => 1,
        8193..=65536 => 2,
        65537..=524288 => 3,
        524289..=4194304 => 4,
        4194305..=33554432 => 5,
        _ => 6,
    }
}

impl CompressionMetrics {
    pub fn new(
        algorithm: AdaptiveAlgorithm,
        input_size: usize,
        output_size: usize,
        compression_time: Duration,
    ) -> Self {
        let compression_ratio = output_size as f64 / input_size as f64;
        let throughput_mbps = (input_size as f64 / (1024.0 * 1024.0)) / compression_time.as_secs_f64();
        
        // Efficiency score combines compression ratio and throughput
        // Lower ratio is better, higher throughput is better
        let ratio_score = (1.0 - compression_ratio).max(0.0);
        let speed_score = (throughput_mbps / 1000.0).min(1.0); // Normalize to reasonable range
        let efficiency_score = (ratio_score * 0.6 + speed_score * 0.4).clamp(0.0, 1.0);
        
        Self {
            algorithm,
            input_size,
            output_size,
            compression_time,
            compression_ratio,
            throughput_mbps,
            efficiency_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_content_analysis() {
        let system_profile = SystemProfile::detect();
        let engine = AdaptiveCompressionEngine::new(system_profile);
        
        // Test highly compressible data (repeated pattern)
        let compressible_data = vec![b'A'; 1024];
        let characteristics = engine.analyze_content(&compressible_data);
        
        assert!(characteristics.repetition_factor > 0.9);
        assert!(characteristics.compressibility_score > 0.8);
        
        // Test random data
        let random_data: Vec<u8> = (0..1024).map(|i| (i * 137) as u8).collect();
        let random_characteristics = engine.analyze_content(&random_data);
        
        assert!(random_characteristics.entropy > characteristics.entropy);
        assert!(random_characteristics.compressibility_score < characteristics.compressibility_score);
    }
    
    #[test]
    fn test_algorithm_selection() {
        let system_profile = SystemProfile::detect();
        let engine = AdaptiveCompressionEngine::new(system_profile);
        
        let high_compression_content = ContentCharacteristics {
            entropy: 0.3,
            repetition_factor: 0.8,
            binary_ratio: 0.1,
            compressibility_score: 0.9,
        };
        
        // Should prefer compression-optimized algorithms for highly compressible content
        let algorithm = engine.select_optimal_algorithm(
            &high_compression_content,
            1024 * 1024,
            9,
            OptimizationTarget::Compression,
        );
        
        assert!(matches!(algorithm, AdaptiveAlgorithm::Zstd | AdaptiveAlgorithm::MaxGzip));
        
        // Should prefer speed for speed optimization
        let speed_algorithm = engine.select_optimal_algorithm(
            &high_compression_content,
            1024 * 1024,
            1,
            OptimizationTarget::Speed,
        );
        
        assert!(matches!(speed_algorithm, AdaptiveAlgorithm::Lz4 | AdaptiveAlgorithm::FastGzip));
    }
    
    #[test]
    fn test_performance_learning() {
        let system_profile = SystemProfile::detect();
        let engine = AdaptiveCompressionEngine::new(system_profile);
        
        // Record some performance metrics
        let metrics = CompressionMetrics::new(
            AdaptiveAlgorithm::Zstd,
            1024 * 1024,
            512 * 1024,
            Duration::from_millis(100),
        );
        
        engine.record_performance(metrics.clone());
        
        // The engine should learn and potentially use this algorithm next time
        let characteristics = ContentCharacteristics {
            entropy: 0.5,
            repetition_factor: 0.6,
            binary_ratio: 0.2,
            compressibility_score: 0.7,
        };
        
        let selected = engine.select_optimal_algorithm(
            &characteristics,
            1024 * 1024,
            6,
            OptimizationTarget::Balanced,
        );
        
        // Should consider the learned performance
        println!("Selected algorithm based on learning: {:?}", selected);
    }
}