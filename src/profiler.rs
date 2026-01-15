use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operation: String,
    pub duration: Duration,
    pub bytes_processed: usize,
    pub thread_id: thread::ThreadId,
    pub timestamp: Instant,
    pub cpu_cycles: Option<u64>,
    pub cache_misses: Option<u64>,
    pub memory_allocations: usize,
}

#[derive(Debug, Clone)]
pub struct ThreadMetrics {
    pub thread_id: thread::ThreadId,
    pub cpu_usage: f64,
    pub memory_usage: usize,
    pub operations_completed: usize,
    pub total_processing_time: Duration,
    pub idle_time: Duration,
    pub contentions: usize,
}

#[derive(Debug, Default)]
pub struct SystemPerformanceProfile {
    pub metrics: Vec<PerformanceMetrics>,
    pub thread_metrics: HashMap<thread::ThreadId, ThreadMetrics>,
    pub compression_ratios: Vec<(usize, f64)>, // (input_size, ratio)
    pub throughput_samples: Vec<(Instant, f64)>, // (timestamp, MB/s)
    pub bottlenecks: Vec<BottleneckReport>,
}

#[derive(Debug, Clone)]
pub struct BottleneckReport {
    pub component: String,
    pub severity: BottleneckSeverity,
    pub description: String,
    pub recommendation: String,
    pub impact_percentage: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckSeverity {
    Critical, // >50% performance impact
    Major,    // 20-50% impact
    Minor,    // 5-20% impact
    Info,     // <5% impact
}

pub struct AdvancedProfiler {
    metrics: Arc<Mutex<SystemPerformanceProfile>>,
    start_time: Instant,
    enabled: bool,
}

impl AdvancedProfiler {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(SystemPerformanceProfile::default())),
            start_time: Instant::now(),
            enabled: true,
        }
    }
    
    pub fn start_operation(&self, operation: &str) -> OperationProfiler {
        OperationProfiler::new(operation.to_string(), self.metrics.clone(), self.enabled)
    }
    
    pub fn record_throughput(&self, bytes_per_second: f64) {
        if !self.enabled { return; }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.throughput_samples.push((Instant::now(), bytes_per_second / 1_000_000.0));
        }
    }
    
    pub fn record_compression_ratio(&self, input_size: usize, output_size: usize) {
        if !self.enabled { return; }
        
        let ratio = output_size as f64 / input_size as f64;
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.compression_ratios.push((input_size, ratio));
        }
    }
    
    pub fn analyze_performance(&self) -> PerformanceAnalysis {
        let metrics = self.metrics.lock().unwrap();
        
        let mut analysis = PerformanceAnalysis::default();
        
        // Analyze throughput trends
        analysis.average_throughput = calculate_average_throughput(&metrics.throughput_samples);
        analysis.peak_throughput = calculate_peak_throughput(&metrics.throughput_samples);
        
        // Analyze thread efficiency
        analysis.thread_efficiency = analyze_thread_efficiency(&metrics.thread_metrics);
        
        // Detect bottlenecks
        analysis.bottlenecks = detect_bottlenecks(&metrics);
        
        // Calculate overall efficiency score
        analysis.efficiency_score = calculate_efficiency_score(&metrics);
        
        analysis
    }
    
    pub fn generate_report(&self) -> String {
        let analysis = self.analyze_performance();
        let metrics = self.metrics.lock().unwrap();
        
        let mut report = String::new();
        report.push_str("=== ADVANCED PERFORMANCE ANALYSIS ===\n\n");
        
        // System overview
        report.push_str(&format!("Total Runtime: {:.2}s\n", self.start_time.elapsed().as_secs_f64()));
        report.push_str(&format!("Operations Recorded: {}\n", metrics.metrics.len()));
        report.push_str(&format!("Efficiency Score: {:.1}%\n\n", analysis.efficiency_score * 100.0));
        
        // Throughput analysis
        report.push_str("THROUGHPUT ANALYSIS:\n");
        report.push_str(&format!("  Average: {:.2} MB/s\n", analysis.average_throughput));
        report.push_str(&format!("  Peak: {:.2} MB/s\n", analysis.peak_throughput));
        report.push_str(&format!("  Utilization: {:.1}%\n\n", analysis.thread_efficiency * 100.0));
        
        // Bottleneck analysis
        if !analysis.bottlenecks.is_empty() {
            report.push_str("BOTTLENECK ANALYSIS:\n");
            for bottleneck in &analysis.bottlenecks {
                let severity_str = match bottleneck.severity {
                    BottleneckSeverity::Critical => "🔴 CRITICAL",
                    BottleneckSeverity::Major => "🟡 MAJOR",
                    BottleneckSeverity::Minor => "🟠 MINOR",
                    BottleneckSeverity::Info => "🔵 INFO",
                };
                report.push_str(&format!("  {} {}: {}\n", severity_str, bottleneck.component, bottleneck.description));
                report.push_str(&format!("    Impact: {:.1}% | Recommendation: {}\n", bottleneck.impact_percentage, bottleneck.recommendation));
            }
            report.push('\n');
        }
        
        // Performance recommendations
        report.push_str("OPTIMIZATION RECOMMENDATIONS:\n");
        let recommendations = generate_recommendations(&analysis);
        for (i, rec) in recommendations.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, rec));
        }
        
        report
    }
    
    pub fn enable_detailed_profiling(&mut self) {
        self.enabled = true;
    }
    
    pub fn disable_profiling(&mut self) {
        self.enabled = false;
    }
}

pub struct OperationProfiler {
    operation: String,
    start_time: Instant,
    start_allocations: usize,
    metrics: Arc<Mutex<SystemPerformanceProfile>>,
    enabled: bool,
}

impl OperationProfiler {
    fn new(operation: String, metrics: Arc<Mutex<SystemPerformanceProfile>>, enabled: bool) -> Self {
        Self {
            operation,
            start_time: Instant::now(),
            start_allocations: get_current_allocations(),
            metrics,
            enabled,
        }
    }
    
    pub fn finish(self, bytes_processed: usize) {
        if !self.enabled { return; }
        
        let duration = self.start_time.elapsed();
        let allocations = get_current_allocations() - self.start_allocations;
        
        let metric = PerformanceMetrics {
            operation: self.operation,
            duration,
            bytes_processed,
            thread_id: thread::current().id(),
            timestamp: self.start_time,
            cpu_cycles: read_cpu_cycles(),
            cache_misses: read_cache_misses(),
            memory_allocations: allocations,
        };
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.metrics.push(metric);
        }
    }
}

#[derive(Debug, Default)]
pub struct PerformanceAnalysis {
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub thread_efficiency: f64,
    pub efficiency_score: f64,
    pub bottlenecks: Vec<BottleneckReport>,
}

fn calculate_average_throughput(samples: &[(Instant, f64)]) -> f64 {
    if samples.is_empty() { return 0.0; }
    samples.iter().map(|(_, throughput)| throughput).sum::<f64>() / samples.len() as f64
}

fn calculate_peak_throughput(samples: &[(Instant, f64)]) -> f64 {
    samples.iter().map(|(_, throughput)| *throughput).fold(0.0, f64::max)
}

fn analyze_thread_efficiency(thread_metrics: &HashMap<thread::ThreadId, ThreadMetrics>) -> f64 {
    if thread_metrics.is_empty() { return 0.0; }
    
    let total_efficiency: f64 = thread_metrics.values()
        .map(|metrics| {
            let total_time = metrics.total_processing_time + metrics.idle_time;
            if total_time.as_nanos() > 0 {
                metrics.total_processing_time.as_secs_f64() / total_time.as_secs_f64()
            } else {
                0.0
            }
        })
        .sum();
    
    total_efficiency / thread_metrics.len() as f64
}

fn detect_bottlenecks(metrics: &SystemPerformanceProfile) -> Vec<BottleneckReport> {
    let mut bottlenecks = Vec::new();
    
    // Analyze operation durations to find slow operations
    let mut operation_times: HashMap<String, Vec<Duration>> = HashMap::new();
    for metric in &metrics.metrics {
        operation_times.entry(metric.operation.clone())
            .or_insert_with(Vec::new)
            .push(metric.duration);
    }
    
    for (operation, durations) in operation_times {
        let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
        let max_duration = durations.iter().max().unwrap();
        
        // If max duration is significantly higher than average, it's a bottleneck
        if max_duration.as_nanos() > avg_duration.as_nanos() * 3 {
            let impact = calculate_bottleneck_impact(&durations);
            let severity = classify_severity(impact);
            
            bottlenecks.push(BottleneckReport {
                component: operation.clone(),
                severity,
                description: format!("Operation '{}' shows high variance in execution time", operation),
                recommendation: generate_operation_recommendation(&operation, impact),
                impact_percentage: impact * 100.0,
            });
        }
    }
    
    // Analyze memory allocation patterns
    let total_allocations: usize = metrics.metrics.iter().map(|m| m.memory_allocations).sum();
    if total_allocations > 1_000_000 { // Arbitrary threshold
        bottlenecks.push(BottleneckReport {
            component: "Memory Management".to_string(),
            severity: BottleneckSeverity::Major,
            description: "High memory allocation rate detected".to_string(),
            recommendation: "Consider using memory pools or larger buffer sizes".to_string(),
            impact_percentage: 25.0,
        });
    }
    
    bottlenecks
}

fn calculate_bottleneck_impact(durations: &[Duration]) -> f64 {
    // Simple heuristic: impact based on variance
    let avg = durations.iter().sum::<Duration>().as_nanos() / durations.len() as u128;
    let variance: f64 = durations.iter()
        .map(|d| {
            let diff = d.as_nanos() as i128 - avg as i128;
            (diff * diff) as f64
        })
        .sum::<f64>() / durations.len() as f64;
    
    let std_dev = variance.sqrt();
    (std_dev / avg as f64).min(1.0) // Normalize to 0-1 range
}

fn classify_severity(impact: f64) -> BottleneckSeverity {
    match impact {
        i if i > 0.5 => BottleneckSeverity::Critical,
        i if i > 0.2 => BottleneckSeverity::Major,
        i if i > 0.05 => BottleneckSeverity::Minor,
        _ => BottleneckSeverity::Info,
    }
}

fn generate_operation_recommendation(operation: &str, _impact: f64) -> String {
    match operation {
        op if op.contains("read") => "Consider using async I/O or larger read buffers".to_string(),
        op if op.contains("compress") => "Try different compression algorithms or chunk sizes".to_string(),
        op if op.contains("write") => "Consider batched writes or faster storage".to_string(),
        _ => format!("Investigate '{}' operation for optimization opportunities", operation),
    }
}

fn calculate_efficiency_score(metrics: &SystemPerformanceProfile) -> f64 {
    // Composite score based on various factors
    let mut score = 1.0;
    
    // Penalize for high variance in operation times
    if !metrics.metrics.is_empty() {
        let durations: Vec<f64> = metrics.metrics.iter().map(|m| m.duration.as_secs_f64()).collect();
        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance = durations.iter().map(|&d| (d - avg).powi(2)).sum::<f64>() / durations.len() as f64;
        let cv = if avg > 0.0 { variance.sqrt() / avg } else { 0.0 };
        score *= 1.0 - cv.min(0.5); // Penalize high coefficient of variation
    }
    
    // Penalize for memory allocations
    let total_allocs: usize = metrics.metrics.iter().map(|m| m.memory_allocations).sum();
    if total_allocs > 100_000 {
        score *= 0.8; // 20% penalty for high allocations
    }
    
    score.max(0.0).min(1.0)
}

fn generate_recommendations(analysis: &PerformanceAnalysis) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if analysis.thread_efficiency < 0.7 {
        recommendations.push("Thread utilization is low - consider reducing thread count or improving work distribution".to_string());
    }
    
    if analysis.efficiency_score < 0.8 {
        recommendations.push("Overall efficiency is suboptimal - review algorithm choices and memory management".to_string());
    }
    
    if analysis.average_throughput < 50.0 {
        recommendations.push("Throughput is below expected levels - consider SIMD optimizations or better algorithms".to_string());
    }
    
    for bottleneck in &analysis.bottlenecks {
        if bottleneck.severity == BottleneckSeverity::Critical {
            recommendations.push(format!("URGENT: Address {} bottleneck - {}", bottleneck.component, bottleneck.recommendation));
        }
    }
    
    if recommendations.is_empty() {
        recommendations.push("Performance is good - consider micro-optimizations for further gains".to_string());
    }
    
    recommendations
}

// Platform-specific performance counter functions
fn read_cpu_cycles() -> Option<u64> {
    // This would require platform-specific implementation
    // For now, return None
    None
}

fn read_cache_misses() -> Option<u64> {
    // This would require platform-specific implementation
    // For now, return None  
    None
}

fn get_current_allocations() -> usize {
    // This would require a custom allocator or profiling tool
    // For now, return 0
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_basic_operation() {
        let profiler = AdvancedProfiler::new();
        
        {
            let op = profiler.start_operation("test_operation");
            std::thread::sleep(Duration::from_millis(10));
            op.finish(1024);
        }
        
        let analysis = profiler.analyze_performance();
        assert!(analysis.efficiency_score > 0.0);
        
        let report = profiler.generate_report();
        assert!(report.contains("ADVANCED PERFORMANCE ANALYSIS"));
    }
    
    #[test]
    fn test_bottleneck_detection() {
        let profiler = AdvancedProfiler::new();
        
        // Simulate operations with varying performance
        for i in 0..10 {
            let op = profiler.start_operation("variable_operation");
            if i % 3 == 0 {
                std::thread::sleep(Duration::from_millis(50)); // Slow operation
            } else {
                std::thread::sleep(Duration::from_millis(5)); // Fast operation
            }
            op.finish(1024);
        }
        
        let analysis = profiler.analyze_performance();
        // Should detect bottleneck due to variance
        assert!(!analysis.bottlenecks.is_empty());
    }
}