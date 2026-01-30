//! LLVM Assembly Analysis Module
//!
//! This module provides tools to analyze and compare LLVM-generated code
//! with our hand-written assembly.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Micro-benchmark a specific code path to measure instruction-level performance
pub struct MicroBenchmark {
    pub name: String,
    pub iterations: u64,
    pub total_cycles: u64,
    pub total_time: Duration,
}

impl MicroBenchmark {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            iterations: 0,
            total_cycles: 0,
            total_time: Duration::ZERO,
        }
    }
    
    pub fn cycles_per_iter(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        self.total_cycles as f64 / self.iterations as f64
    }
    
    pub fn ns_per_iter(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        self.total_time.as_nanos() as f64 / self.iterations as f64
    }
}

/// Analysis of a decode operation
#[derive(Debug, Default)]
pub struct DecodeAnalysis {
    pub literals_decoded: u64,
    pub lengths_decoded: u64,
    pub matches_copied: u64,
    pub bytes_copied: u64,
    pub refills_done: u64,
    pub subtable_lookups: u64,
}

impl DecodeAnalysis {
    pub fn print_summary(&self) {
        eprintln!("\n=== DECODE ANALYSIS ===");
        eprintln!("Literals decoded:   {:>10}", self.literals_decoded);
        eprintln!("Lengths decoded:    {:>10}", self.lengths_decoded);
        eprintln!("Matches copied:     {:>10}", self.matches_copied);
        eprintln!("Bytes copied:       {:>10}", self.bytes_copied);
        eprintln!("Refills done:       {:>10}", self.refills_done);
        eprintln!("Subtable lookups:   {:>10}", self.subtable_lookups);
        
        let total_ops = self.literals_decoded + self.lengths_decoded;
        if total_ops > 0 {
            let literal_pct = self.literals_decoded as f64 / total_ops as f64 * 100.0;
            eprintln!("\nLiteral ratio: {:.1}%", literal_pct);
        }
        
        if self.matches_copied > 0 {
            let avg_match = self.bytes_copied as f64 / self.matches_copied as f64;
            eprintln!("Avg match length: {:.1} bytes", avg_match);
        }
    }
}

/// Compare two decode implementations
pub struct DecoderComparison {
    pub rust_time: Duration,
    pub asm_time: Duration,
    pub rust_correct: bool,
    pub asm_correct: bool,
    pub output_size: usize,
}

impl DecoderComparison {
    pub fn print_summary(&self) {
        eprintln!("\n=== DECODER COMPARISON ===");
        eprintln!("Output size: {} bytes", self.output_size);
        eprintln!("Rust time:   {:?} ({})", 
                 self.rust_time, 
                 if self.rust_correct { "correct" } else { "WRONG" });
        eprintln!("ASM time:    {:?} ({})", 
                 self.asm_time,
                 if self.asm_correct { "correct" } else { "WRONG" });
        
        let rust_mb_s = self.output_size as f64 / self.rust_time.as_secs_f64() / 1_000_000.0;
        let asm_mb_s = self.output_size as f64 / self.asm_time.as_secs_f64() / 1_000_000.0;
        
        eprintln!("Rust: {:.0} MB/s", rust_mb_s);
        eprintln!("ASM:  {:.0} MB/s ({:.1}% of Rust)", asm_mb_s, asm_mb_s / rust_mb_s * 100.0);
    }
}

/// Instruction pattern analysis
#[derive(Debug, Default)]
pub struct InstructionPatterns {
    pub load_sequences: Vec<String>,
    pub store_sequences: Vec<String>,
    pub branch_patterns: Vec<String>,
    pub simd_usage: Vec<String>,
}

/// Analyze the hot path of a decode function
pub fn analyze_hot_path(input: &[u8], output: &mut [u8]) -> DecodeAnalysis {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    // These would be incremented by instrumented code
    static LITERALS: AtomicU64 = AtomicU64::new(0);
    static LENGTHS: AtomicU64 = AtomicU64::new(0);
    static MATCHES: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);
    static REFILLS: AtomicU64 = AtomicU64::new(0);
    static SUBTABLES: AtomicU64 = AtomicU64::new(0);
    
    // Reset counters
    LITERALS.store(0, Ordering::SeqCst);
    LENGTHS.store(0, Ordering::SeqCst);
    MATCHES.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    REFILLS.store(0, Ordering::SeqCst);
    SUBTABLES.store(0, Ordering::SeqCst);
    
    // Run the instrumented decode
    // (In practice, this would use a specially instrumented version)
    let _ = crate::bgzf::inflate_into_pub(input, output);
    
    DecodeAnalysis {
        literals_decoded: LITERALS.load(Ordering::SeqCst),
        lengths_decoded: LENGTHS.load(Ordering::SeqCst),
        matches_copied: MATCHES.load(Ordering::SeqCst),
        bytes_copied: BYTES.load(Ordering::SeqCst),
        refills_done: REFILLS.load(Ordering::SeqCst),
        subtable_lookups: SUBTABLES.load(Ordering::SeqCst),
    }
}

/// Generate a performance profile of the decode loop
pub fn profile_decode_loop(input: &[u8], iterations: usize) -> HashMap<String, Duration> {
    let output_size = input.len() * 10; // Estimate
    let mut output = vec![0u8; output_size];
    
    let mut profile = HashMap::new();
    
    // Profile Rust decoder
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = crate::bgzf::inflate_into_pub(input, &mut output);
    }
    profile.insert("rust_total".to_string(), start.elapsed());
    
    // Profile v4 ASM decoder
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = crate::consume_first_decode::inflate_consume_first_asm_v4(input, &mut output);
    }
    profile.insert("asm_v4_total".to_string(), start.elapsed());
    
    profile
}

/// Estimate cycle counts for different operations
#[cfg(target_arch = "aarch64")]
pub fn estimate_cycle_counts() -> HashMap<&'static str, u32> {
    // Apple M3 approximate cycle counts
    let mut cycles = HashMap::new();
    
    // Memory operations
    cycles.insert("ldr (L1 hit)", 3);
    cycles.insert("ldr (L2 hit)", 10);
    cycles.insert("str", 1);
    cycles.insert("ldp", 4);
    cycles.insert("stp", 2);
    
    // ALU operations
    cycles.insert("add/sub", 1);
    cycles.insert("and/orr/eor", 1);
    cycles.insert("lsr/lsl", 1);
    cycles.insert("ubfx", 1);
    
    // Branches
    cycles.insert("b (taken)", 1);
    cycles.insert("b (not taken)", 1);
    cycles.insert("b (mispredict)", 10);
    cycles.insert("tbz/tbnz", 1);
    
    // Complex
    cycles.insert("mul", 3);
    cycles.insert("div", 10);
    
    cycles
}

#[cfg(target_arch = "x86_64")]
pub fn estimate_cycle_counts() -> HashMap<&'static str, u32> {
    let mut cycles = HashMap::new();
    cycles.insert("mov", 1);
    cycles.insert("add/sub", 1);
    cycles.insert("shr/shl", 1);
    cycles.insert("load (L1)", 4);
    cycles.insert("store", 1);
    cycles
}

/// Generate optimized ASM based on analysis
pub fn generate_optimized_asm_suggestions(analysis: &DecodeAnalysis) -> Vec<String> {
    let mut suggestions = Vec::new();
    
    // Based on literal ratio
    let total_ops = analysis.literals_decoded + analysis.lengths_decoded;
    if total_ops > 0 {
        let literal_pct = analysis.literals_decoded as f64 / total_ops as f64;
        
        if literal_pct > 0.7 {
            suggestions.push(
                "High literal ratio ({:.0}%) - optimize literal path with 8-literal batching"
                    .to_string()
            );
        } else if literal_pct < 0.3 {
            suggestions.push(
                "Low literal ratio ({:.0}%) - optimize match copy path"
                    .to_string()
            );
        }
    }
    
    // Based on average match length
    if analysis.matches_copied > 0 {
        let avg_match = analysis.bytes_copied as f64 / analysis.matches_copied as f64;
        
        if avg_match > 32.0 {
            suggestions.push(
                format!("Long average matches ({:.0} bytes) - use more aggressive SIMD copy", avg_match)
            );
        } else if avg_match < 8.0 {
            suggestions.push(
                format!("Short average matches ({:.0} bytes) - optimize small copy path", avg_match)
            );
        }
    }
    
    // Based on subtable usage
    if analysis.subtable_lookups > 0 {
        let subtable_pct = analysis.subtable_lookups as f64 / total_ops as f64 * 100.0;
        if subtable_pct > 1.0 {
            suggestions.push(
                format!("Subtable usage {:.1}% - optimize subtable path", subtable_pct)
            );
        }
    }
    
    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_micro_benchmark() {
        let mut bench = MicroBenchmark::new("test");
        bench.iterations = 1000;
        bench.total_time = Duration::from_millis(100);
        
        assert!(bench.ns_per_iter() > 0.0);
    }
    
    #[test]
    fn test_decode_analysis_print() {
        let analysis = DecodeAnalysis {
            literals_decoded: 1000,
            lengths_decoded: 200,
            matches_copied: 200,
            bytes_copied: 2000,
            refills_done: 50,
            subtable_lookups: 5,
        };
        
        analysis.print_summary();
    }
    
    #[test]
    fn analyze_silesia_patterns() {
        use std::fs;
        use flate2::read::GzDecoder;
        use std::io::Read;
        
        let data = match fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping - no SILESIA data file");
                return;
            }
        };
        
        let mut decoder = GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        decoder.read_to_end(&mut expected).unwrap();
        
        let deflate_data = &data[10..data.len() - 8];
        
        eprintln!("\n=== SILESIA ANALYSIS ===");
        eprintln!("Compressed: {} bytes", deflate_data.len());
        eprintln!("Uncompressed: {} bytes", expected.len());
        eprintln!("Ratio: {:.2}x", expected.len() as f64 / deflate_data.len() as f64);
        
        // Profile the decoders
        let profile = profile_decode_loop(deflate_data, 3);
        
        for (name, duration) in &profile {
            let mb_s = expected.len() as f64 * 3.0 / duration.as_secs_f64() / 1_000_000.0;
            eprintln!("{}: {:.0} MB/s ({:?})", name, mb_s, duration);
        }
        
        // Print cycle estimates
        let cycles = estimate_cycle_counts();
        eprintln!("\n=== CYCLE ESTIMATES ===");
        for (op, count) in &cycles {
            eprintln!("{}: {} cycles", op, count);
        }
    }
    
    #[test]
    fn compare_rust_vs_v4_detailed() {
        use std::fs;
        use std::time::Instant;
        use flate2::read::GzDecoder;
        use std::io::Read;
        
        let data = match fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping - no SILESIA data file");
                return;
            }
        };
        
        let mut decoder = GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        decoder.read_to_end(&mut expected).unwrap();
        
        let deflate_data = &data[10..data.len() - 8];
        let mut output = vec![0u8; expected.len() + 4096];
        
        // Warmup
        for _ in 0..3 {
            let _ = crate::bgzf::inflate_into_pub(deflate_data, &mut output);
            let _ = crate::consume_first_decode::inflate_consume_first_asm_v4(deflate_data, &mut output);
        }
        
        // Measure Rust
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::bgzf::inflate_into_pub(deflate_data, &mut output);
        }
        let rust_time = start.elapsed() / iterations as u32;
        
        // Verify Rust correctness
        let _ = crate::bgzf::inflate_into_pub(deflate_data, &mut output);
        let rust_correct = output[..expected.len()] == expected[..];
        
        // Measure v4
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::consume_first_decode::inflate_consume_first_asm_v4(deflate_data, &mut output);
        }
        let asm_time = start.elapsed() / iterations as u32;
        
        // Verify v4 correctness
        let _ = crate::consume_first_decode::inflate_consume_first_asm_v4(deflate_data, &mut output);
        let asm_correct = output[..expected.len()] == expected[..];
        
        let comparison = DecoderComparison {
            rust_time,
            asm_time,
            rust_correct,
            asm_correct,
            output_size: expected.len(),
        };
        
        comparison.print_summary();
        
        // Calculate performance gap
        let rust_mb_s = expected.len() as f64 / rust_time.as_secs_f64() / 1_000_000.0;
        let asm_mb_s = expected.len() as f64 / asm_time.as_secs_f64() / 1_000_000.0;
        let gap = (rust_mb_s - asm_mb_s) / rust_mb_s * 100.0;
        
        eprintln!("\n=== PERFORMANCE GAP ANALYSIS ===");
        eprintln!("Gap: {:.1}%", gap);
        eprintln!("To close gap, need to decode {:.0} more MB/s", rust_mb_s - asm_mb_s);
        
        // Time per byte
        let rust_ns_per_byte = rust_time.as_nanos() as f64 / expected.len() as f64;
        let asm_ns_per_byte = asm_time.as_nanos() as f64 / expected.len() as f64;
        
        eprintln!("Rust: {:.2} ns/byte", rust_ns_per_byte);
        eprintln!("ASM:  {:.2} ns/byte", asm_ns_per_byte);
        eprintln!("Difference: {:.2} ns/byte", asm_ns_per_byte - rust_ns_per_byte);
        
        // At 3GHz, that's approximately:
        let cycles_per_byte_rust = rust_ns_per_byte * 3.0; // Assuming 3GHz
        let cycles_per_byte_asm = asm_ns_per_byte * 3.0;
        
        eprintln!("\nEstimated cycles per byte (at 3GHz):");
        eprintln!("Rust: {:.2} cycles/byte", cycles_per_byte_rust);
        eprintln!("ASM:  {:.2} cycles/byte", cycles_per_byte_asm);
    }
    
    #[test]
    fn generate_llvm_analysis_commands() {
        eprintln!("\n=== COMMANDS TO ANALYZE LLVM ===\n");
        
        eprintln!("1. Generate LLVM assembly:");
        eprintln!("   RUSTFLAGS=\"--emit asm\" cargo build --release\n");
        
        eprintln!("2. Run Python analyzer:");
        eprintln!("   python3 scripts/analyze_llvm_asm.py\n");
        
        eprintln!("3. Find the hot function:");
        eprintln!("   grep -n 'decode_huffman_libdeflate_style' target/release/deps/*.s\n");
        
        eprintln!("4. Count instructions:");
        eprintln!("   sed -n 'START,/^[^[:space:]]/p' FILE.s | grep -c '^[[:space:]]'\n");
        
        eprintln!("5. Profile with perf (Linux):");
        eprintln!("   perf stat -e cycles,instructions,branch-misses ./target/release/gzippy -d < test.gz\n");
        
        eprintln!("6. Profile with Instruments (macOS):");
        eprintln!("   xcrun xctrace record --template 'CPU Counters' --launch ./target/release/gzippy -- -d < test.gz\n");
    }
}
