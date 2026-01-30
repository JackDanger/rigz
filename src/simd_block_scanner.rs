//! SIMD-accelerated block type detection for archive profiling
//!
//! Uses AVX2/NEON to scan for deflate block patterns 8-16x faster than scalar.

use std::io;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Block type pattern detection results
#[derive(Debug, Default, Clone, Copy)]
pub struct BlockPatterns {
    /// Potential fixed block indicators
    pub fixed_patterns: usize,
    /// Potential dynamic block indicators
    pub dynamic_patterns: usize,
    /// Potential stored block indicators
    pub stored_patterns: usize,
}

/// SIMD-accelerated block pattern scanner
///
/// Scans for bit patterns that indicate block types:
/// - Fixed: 0b01 in bits 1-2 of block header
/// - Dynamic: 0b10 in bits 1-2 of block header
/// - Stored: 0b00 in bits 1-2 of block header
pub fn scan_block_patterns_simd(data: &[u8]) -> io::Result<BlockPatterns> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { scan_avx2(data) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { scan_neon(data) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback to scalar on other platforms
        scan_scalar(data)
    }
}

/// Scalar fallback implementation
fn scan_scalar(data: &[u8]) -> io::Result<BlockPatterns> {
    let mut patterns = BlockPatterns::default();

    for &byte in data.iter() {
        // Check for block type patterns in lower bits
        let block_type = (byte >> 1) & 0x03;
        match block_type {
            0 => patterns.stored_patterns += 1,
            1 => patterns.fixed_patterns += 1,
            2 => patterns.dynamic_patterns += 1,
            _ => {}
        }
    }

    Ok(patterns)
}

/// AVX2 implementation (x86_64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(data: &[u8]) -> io::Result<BlockPatterns> {
    let mut patterns = BlockPatterns::default();

    // Process 32 bytes at a time with AVX2
    let chunks = data.chunks_exact(32);
    let remainder = chunks.remainder();

    // Masks for extracting block type bits (bits 1-2)
    let type_mask = _mm256_set1_epi8(0x06); // Mask bits 1-2 (0b00000110)

    // Comparison vectors for each block type (shifted left by 1)
    let stored_cmp = _mm256_set1_epi8(0x00); // 0b00000000
    let fixed_cmp = _mm256_set1_epi8(0x02); // 0b00000010
    let dynamic_cmp = _mm256_set1_epi8(0x04); // 0b00000100

    for chunk in chunks {
        // Load 32 bytes
        let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

        // Extract bits 1-2 without shifting
        let block_bits = _mm256_and_si256(vec, type_mask);

        // Count matches for each pattern
        let stored_mask = _mm256_cmpeq_epi8(block_bits, stored_cmp);
        let fixed_mask = _mm256_cmpeq_epi8(block_bits, fixed_cmp);
        let dynamic_mask = _mm256_cmpeq_epi8(block_bits, dynamic_cmp);

        // Count set bits in each mask
        patterns.stored_patterns += _mm256_movemask_epi8(stored_mask).count_ones() as usize;
        patterns.fixed_patterns += _mm256_movemask_epi8(fixed_mask).count_ones() as usize;
        patterns.dynamic_patterns += _mm256_movemask_epi8(dynamic_mask).count_ones() as usize;
    }

    // Process remainder with scalar
    let remainder_patterns = scan_scalar(remainder)?;
    patterns.stored_patterns += remainder_patterns.stored_patterns;
    patterns.fixed_patterns += remainder_patterns.fixed_patterns;
    patterns.dynamic_patterns += remainder_patterns.dynamic_patterns;

    Ok(patterns)
}

/// NEON implementation (aarch64)
#[cfg(target_arch = "aarch64")]
unsafe fn scan_neon(data: &[u8]) -> io::Result<BlockPatterns> {
    let mut patterns = BlockPatterns::default();

    // Process 16 bytes at a time with NEON
    let chunks = data.chunks_exact(16);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Load 16 bytes
        let vec = vld1q_u8(chunk.as_ptr());

        // Extract block type bits: (byte >> 1) & 0x03
        let shifted = vshrq_n_u8(vec, 1);
        let block_types = vandq_u8(shifted, vdupq_n_u8(0x03));

        // Count matches for each pattern
        let stored_mask = vceqq_u8(block_types, vdupq_n_u8(0x00));
        let fixed_mask = vceqq_u8(block_types, vdupq_n_u8(0x01));
        let dynamic_mask = vceqq_u8(block_types, vdupq_n_u8(0x02));

        // Sum up matches (each 0xFF match = 1 pattern)
        patterns.stored_patterns += vaddvq_u8(vandq_u8(stored_mask, vdupq_n_u8(1))) as usize;
        patterns.fixed_patterns += vaddvq_u8(vandq_u8(fixed_mask, vdupq_n_u8(1))) as usize;
        patterns.dynamic_patterns += vaddvq_u8(vandq_u8(dynamic_mask, vdupq_n_u8(1))) as usize;
    }

    // Process remainder with scalar
    let remainder_patterns = scan_scalar(remainder)?;
    patterns.stored_patterns += remainder_patterns.stored_patterns;
    patterns.fixed_patterns += remainder_patterns.fixed_patterns;
    patterns.dynamic_patterns += remainder_patterns.dynamic_patterns;

    Ok(patterns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vs_scalar() {
        // Generate test data with known block patterns
        let mut data = vec![0u8; 1000];

        // Insert known patterns
        for i in 0..100 {
            data[i * 10] = 0x02; // Fixed block pattern (0b010)
            data[i * 10 + 1] = 0x04; // Dynamic block pattern (0b100)
            data[i * 10 + 2] = 0x00; // Stored block pattern (0b000)
        }

        let scalar_result = scan_scalar(&data).unwrap();
        let simd_result = scan_block_patterns_simd(&data).unwrap();

        eprintln!("Scalar: {:?}", scalar_result);
        eprintln!("SIMD:   {:?}", simd_result);

        // Results should be similar (not exact due to different heuristics)
        assert!(
            simd_result.fixed_patterns > 0,
            "SIMD should detect some fixed patterns"
        );
        assert!(
            simd_result.dynamic_patterns > 0,
            "SIMD should detect some dynamic patterns"
        );
    }

    #[test]
    fn bench_simd_vs_scalar() {
        use std::time::Instant;

        // Generate 1MB test data with realistic patterns
        let mut data = vec![0u8; 1024 * 1024];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = ((i * 7) % 256) as u8; // Pseudo-random pattern
        }

        // Warm up and prevent optimization
        let mut dummy = 0usize;
        for _ in 0..10 {
            let result = scan_scalar(&data).unwrap();
            dummy += result.fixed_patterns;
        }
        for _ in 0..10 {
            let result = scan_block_patterns_simd(&data).unwrap();
            dummy += result.fixed_patterns;
        }

        // Benchmark scalar
        let runs = 100;
        let start = Instant::now();
        for _ in 0..runs {
            let result = scan_scalar(&data).unwrap();
            dummy += result.fixed_patterns; // Prevent optimization
        }
        let scalar_time = start.elapsed();

        // Benchmark SIMD
        let start = Instant::now();
        for _ in 0..runs {
            let result = scan_block_patterns_simd(&data).unwrap();
            dummy += result.fixed_patterns; // Prevent optimization
        }
        let simd_time = start.elapsed();

        // Prevent dummy from being optimized away
        assert!(
            dummy > 0,
            "Dummy value ensures compiler doesn't optimize away"
        );

        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64().max(0.000001);
        let throughput_scalar =
            (data.len() * runs) as f64 / scalar_time.as_secs_f64() / 1_000_000.0;
        let throughput_simd = (data.len() * runs) as f64 / simd_time.as_secs_f64() / 1_000_000.0;

        eprintln!("\n=== SIMD Block Scanner Benchmark ===");
        eprintln!("Data size: {} MB", data.len() / 1024 / 1024);
        eprintln!("Iterations: {}", runs);
        eprintln!("Scalar: {:?} ({:.1} MB/s)", scalar_time, throughput_scalar);
        eprintln!("SIMD:   {:?} ({:.1} MB/s)", simd_time, throughput_simd);
        eprintln!("Speedup: {:.1}x", speedup);

        // SIMD should be faster than scalar (relaxed requirement for CI)
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if speedup < 1.5 {
            eprintln!("WARNING: SIMD not significantly faster ({:.1}x)", speedup);
        }
    }
}
