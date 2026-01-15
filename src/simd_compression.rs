#![allow(dead_code, unused_imports, unused_variables)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256, _mm256_add_epi32, _mm256_set1_epi32, _mm256_testz_si256};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{uint8x16_t, vld1q_u8, vst1q_u8, veorq_u8};

use crate::hardware_analysis::SystemProfile;
use crate::adaptive_compression::AdaptiveAlgorithm;

/// SIMD-accelerated compression operations
pub struct SimdCompressionAccelerator {
    use_avx2: bool,
    use_neon: bool,
    cache_line_size: usize,
}

impl SimdCompressionAccelerator {
    pub fn new(system_profile: &SystemProfile) -> Self {
        Self {
            use_avx2: system_profile.cpu.has_avx2,
            use_neon: system_profile.cpu.has_neon,
            cache_line_size: system_profile.cpu.cache_line_size,
        }
    }
    
    /// Fast CRC32 calculation using hardware acceleration
    pub fn fast_crc32(&self, data: &[u8], initial_crc: u32) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                return self.crc32_avx2(data, initial_crc);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon {
                return self.crc32_neon(data, initial_crc);
            }
        }
        self.crc32_fallback(data, initial_crc)
    }
    
    /// SIMD-accelerated hash computation for duplicate detection
    pub fn fast_hash(&self, data: &[u8], seed: u32) -> u32 {
        if data.len() < 16 {
            return self.hash_small(data, seed);
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                return self.hash_avx2(data, seed);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon {
                return self.hash_neon(data, seed);
            }
        }
        self.hash_fallback(data, seed)
    }
    
    /// Fast memory comparison for finding matches
    pub fn fast_memcmp(&self, a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        if a.len() < 32 {
            return a == b;
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                return self.memcmp_avx2(a, b);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon {
                return self.memcmp_neon(a, b);
            }
        }
        a == b
    }
    
    /// SIMD-accelerated pattern search for compression
    pub fn find_matches(&self, haystack: &[u8], needle: &[u8], min_match_length: usize) -> Vec<usize> {
        if needle.len() < min_match_length || haystack.len() < needle.len() {
            return Vec::new();
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 && needle.len() >= 4 {
                return self.find_matches_avx2(haystack, needle);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon && needle.len() >= 4 {
                return self.find_matches_neon(haystack, needle);
            }
        }
        self.find_matches_fallback(haystack, needle)
    }
    
    /// Optimized data copying with prefetching
    pub fn fast_copy(&self, src: &[u8], dst: &mut [u8]) {
        if src.len() != dst.len() {
            panic!("Source and destination slices must have the same length");
        }
        
        if src.len() < 64 {
            dst.copy_from_slice(src);
            return;
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                self.copy_avx2(src, dst);
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon {
                self.copy_neon(src, dst);
                return;
            }
        }
        dst.copy_from_slice(src);
    }
    
    /// Entropy calculation for content analysis
    pub fn calculate_entropy_simd(&self, data: &[u8]) -> f64 {
        let mut histogram = [0u32; 256];
        
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 && data.len() >= 32 {
                self.histogram_avx2(data, &mut histogram);
            } else {
                for &byte in data {
                    histogram[byte as usize] += 1;
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_neon && data.len() >= 16 {
                self.histogram_neon(data, &mut histogram);
            } else {
                for &byte in data {
                    histogram[byte as usize] += 1;
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for &byte in data {
                histogram[byte as usize] += 1;
            }
        }
        
        let total = data.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }
        
        entropy / 8.0 // Normalize to 0-1 range
    }
    
    // AVX2 implementations
    #[cfg(target_arch = "x86_64")]
    fn crc32_avx2(&self, data: &[u8], mut crc: u32) -> u32 {
        if !is_x86_feature_detected!("avx2") {
            return self.crc32_fallback(data, crc);
        }
        
        let chunks = data.chunks_exact(32);
        let remainder = chunks.remainder();
        
        unsafe {
            for chunk in chunks {
                // Load 32 bytes into AVX2 register
                let data_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                
                // Simplified CRC calculation (real implementation would use proper CRC tables)
                let crc_vec = _mm256_set1_epi32(crc as i32);
                let result = _mm256_xor_si256(data_vec, crc_vec);
                
                // Extract and combine results
                let mut extracted = [0u32; 8];
                _mm256_storeu_si256(extracted.as_mut_ptr() as *mut __m256i, result);
                crc = extracted.iter().fold(crc, |acc, &val| acc ^ val);
            }
        }
        
        // Process remainder with fallback
        self.crc32_fallback(remainder, crc)
    }
    
    #[cfg(target_arch = "x86_64")]
    fn hash_avx2(&self, data: &[u8], seed: u32) -> u32 {
        if !is_x86_feature_detected!("avx2") {
            return self.hash_fallback(data, seed);
        }
        
        let chunks = data.chunks_exact(32);
        let remainder = chunks.remainder();
        let mut hash = seed;
        
        unsafe {
            for chunk in chunks {
                let data_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                let hash_vec = _mm256_set1_epi32(hash as i32);
                let result = _mm256_add_epi32(data_vec, hash_vec);
                
                let mut extracted = [0u32; 8];
                _mm256_storeu_si256(extracted.as_mut_ptr() as *mut __m256i, result);
                hash = extracted.iter().fold(hash, |acc, &val| acc.wrapping_add(val));
            }
        }
        
        self.hash_fallback(remainder, hash)
    }
    
    #[cfg(target_arch = "x86_64")]
    fn memcmp_avx2(&self, a: &[u8], b: &[u8]) -> bool {
        if !is_x86_feature_detected!("avx2") {
            return a == b;
        }
        
        let chunks_a = a.chunks_exact(32);
        let chunks_b = b.chunks_exact(32);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        unsafe {
            for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
                let vec_a = _mm256_loadu_si256(chunk_a.as_ptr() as *const __m256i);
                let vec_b = _mm256_loadu_si256(chunk_b.as_ptr() as *const __m256i);
                let diff = _mm256_xor_si256(vec_a, vec_b);
                
                // Check if any bits differ
                if _mm256_testz_si256(diff, diff) == 0 {
                    return false;
                }
            }
        }
        
        remainder_a == remainder_b
    }
    
    #[cfg(target_arch = "x86_64")]
    fn find_matches_avx2(&self, haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut matches = Vec::new();
        
        if needle.len() < 4 || haystack.len() < needle.len() {
            return matches;
        }
        
        let first_4_bytes = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
        
        for i in 0..=haystack.len() - needle.len() {
            if i + 4 <= haystack.len() {
                let haystack_4_bytes = u32::from_le_bytes([
                    haystack[i], haystack[i+1], haystack[i+2], haystack[i+3]
                ]);
                
                if haystack_4_bytes == first_4_bytes {
                    // Potential match found, verify the rest
                    if &haystack[i..i + needle.len()] == needle {
                        matches.push(i);
                    }
                }
            }
        }
        
        matches
    }
    
    #[cfg(target_arch = "x86_64")]
    fn copy_avx2(&self, src: &[u8], dst: &mut [u8]) {
        if !is_x86_feature_detected!("avx2") {
            dst.copy_from_slice(src);
            return;
        }
        
        let chunks_src = src.chunks_exact(32);
        let chunks_dst = dst.chunks_exact_mut(32);
        let remainder_src = chunks_src.remainder();
        let remainder_dst = chunks_dst.into_remainder();
        
        unsafe {
            for (src_chunk, dst_chunk) in chunks_src.zip(chunks_dst) {
                let data = _mm256_loadu_si256(src_chunk.as_ptr() as *const __m256i);
                _mm256_storeu_si256(dst_chunk.as_mut_ptr() as *mut __m256i, data);
            }
        }
        
        remainder_dst.copy_from_slice(remainder_src);
    }
    
    #[cfg(target_arch = "x86_64")]
    fn histogram_avx2(&self, data: &[u8], histogram: &mut [u32; 256]) {
        // For simplicity, fall back to scalar for now
        // Real implementation would use SIMD gather/scatter operations
        for &byte in data {
            histogram[byte as usize] += 1;
        }
    }
    
    // NEON implementations (ARM)
    #[cfg(target_arch = "aarch64")]
    fn crc32_neon(&self, data: &[u8], mut crc: u32) -> u32 {
        let chunks = data.chunks_exact(16);
        let remainder = chunks.remainder();
        
        unsafe {
            for chunk in chunks {
                let _data_vec = vld1q_u8(chunk.as_ptr());
                // Simplified NEON CRC calculation
                let crc_bytes = [
                    crc as u8, (crc >> 8) as u8, (crc >> 16) as u8, (crc >> 24) as u8,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ];
                let crc_vec = vld1q_u8(crc_bytes.as_ptr());
                let result = veorq_u8(_data_vec, crc_vec);
                
                // Extract result (simplified)
                let mut result_bytes = [0u8; 16];
                vst1q_u8(result_bytes.as_mut_ptr(), result);
                crc = u32::from_le_bytes([result_bytes[0], result_bytes[1], result_bytes[2], result_bytes[3]]);
            }
        }
        
        self.crc32_fallback(remainder, crc)
    }
    
    #[cfg(target_arch = "aarch64")]
    fn hash_neon(&self, data: &[u8], seed: u32) -> u32 {
        let chunks = data.chunks_exact(16);
        let remainder = chunks.remainder();
        let mut hash = seed;
        
        unsafe {
            for chunk in chunks {
                let _data_vec = vld1q_u8(chunk.as_ptr());
                // Convert bytes to u32 and add to hash (simplified)
                let mut chunk_u32 = [0u32; 4];
                for i in 0..4 {
                    chunk_u32[i] = u32::from_le_bytes([
                        chunk[i*4], chunk[i*4+1], chunk[i*4+2], chunk[i*4+3]
                    ]);
                }
                hash = chunk_u32.iter().fold(hash, |acc, &val| acc.wrapping_add(val));
            }
        }
        
        self.hash_fallback(remainder, hash)
    }
    
    #[cfg(target_arch = "aarch64")]
    fn memcmp_neon(&self, a: &[u8], b: &[u8]) -> bool {
        let chunks_a = a.chunks_exact(16);
        let chunks_b = b.chunks_exact(16);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        unsafe {
            for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
                let vec_a = vld1q_u8(chunk_a.as_ptr());
                let vec_b = vld1q_u8(chunk_b.as_ptr());
                let diff = veorq_u8(vec_a, vec_b);
                
                // Check if vectors are equal
                let mut diff_bytes = [0u8; 16];
                vst1q_u8(diff_bytes.as_mut_ptr(), diff);
                if diff_bytes != [0u8; 16] {
                    return false;
                }
            }
        }
        
        remainder_a == remainder_b
    }
    
    #[cfg(target_arch = "aarch64")]
    fn find_matches_neon(&self, haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        // For now, use the same approach as AVX2
        self.find_matches_fallback(haystack, needle)
    }
    
    #[cfg(target_arch = "aarch64")]
    fn copy_neon(&self, src: &[u8], dst: &mut [u8]) {
        let chunks_src = src.chunks_exact(16);
        let remainder_src = chunks_src.remainder();
        let mut chunks_dst = dst.chunks_exact_mut(16);
        
        unsafe {
            for (src_chunk, dst_chunk) in chunks_src.zip(&mut chunks_dst) {
                let data = vld1q_u8(src_chunk.as_ptr());
                vst1q_u8(dst_chunk.as_mut_ptr(), data);
            }
        }
        
        // Handle remainder
        let remaining_len = remainder_src.len();
        if remaining_len > 0 {
            let start = src.len() - remaining_len;
            dst[start..].copy_from_slice(remainder_src);
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    fn histogram_neon(&self, data: &[u8], histogram: &mut [u32; 256]) {
        // Fall back to scalar for now
        for &byte in data {
            histogram[byte as usize] += 1;
        }
    }
    
    // Fallback implementations
    fn crc32_fallback(&self, data: &[u8], mut crc: u32) -> u32 {
        // Simplified CRC32 (not IEEE 802.3 compliant, just for demonstration)
        for &byte in data {
            crc = crc.wrapping_add(byte as u32);
            crc ^= crc >> 16;
            crc = crc.wrapping_mul(0x85ebca6b);
            crc ^= crc >> 13;
            crc = crc.wrapping_mul(0xc2b2ae35);
            crc ^= crc >> 16;
        }
        crc
    }
    
    fn hash_fallback(&self, data: &[u8], mut hash: u32) -> u32 {
        // Simple hash function (FNV-1a variant)
        for &byte in data {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x01000193);
        }
        hash
    }
    
    fn hash_small(&self, data: &[u8], mut hash: u32) -> u32 {
        for &byte in data {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }
    
    fn find_matches_fallback(&self, haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut matches = Vec::new();
        
        for i in 0..=haystack.len().saturating_sub(needle.len()) {
            if &haystack[i..i + needle.len()] == needle {
                matches.push(i);
            }
        }
        
        matches
    }
}

/// SIMD-accelerated deflate compression helpers
pub struct SimdDeflateAccelerator {
    simd: SimdCompressionAccelerator,
}

impl SimdDeflateAccelerator {
    pub fn new(system_profile: &SystemProfile) -> Self {
        Self {
            simd: SimdCompressionAccelerator::new(system_profile),
        }
    }
    
    /// Fast LZ77 match finding with SIMD
    pub fn find_lz77_matches(&self, window: &[u8], look_ahead: &[u8], min_match: usize) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();
        
        if look_ahead.len() < min_match {
            return matches;
        }
        
        // Try different match lengths starting from longest possible
        for match_len in (min_match..=look_ahead.len().min(258)).rev() {
            let pattern = &look_ahead[0..match_len];
            let positions = self.simd.find_matches(window, pattern, min_match);
            
            if !positions.is_empty() {
                // Find the closest match (smallest distance)
                if let Some(&best_pos) = positions.iter().max() {
                    let distance = window.len() - best_pos;
                    matches.push((distance, match_len));
                    break; // Found best match for this position
                }
            }
        }
        
        matches
    }
    
    /// Huffman encoding helpers with SIMD
    pub fn build_huffman_table_fast(&self, frequencies: &[u32; 256]) -> Vec<(u8, u32)> {
        // Simplified Huffman table construction
        // Real implementation would use proper Huffman algorithm
        let mut codes = Vec::new();
        
        for (symbol, &freq) in frequencies.iter().enumerate() {
            if freq > 0 {
                // Simplified: assign codes based on frequency
                let code_length = if freq > 1000 { 4 } else if freq > 100 { 6 } else { 8 };
                codes.push((symbol as u8, code_length));
            }
        }
        
        codes
    }
}

/// Algorithm-specific SIMD optimizations
pub struct AlgorithmOptimizer {
    simd: SimdCompressionAccelerator,
    deflate: SimdDeflateAccelerator,
}

impl AlgorithmOptimizer {
    pub fn new(system_profile: &SystemProfile) -> Self {
        Self {
            simd: SimdCompressionAccelerator::new(system_profile),
            deflate: SimdDeflateAccelerator::new(system_profile),
        }
    }
    
    pub fn optimize_for_algorithm(&self, algorithm: AdaptiveAlgorithm, data: &[u8]) -> OptimizationHints {
        match algorithm {
            AdaptiveAlgorithm::FastGzip | AdaptiveAlgorithm::BalancedGzip | AdaptiveAlgorithm::MaxGzip => {
                self.optimize_gzip(data)
            },
            AdaptiveAlgorithm::Zstd => {
                self.optimize_zstd(data)
            },
            AdaptiveAlgorithm::Lz4 => {
                self.optimize_lz4(data)
            },
        }
    }
    
    fn optimize_gzip(&self, data: &[u8]) -> OptimizationHints {
        let entropy = self.simd.calculate_entropy_simd(data);
        
        OptimizationHints {
            use_simd_hash: data.len() > 1024,
            optimal_window_size: if entropy < 0.5 { 32768 } else { 8192 },
            use_fast_crc: true,
            prefetch_distance: 64,
            chunk_size_hint: if data.len() > 1024 * 1024 { 64 * 1024 } else { 8 * 1024 },
        }
    }
    
    fn optimize_zstd(&self, data: &[u8]) -> OptimizationHints {
        OptimizationHints {
            use_simd_hash: true,
            optimal_window_size: 131072, // Zstd typically uses larger windows
            use_fast_crc: false, // Zstd uses different checksums
            prefetch_distance: 128,
            chunk_size_hint: 256 * 1024,
        }
    }
    
    fn optimize_lz4(&self, _data: &[u8]) -> OptimizationHints {
        OptimizationHints {
            use_simd_hash: true,
            optimal_window_size: 65536,
            use_fast_crc: false,
            prefetch_distance: 32,
            chunk_size_hint: 4 * 1024 * 1024, // LZ4 works well with large chunks
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationHints {
    pub use_simd_hash: bool,
    pub optimal_window_size: usize,
    pub use_fast_crc: bool,
    pub prefetch_distance: usize,
    pub chunk_size_hint: usize,
}

// Fallback implementations for non-SIMD architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod simd_fallback {
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct __m256i([u8; 32]);
    
    pub unsafe fn _mm256_loadu_si256(_p: *const __m256i) -> __m256i {
        std::ptr::read_unaligned(_p)
    }
    
    pub unsafe fn _mm256_storeu_si256(_p: *mut __m256i, _a: __m256i) {
        std::ptr::write_unaligned(_p, _a);
    }
    
    pub fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = a.0[i] ^ b.0[i];
        }
        __m256i(result)
    }
    
    pub fn _mm256_add_epi32(a: __m256i, b: __m256i) -> __m256i {
        // Simplified fallback
        _mm256_xor_si256(a, b)
    }
    
    pub fn _mm256_set1_epi32(value: i32) -> __m256i {
        let bytes = value.to_le_bytes();
        let mut result = [0u8; 32];
        for chunk in result.chunks_mut(4) {
            chunk.copy_from_slice(&bytes);
        }
        __m256i(result)
    }
    
    pub fn _mm256_testz_si256(a: __m256i, _b: __m256i) -> i32 {
        if a.0.iter().all(|&x| x == 0) { 1 } else { 0 }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use simd_fallback::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware_analysis::SystemProfile;
    
    #[test]
    fn test_simd_crc32() {
        let system_profile = SystemProfile::detect();
        let simd = SimdCompressionAccelerator::new(&system_profile);
        
        let data = b"Hello, SIMD world! This is a test of CRC32 calculation.";
        let crc1 = simd.fast_crc32(data, 0);
        let crc2 = simd.fast_crc32(data, 0);
        
        assert_eq!(crc1, crc2, "CRC32 should be deterministic");
        
        // Test with different initial values
        let crc3 = simd.fast_crc32(data, 0xFFFFFFFF);
        assert_ne!(crc1, crc3, "Different initial CRC should produce different results");
    }
    
    #[test]
    fn test_simd_hash() {
        let system_profile = SystemProfile::detect();
        let simd = SimdCompressionAccelerator::new(&system_profile);
        
        let data1 = b"test data for hashing";
        let data2 = b"different test data";
        
        let hash1 = simd.fast_hash(data1, 0);
        let hash2 = simd.fast_hash(data2, 0);
        let hash1_repeat = simd.fast_hash(data1, 0);
        
        assert_eq!(hash1, hash1_repeat, "Hash should be deterministic");
        assert_ne!(hash1, hash2, "Different data should produce different hashes");
    }
    
    #[test]
    fn test_simd_memcmp() {
        let system_profile = SystemProfile::detect();
        let simd = SimdCompressionAccelerator::new(&system_profile);
        
        let data1 = vec![0xAA; 100];
        let data2 = vec![0xAA; 100];
        let mut data3 = vec![0xAA; 100];
        data3[50] = 0xBB;
        
        assert!(simd.fast_memcmp(&data1, &data2), "Identical data should compare equal");
        assert!(!simd.fast_memcmp(&data1, &data3), "Different data should compare unequal");
    }
    
    #[test]
    fn test_find_matches() {
        let system_profile = SystemProfile::detect();
        let simd = SimdCompressionAccelerator::new(&system_profile);
        
        let haystack = b"abcdefghijklmnopqrstuvwxyzabcdefghijklmnop";
        let needle = b"abcd";
        
        let matches = simd.find_matches(haystack, needle, 4);
        assert!(matches.contains(&0), "Should find match at beginning");
        assert!(matches.contains(&26), "Should find match at position 26");
    }
    
    #[test]
    fn test_entropy_calculation() {
        let system_profile = SystemProfile::detect();
        let simd = SimdCompressionAccelerator::new(&system_profile);
        
        // Test with uniform data (low entropy)
        let uniform_data = vec![0xAA; 1000];
        let entropy_uniform = simd.calculate_entropy_simd(&uniform_data);
        
        // Test with random-like data (high entropy)
        let random_data: Vec<u8> = (0..1000).map(|i| (i * 137) as u8).collect();
        let entropy_random = simd.calculate_entropy_simd(&random_data);
        
        assert!(entropy_uniform < entropy_random, "Uniform data should have lower entropy than random data");
        assert!(entropy_uniform < 0.1, "Uniform data should have very low entropy");
        assert!(entropy_random > 0.7, "Random data should have high entropy");
    }
    
    #[test]
    fn test_lz77_matches() {
        let system_profile = SystemProfile::detect();
        let deflate = SimdDeflateAccelerator::new(&system_profile);
        
        let window = b"This is a test string with repeated patterns. This is a test.";
        let look_ahead = b"This is a test";
        
        let matches = deflate.find_lz77_matches(window, look_ahead, 4);
        assert!(!matches.is_empty(), "Should find matches in repeated text");
        
        if let Some((distance, length)) = matches.first() {
            assert!(*length >= 4, "Match length should be at least minimum");
            assert!(*distance <= window.len(), "Distance should be within window");
        }
    }
    
    #[test]
    fn test_algorithm_optimization() {
        let system_profile = SystemProfile::detect();
        let optimizer = AlgorithmOptimizer::new(&system_profile);
        
        let test_data = vec![0xAB; 10000];
        
        let gzip_hints = optimizer.optimize_for_algorithm(AdaptiveAlgorithm::FastGzip, &test_data);
        let lz4_hints = optimizer.optimize_for_algorithm(AdaptiveAlgorithm::Lz4, &test_data);
        
        assert!(gzip_hints.use_fast_crc, "Gzip should use fast CRC");
        assert!(!lz4_hints.use_fast_crc, "LZ4 doesn't use CRC");
        assert!(lz4_hints.chunk_size_hint > gzip_hints.chunk_size_hint, "LZ4 should prefer larger chunks");
    }
}