//! SIMD-accelerated LZ77 copy operations
//!
//! This module provides ultra-fast memory copy routines optimized for
//! the overlapping copy patterns common in LZ77 decompression.
//!
//! Key optimizations:
//! 1. Inline assembly for x86_64 AVX2 (32-byte copies)
//! 2. NEON intrinsics for ARM64 (16-byte copies)
//! 3. Pattern expansion for small distances (1-8 bytes)
//! 4. Overlapping copy handling without branches

#![allow(dead_code)]

// =============================================================================
// x86_64 AVX2 Implementation
// =============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2 {
    use std::arch::x86_64::*;

    /// Copy 32 bytes using AVX2 (inline assembly version)
    #[inline(always)]
    pub unsafe fn copy_32(src: *const u8, dst: *mut u8) {
        // Use inline assembly for maximum performance
        core::arch::asm!(
            "vmovdqu ymm0, [{src}]",
            "vmovdqu [{dst}], ymm0",
            src = in(reg) src,
            dst = in(reg) dst,
            out("ymm0") _,
            options(nostack, preserves_flags)
        );
    }

    /// Copy 64 bytes using two AVX2 operations
    #[inline(always)]
    pub unsafe fn copy_64(src: *const u8, dst: *mut u8) {
        core::arch::asm!(
            "vmovdqu ymm0, [{src}]",
            "vmovdqu ymm1, [{src} + 32]",
            "vmovdqu [{dst}], ymm0",
            "vmovdqu [{dst} + 32], ymm1",
            src = in(reg) src,
            dst = in(reg) dst,
            out("ymm0") _,
            out("ymm1") _,
            options(nostack, preserves_flags)
        );
    }

    /// Broadcast a single byte to all 32 positions and store
    #[inline(always)]
    pub unsafe fn fill_byte_32(byte: u8, dst: *mut u8) {
        let pattern = _mm256_set1_epi8(byte as i8);
        _mm256_storeu_si256(dst as *mut __m256i, pattern);
    }

    /// Broadcast a 2-byte pattern to all 32 bytes and store
    #[inline(always)]
    pub unsafe fn fill_word_32(word: u16, dst: *mut u8) {
        let pattern = _mm256_set1_epi16(word as i16);
        _mm256_storeu_si256(dst as *mut __m256i, pattern);
    }

    /// Broadcast a 4-byte pattern to all 32 bytes and store  
    #[inline(always)]
    pub unsafe fn fill_dword_32(dword: u32, dst: *mut u8) {
        let pattern = _mm256_set1_epi32(dword as i32);
        _mm256_storeu_si256(dst as *mut __m256i, pattern);
    }

    /// Broadcast an 8-byte pattern to all 32 bytes and store
    #[inline(always)]
    pub unsafe fn fill_qword_32(qword: u64, dst: *mut u8) {
        let pattern = _mm256_set1_epi64x(qword as i64);
        _mm256_storeu_si256(dst as *mut __m256i, pattern);
    }
}

// =============================================================================
// x86_64 SSE2 Fallback
// =============================================================================

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
mod sse2 {
    use std::arch::x86_64::*;

    /// Copy 16 bytes using SSE2
    #[inline(always)]
    pub unsafe fn copy_16(src: *const u8, dst: *mut u8) {
        let data = _mm_loadu_si128(src as *const __m128i);
        _mm_storeu_si128(dst as *mut __m128i, data);
    }

    /// Copy 32 bytes using two SSE2 operations
    #[inline(always)]
    pub unsafe fn copy_32(src: *const u8, dst: *mut u8) {
        copy_16(src, dst);
        copy_16(src.add(16), dst.add(16));
    }
}

// =============================================================================
// ARM64 NEON Implementation
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Copy 16 bytes using NEON
    #[inline(always)]
    pub unsafe fn copy_16(src: *const u8, dst: *mut u8) {
        let data = vld1q_u8(src);
        vst1q_u8(dst, data);
    }

    /// Copy 32 bytes using two NEON operations
    #[inline(always)]
    pub unsafe fn copy_32(src: *const u8, dst: *mut u8) {
        copy_16(src, dst);
        copy_16(src.add(16), dst.add(16));
    }

    /// Copy 64 bytes using four NEON operations
    #[inline(always)]
    pub unsafe fn copy_64(src: *const u8, dst: *mut u8) {
        // Load all 4 vectors first to maximize memory parallelism
        let v0 = vld1q_u8(src);
        let v1 = vld1q_u8(src.add(16));
        let v2 = vld1q_u8(src.add(32));
        let v3 = vld1q_u8(src.add(48));
        // Store all 4 vectors
        vst1q_u8(dst, v0);
        vst1q_u8(dst.add(16), v1);
        vst1q_u8(dst.add(32), v2);
        vst1q_u8(dst.add(48), v3);
    }

    /// Broadcast a single byte to 16 positions (RLE optimization)
    #[inline(always)]
    pub unsafe fn fill_byte_16(byte: u8, dst: *mut u8) {
        let pattern = vdupq_n_u8(byte);
        vst1q_u8(dst, pattern);
    }

    /// Broadcast a single byte to 32 positions
    #[inline(always)]
    pub unsafe fn fill_byte_32(byte: u8, dst: *mut u8) {
        let pattern = vdupq_n_u8(byte);
        vst1q_u8(dst, pattern);
        vst1q_u8(dst.add(16), pattern);
    }

    /// Broadcast a 2-byte pattern to 16 bytes
    #[inline(always)]
    pub unsafe fn fill_word_16(word: u16, dst: *mut u8) {
        let pattern = vdupq_n_u16(word);
        vst1q_u8(dst, vreinterpretq_u8_u16(pattern));
    }

    /// Broadcast a 4-byte pattern to 16 bytes
    #[inline(always)]
    pub unsafe fn fill_dword_16(dword: u32, dst: *mut u8) {
        let pattern = vdupq_n_u32(dword);
        vst1q_u8(dst, vreinterpretq_u8_u32(pattern));
    }

    /// Broadcast an 8-byte pattern to 16 bytes
    #[inline(always)]
    pub unsafe fn fill_qword_16(qword: u64, dst: *mut u8) {
        let pattern = vdupq_n_u64(qword);
        vst1q_u8(dst, vreinterpretq_u8_u64(pattern));
    }

    /// Optimized LZ77 copy for distance >= 16 (non-overlapping)
    /// Copies `len` bytes, may overwrite up to 15 extra bytes
    #[inline(always)]
    pub unsafe fn lz77_copy_nonoverlap(src: *const u8, dst: *mut u8, len: usize) {
        let mut i = 0;
        // Unroll for common cases
        while i + 64 <= len {
            copy_64(src.add(i), dst.add(i));
            i += 64;
        }
        while i + 16 <= len {
            copy_16(src.add(i), dst.add(i));
            i += 16;
        }
        // Handle remainder with final potentially-overlapping copy
        if i < len {
            copy_16(src.add(len - 16), dst.add(len - 16));
        }
    }
}

// =============================================================================
// Portable Fallback
// =============================================================================

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod portable {
    #[inline(always)]
    pub unsafe fn copy_16(src: *const u8, dst: *mut u8) {
        std::ptr::copy_nonoverlapping(src, dst, 16);
    }

    #[inline(always)]
    pub unsafe fn copy_32(src: *const u8, dst: *mut u8) {
        std::ptr::copy_nonoverlapping(src, dst, 32);
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Fast LZ77 copy with pattern expansion and SIMD
///
/// This is the main entry point for LZ77 back-reference copies.
/// It handles all distance/length combinations optimally.
#[inline(always)]
pub fn lz77_copy_fast(output: &mut Vec<u8>, distance: usize, length: usize) {
    // Reserve space
    output.reserve(length);
    let out_pos = output.len();

    unsafe {
        output.set_len(out_pos + length);
        let ptr = output.as_mut_ptr();
        let dst = ptr.add(out_pos);
        let src = ptr.add(out_pos - distance);

        if distance >= 32 {
            // Non-overlapping or large distance: use SIMD copy
            copy_large(src, dst, length);
        } else if distance == 1 {
            // RLE: single byte repeat (very common)
            fill_rle(dst, *src, length);
        } else if distance < 8 {
            // Small pattern: expand and repeat
            copy_small_pattern(src, dst, distance, length);
        } else {
            // Medium distance (8-31): overlapping SIMD
            copy_overlapping(src, dst, distance, length);
        }
    }
}

/// Copy for large distances (>= 32, non-overlapping or large overlap)
#[inline(always)]
unsafe fn copy_large(mut src: *const u8, mut dst: *mut u8, mut length: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // Use 64-byte copies for large transfers
        while length >= 64 {
            avx2::copy_64(src, dst);
            src = src.add(64);
            dst = dst.add(64);
            length -= 64;
        }

        while length >= 32 {
            avx2::copy_32(src, dst);
            src = src.add(32);
            dst = dst.add(32);
            length -= 32;
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        while length >= 32 {
            sse2::copy_32(src, dst);
            src = src.add(32);
            dst = dst.add(32);
            length -= 32;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        while length >= 32 {
            neon::copy_32(src, dst);
            src = src.add(32);
            dst = dst.add(32);
            length -= 32;
        }
    }

    // Handle remainder
    if length > 0 {
        std::ptr::copy_nonoverlapping(src, dst, length);
    }
}

/// Fill with RLE (single byte repeat) - distance = 1
#[inline(always)]
unsafe fn fill_rle(dst: *mut u8, byte: u8, mut length: usize) {
    let mut p = dst;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        while length >= 32 {
            avx2::fill_byte_32(byte, p);
            p = p.add(32);
            length -= 32;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        while length >= 16 {
            neon::fill_byte_16(byte, p);
            p = p.add(16);
            length -= 16;
        }
    }

    // Remainder
    while length > 0 {
        *p = byte;
        p = p.add(1);
        length -= 1;
    }
}

/// Copy small pattern (distance 2-7) with SIMD pattern expansion
/// CRITICAL: Only distances that divide evenly into 8 can use SIMD broadcast
/// (distances 1, 2, 4, 8). Others must use byte-by-byte copy.
#[inline(always)]
unsafe fn copy_small_pattern(src: *const u8, dst: *mut u8, distance: usize, mut length: usize) {
    match distance {
        2 => {
            // Distance 2 divides 8 evenly - can use SIMD
            let a = *src;
            let b = *src.add(1);
            let pattern = u64::from_le_bytes([a, b, a, b, a, b, a, b]);
            let mut p = dst;

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                while length >= 32 {
                    avx2::fill_qword_32(pattern, p);
                    p = p.add(32);
                    length -= 32;
                }
            }

            while length >= 8 {
                (p as *mut u64).write_unaligned(pattern);
                p = p.add(8);
                length -= 8;
            }

            // Remainder
            for i in 0..length {
                *p.add(i) = *src.add(i % 2);
            }
        }
        4 => {
            // Distance 4 divides 8 evenly - can use SIMD
            let a = *src;
            let b = *src.add(1);
            let c = *src.add(2);
            let d = *src.add(3);
            let pattern = u64::from_le_bytes([a, b, c, d, a, b, c, d]);
            let mut p = dst;

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                while length >= 32 {
                    avx2::fill_qword_32(pattern, p);
                    p = p.add(32);
                    length -= 32;
                }
            }

            while length >= 8 {
                (p as *mut u64).write_unaligned(pattern);
                p = p.add(8);
                length -= 8;
            }

            // Remainder
            for i in 0..length {
                *p.add(i) = *src.add(i % 4);
            }
        }
        _ => {
            // Distances 3, 5, 6, 7 don't divide 8 evenly
            // Must use byte-by-byte copy to ensure correctness
            for i in 0..length {
                *dst.add(i) = *src.add(i % distance);
            }
        }
    }
}

/// Copy with medium overlap (distance 8-31)
/// Uses chunked copying to handle overlap safely
#[inline(always)]
unsafe fn copy_overlapping(src: *const u8, dst: *mut u8, distance: usize, mut length: usize) {
    let mut s = src;
    let mut d = dst;

    // For distances 8-15, copy 8 bytes at a time
    // For distances 16-31, copy 16 bytes at a time
    if distance >= 16 {
        // Can safely copy 16 bytes at a time
        #[cfg(target_arch = "x86_64")]
        {
            while length >= 16 {
                let chunk = (s as *const u128).read_unaligned();
                (d as *mut u128).write_unaligned(chunk);
                s = s.add(16);
                d = d.add(16);
                length -= 16;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            while length >= 16 {
                neon::copy_16(s, d);
                s = s.add(16);
                d = d.add(16);
                length -= 16;
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            while length >= 16 {
                std::ptr::copy_nonoverlapping(s, d, 16);
                s = s.add(16);
                d = d.add(16);
                length -= 16;
            }
        }
    } else {
        // distance 8-15: copy 8 bytes at a time
        while length >= 8 {
            let chunk = (s as *const u64).read_unaligned();
            (d as *mut u64).write_unaligned(chunk);
            s = s.add(8);
            d = d.add(8);
            length -= 8;
        }
    }

    // Copy remainder byte by byte
    for i in 0..length {
        *d.add(i) = *s.add(i);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz77_copy_rle() {
        let mut output = vec![b'A'];
        lz77_copy_fast(&mut output, 1, 10);
        assert_eq!(output, b"AAAAAAAAAAA");
    }

    #[test]
    fn test_lz77_copy_pattern() {
        let mut output = b"AB".to_vec();
        lz77_copy_fast(&mut output, 2, 8);
        assert_eq!(output, b"ABABABABAB");
    }

    #[test]
    fn test_lz77_copy_large() {
        let mut output: Vec<u8> = (0..100).collect();
        let expected: Vec<u8> = (0..100).chain(0..100).collect();
        lz77_copy_fast(&mut output, 100, 100);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_lz77_copy_overlapping() {
        let mut output = b"Hello".to_vec();
        lz77_copy_fast(&mut output, 5, 15);
        assert_eq!(output, b"HelloHelloHelloHello");
    }

    #[test]
    fn test_benchmark_copy() {
        // Create a large buffer
        let mut output: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        for _ in 0..10000 {
            output.truncate(1000);
            lz77_copy_fast(&mut output, 100, 900);
        }
        let elapsed = start.elapsed();

        println!("\n=== SIMD Copy Benchmark ===");
        println!("10K copies of 900 bytes: {:?}", elapsed);
        println!("Per copy: {:?}", elapsed / 10000);
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn test_rle_copy() {
        // RLE: distance 1, repeat single byte
        let mut buf = vec![b'A'];
        lz77_copy_fast(&mut buf, 1, 32);
        assert_eq!(buf.len(), 33);
        assert!(buf.iter().all(|&b| b == b'A'), "All bytes should be 'A'");
    }

    #[test]
    fn test_pattern_copy_dist_2() {
        // Distance 2: repeat 2-byte pattern
        let mut buf = vec![b'A', b'B'];
        lz77_copy_fast(&mut buf, 2, 32);
        assert_eq!(buf.len(), 34);
        for (i, &byte) in buf.iter().enumerate() {
            let expected = if i % 2 == 0 { b'A' } else { b'B' };
            assert_eq!(byte, expected, "Pattern mismatch at {}", i);
        }
    }

    #[test]
    fn test_large_copy() {
        // Large distance, large length
        let mut buf: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let original = buf.clone();
        lz77_copy_fast(&mut buf, 500, 500);
        assert_eq!(buf.len(), 1500);
        // First 1000 bytes unchanged
        assert_eq!(&buf[..1000], &original[..]);
        // Next 500 bytes are copy from offset 500
        assert_eq!(&buf[1000..1500], &original[500..1000]);
    }

    #[test]
    fn test_overlapping_copy() {
        // Overlapping: distance < length
        let mut buf = vec![1u8, 2, 3, 4];
        lz77_copy_fast(&mut buf, 4, 12);
        assert_eq!(buf.len(), 16);
        // Should repeat: 1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4
        for (i, &byte) in buf.iter().enumerate() {
            assert_eq!(byte, ((i % 4) + 1) as u8, "Mismatch at {}", i);
        }
    }
}
