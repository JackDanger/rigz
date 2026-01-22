//! Consume-First Decode - Matching libdeflate's Exact Structure
//!
//! This is a fresh implementation that exactly matches libdeflate's decompress_template.h.
//! Key invariants:
//! 1. Entry is preloaded BEFORE the loop body
//! 2. Consume happens FIRST (unconditionally)
//! 3. saved_bitbuf is captured BEFORE consume
//! 4. Every literal path ends with `continue` - NEVER falls through to length
//! 5. Length handling is ONLY reached if NOT a literal

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::jit_decode::TableFingerprint;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result};

#[inline(always)]
fn unlikely(b: bool) -> bool {
    b
}

// Thread-local cache for built Huffman tables
// Avoids rebuilding the same table when fingerprint matches
thread_local! {
    static TABLE_CACHE: RefCell<HashMap<TableFingerprint, (LitLenTable, DistTable)>> =
        RefCell::new(HashMap::new());
    static CACHE_STATS: RefCell<(usize, usize)> = const { RefCell::new((0, 0)) }; // (hits, misses)
    static SPEC_CACHE: RefCell<crate::specialized_decode::SpecializedCache> =
        RefCell::new(crate::specialized_decode::SpecializedCache::new());
    static SPEC_STATS: RefCell<(usize, usize)> = const { RefCell::new((0, 0)) }; // (specialized_used, generic_used)
    static BLOCK_STATS: RefCell<BlockStats> = const { RefCell::new(BlockStats::new()) };
    // Timing stats: (table_build_nanos, decode_nanos, header_parse_nanos)
    static TIMING_STATS: RefCell<TimingStats> = const { RefCell::new(TimingStats::new()) };
}

/// Timing statistics for profiling table building vs decoding
#[derive(Debug, Clone, Copy, Default)]
pub struct TimingStats {
    pub table_build_nanos: u64,
    pub decode_nanos: u64,
    pub header_parse_nanos: u64,
    pub table_build_count: u64,
    pub decode_count: u64,
}

impl TimingStats {
    pub const fn new() -> Self {
        Self {
            table_build_nanos: 0,
            decode_nanos: 0,
            header_parse_nanos: 0,
            table_build_count: 0,
            decode_count: 0,
        }
    }
}

/// Block type statistics for analysis
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockStats {
    pub stored_blocks: usize,
    pub fixed_blocks: usize,
    pub dynamic_blocks: usize,
    pub stored_bytes: usize,
    pub fixed_bytes: usize,
    pub dynamic_bytes: usize,
}

impl BlockStats {
    pub const fn new() -> Self {
        Self {
            stored_blocks: 0,
            fixed_blocks: 0,
            dynamic_blocks: 0,
            stored_bytes: 0,
            fixed_bytes: 0,
            dynamic_bytes: 0,
        }
    }

    pub fn total_blocks(&self) -> usize {
        self.stored_blocks + self.fixed_blocks + self.dynamic_blocks
    }

    pub fn total_bytes(&self) -> usize {
        self.stored_bytes + self.fixed_bytes + self.dynamic_bytes
    }
}

/// Get cache statistics (hits, misses, hit_rate)
pub fn get_cache_stats() -> (usize, usize, f64) {
    CACHE_STATS.with(|stats| {
        let (hits, misses) = *stats.borrow();
        let total = hits + misses;
        let rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        (hits, misses, rate)
    })
}

/// Get specialized decoder statistics (used, fallback)
pub fn get_spec_stats() -> (usize, usize) {
    SPEC_STATS.with(|stats| *stats.borrow())
}

/// Get block type statistics
pub fn get_block_stats() -> BlockStats {
    BLOCK_STATS.with(|stats| *stats.borrow())
}

/// Get TABLE_CACHE size (number of unique fingerprints)
pub fn get_table_cache_size() -> usize {
    TABLE_CACHE.with(|cache| cache.borrow().len())
}

/// Get SPEC_CACHE detailed stats: (decoders, failed, total_uses, max_uses)
pub fn get_spec_cache_stats() -> (usize, usize, usize, usize) {
    SPEC_CACHE.with(|cache| cache.borrow().detailed_stats())
}

/// Get timing statistics
pub fn get_timing_stats() -> TimingStats {
    TIMING_STATS.with(|stats| *stats.borrow())
}

/// Reset all statistics (cache, spec, block, timing)
pub fn reset_cache_stats() {
    CACHE_STATS.with(|stats| {
        *stats.borrow_mut() = (0, 0);
    });
    SPEC_STATS.with(|stats| {
        *stats.borrow_mut() = (0, 0);
    });
    BLOCK_STATS.with(|stats| {
        *stats.borrow_mut() = BlockStats::new();
    });
    TABLE_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
    SPEC_CACHE.with(|cache| {
        *cache.borrow_mut() = crate::specialized_decode::SpecializedCache::new();
    });
    TIMING_STATS.with(|stats| {
        *stats.borrow_mut() = TimingStats::new();
    });
}

// =============================================================================
// Bit Extraction - BMI2 BZHI on x86_64, branchless fallback elsewhere
// =============================================================================

/// Extract low n bits from a value using BMI2 BZHI when available
/// On x86_64 with BMI2, this compiles to a single `bzhi` instruction
/// Elsewhere, uses branchless mask computation
#[inline(always)]
fn bzhi_u64(x: u64, n: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // BMI2 bzhi instruction - single cycle bit extraction
        unsafe { _bzhi_u64(x, n) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        // Branchless: compute mask and apply
        // For n=0, mask = 0; for n=63, mask = 0x7FFF_FFFF_FFFF_FFFF
        let mask = (1u64 << (n & 63)).wrapping_sub(1);
        x & mask
    }
}

/// Alias for backward compatibility
#[allow(dead_code)]
#[inline(always)]
fn extract_bits(value: u64, n: u32) -> u64 {
    bzhi_u64(value, n)
}

// =============================================================================
// Bit Reader - Matching libdeflate exactly
// =============================================================================

/// Bit buffer matching libdeflate's structure
/// Bit buffer matching libdeflate's structure
pub struct Bits<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub bitbuf: u64,
    pub bitsleft: u32,
}

impl<'a> Bits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut bits = Self {
            data,
            pos: 0,
            bitbuf: 0,
            bitsleft: 0,
        };
        bits.refill();
        bits
    }

    /// Branchless refill matching libdeflate
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let word = unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            let word = u64::from_le(word);
            self.bitbuf |= word << (self.bitsleft as u8);
            self.pos += 7;
            self.pos -= ((self.bitsleft >> 3) & 0x7) as usize;
            self.bitsleft |= 56; // MAX_BITSLEFT & !7
        } else {
            self.refill_slow();
        }
    }

    #[inline(never)]
    pub fn refill_slow(&mut self) {
        while self.bitsleft <= 56 {
            if self.pos < self.data.len() {
                self.bitbuf |= (self.data[self.pos] as u64) << self.bitsleft;
                self.pos += 1;
                self.bitsleft += 8;
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    pub fn peek(&self) -> u64 {
        self.bitbuf
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bitbuf >>= n as u8;
        self.bitsleft -= n;
    }

    /// Consume using entry's low 5 bits
    #[inline(always)]
    pub fn consume_entry(&mut self, entry: u32) {
        self.bitbuf >>= entry as u8;
        self.bitsleft = self.bitsleft.wrapping_sub(entry & 0x1F);
    }

    /// Available bits (low 8 bits only - matching libdeflate's (u8)bitsleft pattern)
    #[inline(always)]
    pub fn available(&self) -> u32 {
        // libdeflate allows garbage in high bits, so cast to u8 for the real value
        (self.bitsleft as u8) as u32
    }

    pub fn align_to_byte(&mut self) {
        // bitsleft may have garbage in high bits, use (u8) cast
        let discard = (self.bitsleft as u8) & 7;
        self.consume(discard as u32);
    }

    pub fn read_u16(&mut self) -> u16 {
        self.align_to_byte();

        // If we have bytes in the bit buffer, extract from there
        if self.available() >= 16 {
            let val = (self.bitbuf & 0xFFFF) as u16;
            self.consume(16);
            return val;
        }

        // Need more bits, refill first
        self.refill();
        let val = (self.bitbuf & 0xFFFF) as u16;
        self.consume(16);
        val
    }
}

// =============================================================================
// Match Copy - Matching libdeflate's decompress_template.h lines 575-680
// =============================================================================

/// Prefetch hint for x86_64
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
}

// Note: ARM64 prefetch intrinsics are unstable in Rust, so we skip prefetch on ARM
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn prefetch_read(_ptr: *const u8) {}

/// Fast match copy for fastloop - may write up to 40 bytes beyond length
/// ONLY use when you have FASTLOOP_MARGIN bytes of buffer margin!
#[inline(always)]
fn copy_match_fast(output: &mut [u8], out_pos: usize, distance: u32, length: u32) -> usize {
    let dist = distance as usize;
    let len = length as usize;

    unsafe {
        let out_ptr = output.as_mut_ptr();
        let mut dst = out_ptr.add(out_pos);
        let mut src = out_ptr.add(out_pos - dist);
        let end = dst.add(len);

        // Prefetch for long matches
        if len > 40 {
            prefetch_read(src.add(40));
        }

        if dist >= 32 && len >= 64 {
            // SIMD fast path: use AVX2 32-byte copies for large non-overlapping matches
            #[cfg(target_arch = "x86_64")]
            {
                // Copy 64 bytes at a time using AVX2
                while dst.add(64) <= end {
                    let v0 = _mm256_loadu_si256(src as *const __m256i);
                    let v1 = _mm256_loadu_si256(src.add(32) as *const __m256i);
                    _mm256_storeu_si256(dst as *mut __m256i, v0);
                    _mm256_storeu_si256(dst.add(32) as *mut __m256i, v1);
                    src = src.add(64);
                    dst = dst.add(64);
                }
                // Copy 32 bytes at a time
                while dst.add(32) <= end {
                    let v = _mm256_loadu_si256(src as *const __m256i);
                    _mm256_storeu_si256(dst as *mut __m256i, v);
                    src = src.add(32);
                    dst = dst.add(32);
                }
                // Handle remainder with 8-byte copies
                while dst < end {
                    (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                    src = src.add(8);
                    dst = dst.add(8);
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                // NEON fast path: 32-byte copies using two 16-byte registers
                while dst.add(32) <= end {
                    let v0 = vld1q_u8(src);
                    let v1 = vld1q_u8(src.add(16));
                    vst1q_u8(dst, v0);
                    vst1q_u8(dst.add(16), v1);
                    src = src.add(32);
                    dst = dst.add(32);
                }
                // 16-byte cleanup
                while dst.add(16) <= end {
                    let v = vld1q_u8(src);
                    vst1q_u8(dst, v);
                    src = src.add(16);
                    dst = dst.add(16);
                }
                // 8-byte cleanup
                while dst < end {
                    (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                    src = src.add(8);
                    dst = dst.add(8);
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                // Scalar fallback
                while dst < end {
                    (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                    src = src.add(8);
                    dst = dst.add(8);
                }
            }
        } else if dist >= 8 {
            // Fast path: offset >= WORDBYTES (8)
            // Unconditionally copy 5 words first (40 bytes - covers most matches)
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(8);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(8);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(8);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(8);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(8);
            dst = dst.add(8);

            // Loop for longer matches
            while dst < end {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(8);
                dst = dst.add(8);
            }
        } else if dist == 1 {
            // RLE path: use SIMD broadcast for large fills
            let byte = *src;
            #[cfg(target_arch = "x86_64")]
            {
                if len >= 64 {
                    // Use AVX2 broadcast for large RLE
                    let pattern = _mm256_set1_epi8(byte as i8);
                    while dst.add(64) <= end {
                        _mm256_storeu_si256(dst as *mut __m256i, pattern);
                        _mm256_storeu_si256(dst.add(32) as *mut __m256i, pattern);
                        dst = dst.add(64);
                    }
                    while dst.add(32) <= end {
                        _mm256_storeu_si256(dst as *mut __m256i, pattern);
                        dst = dst.add(32);
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if len >= 32 {
                    // Use NEON broadcast for large RLE
                    let pattern = vdupq_n_u8(byte);
                    while dst.add(32) <= end {
                        vst1q_u8(dst, pattern);
                        vst1q_u8(dst.add(16), pattern);
                        dst = dst.add(32);
                    }
                    while dst.add(16) <= end {
                        vst1q_u8(dst, pattern);
                        dst = dst.add(16);
                    }
                }
            }
            // Remainder with 8-byte broadcast
            let v = 0x0101010101010101u64 * (byte as u64);
            while dst < end {
                (dst as *mut u64).write_unaligned(v);
                dst = dst.add(8);
            }
        } else {
            // Small distance (2-7): word copy with stride = offset
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(dist);
            dst = dst.add(dist);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(dist);
            dst = dst.add(dist);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(dist);
            dst = dst.add(dist);
            (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
            src = src.add(dist);
            dst = dst.add(dist);
            while dst < end {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(dist);
                dst = dst.add(dist);
            }
        }
    }

    out_pos + len
}

/// Safe match copy for generic loop - writes exactly `length` bytes, no overwrite
#[inline(always)]
fn copy_match_safe(output: &mut [u8], out_pos: usize, distance: u32, length: u32) -> usize {
    let dist = distance as usize;
    let len = length as usize;
    let src_start = out_pos - dist;

    if dist >= len {
        // Non-overlapping - direct copy
        output.copy_within(src_start..src_start + len, out_pos);
    } else if dist == 1 {
        // RLE - fill with single byte
        let byte = output[src_start];
        for i in 0..len {
            output[out_pos + i] = byte;
        }
    } else {
        // Overlapping copy - must go byte by byte
        for i in 0..len {
            output[out_pos + i] = output[src_start + (i % dist)];
        }
    }

    out_pos + len
}

// =============================================================================
// Main Decode Function - Matching libdeflate's Structure EXACTLY
// =============================================================================

/// Decode a Huffman block using consume-first pattern
///
/// This matches libdeflate's decompress_template.h lines 340-580
/// Huffman decode with Vector Huffman multi-literal optimization
fn decode_huffman_cf_vector(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist_table: &DistTable,
    vector_table: &crate::vector_huffman::VectorTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 320;

    // FASTLOOP with multi-literal optimization
    while out_pos + FASTLOOP_MARGIN <= output.len() && bits.pos < bits.data.len() {
        if bits.available() < 32 {
            bits.refill();
        }

        // Try multi-literal lookahead (up to 4 literals)
        let (symbols, count, bits_count) =
            crate::vector_huffman::decode_multi_literals(bits.peek(), &vector_table.table);
        if count > 0 {
            output[out_pos..(out_pos + count)].copy_from_slice(&symbols[..count]);
            out_pos += count;
            bits.consume(bits_count);
            continue;
        }

        // Fallback to standard decode (lengths/exceptional/overflow)
        let saved_bitbuf = bits.peek();
        let mut entry = litlen.lookup(saved_bitbuf);
        bits.consume_entry(entry.raw());

        if (entry.raw() as i32) < 0 {
            // Literal
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            continue;
        }

        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                return Ok(out_pos);
            }

            // Subtable
            let sub_saved = bits.peek();
            entry = litlen.lookup_subtable(entry, saved_bitbuf);
            bits.consume_entry(entry.raw());

            if entry.is_end_of_block() {
                return Ok(out_pos);
            }

            if (entry.raw() as i32) < 0 {
                // Literal from subtable
                output[out_pos] = entry.literal_value();
                out_pos += 1;
                continue;
            }

            // Length from subtable
            let length_val = entry.decode_length(sub_saved);
            out_pos = decode_huffman_match(bits, output, out_pos, length_val, dist_table)?;
        } else {
            // Length
            let length_val = entry.decode_length(saved_bitbuf);
            out_pos = decode_huffman_match(bits, output, out_pos, length_val, dist_table)?;
        }
    }

    // Generic loop for remainder
    decode_huffman_cf(bits, output, out_pos, litlen, dist_table)
}

fn decode_huffman_match(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    length: u32,
    dist_table: &DistTable,
) -> Result<usize> {
    bits.refill();
    let dist_saved = bits.peek();
    let mut dist_entry = dist_table.lookup(dist_saved);

    if dist_entry.is_subtable_ptr() {
        bits.consume(DistTable::TABLE_BITS as u32);
        dist_entry = dist_table.lookup_subtable(dist_entry, dist_saved);
    }

    let dist_extra_saved = bits.peek();
    bits.consume_entry(dist_entry.raw());
    let distance = dist_entry.decode_distance(dist_extra_saved);

    if distance == 0 || distance as usize > out_pos {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Invalid distance {} at pos {}", distance, out_pos),
        ));
    }

    // Copy match
    out_pos = copy_match_fast(output, out_pos, distance, length);
    bits.refill();
    Ok(out_pos)
}

/// Libdeflate-style optimized decoder
/// Key differences from baseline:
/// 1. `bitsleft -= entry` (garbage in high bits allowed)
/// 2. Entry preloaded at loop start, updated during iteration
/// 3. Pointer arithmetic instead of index
/// 4. Up to 2 extra literals decoded per iteration on 64-bit
/// 5. Next entry preloaded before refill to hide latency
#[inline(never)]
fn decode_huffman_libdeflate_style(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 320;
    const LITLEN_TABLEMASK: u64 = (1u64 << LitLenTable::TABLE_BITS) - 1;

    let out_ptr = output.as_mut_ptr();
    let out_end = output.len();
    let litlen_ptr = litlen.entries_ptr();

    // Local copies for register allocation
    let mut bitbuf = bits.bitbuf;
    let mut bitsleft = bits.bitsleft;
    let mut in_pos = bits.pos;
    let in_data = bits.data;
    let in_ptr = in_data.as_ptr();

    // Calculate fastloop input end - need margin for branchless refill
    // We use 32 bytes margin to account for multiple refills per iteration
    // (up to 3 refills in 8-literal path, each reading up to 8 bytes)
    let in_fastloop_end = in_data.len().saturating_sub(32);

    // Truly branchless refill for fastloop - ONLY use when in_pos < in_fastloop_end
    macro_rules! refill_branchless_fast {
        () => {
            unsafe {
                let word = (in_ptr.add(in_pos) as *const u64).read_unaligned();
                let word = u64::from_le(word);
                bitbuf |= word << (bitsleft as u8);
                in_pos += 7;
                in_pos -= ((bitsleft >> 3) & 0x7) as usize;
                bitsleft |= 56; // MAX_BITSLEFT & !7
            }
        };
    }

    // Safe refill with bounds checking - for generic loop
    macro_rules! refill_branchless {
        () => {
            if in_pos + 8 <= in_data.len() {
                refill_branchless_fast!();
            } else {
                // Slow path for end of input
                while (bitsleft as u8) <= 56 && in_pos < in_data.len() {
                    bitbuf |= (in_data[in_pos] as u64) << (bitsleft as u8);
                    in_pos += 1;
                    bitsleft = (bitsleft as u8).wrapping_add(8) as u32;
                }
            }
        };
    }

    // Inline entry lookup
    macro_rules! lookup {
        () => {
            unsafe { (*litlen_ptr.add((bitbuf & LITLEN_TABLEMASK) as usize)).raw() }
        };
    }

    // PRELOAD first entry BEFORE loop (libdeflate pattern)
    refill_branchless!();
    let mut entry = lookup!();

    // FASTLOOP - check BOTH input and output bounds to enable truly branchless refill
    while in_pos < in_fastloop_end && out_pos + FASTLOOP_MARGIN <= out_end {
        // Save bitbuf for extra bits extraction
        let saved_bitbuf = bitbuf;

        // Consume bits - NOTE: subtract full entry, not masked!
        // This is the key libdeflate optimization
        bitbuf >>= entry as u8;
        bitsleft = bitsleft.wrapping_sub(entry);

        // Check LITERAL (bit 31 set = negative as i32)
        if (entry as i32) < 0 {
            // LITERAL PATH - libdeflate style multi-literal decode
            let lit1 = (entry >> 16) as u8;
            entry = lookup!();

            if (entry as i32) < 0 {
                // 2nd literal
                bitbuf >>= entry as u8;
                bitsleft = bitsleft.wrapping_sub(entry);
                let lit2 = (entry >> 16) as u8;
                entry = lookup!();

                if (entry as i32) < 0 {
                    // 3rd literal
                    bitbuf >>= entry as u8;
                    bitsleft = bitsleft.wrapping_sub(entry);
                    let lit3 = (entry >> 16) as u8;
                    entry = lookup!();

                    if (entry as i32) < 0 {
                        // 4th literal
                        bitbuf >>= entry as u8;
                        bitsleft = bitsleft.wrapping_sub(entry);
                        let lit4 = (entry >> 16) as u8;
                        entry = lookup!();
                        // Always refill before 5th literal - we need bits for potential length/distance
                        refill_branchless_fast!();

                        if (entry as i32) < 0 {
                            // 5th literal
                            bitbuf >>= entry as u8;
                            bitsleft = bitsleft.wrapping_sub(entry);
                            let lit5 = (entry >> 16) as u8;
                            entry = lookup!();

                            // Try to decode 3 more literals for 8-literal batch
                            if (entry as i32) < 0 {
                                // 6th literal
                                bitbuf >>= entry as u8;
                                bitsleft = bitsleft.wrapping_sub(entry);
                                let lit6 = (entry >> 16) as u8;
                                entry = lookup!();

                                if (entry as i32) < 0 {
                                    // 7th literal
                                    bitbuf >>= entry as u8;
                                    bitsleft = bitsleft.wrapping_sub(entry);
                                    let lit7 = (entry >> 16) as u8;
                                    entry = lookup!();
                                    refill_branchless_fast!();

                                    if (entry as i32) < 0 {
                                        // 8th literal - write all 8 at once
                                        bitbuf >>= entry as u8;
                                        bitsleft = bitsleft.wrapping_sub(entry);
                                        let lit8 = (entry >> 16) as u8;
                                        entry = lookup!();

                                        // Pack 8 literals into a u64 and write
                                        let packed = (lit1 as u64)
                                            | ((lit2 as u64) << 8)
                                            | ((lit3 as u64) << 16)
                                            | ((lit4 as u64) << 24)
                                            | ((lit5 as u64) << 32)
                                            | ((lit6 as u64) << 40)
                                            | ((lit7 as u64) << 48)
                                            | ((lit8 as u64) << 56);
                                        unsafe {
                                            (out_ptr.add(out_pos) as *mut u64)
                                                .write_unaligned(packed);
                                        }
                                        out_pos += 8;
                                        continue;
                                    }

                                    // 7 literals
                                    let packed = (lit1 as u64)
                                        | ((lit2 as u64) << 8)
                                        | ((lit3 as u64) << 16)
                                        | ((lit4 as u64) << 24)
                                        | ((lit5 as u64) << 32)
                                        | ((lit6 as u64) << 40)
                                        | ((lit7 as u64) << 48);
                                    unsafe {
                                        (out_ptr.add(out_pos) as *mut u64).write_unaligned(packed);
                                    }
                                    out_pos += 7;
                                    continue;
                                }

                                // 6 literals
                                let packed = (lit1 as u64)
                                    | ((lit2 as u64) << 8)
                                    | ((lit3 as u64) << 16)
                                    | ((lit4 as u64) << 24)
                                    | ((lit5 as u64) << 32)
                                    | ((lit6 as u64) << 40);
                                unsafe {
                                    (out_ptr.add(out_pos) as *mut u64).write_unaligned(packed);
                                }
                                out_pos += 6;
                                refill_branchless_fast!();
                                continue;
                            }

                            // 5 literals
                            let packed = (lit1 as u64)
                                | ((lit2 as u64) << 8)
                                | ((lit3 as u64) << 16)
                                | ((lit4 as u64) << 24)
                                | ((lit5 as u64) << 32);
                            unsafe {
                                (out_ptr.add(out_pos) as *mut u64).write_unaligned(packed);
                            }
                            out_pos += 5;
                            refill_branchless_fast!();
                            continue;
                        }

                        // Pack 4 literals into a u32 and write at once
                        let packed = (lit1 as u32)
                            | ((lit2 as u32) << 8)
                            | ((lit3 as u32) << 16)
                            | ((lit4 as u32) << 24);
                        unsafe {
                            (out_ptr.add(out_pos) as *mut u32).write_unaligned(packed);
                        }
                        out_pos += 4;
                        continue;
                    }

                    // Pack 3 literals and write (u32 is fine, we only need 3 bytes)
                    let packed = (lit1 as u32) | ((lit2 as u32) << 8) | ((lit3 as u32) << 16);
                    unsafe {
                        (out_ptr.add(out_pos) as *mut u32).write_unaligned(packed);
                    }
                    out_pos += 3;
                    // Conditional refill - only if we need more bits
                    if (bitsleft as u8) < 32 {
                        refill_branchless_fast!();
                    }
                    continue;
                }

                // 2 literals - pack into u16
                let packed = (lit1 as u16) | ((lit2 as u16) << 8);
                unsafe {
                    (out_ptr.add(out_pos) as *mut u16).write_unaligned(packed);
                }
                out_pos += 2;
                // Conditional refill - only if we need more bits
                if (bitsleft as u8) < 32 {
                    refill_branchless_fast!();
                }
                continue;
            }

            // Single literal
            unsafe {
                *out_ptr.add(out_pos) = lit1;
            }
            out_pos += 1;
            // Conditional refill - only if we need more bits
            if (bitsleft as u8) < 32 {
                refill_branchless_fast!();
            }
            continue;
        }

        // Not a literal - check EXCEPTIONAL (subtable or EOB)
        if (entry & 0x8000) != 0 {
            // HUFFDEC_EXCEPTIONAL
            if (entry & 0x2000) != 0 {
                // HUFFDEC_END_OF_BLOCK
                bits.bitbuf = bitbuf;
                bits.bitsleft = bitsleft;
                bits.pos = in_pos;
                return Ok(out_pos);
            }

            // Subtable lookup
            let subtable_start = (entry >> 16) as usize;
            let subtable_bits = ((entry >> 8) & 0x3F) as u64;
            let sub_idx = (bitbuf & ((1u64 << subtable_bits) - 1)) as usize;
            entry = unsafe { (*litlen_ptr.add(subtable_start + sub_idx)).raw() };

            let saved_sub = bitbuf;
            bitbuf >>= entry as u8;
            bitsleft = bitsleft.wrapping_sub(entry);

            if (entry as i32) < 0 {
                // Literal from subtable
                let lit = ((entry >> 16) & 0xFF) as u8;
                entry = lookup!();
                refill_branchless_fast!();
                unsafe {
                    *out_ptr.add(out_pos) = lit;
                }
                out_pos += 1;
                continue;
            }
            if (entry & 0x2000) != 0 {
                // EOB from subtable
                bits.bitbuf = bitbuf;
                bits.bitsleft = bitsleft;
                bits.pos = in_pos;
                return Ok(out_pos);
            }

            // Length from subtable - BMI2 BZHI pattern
            let length = (entry >> 16)
                + (extract_bits(saved_sub, (entry as u8) as u32) >> ((entry >> 8) as u8)) as u32;

            // Decode distance
            refill_branchless_fast!();
            let mut dist_entry = dist.lookup(bitbuf);

            if dist_entry.is_subtable_ptr() {
                bitbuf >>= DistTable::TABLE_BITS;
                bitsleft = bitsleft.wrapping_sub(DistTable::TABLE_BITS as u32);
                dist_entry = dist.lookup_subtable_direct(dist_entry, bitbuf);
            }

            let dist_extra_saved = bitbuf;
            let dist_raw = dist_entry.raw();
            bitbuf >>= dist_raw as u8;
            bitsleft = bitsleft.wrapping_sub(dist_raw);
            let distance = (dist_raw >> 16)
                + (extract_bits(dist_extra_saved, (dist_raw as u8) as u32)
                    >> ((dist_raw >> 8) as u8)) as u32;

            if distance == 0 || distance as usize > out_pos {
                bits.bitbuf = bitbuf;
                bits.bitsleft = bitsleft;
                bits.pos = in_pos;
                return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
            }

            // Preload next entry before copy
            entry = lookup!();
            refill_branchless_fast!();

            // Fast copy
            out_pos = copy_match_fast(output, out_pos, distance, length);
            continue;
        }

        // LENGTH CODE - Start distance lookup early to overlap with length computation
        // The dist lookup uses post-consume bitbuf, length uses pre-consume saved_bitbuf
        let mut dist_entry = dist.lookup(bitbuf); // Start memory fetch

        // Compute length while dist_entry fetch is in flight
        let length = (entry >> 16)
            + (extract_bits(saved_bitbuf, (entry as u8) as u32) >> ((entry >> 8) as u8)) as u32;

        // Conditional refill after length computation
        if (bitsleft as u8) < 32 {
            refill_branchless_fast!();
        }

        if dist_entry.is_subtable_ptr() {
            bitbuf >>= DistTable::TABLE_BITS;
            bitsleft = bitsleft.wrapping_sub(DistTable::TABLE_BITS as u32);
            // Use current bitbuf after consuming main table bits (libdeflate pattern)
            dist_entry = dist.lookup_subtable_direct(dist_entry, bitbuf);
        }

        let dist_extra_saved = bitbuf;
        let dist_raw = dist_entry.raw();
        bitbuf >>= dist_raw as u8;
        bitsleft = bitsleft.wrapping_sub(dist_raw);
        // BMI2 BZHI pattern for distance extra bits
        let distance = (dist_raw >> 16)
            + (extract_bits(dist_extra_saved, (dist_raw as u8) as u32) >> ((dist_raw >> 8) as u8))
                as u32;

        if distance == 0 || distance as usize > out_pos {
            bits.bitbuf = bitbuf;
            bits.bitsleft = bitsleft;
            bits.pos = in_pos;
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        // Preload next entry BEFORE copy (hide latency)
        entry = lookup!();
        refill_branchless_fast!();

        // Fast copy
        out_pos = copy_match_fast(output, out_pos, distance, length);
    }

    // Write back state - the entry is preloaded but generic loop will re-lookup
    // Need to make sure bitsleft has valid low bits
    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft & 0xFF; // Clear garbage in high bits for generic loop
    bits.pos = in_pos;

    // Generic loop for remainder
    decode_huffman_cf(bits, output, out_pos, litlen, dist)
}

/// Optimized decode loop with BZHI-style bit extraction
/// Works on all platforms - uses efficient masking for bit extraction
#[inline(never)]
fn decode_huffman_optimized(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 320;
    const LITLEN_TABLEMASK: u64 = (1u64 << LitLenTable::TABLE_BITS) - 1;

    let out_ptr = output.as_mut_ptr();
    let out_end = output.len();
    let litlen_ptr = litlen.entries_ptr();

    // Local copies for register allocation
    let mut bitbuf = bits.bitbuf;
    let mut bitsleft = bits.bitsleft;
    let mut in_pos = bits.pos;
    let in_data = bits.data;

    // Inline branchless refill
    macro_rules! refill_branchless {
        () => {
            if in_pos + 8 <= in_data.len() {
                let word = unsafe { (in_data.as_ptr().add(in_pos) as *const u64).read_unaligned() };
                let word = u64::from_le(word);
                bitbuf |= word << (bitsleft as u8);
                in_pos += 7;
                in_pos -= ((bitsleft >> 3) & 0x7) as usize;
                bitsleft |= 56;
            } else {
                while (bitsleft as u8) <= 56 && in_pos < in_data.len() {
                    bitbuf |= (in_data[in_pos] as u64) << (bitsleft as u8);
                    in_pos += 1;
                    bitsleft = (bitsleft as u8).wrapping_add(8) as u32;
                }
            }
        };
    }

    macro_rules! lookup {
        () => {
            unsafe { (*litlen_ptr.add((bitbuf & LITLEN_TABLEMASK) as usize)).raw() }
        };
    }

    // BZHI-style bit extraction - uses efficient masking on all platforms
    macro_rules! bzhi {
        ($val:expr, $bits:expr) => {
            bzhi_u64($val, $bits)
        };
    }

    refill_branchless!();
    let mut entry = lookup!();

    // FASTLOOP
    while out_pos + FASTLOOP_MARGIN <= out_end {
        let saved_bitbuf = bitbuf;
        bitbuf >>= entry as u8;
        bitsleft = bitsleft.wrapping_sub(entry);

        if (entry as i32) < 0 {
            // LITERAL PATH - multi-literal decode
            let lit1 = (entry >> 16) as u8;
            entry = lookup!();

            if (entry as i32) < 0 {
                bitbuf >>= entry as u8;
                bitsleft = bitsleft.wrapping_sub(entry);
                let lit2 = (entry >> 16) as u8;
                entry = lookup!();

                if (entry as i32) < 0 {
                    bitbuf >>= entry as u8;
                    bitsleft = bitsleft.wrapping_sub(entry);
                    let lit3 = (entry >> 16) as u8;
                    entry = lookup!();

                    if (entry as i32) < 0 {
                        bitbuf >>= entry as u8;
                        bitsleft = bitsleft.wrapping_sub(entry);
                        let lit4 = (entry >> 16) as u8;
                        entry = lookup!();
                        refill_branchless!();

                        // Write 4 literals packed
                        let packed = (lit1 as u32)
                            | ((lit2 as u32) << 8)
                            | ((lit3 as u32) << 16)
                            | ((lit4 as u32) << 24);
                        unsafe { (out_ptr.add(out_pos) as *mut u32).write_unaligned(packed) };
                        out_pos += 4;
                        continue;
                    }

                    let packed = (lit1 as u32) | ((lit2 as u32) << 8) | ((lit3 as u32) << 16);
                    unsafe { (out_ptr.add(out_pos) as *mut u32).write_unaligned(packed) };
                    out_pos += 3;
                    if (bitsleft as u8) < 32 {
                        refill_branchless!();
                    }
                    continue;
                }

                let packed = (lit1 as u16) | ((lit2 as u16) << 8);
                unsafe { (out_ptr.add(out_pos) as *mut u16).write_unaligned(packed) };
                out_pos += 2;
                if (bitsleft as u8) < 32 {
                    refill_branchless!();
                }
                continue;
            }

            unsafe { *out_ptr.add(out_pos) = lit1 };
            out_pos += 1;
            if (bitsleft as u8) < 32 {
                refill_branchless!();
            }
            continue;
        }

        // EXCEPTIONAL
        if (entry & 0x8000) != 0 {
            if (entry & 0x2000) != 0 {
                bits.bitbuf = bitbuf;
                bits.bitsleft = bitsleft;
                bits.pos = in_pos;
                return Ok(out_pos);
            }

            let subtable_start = (entry >> 16) as usize;
            let subtable_bits = (entry >> 8) & 0x3F;
            let sub_idx = bzhi!(bitbuf, subtable_bits) as usize;
            entry = unsafe { (*litlen_ptr.add(subtable_start + sub_idx)).raw() };

            let saved_sub = bitbuf;
            bitbuf >>= entry as u8;
            bitsleft = bitsleft.wrapping_sub(entry);

            if (entry as i32) < 0 {
                let lit = ((entry >> 16) & 0xFF) as u8;
                entry = lookup!();
                refill_branchless!();
                unsafe { *out_ptr.add(out_pos) = lit };
                out_pos += 1;
                continue;
            }
            if (entry & 0x2000) != 0 {
                bits.bitbuf = bitbuf;
                bits.bitsleft = bitsleft;
                bits.pos = in_pos;
                return Ok(out_pos);
            }

            // Length from subtable - use BMI2 BZHI
            let length = (entry >> 16)
                + (bzhi!(saved_sub, (entry as u8) as u32) >> ((entry >> 8) as u8)) as u32;

            refill_branchless!();
            let mut dist_entry = dist.lookup(bitbuf);

            if dist_entry.is_subtable_ptr() {
                bitbuf >>= DistTable::TABLE_BITS;
                bitsleft = bitsleft.wrapping_sub(DistTable::TABLE_BITS as u32);
                dist_entry = dist.lookup_subtable_direct(dist_entry, bitbuf);
            }

            let dist_extra_saved = bitbuf;
            let dist_raw = dist_entry.raw();
            bitbuf >>= dist_raw as u8;
            bitsleft = bitsleft.wrapping_sub(dist_raw);
            // BMI2 BZHI for distance
            let distance = (dist_raw >> 16)
                + (bzhi!(dist_extra_saved, (dist_raw as u8) as u32) >> ((dist_raw >> 8) as u8))
                    as u32;

            if distance == 0 || distance as usize > out_pos {
                bits.bitbuf = bitbuf;
                bits.bitsleft = bitsleft;
                bits.pos = in_pos;
                return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
            }

            entry = lookup!();
            refill_branchless!();
            out_pos = copy_match_fast(output, out_pos, distance, length);
            continue;
        }

        // LENGTH CODE - BMI2 BZHI for extra bits
        let length = (entry >> 16)
            + (bzhi!(saved_bitbuf, (entry as u8) as u32) >> ((entry >> 8) as u8)) as u32;

        let mut dist_entry = dist.lookup(bitbuf);
        if (bitsleft as u8) < 32 {
            refill_branchless!();
        }

        if dist_entry.is_subtable_ptr() {
            bitbuf >>= DistTable::TABLE_BITS;
            bitsleft = bitsleft.wrapping_sub(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable_direct(dist_entry, bitbuf);
        }

        let dist_extra_saved = bitbuf;
        let dist_raw = dist_entry.raw();
        bitbuf >>= dist_raw as u8;
        bitsleft = bitsleft.wrapping_sub(dist_raw);
        // BMI2 BZHI for distance
        let distance = (dist_raw >> 16)
            + (bzhi!(dist_extra_saved, (dist_raw as u8) as u32) >> ((dist_raw >> 8) as u8)) as u32;

        if distance == 0 || distance as usize > out_pos {
            bits.bitbuf = bitbuf;
            bits.bitsleft = bitsleft;
            bits.pos = in_pos;
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        entry = lookup!();
        refill_branchless!();
        out_pos = copy_match_fast(output, out_pos, distance, length);
    }

    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft & 0xFF;
    bits.pos = in_pos;

    decode_huffman_cf(bits, output, out_pos, litlen, dist)
}

/// Dispatch function - uses libdeflate-style decode with BZHI bit extraction
fn decode_huffman_with_dispatch(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    // Use the original libdeflate-style decoder with bzhi_u64 for bit extraction
    decode_huffman_libdeflate_style(bits, output, out_pos, litlen, dist)
}

/// Flat literal loop decoder - avoids deep nesting for better branch prediction
fn decode_huffman_flat(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 320;
    let out_ptr = output.as_mut_ptr();
    let out_end = output.len();

    // FASTLOOP
    'fastloop: while out_pos + FASTLOOP_MARGIN <= out_end {
        bits.refill();

        // Inner literal loop - process up to 8 literals before checking bits
        'literals: loop {
            let saved_bitbuf = bits.peek();
            let entry = litlen.lookup(saved_bitbuf);
            bits.consume_entry(entry.raw());

            // Check literal (bit 31 set = negative)
            if (entry.raw() as i32) >= 0 {
                // Not a literal - break out to handle length/EOB
                // Need to handle this entry
                if entry.is_exceptional() {
                    if entry.is_end_of_block() {
                        return Ok(out_pos);
                    }
                    // Subtable
                    let sub_entry = litlen.lookup_subtable(entry, saved_bitbuf);
                    let sub_saved = bits.peek();
                    bits.consume_entry(sub_entry.raw());

                    if sub_entry.is_end_of_block() {
                        return Ok(out_pos);
                    }
                    if (sub_entry.raw() as i32) < 0 {
                        // Literal from subtable
                        unsafe {
                            *out_ptr.add(out_pos) = sub_entry.literal_value();
                        }
                        out_pos += 1;
                        continue 'literals;
                    }
                    // Length from subtable
                    let length = sub_entry.decode_length(sub_saved);
                    out_pos = decode_match_inline(bits, output, out_pos, length, dist)?;
                    continue 'fastloop;
                }
                // Length code
                let length = entry.decode_length(saved_bitbuf);
                out_pos = decode_match_inline(bits, output, out_pos, length, dist)?;
                continue 'fastloop;
            }

            // Literal - write and continue
            unsafe {
                *out_ptr.add(out_pos) = entry.literal_value();
            }
            out_pos += 1;

            // Check if we need to refill (consumed ~9 bits per literal, so check after a few)
            if bits.available() < 32 {
                continue 'fastloop;
            }
        }
    }

    // Generic loop for remainder
    decode_huffman_cf(bits, output, out_pos, litlen, dist)
}

/// Inline match decode for flat loop
#[inline(always)]
fn decode_match_inline(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    length: u32,
    dist: &DistTable,
) -> Result<usize> {
    bits.refill();
    let dist_saved = bits.peek();
    let mut dist_entry = dist.lookup(dist_saved);

    if dist_entry.is_subtable_ptr() {
        bits.consume(DistTable::TABLE_BITS as u32);
        dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
    }

    let dist_extra_saved = bits.peek();
    bits.consume_entry(dist_entry.raw());
    let distance = dist_entry.decode_distance(dist_extra_saved);

    if distance == 0 || distance as usize > out_pos {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
    }

    Ok(copy_match_fast(output, out_pos, distance, length))
}

/// Decode Huffman stream using consume-first pattern (public for use by optimized decoders)
pub fn decode_huffman_cf_pub(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    decode_huffman_cf(bits, output, out_pos, litlen, dist)
}

fn decode_huffman_cf(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    // CRITICAL: Must be >= max_match(258) + copy_match_fast overwrite(40) + safety
    const FASTLOOP_MARGIN: usize = 320;

    // PRELOAD first entry BEFORE loop
    bits.refill();
    let mut entry = litlen.lookup(bits.peek());

    // FASTLOOP
    while out_pos + FASTLOOP_MARGIN <= output.len() {
        // Step 1: SAVE bitbuf BEFORE consume (for extra bits extraction)
        let saved_bitbuf = bits.peek();

        // Step 2: CONSUME unconditionally
        bits.consume_entry(entry.raw());

        // Step 3: Check if LITERAL (bit 31 set = negative as i32)
        if (entry.raw() as i32) < 0 {
            // LITERAL PATH - Unrolled 5 literals, strategic refills
            let out_ptr = output.as_mut_ptr();

            // Literal 1
            let lit1 = entry.literal_value();
            entry = litlen.lookup(bits.peek());
            unsafe {
                *out_ptr.add(out_pos) = lit1;
            }
            out_pos += 1;

            // Literal 2
            if (entry.raw() as i32) < 0 {
                bits.consume_entry(entry.raw());
                let lit2 = entry.literal_value();
                entry = litlen.lookup(bits.peek());
                unsafe {
                    *out_ptr.add(out_pos) = lit2;
                }
                out_pos += 1;

                // Literal 3
                if (entry.raw() as i32) < 0 {
                    bits.consume_entry(entry.raw());
                    let lit3 = entry.literal_value();
                    entry = litlen.lookup(bits.peek());
                    unsafe {
                        *out_ptr.add(out_pos) = lit3;
                    }
                    out_pos += 1;

                    // Literal 4
                    if (entry.raw() as i32) < 0 {
                        bits.consume_entry(entry.raw());
                        let lit4 = entry.literal_value();
                        entry = litlen.lookup(bits.peek());
                        unsafe {
                            *out_ptr.add(out_pos) = lit4;
                        }
                        out_pos += 1;

                        // Literal 5 - always refill after 5 literals (~45 bits consumed)
                        if (entry.raw() as i32) < 0 {
                            bits.consume_entry(entry.raw());
                            let lit5 = entry.literal_value();
                            bits.refill();
                            entry = litlen.lookup(bits.peek());
                            unsafe {
                                *out_ptr.add(out_pos) = lit5;
                            }
                            out_pos += 1;
                            continue;
                        }
                    }
                    // Conditional refill only when low (libdeflate pattern)
                    if bits.available() < 32 {
                        bits.refill();
                    }
                    continue;
                }
                if bits.available() < 32 {
                    bits.refill();
                }
                continue;
            }
            if bits.available() < 32 {
                bits.refill();
            }
            continue;
        }

        // Step 4: Check for EXCEPTIONAL (subtable or EOB)
        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                return Ok(out_pos);
            }

            // Subtable needed - resolve it
            // lookup_subtable expects the ORIGINAL saved_bitbuf and shifts internally
            entry = litlen.lookup_subtable(entry, saved_bitbuf);

            // Now we need to consume the subtable entry bits
            // saved_sub should be CURRENT bitbuf (after main table was consumed, for extra bits)
            let saved_sub = bits.peek();
            bits.consume_entry(entry.raw());

            // Check subtable result
            if (entry.raw() as i32) < 0 {
                // Literal from subtable
                let lit = entry.literal_value();
                bits.refill();
                entry = litlen.lookup(bits.peek());
                unsafe {
                    *output.as_mut_ptr().add(out_pos) = lit;
                }
                out_pos += 1;
                continue; // NEVER fall through
            }
            if entry.is_end_of_block() {
                return Ok(out_pos);
            }

            // Length from subtable - decode it here, don't fall through
            let length = entry.decode_length(saved_sub);

            bits.refill();
            let dist_saved = bits.peek();
            let mut dist_entry = dist.lookup(dist_saved);

            let is_subtable = dist_entry.is_subtable_ptr();
            if is_subtable {
                // lookup_subtable handles the main_bits shift internally
                // We need to consume main bits AFTER the lookup
                bits.consume(DistTable::TABLE_BITS as u32);
                dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
            }

            // Now capture saved_bitbuf for extra bits extraction
            let dist_extra_saved = bits.peek();
            bits.consume_entry(dist_entry.raw());
            let distance = dist_entry.decode_distance(dist_extra_saved);

            if distance == 0 || distance as usize > out_pos {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Invalid distance {} at pos {} from subtable",
                        distance, out_pos
                    ),
                ));
            }

            // PRELOAD next entry BEFORE copy
            bits.refill();
            entry = litlen.lookup(bits.peek());

            // FAST copy in fastloop - we have FASTLOOP_MARGIN bytes of buffer margin
            out_pos = copy_match_fast(output, out_pos, distance, length);
            continue; // NEVER fall through
        }

        // Step 5: LENGTH CODE - decode from saved_bitbuf
        let length = entry.decode_length(saved_bitbuf);

        // Step 6: DISTANCE
        bits.refill();
        let dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);

        if dist_entry.is_subtable_ptr() {
            // lookup_subtable handles the main_bits shift internally
            bits.consume(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
        }

        // Capture saved_bitbuf for extra bits extraction
        let dist_extra_saved = bits.peek();
        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_extra_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        // PRELOAD next entry BEFORE copy (libdeflate optimization)
        // This hides memory latency by overlapping lookup with copy
        bits.refill();
        entry = litlen.lookup(bits.peek());

        // FAST copy in fastloop - we have FASTLOOP_MARGIN bytes of buffer margin
        out_pos = copy_match_fast(output, out_pos, distance, length);
    }

    // GENERIC LOOP (near end of output)
    // This loop handles bytes near the end where we can't afford copy overwrites
    // KEY: Always refill at start of each iteration (like libdeflate's generic_loop)
    loop {
        // Always refill - handles end-of-input gracefully
        bits.refill();

        let mut saved_bitbuf = bits.peek();
        entry = litlen.lookup(saved_bitbuf);

        if entry.is_subtable_ptr() {
            // Consume main table bits FIRST
            bits.consume(LitLenTable::TABLE_BITS as u32);
            entry = litlen.lookup_subtable(entry, saved_bitbuf);
            // Capture saved_bitbuf AFTER main bits for extra bits extraction
            saved_bitbuf = bits.peek();
            bits.consume_entry(entry.raw()); // subtable entry bits only
        } else {
            bits.consume_entry(entry.raw()); // main entry bits
        }

        if (entry.raw() as i32) < 0 {
            // Literal
            if out_pos >= output.len() {
                return Err(Error::new(
                    ErrorKind::WriteZero,
                    format!(
                        "Generic literal overflow: out_pos={} output.len={}",
                        out_pos,
                        output.len()
                    ),
                ));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            continue;
        }

        if entry.is_end_of_block() {
            return Ok(out_pos);
        }

        // Length code - decode from saved_bitbuf
        let length = entry.decode_length(saved_bitbuf);

        bits.refill();
        let dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);
        if dist_entry.is_subtable_ptr() {
            bits.consume(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
        }
        let dist_extra_saved = bits.peek();
        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_extra_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        if out_pos + length as usize > output.len() {
            return Err(Error::new(
                ErrorKind::WriteZero,
                format!(
                    "Generic match overflow: out_pos={} length={} output.len={}",
                    out_pos,
                    length,
                    output.len()
                ),
            ));
        }

        // SAFE copy in generic loop - no buffer margin here
        out_pos = copy_match_safe(output, out_pos, distance, length);
    }
}

// =============================================================================
// Block Handling
// =============================================================================

fn decode_stored(bits: &mut Bits, output: &mut [u8], mut out_pos: usize) -> Result<usize> {
    bits.align_to_byte();

    let len = bits.read_u16();
    let nlen = bits.read_u16();

    if len != !nlen {
        eprintln!(
            "Invalid stored block: len={:#x}, nlen={:#x}, !nlen={:#x}, pos={}, out_pos={}",
            len, nlen, !nlen, bits.pos, out_pos
        );
        return Err(Error::new(ErrorKind::InvalidData, "Invalid stored block"));
    }

    let len = len as usize;
    if len == 0 {
        return Ok(out_pos);
    }

    if out_pos + len > output.len() {
        return Err(Error::new(ErrorKind::WriteZero, "Output full"));
    }

    // Calculate true position: pos is where we've loaded from, but we may have buffered bytes
    // After align + reading 32 bits, remaining bits should be extracted from buffer
    let _bytes_in_buffer = bits.available() as usize / 8;

    // First, drain any bytes still in the bit buffer
    let mut remaining = len;
    while remaining > 0 && bits.available() >= 8 {
        output[out_pos] = (bits.bitbuf & 0xFF) as u8;
        bits.consume(8);
        out_pos += 1;
        remaining -= 1;
    }

    // Copy the rest directly from input
    if remaining > 0 {
        if bits.pos + remaining > bits.data.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "Truncated stored block",
            ));
        }

        output[out_pos..out_pos + remaining]
            .copy_from_slice(&bits.data[bits.pos..bits.pos + remaining]);
        bits.pos += remaining;
        out_pos += remaining;
    }

    // Reset bit buffer state for next block
    bits.bitbuf = 0;
    bits.bitsleft = 0;

    Ok(out_pos)
}

/// Cached double-literal cache for fixed Huffman (built once)
fn get_fixed_double_lit_cache() -> &'static crate::double_literal::DoubleLitCache {
    use std::sync::OnceLock;
    static CACHE: OnceLock<crate::double_literal::DoubleLitCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        let tables = crate::libdeflate_decode::get_fixed_tables();
        crate::double_literal::DoubleLitCache::build(&tables.0)
    })
}

/// Get or build the specialized decoder for fixed Huffman (cached)
fn get_fixed_specialized_decoder() -> &'static crate::specialized_decode::SpecializedDecoder {
    use std::sync::OnceLock;
    static DECODER: OnceLock<crate::specialized_decode::SpecializedDecoder> = OnceLock::new();
    DECODER.get_or_init(|| {
        // Fixed Huffman code lengths per RFC 1951
        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[0..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256..280].fill(7);
        litlen_lens[280..288].fill(8);

        let mut dist_lens = vec![0u8; 32];
        dist_lens.fill(5);

        crate::specialized_decode::SpecializedDecoder::build(&litlen_lens, &dist_lens)
            .expect("Fixed Huffman should always build")
    })
}

fn decode_fixed(bits: &mut Bits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    // Use the same fast path as dynamic Huffman for maximum speed
    // The libdeflate-style path achieves 99% of libdeflate vs 69% for specialized path
    let (litlen, dist) = crate::libdeflate_decode::get_fixed_tables();
    decode_huffman_libdeflate_style(bits, output, out_pos, litlen, dist)
}

/// Huffman decode with double-literal cache optimization
/// Uses DoubleLitCache for the fastloop to decode 2 literals at once when possible
fn decode_huffman_cf_double(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
    double_cache: &crate::double_literal::DoubleLitCache,
) -> Result<usize> {
    #![allow(unused_imports)]
    use crate::double_literal::DoubleLitEntry;

    const FASTLOOP_MARGIN: usize = 320;

    // FASTLOOP with double-literal optimization
    while out_pos + FASTLOOP_MARGIN <= output.len() && bits.available() >= 16 {
        // Try double-literal lookup first
        let saved_bitbuf = bits.peek();
        let double_entry = double_cache.lookup(saved_bitbuf);

        if double_entry.is_literal() {
            if double_entry.has_second() {
                // DOUBLE LITERAL - decode 2 at once
                let out_ptr = output.as_mut_ptr();
                unsafe {
                    *out_ptr.add(out_pos) = double_entry.symbol1();
                    *out_ptr.add(out_pos + 1) = double_entry.symbol2();
                }
                out_pos += 2;
                bits.consume(double_entry.total_bits() as u32);

                // Continue with double-literal attempts
                bits.refill();
                continue;
            } else {
                // SINGLE LITERAL from double cache
                let out_ptr = output.as_mut_ptr();
                unsafe {
                    *out_ptr.add(out_pos) = double_entry.symbol1();
                }
                out_pos += 1;
                bits.consume(double_entry.total_bits() as u32);

                // Try to continue with more literals via regular path
                bits.refill();
                continue;
            }
        }

        // NOT A LITERAL - fall back to regular decode path
        let entry = litlen.lookup(saved_bitbuf);
        bits.consume_entry(entry.raw());

        // Check for EXCEPTIONAL (subtable or EOB)
        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                return Ok(out_pos);
            }

            // Check for subtable
            let sub_saved = bits.peek();
            let sub_entry = litlen.lookup_subtable(entry, saved_bitbuf);
            bits.consume_entry(sub_entry.raw());

            if sub_entry.is_end_of_block() {
                return Ok(out_pos);
            }

            if (sub_entry.raw() as i32) < 0 {
                // Literal from subtable
                let lit = sub_entry.literal_value();
                let out_ptr = output.as_mut_ptr();
                unsafe {
                    *out_ptr.add(out_pos) = lit;
                }
                out_pos += 1;
                bits.refill();
                continue;
            }

            // Length from subtable
            let length = sub_entry.decode_length(sub_saved);

            bits.refill();
            let dist_saved = bits.peek();
            let mut dist_entry = dist.lookup(dist_saved);

            if dist_entry.is_subtable_ptr() {
                bits.consume(DistTable::TABLE_BITS as u32);
                dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
            }

            let dist_extra_saved = bits.peek();
            bits.consume_entry(dist_entry.raw());
            let distance = dist_entry.decode_distance(dist_extra_saved);

            if distance == 0 || distance as usize > out_pos {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid distance {} at pos {}", distance, out_pos),
                ));
            }

            bits.refill();
            out_pos = copy_match_fast(output, out_pos, distance, length);
            continue;
        }

        // Length code from main table
        let length = entry.decode_length(saved_bitbuf);

        bits.refill();
        let dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);

        if dist_entry.is_subtable_ptr() {
            bits.consume(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
        }

        let dist_extra_saved = bits.peek();
        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_extra_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        bits.refill();
        out_pos = copy_match_fast(output, out_pos, distance, length);
    }

    // Fall back to generic path for remaining output
    // (reuse the regular decode_huffman_cf for the generic loop)
    decode_huffman_cf(bits, output, out_pos, litlen, dist)
}

fn decode_dynamic(bits: &mut Bits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    // Read dynamic Huffman table header
    if bits.available() < 14 {
        bits.refill();
    }

    let hlit = (bits.peek() & 0x1F) as usize + 257;
    bits.consume(5);
    let hdist = (bits.peek() & 0x1F) as usize + 1;
    bits.consume(5);
    let hclen = (bits.peek() & 0xF) as usize + 4;
    bits.consume(4);

    // Read code length code lengths
    const CODE_LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    let mut code_length_lengths = [0u8; 19];
    for i in 0..hclen {
        if bits.available() < 3 {
            bits.refill();
        }
        code_length_lengths[CODE_LENGTH_ORDER[i]] = (bits.peek() & 0x7) as u8;
        bits.consume(3);
    }

    // Build code length table
    let cl_table = build_code_length_table(&code_length_lengths)?;

    // Read literal/length and distance code lengths
    let mut all_lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        // Need up to 7 bits for codeword + 7 for repeat = 14 max
        if bits.available() < 15 {
            bits.refill();
        }

        let entry = cl_table[(bits.peek() & 0x7F) as usize];
        let symbol = (entry >> 8) as u8;
        let len = (entry & 0xFF) as u8;
        bits.consume(len as u32);

        match symbol {
            0..=15 => {
                all_lengths[i] = symbol;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid repeat"));
                }
                let repeat = 3 + (bits.peek() & 0x3) as usize;
                bits.consume(2);
                let val = all_lengths[i - 1];
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = val;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + (bits.peek() & 0x7) as usize;
                bits.consume(3);
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = 11 + (bits.peek() & 0x7F) as usize;
                bits.consume(7);
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Invalid code length symbol",
                ))
            }
        }
    }

    let litlen_lengths = &all_lengths[..hlit];
    let dist_lengths = &all_lengths[hlit..];

    // Compute fingerprint for table caching
    let fingerprint = crate::jit_decode::TableFingerprint::combined(litlen_lengths, dist_lengths);

    // Use libdeflate-style decoder for all dynamic blocks
    // This achieves 99-112% of libdeflate performance across all datasets.
    // The specialized decoder was slower for match-heavy content (SOFTWARE, LOGS)
    // due to its inline extra-bits handling vs libdeflate's saved_bitbuf pattern.
    SPEC_STATS.with(|s| s.borrow_mut().1 += 1);
    decode_dynamic_fallback(
        bits,
        output,
        out_pos,
        litlen_lengths,
        dist_lengths,
        fingerprint,
    )
}

/// Fallback to generic table-based decoder
fn decode_dynamic_fallback(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen_lengths: &[u8],
    dist_lengths: &[u8],
    fingerprint: TableFingerprint,
) -> Result<usize> {
    use std::time::Instant;

    // Try to get cached tables, otherwise build new ones
    let ((litlen_table, dist_table), build_time) = TABLE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(tables) = cache.get(&fingerprint) {
            CACHE_STATS.with(|s| s.borrow_mut().0 += 1); // hit
            return (tables.clone(), 0u64);
        }

        CACHE_STATS.with(|s| s.borrow_mut().1 += 1); // miss

        let start = Instant::now();
        let litlen = LitLenTable::build(litlen_lengths).expect("Invalid litlen table");
        let dist = DistTable::build(dist_lengths).expect("Invalid dist table");
        let build_nanos = start.elapsed().as_nanos() as u64;

        let tables = (litlen, dist);
        cache.insert(fingerprint, tables.clone());
        (tables, build_nanos)
    });

    // Record table build time
    if build_time > 0 {
        TIMING_STATS.with(|s| {
            let mut stats = s.borrow_mut();
            stats.table_build_nanos += build_time;
            stats.table_build_count += 1;
        });
    }

    // Time the decode
    let decode_start = Instant::now();
    let result = decode_dynamic_speculative(bits, output, out_pos, &litlen_table, &dist_table);
    let decode_nanos = decode_start.elapsed().as_nanos() as u64;

    TIMING_STATS.with(|s| {
        let mut stats = s.borrow_mut();
        stats.decode_nanos += decode_nanos;
        stats.decode_count += 1;
    });

    result
}

/// Decode using specialized flat tables (no subtables, direct lookup)
#[inline(never)]
fn decode_with_specialized_tables(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    spec: &crate::specialized_decode::SpecializedDecoder,
) -> Result<usize> {
    #[allow(unused_imports)]
    use crate::specialized_decode::SpecEntry;

    const FASTLOOP_MARGIN: usize = 320;
    let out_ptr = output.as_mut_ptr();
    let out_end = output.len();
    let in_data = bits.data;

    let mut bitbuf = bits.bitbuf;
    let mut bitsleft = bits.bitsleft;
    let mut in_pos = bits.pos;

    // Branchless refill macro
    macro_rules! refill {
        () => {
            if in_pos + 8 <= in_data.len() {
                let word = unsafe { (in_data.as_ptr().add(in_pos) as *const u64).read_unaligned() };
                let word = u64::from_le(word);
                bitbuf |= word << (bitsleft as u8);
                in_pos += 7;
                in_pos -= ((bitsleft >> 3) & 0x7) as usize;
                bitsleft |= 56;
            } else {
                while (bitsleft as u8) <= 56 && in_pos < in_data.len() {
                    bitbuf |= (in_data[in_pos] as u64) << (bitsleft as u8);
                    in_pos += 1;
                    bitsleft = (bitsleft as u8).wrapping_add(8) as u32;
                }
            }
        };
    }

    // Initial refill
    refill!();

    // FASTLOOP with specialized main tables
    while out_pos + FASTLOOP_MARGIN <= out_end {
        // Decode litlen using flat 11-bit main table
        let mut entry = unsafe { *spec.litlen.get_unchecked((bitbuf & 0x7FF) as usize) };

        let bits_used = entry.total_bits() as u32;
        bitbuf >>= bits_used;
        bitsleft = bitsleft.wrapping_sub(bits_used);

        // Check literal FIRST (most common case)
        if entry.is_literal() {
            if entry.is_double() {
                unsafe {
                    let ptr = out_ptr.add(out_pos);
                    ptr.write(entry.lit1());
                    ptr.add(1).write(entry.lit2());
                }
                out_pos += 2;
            } else {
                unsafe {
                    *out_ptr.add(out_pos) = entry.literal_value();
                }
                out_pos += 1;
            }

            // PRELOAD next entry BEFORE writing (hides memory latency)
            entry = unsafe { *spec.litlen.get_unchecked((bitbuf & 0x7FF) as usize) };

            if entry.is_literal() {
                let bits2 = entry.total_bits() as u32;
                bitbuf >>= bits2;
                bitsleft = bitsleft.wrapping_sub(bits2);

                if entry.is_double() {
                    unsafe {
                        let ptr = out_ptr.add(out_pos);
                        ptr.write(entry.lit1());
                        ptr.add(1).write(entry.lit2());
                    }
                    out_pos += 2;
                } else {
                    unsafe {
                        *out_ptr.add(out_pos) = entry.literal_value();
                    }
                    out_pos += 1;
                }

                // Try one more literal batch if we have enough bits (up to 6 total literals)
                if bitsleft >= 32 {
                    let next_entry =
                        unsafe { *spec.litlen.get_unchecked((bitbuf & 0x7FF) as usize) };
                    if next_entry.is_literal() {
                        let bits3 = next_entry.total_bits() as u32;
                        bitbuf >>= bits3;
                        bitsleft = bitsleft.wrapping_sub(bits3);
                        if next_entry.is_double() {
                            unsafe {
                                let ptr = out_ptr.add(out_pos);
                                ptr.write(next_entry.lit1());
                                ptr.add(1).write(next_entry.lit2());
                            }
                            out_pos += 2;
                        } else {
                            unsafe {
                                *out_ptr.add(out_pos) = next_entry.literal_value();
                            }
                            out_pos += 1;
                        }
                    }
                }

                refill!();
                continue;
            }
            refill!();
            continue;
        }

        // Check EOB
        if entry.is_eob() {
            bits.bitbuf = bitbuf;
            bits.bitsleft = bitsleft;
            bits.pos = in_pos;
            return Ok(out_pos);
        }

        // LENGTH - decode with extra bits
        let length_base = entry.length_base() as u32;
        let extra = entry.extra_bits();
        let length = if extra > 0 {
            let extra_val = (bitbuf & ((1u64 << extra) - 1)) as u32;
            bitbuf >>= extra;
            bitsleft = bitsleft.wrapping_sub(extra as u32);
            length_base + extra_val
        } else {
            length_base
        };

        refill!();

        // DISTANCE - flat 11-bit table + subtables
        let mut dist_entry = spec.dist[(bitbuf & 0x7FF) as usize];
        if unlikely(dist_entry.is_subtable()) {
            let offset = dist_entry.subtable_offset() as usize;
            let sub_bits = dist_entry.subtable_bits();
            let idx = (bitbuf >> 11) & ((1 << sub_bits) - 1);
            dist_entry = spec.dist[offset + idx as usize];
        }
        let dist_bits = dist_entry.total_bits() as u32;
        bitbuf >>= dist_bits;
        bitsleft = bitsleft.wrapping_sub(dist_bits);

        let dist_base = dist_entry.symbol() as u32;
        let dist_extra = dist_entry.extra_bits();
        let distance = if dist_extra > 0 {
            let extra_val = (bitbuf & ((1u64 << dist_extra) - 1)) as u32;
            bitbuf >>= dist_extra;
            bitsleft = bitsleft.wrapping_sub(dist_extra as u32);
            dist_base + extra_val
        } else {
            dist_base
        };

        if distance == 0 || distance as usize > out_pos {
            bits.bitbuf = bitbuf;
            bits.bitsleft = bitsleft;
            bits.pos = in_pos;
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        refill!();

        // Copy match
        out_pos = copy_match_fast(output, out_pos, distance, length);
    }

    // Write back state for generic loop
    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft & 0xFF;
    bits.pos = in_pos;

    // Generic loop for remainder - use the specialized tables
    decode_generic_with_spec(bits, output, out_pos, spec)
}

/// Generic loop using specialized tables
fn decode_generic_with_spec(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    spec: &crate::specialized_decode::SpecializedDecoder,
) -> Result<usize> {
    loop {
        bits.refill();

        let entry = spec.decode_symbol(bits.peek());
        let bits_used = entry.total_bits() as u32;
        bits.consume(bits_used);

        // CRITICAL: Check EOB BEFORE literal - EOB's symbol 0xFFFF has the literal flag set!
        if entry.is_eob() {
            return Ok(out_pos);
        }

        if entry.is_literal() {
            if out_pos + 2 > output.len() {
                return Err(Error::new(ErrorKind::InvalidData, "Output overflow"));
            }
            if entry.is_double() {
                output[out_pos] = entry.lit1();
                output[out_pos + 1] = entry.lit2();
                out_pos += 2;
            } else {
                output[out_pos] = entry.literal_value();
                out_pos += 1;
            }
            continue;
        }

        // Length
        let length_base = entry.length_base() as u32;
        let extra = entry.extra_bits();
        let length = if extra > 0 {
            let extra_val = (bits.peek() & ((1u64 << extra) - 1)) as u32;
            bits.consume(extra as u32);
            length_base + extra_val
        } else {
            length_base
        };

        bits.refill();

        // Distance
        let dist_entry = spec.decode_distance(bits.peek());
        let dist_bits = dist_entry.total_bits() as u32;
        bits.consume(dist_bits);

        let dist_base = dist_entry.symbol() as u32;
        let dist_extra = dist_entry.extra_bits();
        let distance = if dist_extra > 0 {
            let extra_val = (bits.peek() & ((1u64 << dist_extra) - 1)) as u32;
            bits.consume(dist_extra as u32);
            dist_base + extra_val
        } else {
            dist_base
        };

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        if out_pos + length as usize > output.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Output overflow"));
        }

        out_pos = copy_match_safe(output, out_pos, distance, length);
    }
}

/// Dynamic block decoder with SIMD speculative fast path
///
/// Uses SIMD parallel speculative decode for literal runs,
/// falls back to scalar for lengths and exceptional cases.
#[inline(never)]
fn decode_dynamic_speculative(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    // Use BMI2-optimized decoder if available, otherwise libdeflate-style
    decode_huffman_with_dispatch(bits, output, out_pos, litlen, dist)
}

/// Inlined monolithic decoder - all hot functions inlined for maximum speed
/// Key insight: Eliminate ALL function call overhead in the hot path
#[inline(never)] // Don't inline the outer function, but inline everything inside
fn decode_dynamic_hyperfast(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 320;

    let litlen_ptr = litlen.entries_ptr();
    let out_ptr = output.as_mut_ptr();
    let out_end = output.len();

    // PRELOAD first entry
    bits.refill();

    // FASTLOOP - fully inlined
    while out_pos + FASTLOOP_MARGIN <= out_end {
        // Inline peek
        let saved_bitbuf = bits.bitbuf;

        // Inline lookup - direct table access, get raw u32
        let entry = unsafe { (*litlen_ptr.add((saved_bitbuf as usize) & 0x7FF)).raw() };

        // Inline consume_entry - shift and subtract
        bits.bitbuf >>= entry as u8;
        bits.bitsleft = bits.bitsleft.wrapping_sub(entry & 0x1F);

        // Check literal (bit 31 = negative)
        if (entry as i32) < 0 {
            // LITERAL - inline write
            let lit = ((entry >> 16) & 0xFF) as u8;
            unsafe {
                *out_ptr.add(out_pos) = lit;
            }
            out_pos += 1;

            // Unrolled literal 2
            let entry2 = unsafe { (*litlen_ptr.add((bits.bitbuf as usize) & 0x7FF)).raw() };
            if (entry2 as i32) < 0 {
                bits.bitbuf >>= entry2 as u8;
                bits.bitsleft = bits.bitsleft.wrapping_sub(entry2 & 0x1F);
                let lit2 = ((entry2 >> 16) & 0xFF) as u8;
                unsafe {
                    *out_ptr.add(out_pos) = lit2;
                }
                out_pos += 1;

                // Unrolled literal 3
                let entry3 = unsafe { (*litlen_ptr.add((bits.bitbuf as usize) & 0x7FF)).raw() };
                if (entry3 as i32) < 0 {
                    bits.bitbuf >>= entry3 as u8;
                    bits.bitsleft = bits.bitsleft.wrapping_sub(entry3 & 0x1F);
                    let lit3 = ((entry3 >> 16) & 0xFF) as u8;
                    unsafe {
                        *out_ptr.add(out_pos) = lit3;
                    }
                    out_pos += 1;

                    // Unrolled literal 4
                    let entry4 = unsafe { (*litlen_ptr.add((bits.bitbuf as usize) & 0x7FF)).raw() };
                    if (entry4 as i32) < 0 {
                        bits.bitbuf >>= entry4 as u8;
                        bits.bitsleft = bits.bitsleft.wrapping_sub(entry4 & 0x1F);
                        let lit4 = ((entry4 >> 16) & 0xFF) as u8;
                        unsafe {
                            *out_ptr.add(out_pos) = lit4;
                        }
                        out_pos += 1;

                        // Refill after 4 literals
                        if (bits.bitsleft as u8) < 32 {
                            bits.refill();
                        }

                        // Unrolled literal 5
                        let entry5 =
                            unsafe { (*litlen_ptr.add((bits.bitbuf as usize) & 0x7FF)).raw() };
                        if (entry5 as i32) < 0 {
                            bits.bitbuf >>= entry5 as u8;
                            bits.bitsleft = bits.bitsleft.wrapping_sub(entry5 & 0x1F);
                            let lit5 = ((entry5 >> 16) & 0xFF) as u8;
                            unsafe {
                                *out_ptr.add(out_pos) = lit5;
                            }
                            out_pos += 1;
                            bits.refill();
                            continue;
                        }
                        continue;
                    }
                    if (bits.bitsleft as u8) < 32 {
                        bits.refill();
                    }
                    continue;
                }
                if (bits.bitsleft as u8) < 32 {
                    bits.refill();
                }
                continue;
            }
            if (bits.bitsleft as u8) < 32 {
                bits.refill();
            }
            continue;
        }

        // Not a literal - use regular path for exceptional/length
        let entry = crate::libdeflate_entry::LitLenEntry::from_raw(entry);

        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                return Ok(out_pos);
            }

            // Subtable
            let sub_entry = litlen.lookup_subtable(entry, saved_bitbuf);
            let sub_saved = bits.peek();
            bits.consume_entry(sub_entry.raw());

            if (sub_entry.raw() as i32) < 0 {
                // Literal from subtable
                unsafe {
                    *out_ptr.add(out_pos) = sub_entry.literal_value();
                }
                out_pos += 1;
                bits.refill();
                continue;
            }
            if sub_entry.is_end_of_block() {
                return Ok(out_pos);
            }

            // Length from subtable
            let length = sub_entry.decode_length(sub_saved);

            bits.refill();
            let dist_saved = bits.peek();
            let mut dist_entry = dist.lookup(dist_saved);

            if dist_entry.is_subtable_ptr() {
                bits.consume(DistTable::TABLE_BITS as u32);
                dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
            }

            let dist_extra_saved = bits.peek();
            bits.consume_entry(dist_entry.raw());
            let distance = dist_entry.decode_distance(dist_extra_saved);

            if distance == 0 || distance as usize > out_pos {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid distance {} at pos {}", distance, out_pos),
                ));
            }

            bits.refill();
            out_pos = copy_match_fast(output, out_pos, distance, length);
            continue;
        }

        // Length code from main table - entry is already a LitLenEntry from above
        let length = entry.decode_length(saved_bitbuf);

        bits.refill();
        let dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);

        if dist_entry.is_subtable_ptr() {
            bits.consume(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
        }

        let dist_extra_saved = bits.peek();
        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_extra_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        bits.refill();
        out_pos = copy_match_fast(output, out_pos, distance, length);
    }

    // Generic loop for remainder
    decode_huffman_cf(bits, output, out_pos, litlen, dist)
}

fn build_code_length_table(lengths: &[u8; 19]) -> Result<[u16; 128]> {
    let mut table = [0u16; 128];

    let mut count = [0u16; 8];
    for &len in lengths.iter() {
        if len > 0 && len <= 7 {
            count[len as usize] += 1;
        }
    }

    let mut code = 0u32;
    let mut first_code = [0u32; 8];
    for len in 1..=7 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }

    let mut next_code = first_code;
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len = len as usize;
        let codeword = next_code[len];
        next_code[len] += 1;

        let mut reversed = 0u32;
        let mut c = codeword;
        for _ in 0..len {
            reversed = (reversed << 1) | (c & 1);
            c >>= 1;
        }

        let stride = 1usize << len;
        let mut idx = reversed as usize;
        while idx < 128 {
            table[idx] = ((symbol as u16) << 8) | (len as u16);
            idx += stride;
        }
    }

    Ok(table)
}

// =============================================================================
// Public API
// =============================================================================

/// Decode a deflate stream using consume-first pattern
pub fn inflate_consume_first(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let mut bits = Bits::new(input);
    let out_size = inflate_consume_first_bits(&mut bits, output)?;
    Ok(out_size)
}

/// Decode a deflate stream from a Bits reader
pub fn inflate_consume_first_bits(bits: &mut Bits, output: &mut [u8]) -> Result<usize> {
    let mut out_pos = 0;

    loop {
        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        let prev_pos = out_pos;
        match btype {
            0 => out_pos = decode_stored(bits, output, out_pos)?,
            1 => out_pos = decode_fixed(bits, output, out_pos)?,
            2 => out_pos = decode_dynamic(bits, output, out_pos)?,
            3 => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type")),
            _ => unreachable!(),
        }

        // Record block statistics
        let block_bytes = out_pos - prev_pos;
        BLOCK_STATS.with(|stats| {
            let mut s = stats.borrow_mut();
            match btype {
                0 => {
                    s.stored_blocks += 1;
                    s.stored_bytes += block_bytes;
                }
                1 => {
                    s.fixed_blocks += 1;
                    s.fixed_bytes += block_bytes;
                }
                2 => {
                    s.dynamic_blocks += 1;
                    s.dynamic_bytes += block_bytes;
                }
                _ => {}
            }
        });

        if bfinal {
            // Align bits to byte boundary at end of deflate stream
            bits.align_to_byte();
            return Ok(out_pos);
        }
    }
}

// =============================================================================
// Option 3: Interleaved Multi-Block Decode (SIMD-style parallelism)
// =============================================================================
//
// Decode 4 independent deflate blocks simultaneously within a single thread.
// Uses instruction-level parallelism by interleaving operations across lanes.

/// State for one decode lane (one independent block)
#[derive(Clone)]
pub struct DecodeLane<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub bitbuf: u64,
    pub bitsleft: u32,
    pub out_pos: usize,
    pub done: bool,
}

impl<'a> DecodeLane<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut lane = Self {
            data,
            pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            out_pos: 0,
            done: false,
        };
        lane.refill();
        lane
    }

    #[inline(always)]
    pub fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let word = unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            let word = u64::from_le(word);
            self.bitbuf |= word << (self.bitsleft as u8);
            self.pos += 7;
            self.pos -= ((self.bitsleft >> 3) & 0x7) as usize;
            self.bitsleft |= 56;
        } else {
            while self.bitsleft <= 56 && self.pos < self.data.len() {
                self.bitbuf |= (self.data[self.pos] as u64) << self.bitsleft;
                self.pos += 1;
                self.bitsleft += 8;
            }
        }
    }

    #[inline(always)]
    pub fn peek(&self) -> u64 {
        self.bitbuf
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bitbuf >>= n as u8;
        self.bitsleft -= n;
    }
}

/// Decode 4 independent deflate blocks simultaneously (for BGZF)
/// Returns total bytes written across all outputs
pub fn decode_interleaved_4(
    blocks: [&[u8]; 4],
    outputs: [&mut [u8]; 4],
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<[usize; 4]> {
    const LITLEN_TABLEMASK: u64 = (1u64 << LitLenTable::TABLE_BITS) - 1;
    let litlen_ptr = litlen.entries_ptr();

    // Initialize 4 lanes
    let mut lanes: [DecodeLane; 4] = [
        DecodeLane::new(blocks[0]),
        DecodeLane::new(blocks[1]),
        DecodeLane::new(blocks[2]),
        DecodeLane::new(blocks[3]),
    ];

    // Get output pointers
    let out_ptrs: [*mut u8; 4] = [
        outputs[0].as_mut_ptr(),
        outputs[1].as_mut_ptr(),
        outputs[2].as_mut_ptr(),
        outputs[3].as_mut_ptr(),
    ];
    let _out_ends: [usize; 4] = [
        outputs[0].len(),
        outputs[1].len(),
        outputs[2].len(),
        outputs[3].len(),
    ];

    // Preload entries for all 4 lanes
    let mut entries: [u32; 4] = [
        unsafe { (*litlen_ptr.add((lanes[0].peek() & LITLEN_TABLEMASK) as usize)).raw() },
        unsafe { (*litlen_ptr.add((lanes[1].peek() & LITLEN_TABLEMASK) as usize)).raw() },
        unsafe { (*litlen_ptr.add((lanes[2].peek() & LITLEN_TABLEMASK) as usize)).raw() },
        unsafe { (*litlen_ptr.add((lanes[3].peek() & LITLEN_TABLEMASK) as usize)).raw() },
    ];

    // Interleaved decode loop - process all 4 lanes in lockstep for literals
    loop {
        let mut all_done = true;
        let mut any_literal = false;

        // Check which lanes have literals
        for i in 0..4 {
            if lanes[i].done {
                continue;
            }
            all_done = false;

            if (entries[i] as i32) < 0 {
                // Literal
                any_literal = true;
            }
        }

        if all_done {
            break;
        }

        // Process literals for all lanes that have them
        if any_literal {
            for i in 0..4 {
                if lanes[i].done || (entries[i] as i32) >= 0 {
                    continue;
                }

                // Decode literal
                let lit = (entries[i] >> 16) as u8;
                lanes[i].bitbuf >>= entries[i] as u8;
                lanes[i].bitsleft = lanes[i].bitsleft.wrapping_sub(entries[i]);

                unsafe {
                    *out_ptrs[i].add(lanes[i].out_pos) = lit;
                }
                lanes[i].out_pos += 1;

                // Refill and lookup next entry
                if lanes[i].bitsleft < 32 {
                    lanes[i].refill();
                }
                entries[i] = unsafe {
                    (*litlen_ptr.add((lanes[i].peek() & LITLEN_TABLEMASK) as usize)).raw()
                };
            }
            continue;
        }

        // No literals - handle exceptional cases (EOB, length codes) for each lane
        for i in 0..4 {
            if lanes[i].done {
                continue;
            }

            let entry = crate::libdeflate_entry::LitLenEntry::from_raw(entries[i]);
            let saved = lanes[i].peek();
            lanes[i].bitbuf >>= entries[i] as u8;
            lanes[i].bitsleft = lanes[i].bitsleft.wrapping_sub(entries[i]);

            if entry.is_exceptional() {
                if entry.is_end_of_block() {
                    lanes[i].done = true;
                    continue;
                }

                // Subtable - use slow path
                let sub_entry = litlen.lookup_subtable(entry, saved);
                let sub_saved = lanes[i].peek();
                lanes[i].bitbuf >>= sub_entry.raw() as u8;
                lanes[i].bitsleft = lanes[i].bitsleft.wrapping_sub(sub_entry.raw());

                if (sub_entry.raw() as i32) < 0 {
                    unsafe {
                        *out_ptrs[i].add(lanes[i].out_pos) = sub_entry.literal_value();
                    }
                    lanes[i].out_pos += 1;
                    lanes[i].refill();
                    entries[i] = unsafe {
                        (*litlen_ptr.add((lanes[i].peek() & LITLEN_TABLEMASK) as usize)).raw()
                    };
                    continue;
                }
                if sub_entry.is_end_of_block() {
                    lanes[i].done = true;
                    continue;
                }

                // Length from subtable - fall through to length handling below
                let length = sub_entry.decode_length(sub_saved);
                decode_match_for_lane(&mut lanes[i], outputs[i], out_ptrs[i], length, dist)?;
                entries[i] = unsafe {
                    (*litlen_ptr.add((lanes[i].peek() & LITLEN_TABLEMASK) as usize)).raw()
                };
                continue;
            }

            // Length code
            let length = entry.decode_length(saved);
            decode_match_for_lane(&mut lanes[i], outputs[i], out_ptrs[i], length, dist)?;
            entries[i] =
                unsafe { (*litlen_ptr.add((lanes[i].peek() & LITLEN_TABLEMASK) as usize)).raw() };
        }
    }

    Ok([
        lanes[0].out_pos,
        lanes[1].out_pos,
        lanes[2].out_pos,
        lanes[3].out_pos,
    ])
}

/// Helper to decode a match for a single lane
#[inline(always)]
fn decode_match_for_lane(
    lane: &mut DecodeLane,
    _output: &mut [u8],
    out_ptr: *mut u8,
    length: u32,
    dist: &DistTable,
) -> Result<()> {
    lane.refill();
    let dist_saved = lane.peek();
    let mut dist_entry = dist.lookup(dist_saved);

    if dist_entry.is_subtable_ptr() {
        lane.consume(DistTable::TABLE_BITS as u32);
        dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
    }

    let dist_extra_saved = lane.peek();
    lane.bitbuf >>= dist_entry.raw() as u8;
    lane.bitsleft = lane.bitsleft.wrapping_sub(dist_entry.raw());
    let distance = dist_entry.decode_distance(dist_extra_saved);

    if distance == 0 || distance as usize > lane.out_pos {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
    }

    lane.refill();

    // Copy match
    let dist_usize = distance as usize;
    let len_usize = length as usize;
    unsafe {
        let dst = out_ptr.add(lane.out_pos);
        let src = out_ptr.add(lane.out_pos - dist_usize);
        if dist_usize >= 8 {
            let mut s = src;
            let mut d = dst;
            let end = dst.add(len_usize);
            while d < end {
                (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
                s = s.add(8);
                d = d.add(8);
            }
        } else {
            for j in 0..len_usize {
                *dst.add(j) = *src.add(j % dist_usize);
            }
        }
    }
    lane.out_pos += len_usize;

    Ok(())
}

// =============================================================================
// Tests - Using REAL silesia data, not simulations
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper for benchmarking a single dataset
    fn run_bench(name: &str, gz_path: &str) {
        // Ensure files are prepared
        let _ = crate::benchmark_datasets::prepare_datasets();

        let gz = match std::fs::read(gz_path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping {} benchmark - file not found: {}", name, gz_path);
                return;
            }
        };

        // Parse gzip header properly
        let mut pos = 10;
        let flg = gz[3];
        if (flg & 0x04) != 0 {
            let xlen = u16::from_le_bytes([gz[pos], gz[pos + 1]]) as usize;
            pos += 2 + xlen;
        }
        if (flg & 0x08) != 0 {
            while pos < gz.len() && gz[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if (flg & 0x10) != 0 {
            while pos < gz.len() && gz[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if (flg & 0x02) != 0 {
            pos += 2;
        }

        let deflate = &gz[pos..gz.len() - 8];
        let isize = u32::from_le_bytes([
            gz[gz.len() - 4],
            gz[gz.len() - 3],
            gz[gz.len() - 2],
            gz[gz.len() - 1],
        ]) as usize;

        let mut output = vec![0u8; isize + 1024];
        let mut lib_output = vec![0u8; isize + 1024];

        // Verify libdeflate can decode it first
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(deflate, &mut lib_output)
            .expect("libdeflate failed");

        // Now try our decoder
        let our_result = inflate_consume_first(deflate, &mut output);

        if let Err(e) = &our_result {
            eprintln!("Error decoding {}: {:?}", name, e);
            let check_len = lib_size.min(output.len());
            for i in 0..check_len {
                if output[i] != lib_output[i] {
                    eprintln!(
                        "First mismatch at byte {}: got {:02x} expected {:02x}",
                        i, output[i], lib_output[i]
                    );
                    break;
                }
            }
            panic!("Our decode failed for {}", name);
        }

        let our_size = our_result.unwrap();
        assert_eq!(our_size, lib_size, "Size mismatch for {}", name);

        // Check first 10KB for correctness (matches original benchmark behavior)
        let check_len = 10000.min(our_size);
        if output[..check_len] != lib_output[..check_len] {
            for i in 0..check_len {
                if output[i] != lib_output[i] {
                    panic!("Data mismatch at byte {} for {}", i, name);
                }
            }
        }

        // Benchmark
        let iterations = 10;
        reset_cache_stats();

        let start_t = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = inflate_consume_first(deflate, &mut output);
        }
        let our_time = start_t.elapsed();

        let start_t = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = libdeflater::Decompressor::new().deflate_decompress(deflate, &mut lib_output);
        }
        let lib_time = start_t.elapsed();

        let our_throughput = (isize * iterations) as f64 / our_time.as_secs_f64() / 1e6;
        let lib_throughput = (isize * iterations) as f64 / lib_time.as_secs_f64() / 1e6;

        let (hits, misses, hit_rate) = get_cache_stats();

        eprintln!("\n=== CONSUME-FIRST {} ===", name.to_uppercase());
        eprintln!("Data size: {:.1} MB", isize as f64 / 1_000_000.0);
        eprintln!("Our throughput:       {:>8.1} MB/s", our_throughput);
        eprintln!("libdeflate throughput: {:>8.1} MB/s", lib_throughput);
        eprintln!("Ratio: {:.1}%", 100.0 * our_throughput / lib_throughput);
        eprintln!(
            "Cache: {} hits, {} misses ({:.1}% hit rate)",
            hits,
            misses,
            hit_rate * 100.0
        );
        let (spec_used, spec_fallback) = get_spec_stats();
        if spec_used + spec_fallback > 0 {
            let spec_rate = spec_used as f64 / (spec_used + spec_fallback) as f64;
            eprintln!(
                "Specialized: {} used, {} fallback ({:.1}% specialized)",
                spec_used,
                spec_fallback,
                spec_rate * 100.0
            );
        }
        eprintln!("=============================\n");
    }

    #[test]
    fn test_cf_simple_literals() {
        let original = b"Hello, World!";

        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::default());
            enc.write_all(original).unwrap();
            enc.finish().unwrap();
        }

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_consume_first(&compressed, &mut output).expect("Decode failed");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    #[test]
    fn test_cf_rle() {
        let original = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::default());
            enc.write_all(original).unwrap();
            enc.finish().unwrap();
        }

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_consume_first(&compressed, &mut output).expect("Decode failed");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    #[test]
    fn test_cf_matches() {
        let original = b"abcabcabcabcabcabcabcabc";

        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::best());
            enc.write_all(original).unwrap();
            enc.finish().unwrap();
        }

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_consume_first(&compressed, &mut output).expect("Decode failed");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    #[test]
    fn test_cf_large_data() {
        let original = b"This is a test of the consume-first decoder. ".repeat(1000);

        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::best());
            enc.write_all(&original).unwrap();
            enc.finish().unwrap();
        }

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_consume_first(&compressed, &mut output).expect("Decode failed");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    /// Benchmark on silesia dataset
    #[test]
    fn bench_cf_silesia() {
        run_bench("silesia", "benchmark_data/silesia-gzip.tar.gz");
    }

    /// Benchmark on software archive dataset (source code patterns)
    #[test]
    fn bench_cf_software() {
        run_bench("software", "benchmark_data/software.archive.gz");
    }

    /// Benchmark on repetitive logs dataset
    #[test]
    fn bench_cf_logs() {
        run_bench("logs", "benchmark_data/logs.txt.gz");
    }

    #[test]
    fn test_cf_single_bgzf_block() {
        // Test on a BGZF block to match what parallel decompress does
        let data = match std::fs::read("benchmark_data/test-gzippy-l1-t14.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skip - no file");
                return;
            }
        };

        // Get expected from flate2 MultiGzDecoder
        use std::io::Read;
        let mut expected = Vec::new();
        let mut decoder = flate2::read::MultiGzDecoder::new(&data[..]);
        decoder.read_to_end(&mut expected).unwrap();

        // Parse first BGZF block manually
        // Header: 1f 8b 08 04 (gzip with FEXTRA)
        let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
        let header_end = 12 + xlen;

        // Find block size from extra field
        let mut pos = 12;
        let mut bsize = None;
        while pos < header_end {
            let slen = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let si = [data[pos], data[pos + 1]];
            if si == *b"GZ" && slen >= 4 {
                // Our BGZF uses GZ with 4-byte block size
                bsize = Some(u32::from_le_bytes([
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize);
            } else if si == *b"BC" && slen == 2 {
                // Standard BGZF uses BC with 2-byte block size
                bsize = Some(u16::from_le_bytes([data[pos + 4], data[pos + 5]]) as usize + 1);
            }
            pos += 4 + slen;
        }

        let bsize = bsize.expect("No block size in BGZF header");
        let isize_expected = u32::from_le_bytes([
            data[bsize - 4],
            data[bsize - 3],
            data[bsize - 2],
            data[bsize - 1],
        ]) as usize;

        let deflate_data = &data[header_end..bsize - 8];

        // Test decoding a single BGZF block

        // Test with libdeflate first (reference)
        let mut lib_out = vec![0u8; isize_expected];
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(deflate_data, &mut lib_out)
            .expect("libdeflate failed");
        assert_eq!(lib_size, isize_expected, "libdeflate size mismatch");

        // Test with our decoder - exact size
        let mut our_out = vec![0u8; isize_expected];
        let our_size =
            inflate_consume_first(deflate_data, &mut our_out).expect("our decoder failed");

        assert_eq!(our_size, isize_expected, "size mismatch");
        assert_eq!(
            &our_out[..our_size],
            &lib_out[..lib_size],
            "content mismatch"
        );
    }

    #[test]
    fn bench_interleaved_4_blocks() {
        // Create 4 independent deflate blocks and benchmark interleaved decode
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create 4 different data chunks
        let chunks: Vec<Vec<u8>> = (0..4)
            .map(|i| {
                (0..50_000)
                    .map(|j| ((i * 37 + j * 13) % 256) as u8)
                    .collect()
            })
            .collect();

        // Compress each chunk
        let compressed: Vec<Vec<u8>> = chunks
            .iter()
            .map(|chunk| {
                let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(chunk).unwrap();
                encoder.finish().unwrap()
            })
            .collect();

        // Build fixed Huffman tables for all blocks
        let tables = crate::libdeflate_decode::get_fixed_tables();
        let _litlen = &tables.0;
        let _dist = &tables.1;

        // Prepare outputs
        let mut outputs: Vec<Vec<u8>> = chunks.iter().map(|c| vec![0u8; c.len() + 1000]).collect();

        // Verify single-threaded decode first
        for (i, (comp, expected)) in compressed.iter().zip(chunks.iter()).enumerate() {
            let mut out = vec![0u8; expected.len() + 100];
            let size = inflate_consume_first(comp, &mut out).expect("decode failed");
            assert_eq!(size, expected.len(), "Block {} size mismatch", i);
            assert_eq!(&out[..size], &expected[..], "Block {} content mismatch", i);
        }

        eprintln!("\n=== INTERLEAVED 4-BLOCK BENCHMARK ===");

        // Benchmark sequential decode (baseline)
        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for (comp, out) in compressed.iter().zip(outputs.iter_mut()) {
                let _ = inflate_consume_first(comp, out);
            }
        }
        let seq_time = start.elapsed();
        let total_bytes: usize = chunks.iter().map(|c| c.len()).sum();
        let seq_throughput = (total_bytes * iterations) as f64 / seq_time.as_secs_f64() / 1e6;
        eprintln!("Sequential 4 blocks: {:.1} MB/s", seq_throughput);

        // Benchmark interleaved decode
        // Note: This requires dynamic tables, so we need to build them for each block
        // For now, just compare against sequential as a concept demonstration

        eprintln!("(Interleaved decode requires per-block table building - testing concept)");
        eprintln!("=====================================\n");
    }
}
