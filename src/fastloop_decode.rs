//! Fastloop decoder inspired by libdeflate
//!
//! Key optimizations from libdeflate:
//! 1. Fastloop with bounds checks in loop condition (not in body)
//! 2. Multi-literal decode (up to 3 literals per iteration)
//! 3. Preloading next table entry before current iteration completes
//! 4. Distance=1 memset optimization
//! 5. Branchless bit refill

#![allow(dead_code)]

use crate::combined_lut::{CombinedLUT, DIST_END_OF_BLOCK, DIST_LITERAL, DIST_SLOW_PATH};
use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};
use crate::two_level_table::TwoLevelTable;
use std::io;

// Constants for fastloop bounds
const FASTLOOP_MAX_BYTES_WRITTEN: usize = 2 + 258 + 16; // 2 literals + max match + copy overrun
const FASTLOOP_MAX_BYTES_READ: usize = 32; // Worst case bits consumed per iteration

/// Optimized bit reader with branchless refill
pub struct FastBitsOptimized<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
}

impl<'a> FastBitsOptimized<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut fb = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
        };
        fb.refill();
        fb
    }

    /// Branchless refill (from libdeflate REFILL_BITS_BRANCHLESS)
    /// This is the key to high performance
    #[inline(always)]
    pub fn refill_branchless(&mut self) {
        if self.pos + 8 <= self.data.len() {
            // Load 8 bytes unconditionally
            let bytes =
                unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            self.buf |= bytes.to_le() << (self.bits as u8);
            // Advance position based on how much space we had
            // This is the key branchless trick: advance by (64 - bits) / 8
            // but clamped appropriately
            let consumed = (64 - self.bits) / 8;
            self.pos += consumed as usize;
            self.bits |= 56; // Set bits to at least 56 (MAX_BITSLEFT & ~7)
        } else {
            // Slow path at end of stream
            while self.bits <= 56 && self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << self.bits;
                self.pos += 1;
                self.bits += 8;
            }
        }
    }

    /// Standard refill (for generic loop)
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.bits > 56 {
            return;
        }
        self.refill_branchless();
    }

    #[inline(always)]
    pub fn buffer(&self) -> u64 {
        self.buf
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bits = self.bits.saturating_sub(n);
    }

    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        val
    }

    #[inline(always)]
    pub fn bits_available(&self) -> u32 {
        self.bits
    }

    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len() && self.bits == 0
    }
}

/// Optimized LZ77 copy with distance=1 memset
#[inline(always)]
fn copy_match_fast(output: &mut [u8], out_pos: usize, distance: usize, length: usize) -> usize {
    debug_assert!(distance <= out_pos);
    debug_assert!(out_pos + length <= output.len());

    let src_start = out_pos - distance;

    unsafe {
        let dst = output.as_mut_ptr().add(out_pos);
        let src = output.as_ptr().add(src_start);

        if distance == 1 {
            // Very common: RLE (single byte repeat)
            // Use memset semantics
            let byte = *src;
            std::ptr::write_bytes(dst, byte, length);
        } else if distance >= length {
            // Non-overlapping: use memcpy
            std::ptr::copy_nonoverlapping(src, dst, length);
        } else if distance >= 8 {
            // Overlapping but distance >= 8: chunk copy
            let mut remaining = length;
            let mut d = dst;
            let mut s = src;
            while remaining >= 8 {
                let chunk = (s as *const u64).read_unaligned();
                (d as *mut u64).write_unaligned(chunk);
                d = d.add(8);
                s = s.add(8);
                remaining -= 8;
            }
            // Copy remainder
            for i in 0..remaining {
                *d.add(i) = *s.add(i);
            }
        } else {
            // Small distance (2-7): byte-by-byte
            for i in 0..length {
                *dst.add(i) = *src.add(i % distance);
            }
        }
    }

    out_pos + length
}

/// Fastloop decode - the main optimization
///
/// This function uses the libdeflate architecture:
/// 1. Check bounds ONCE at loop start for worst-case iteration
/// 2. Decode up to 3 literals per iteration
/// 3. Preload next entry before completing current
/// 4. Fall back to generic loop near buffer ends
#[inline(never)]
pub fn decode_huffman_fastloop(
    bits: &mut FastBitsOptimized,
    output: &mut [u8],
    mut out_pos: usize,
    combined_lut: &CombinedLUT,
    lit_len_table: &TwoLevelTable,
    dist_table: &TwoLevelTable,
) -> io::Result<usize> {
    // Fastloop bounds
    let out_fastloop_end = output.len().saturating_sub(FASTLOOP_MAX_BYTES_WRITTEN);

    // Main fastloop
    while out_pos < out_fastloop_end && bits.bits_available() >= 32 {
        bits.refill_branchless();

        let entry = combined_lut.decode(bits.buffer());

        // Long code fallback (rare)
        if entry.bits_to_skip == 0 {
            return decode_huffman_generic(bits, output, out_pos, lit_len_table, dist_table);
        }

        bits.consume(entry.bits_to_skip as u32);

        match entry.distance {
            DIST_LITERAL => {
                // Output literal
                output[out_pos] = entry.symbol_or_length;
                out_pos += 1;

                // Multi-literal optimization: try to decode 2 more fast literals
                if bits.bits_available() >= 24 {
                    let entry2 = combined_lut.decode(bits.buffer());
                    if entry2.bits_to_skip > 0 && entry2.distance == DIST_LITERAL {
                        bits.consume(entry2.bits_to_skip as u32);
                        output[out_pos] = entry2.symbol_or_length;
                        out_pos += 1;

                        if bits.bits_available() >= 12 {
                            let entry3 = combined_lut.decode(bits.buffer());
                            if entry3.bits_to_skip > 0 && entry3.distance == DIST_LITERAL {
                                bits.consume(entry3.bits_to_skip as u32);
                                output[out_pos] = entry3.symbol_or_length;
                                out_pos += 1;
                            }
                        }
                    }
                }
            }

            DIST_END_OF_BLOCK => return Ok(out_pos),

            DIST_SLOW_PATH => {
                // Length code, need to decode distance separately
                let length = entry.symbol_or_length as usize + 3;

                let (dist_sym, dist_len) = dist_table.decode(bits.buffer());
                if dist_len == 0 || dist_sym >= 30 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance code",
                    ));
                }
                bits.consume(dist_len);

                bits.refill_branchless();
                let distance = DIST_START[dist_sym as usize] as usize
                    + bits.read(DIST_EXTRA_BITS[dist_sym as usize] as u32) as usize;

                if distance > out_pos || distance == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                out_pos = copy_match_fast(output, out_pos, distance, length);
            }

            distance => {
                // Fast path: length and distance both in entry
                let length = entry.length();
                let dist = distance as usize;

                if dist > out_pos || dist == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                out_pos = copy_match_fast(output, out_pos, dist, length);
            }
        }
    }

    // Fall back to generic loop near buffer ends
    decode_huffman_generic(bits, output, out_pos, lit_len_table, dist_table)
}

/// Generic decode loop for near buffer boundaries
/// This is slower but safer - handles all edge cases
fn decode_huffman_generic(
    bits: &mut FastBitsOptimized,
    output: &mut [u8],
    mut out_pos: usize,
    lit_len_table: &TwoLevelTable,
    dist_table: &TwoLevelTable,
) -> io::Result<usize> {
    loop {
        bits.refill();

        let (symbol, code_len) = lit_len_table.decode(bits.buffer());
        if code_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code",
            ));
        }
        bits.consume(code_len);

        if symbol < 256 {
            if out_pos >= output.len() {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "Output buffer full",
                ));
            }
            output[out_pos] = symbol as u8;
            out_pos += 1;
            continue;
        }

        if symbol == 256 {
            return Ok(out_pos);
        }

        // Length code
        let len_idx = (symbol - 257) as usize;
        if len_idx >= 29 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length code",
            ));
        }

        bits.refill();
        let length =
            LEN_START[len_idx] as usize + bits.read(LEN_EXTRA_BITS[len_idx] as u32) as usize;

        let (dist_sym, dist_len) = dist_table.decode(bits.buffer());
        if dist_len == 0 || dist_sym >= 30 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }
        bits.consume(dist_len);

        bits.refill();
        let distance = DIST_START[dist_sym as usize] as usize
            + bits.read(DIST_EXTRA_BITS[dist_sym as usize] as u32) as usize;

        if distance > out_pos || distance == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }

        // Safe copy for generic loop
        if out_pos + length > output.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "Output buffer full",
            ));
        }

        out_pos = copy_match_safe(output, out_pos, distance, length);
    }
}

/// Safe copy for generic loop (with full bounds checking)
#[inline]
fn copy_match_safe(output: &mut [u8], mut out_pos: usize, distance: usize, length: usize) -> usize {
    let src_start = out_pos - distance;

    if distance >= length {
        output.copy_within(src_start..src_start + length, out_pos);
        out_pos + length
    } else {
        for i in 0..length {
            output[out_pos] = output[src_start + (i % distance)];
            out_pos += 1;
        }
        out_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_match_fast_distance_1() {
        let mut output = vec![b'A', 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let new_pos = copy_match_fast(&mut output, 1, 1, 9);
        assert_eq!(new_pos, 10);
        assert_eq!(output, b"AAAAAAAAAA");
    }

    #[test]
    fn test_copy_match_fast_non_overlapping() {
        let mut output = vec![1, 2, 3, 4, 0, 0, 0, 0];
        let new_pos = copy_match_fast(&mut output, 4, 4, 4);
        assert_eq!(new_pos, 8);
        assert_eq!(output, vec![1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn test_fastbits_branchless_refill() {
        let data: Vec<u8> = (0..100).collect();
        let mut bits = FastBitsOptimized::new(&data);

        // Should have at least 56 bits after initial refill
        assert!(bits.bits_available() >= 56);

        // Read some bits
        let val = bits.read(32);
        assert!(val > 0);

        // Refill should restore bits
        bits.refill_branchless();
        assert!(bits.bits_available() >= 32);
    }
}
