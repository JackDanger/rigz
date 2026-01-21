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

use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::io::{Error, ErrorKind, Result};

// =============================================================================
// BMI2 Optimizations (x86_64 only)
// =============================================================================

/// Extract low n bits from a value using BMI2 if available
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn extract_bits(value: u64, n: u32) -> u64 {
    unsafe { core::arch::x86_64::_bzhi_u64(value, n) }
}

#[allow(dead_code)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn extract_bits(value: u64, n: u32) -> u64 {
    value & ((1u64 << n) - 1)
}

// =============================================================================
// Bit Reader - Matching libdeflate exactly
// =============================================================================

/// Bit buffer matching libdeflate's structure
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u64,
    bitsleft: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
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
    fn refill(&mut self) {
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
    fn refill_slow(&mut self) {
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
    fn peek(&self) -> u64 {
        self.bitbuf
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.bitbuf >>= n as u8;
        self.bitsleft -= n;
    }

    /// Consume using entry's low 5 bits
    #[inline(always)]
    fn consume_entry(&mut self, entry: u32) {
        self.bitbuf >>= entry as u8;
        self.bitsleft = self.bitsleft.wrapping_sub(entry & 0x1F);
    }

    /// Available bits (low 8 bits only - matching libdeflate's (u8)bitsleft pattern)
    #[inline(always)]
    fn available(&self) -> u32 {
        // libdeflate allows garbage in high bits, so cast to u8 for the real value
        (self.bitsleft as u8) as u32
    }

    fn align_to_byte(&mut self) {
        // bitsleft may have garbage in high bits, use (u8) cast
        let discard = (self.bitsleft as u8) & 7;
        self.consume(discard as u32);
    }

    fn read_u16(&mut self) -> u16 {
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

        if dist >= 8 {
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
            // RLE path: broadcast byte and copy words (auto-vectorizes)
            let v = 0x0101010101010101u64 * (*src as u64);
            (dst as *mut u64).write_unaligned(v);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned(v);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned(v);
            dst = dst.add(8);
            (dst as *mut u64).write_unaligned(v);
            dst = dst.add(8);
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

fn decode_fixed(bits: &mut Bits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    let tables = crate::libdeflate_decode::get_fixed_tables();
    // USE the DoubleLitCache for fixed blocks - decodes 2 literals per lookup
    let double_cache = get_fixed_double_lit_cache();
    decode_huffman_cf_double(bits, output, out_pos, &tables.0, &tables.1, double_cache)
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

    let litlen_table = LitLenTable::build(litlen_lengths)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid litlen table"))?;
    let dist_table = DistTable::build(dist_lengths)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid dist table"))?;

    decode_huffman_cf(bits, output, out_pos, &litlen_table, &dist_table)
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
    let mut out_pos = 0;
    #[allow(unused_variables)]
    let mut block_count = 0;

    loop {
        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);
        block_count += 1;

        match btype {
            0 => out_pos = decode_stored(&mut bits, output, out_pos)?,
            1 => out_pos = decode_fixed(&mut bits, output, out_pos)?,
            2 => out_pos = decode_dynamic(&mut bits, output, out_pos)?,
            3 => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type")),
            _ => unreachable!(),
        }

        if bfinal {
            return Ok(out_pos);
        }
    }
}

// =============================================================================
// Tests - Using REAL silesia data, not simulations
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

    /// REAL benchmark on silesia - the only valid benchmark
    #[test]
    fn bench_cf_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping silesia benchmark - file not found");
                return;
            }
        };

        // Parse gzip header properly
        let mut pos = 10; // Skip magic, method, flags, mtime, xfl, os
        let flg = gz[3];
        if (flg & 0x04) != 0 {
            // FEXTRA
            let xlen = u16::from_le_bytes([gz[pos], gz[pos + 1]]) as usize;
            pos += 2 + xlen;
        }
        if (flg & 0x08) != 0 {
            // FNAME
            while pos < gz.len() && gz[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if (flg & 0x10) != 0 {
            // FCOMMENT
            while pos < gz.len() && gz[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if (flg & 0x02) != 0 {
            // FHCRC
            pos += 2;
        }

        let start = pos;
        let end = gz.len() - 8;
        let deflate = &gz[start..end];
        let isize = u32::from_le_bytes([
            gz[gz.len() - 4],
            gz[gz.len() - 3],
            gz[gz.len() - 2],
            gz[gz.len() - 1],
        ]) as usize;

        eprintln!(
            "Deflate stream: start={} end={} len={} isize={}",
            start,
            end,
            deflate.len(),
            isize
        );
        eprintln!(
            "First 20 deflate bytes: {:02x?}",
            &deflate[..20.min(deflate.len())]
        );

        let mut output = vec![0u8; isize + 1000];
        let mut lib_output = vec![0u8; isize + 1000];

        // Verify libdeflate can decode it first
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(deflate, &mut lib_output)
            .expect("libdeflate failed");
        eprintln!("libdeflate decoded {} bytes successfully", lib_size);
        eprintln!(
            "First 20 output bytes: {:02x?}",
            &lib_output[..20.min(lib_size)]
        );

        // Now try our decoder
        let our_result = inflate_consume_first(deflate, &mut output);

        if let Err(e) = &our_result {
            eprintln!("Error: {:?}", e);
            // Check how many bytes match
            let check_len = lib_size.min(output.len());
            let mut first_mismatch = None;
            for i in 0..check_len {
                if output[i] != lib_output[i] {
                    first_mismatch = Some(i);
                    eprintln!(
                        "First mismatch at byte {}: got {:02x} expected {:02x}",
                        i, output[i], lib_output[i]
                    );
                    let start = i.saturating_sub(5);
                    let end = (i + 10).min(check_len);
                    eprintln!("Our bytes around mismatch: {:02x?}", &output[start..end]);
                    eprintln!(
                        "Lib bytes around mismatch: {:02x?}",
                        &lib_output[start..end]
                    );
                    break;
                }
            }
            if let Some(pos) = first_mismatch {
                eprintln!("First {} bytes matched before mismatch", pos);
            } else {
                // Count how many non-zero bytes we decoded
                let decoded = output
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, &b)| b != 0)
                    .map(|(i, _)| i + 1)
                    .unwrap_or(0);
                eprintln!("Decoded {} bytes, all match!", decoded);
            }
        }

        let our_size = our_result.expect("Our decode failed");

        assert_eq!(our_size, lib_size, "Size mismatch");

        // Check first 10KB
        for i in 0..10000.min(our_size) {
            if output[i] != lib_output[i] {
                panic!(
                    "Mismatch at byte {}: got {} expected {}",
                    i, output[i], lib_output[i]
                );
            }
        }

        // Benchmark with more iterations for stability
        let iterations = 10;

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

        eprintln!("\n=== CONSUME-FIRST SILESIA ===");
        eprintln!("Data size: {} MB", isize / 1_000_000);
        eprintln!("Our throughput:       {:>8.1} MB/s", our_throughput);
        eprintln!("libdeflate throughput: {:>8.1} MB/s", lib_throughput);
        eprintln!("Ratio: {:.1}%", 100.0 * our_throughput / lib_throughput);
        eprintln!("=============================\n");
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
}
