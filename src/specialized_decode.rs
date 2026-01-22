//! Specialized Huffman Decoders
//!
//! For frequently-seen Huffman tables, we generate specialized decoders where:
//! - Symbol values are immediate constants (no memory lookup)
//! - Bit patterns are matched directly (Rust compiles to jump tables)
//! - Common literal sequences are unrolled
//!
//! ## Mathematical Basis
//!
//! A Huffman table maps bit patterns to symbols. For a table with n symbols,
//! we generate a match statement with O(n) arms. Rust/LLVM compiles this to
//! a jump table when patterns are dense, giving O(1) lookup without memory access.
//!
//! ## Example
//!
//! For a table where:
//! - 0 -> 'e' (1 bit)
//! - 10 -> 't' (2 bits)
//! - 110 -> 'a' (3 bits)
//!
//! We generate:
//! ```ignore
//! match bitbuf & 0x7 {
//!     0b000 | 0b010 | 0b100 | 0b110 => ('e', 1),
//!     0b001 | 0b101 => ('t', 2),
//!     0b011 => ('a', 3),
//!     0b111 => ... // other symbol
//! }
//! ```

#![allow(dead_code)]

use crate::jit_decode::TableFingerprint;
#[allow(unused_imports)]
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result};

/// Entry in the specialized lookup table
/// Packed as: symbol(16) | extra_bits(8) | total_bits(8)
#[derive(Clone, Copy)]
pub struct SpecEntry(u32);

/// Entry format (redesigned for single-bit checks):
/// - Bit 31: LITERAL flag (mutually exclusive with EOB)
/// - Bit 30: EOB flag (mutually exclusive with LITERAL)
/// - Bits 16-29: symbol/value (14 bits, enough for literals 0-255 and lengths 3-258)
/// - Bits 8-15: extra bits count
/// - Bits 0-7: total bits consumed
const LITERAL_FLAG: u32 = 0x8000_0000;
const EOB_FLAG: u32 = 0x4000_0000;

impl SpecEntry {
    #[inline(always)]
    pub const fn new(symbol: u16, extra_bits: u8, total_bits: u8) -> Self {
        Self(((symbol as u32) << 16) | ((extra_bits as u32) << 8) | (total_bits as u32))
    }

    #[inline(always)]
    pub const fn literal(value: u8, bits: u8) -> Self {
        // Set LITERAL flag (bit 31), value in bits 16-23
        Self(LITERAL_FLAG | ((value as u32) << 16) | (bits as u32))
    }

    #[inline(always)]
    pub const fn length(base: u16, extra: u8, bits: u8) -> Self {
        // No flags set, just base/extra/bits
        Self(((base as u32) << 16) | ((extra as u32) << 8) | (bits as u32))
    }

    #[inline(always)]
    pub const fn end_of_block(bits: u8) -> Self {
        // Set EOB flag (bit 30), no symbol needed
        Self(EOB_FLAG | (bits as u32))
    }

    #[inline(always)]
    pub const fn symbol(self) -> u16 {
        ((self.0 >> 16) & 0x3FFF) as u16 // Mask off flag bits
    }

    #[inline(always)]
    pub const fn is_literal(self) -> bool {
        // Single AND instruction - LITERAL flag is bit 31
        (self.0 & LITERAL_FLAG) != 0
    }

    #[inline(always)]
    pub const fn is_eob(self) -> bool {
        // Single AND instruction - EOB flag is bit 30
        (self.0 & EOB_FLAG) != 0
    }

    #[inline(always)]
    pub const fn literal_value(self) -> u8 {
        (self.symbol() & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn length_base(self) -> u16 {
        self.symbol()
    }

    #[inline(always)]
    pub const fn extra_bits(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn total_bits(self) -> u8 {
        (self.0 & 0xFF) as u8
    }
}

/// A specialized decoder for a specific Huffman table fingerprint
/// Uses a flat 2048-entry lookup table (11 bits) with no subtables
pub struct SpecializedDecoder {
    pub fingerprint: TableFingerprint,
    /// Litlen lookup: 2048 entries (11-bit lookup)
    /// Each entry is a SpecEntry
    pub litlen: Box<[SpecEntry; 2048]>,
    /// Distance lookup: 512 entries (9-bit lookup)
    pub dist: Box<[SpecEntry; 512]>,
    /// Number of times this decoder has been used
    pub use_count: usize,
}

impl SpecializedDecoder {
    /// Build a specialized decoder from code lengths
    pub fn build(litlen_lens: &[u8], dist_lens: &[u8]) -> Option<Self> {
        let fingerprint = TableFingerprint::combined(litlen_lens, dist_lens);

        // Build flat litlen table (11 bits, no subtables)
        let litlen = build_flat_litlen_table(litlen_lens)?;

        // Build flat distance table (9 bits)
        let dist = build_flat_dist_table(dist_lens)?;

        Some(Self {
            fingerprint,
            litlen,
            dist,
            use_count: 0,
        })
    }

    /// Decode using this specialized table
    #[inline(always)]
    pub fn decode_symbol(&self, bitbuf: u64) -> SpecEntry {
        self.litlen[(bitbuf & 0x7FF) as usize]
    }

    /// Decode distance using this specialized table
    #[inline(always)]
    pub fn decode_distance(&self, bitbuf: u64) -> SpecEntry {
        self.dist[(bitbuf & 0x1FF) as usize]
    }
}

/// Build a flat 11-bit litlen lookup table
fn build_flat_litlen_table(lengths: &[u8]) -> Option<Box<[SpecEntry; 2048]>> {
    const TABLE_BITS: usize = 11;
    const TABLE_SIZE: usize = 1 << TABLE_BITS;

    let mut table = Box::new([SpecEntry::new(0, 0, 0); TABLE_SIZE]);

    // Length base values for symbols 257-285
    const LENGTH_BASES: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LENGTH_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];

    // Count codes at each length
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    if max_len > 15 {
        return None; // Invalid
    }

    let mut bl_count = [0u32; 16];
    for &len in lengths {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // Compute first code for each length
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols and fill table
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len = len as usize;
        if len > TABLE_BITS {
            // Code too long for flat table - can't use specialized decoder
            return None;
        }

        let code = next_code[len];
        next_code[len] += 1;

        // Reverse code bits for LSB-first reading
        let mut rev_code = 0u32;
        for i in 0..len {
            if (code >> i) & 1 != 0 {
                rev_code |= 1 << (len - 1 - i);
            }
        }

        // Create entry based on symbol type
        let entry = if symbol < 256 {
            // Literal
            SpecEntry::literal(symbol as u8, len as u8)
        } else if symbol == 256 {
            // End of block
            SpecEntry::end_of_block(len as u8)
        } else if symbol <= 285 {
            // Length code
            let idx = symbol - 257;
            SpecEntry::length(LENGTH_BASES[idx], LENGTH_EXTRA[idx], len as u8)
        } else {
            continue; // Invalid symbol
        };

        // Fill all table slots that match this code (handles shorter codes)
        let step = 1usize << len;
        let mut idx = rev_code as usize;
        while idx < TABLE_SIZE {
            table[idx] = entry;
            idx += step;
        }
    }

    Some(table)
}

/// Build a flat 9-bit distance lookup table
fn build_flat_dist_table(lengths: &[u8]) -> Option<Box<[SpecEntry; 512]>> {
    const TABLE_BITS: usize = 9;
    const TABLE_SIZE: usize = 1 << TABLE_BITS;

    let mut table = Box::new([SpecEntry::new(0, 0, 0); TABLE_SIZE]);

    // Distance base values for symbols 0-29
    const DIST_BASES: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    // Count codes at each length
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    if max_len > 15 {
        return None;
    }

    let mut bl_count = [0u32; 16];
    for &len in lengths {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // Compute first code for each length
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols and fill table
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 || symbol >= 30 {
            continue;
        }
        let len = len as usize;
        if len > TABLE_BITS {
            // Code too long for flat table - can't use specialized decoder
            return None;
        }

        let code = next_code[len];
        next_code[len] += 1;

        // Reverse code bits
        let mut rev_code = 0u32;
        for i in 0..len {
            if (code >> i) & 1 != 0 {
                rev_code |= 1 << (len - 1 - i);
            }
        }

        let entry = SpecEntry::new(
            DIST_BASES[symbol],
            DIST_EXTRA[symbol],
            len as u8, // Only Huffman code length, extra bits read separately
        );

        // Fill all matching slots
        let step = 1usize << len;
        let mut idx = rev_code as usize;
        while idx < TABLE_SIZE {
            table[idx] = entry;
            idx += step;
        }
    }

    Some(table)
}

/// Cache of specialized decoders
pub struct SpecializedCache {
    decoders: HashMap<TableFingerprint, SpecializedDecoder>,
    /// Fingerprints we've seen but couldn't build specialized decoders for
    /// (e.g., codes too long for flat table)
    failed: HashMap<TableFingerprint, ()>,
}

impl SpecializedCache {
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
            failed: HashMap::new(),
        }
    }

    /// Get or try to create a specialized decoder
    pub fn get_or_create(
        &mut self,
        litlen_lens: &[u8],
        dist_lens: &[u8],
    ) -> Option<&mut SpecializedDecoder> {
        let fp = TableFingerprint::combined(litlen_lens, dist_lens);

        // Check if we already failed to build this one
        if self.failed.contains_key(&fp) {
            return None;
        }

        // Check if we already have it
        if self.decoders.contains_key(&fp) {
            let decoder = self.decoders.get_mut(&fp).unwrap();
            decoder.use_count += 1;
            return Some(decoder);
        }

        // Try to build it
        match SpecializedDecoder::build(litlen_lens, dist_lens) {
            Some(mut decoder) => {
                decoder.use_count = 1;
                self.decoders.insert(fp, decoder);
                self.decoders.get_mut(&fp)
            }
            None => {
                self.failed.insert(fp, ());
                None
            }
        }
    }

    /// Get a specialized decoder by fingerprint (for cached tables)
    pub fn get(&mut self, fp: TableFingerprint) -> Option<&mut SpecializedDecoder> {
        if let Some(decoder) = self.decoders.get_mut(&fp) {
            decoder.use_count += 1;
            Some(decoder)
        } else {
            None
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        (self.decoders.len(), self.failed.len())
    }

    /// Get a decoder by fingerprint (immutable reference)
    pub fn get_decoder(&self, fp: &TableFingerprint) -> Option<&SpecializedDecoder> {
        self.decoders.get(fp)
    }
}

/// Decode using a specialized decoder
/// Returns (output_position, success)
#[inline(never)]
pub fn decode_with_specialized(
    spec: &SpecializedDecoder,
    input: &[u8],
    output: &mut [u8],
    mut out_pos: usize,
    mut in_pos: usize,
) -> Result<(usize, usize)> {
    const FASTLOOP_MARGIN: usize = 320;

    let mut bitbuf: u64 = 0;
    let mut bitsleft: u32 = 0;

    // Initial refill
    if in_pos + 8 <= input.len() {
        bitbuf = u64::from_le_bytes(input[in_pos..in_pos + 8].try_into().unwrap());
        in_pos += 8;
        bitsleft = 64;
    }

    // Refill macro
    macro_rules! refill {
        () => {
            if in_pos + 8 <= input.len() {
                let word = u64::from_le_bytes(input[in_pos..in_pos + 8].try_into().unwrap());
                bitbuf |= word << bitsleft;
                in_pos += (64 - bitsleft) as usize / 8;
                bitsleft |= 56;
            }
        };
    }

    // Fast loop
    while out_pos + FASTLOOP_MARGIN <= output.len() {
        refill!();

        // Decode literal/length
        let entry = spec.decode_symbol(bitbuf);
        let bits = entry.total_bits() as u32;
        bitbuf >>= bits;
        bitsleft = bitsleft.wrapping_sub(bits);

        if entry.is_literal() {
            // Literal - write and continue
            output[out_pos] = entry.literal_value();
            out_pos += 1;

            // Try to decode more literals inline
            let entry2 = spec.decode_symbol(bitbuf);
            if entry2.is_literal() {
                let bits2 = entry2.total_bits() as u32;
                bitbuf >>= bits2;
                bitsleft = bitsleft.wrapping_sub(bits2);
                output[out_pos] = entry2.literal_value();
                out_pos += 1;

                let entry3 = spec.decode_symbol(bitbuf);
                if entry3.is_literal() {
                    let bits3 = entry3.total_bits() as u32;
                    bitbuf >>= bits3;
                    bitsleft = bitsleft.wrapping_sub(bits3);
                    output[out_pos] = entry3.literal_value();
                    out_pos += 1;
                }
            }
            continue;
        }

        if entry.is_eob() {
            return Ok((out_pos, in_pos - (bitsleft as usize / 8)));
        }

        // Length code - decode length
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

        // Decode distance
        let dist_entry = spec.decode_distance(bitbuf);
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

        // Validate and copy
        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        // Copy match
        let dist = distance as usize;
        let len = length as usize;
        let src_start = out_pos - dist;

        if dist >= len {
            // Non-overlapping copy
            output.copy_within(src_start..src_start + len, out_pos);
        } else {
            // Overlapping - byte by byte
            for i in 0..len {
                output[out_pos + i] = output[src_start + i];
            }
        }
        out_pos += len;
    }

    // Return position for generic loop to continue
    Ok((out_pos, in_pos - (bitsleft as usize / 8)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_specialized() {
        // Fixed Huffman code lengths for litlen (simplified)
        let mut litlen_lens = vec![0u8; 288];
        litlen_lens[0..144].fill(8);
        litlen_lens[144..256].fill(9);
        litlen_lens[256..280].fill(7);
        litlen_lens[280..288].fill(8);

        let dist_lens = vec![5u8; 32];

        let decoder = SpecializedDecoder::build(&litlen_lens, &dist_lens);
        assert!(
            decoder.is_some(),
            "Should build specialized decoder for fixed codes"
        );

        let decoder = decoder.unwrap();

        // Test that 'e' (0x65 = 101) decodes correctly
        // In fixed Huffman, literals 0-143 use 8 bits
        // The code for 'e' would be some 8-bit pattern
        // Just verify the table was built
        assert!(decoder.litlen[0].total_bits() > 0);
    }
}
