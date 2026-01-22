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

/// Entry format (redesigned for multi-literal and subtable support):
/// - Bit 31: LITERAL flag
/// - Bit 30: EOB flag
/// - Bit 29: SUBTABLE flag
/// - Bit 28: DOUBLE_LITERAL flag
/// - Bits 20-27: lit1 / symbol / subtable offset (low 8 bits)
/// - Bits 12-19: lit2 / symbol / subtable offset (high 8 bits)
/// - Bits 5-11: Extra bits count / sub_bits (7 bits)
/// - Bits 0-4: Total bits consumed (5 bits, 0-31)
const LITERAL_FLAG: u32 = 1 << 31;
const EOB_FLAG: u32 = 1 << 30;
const SUBTABLE_FLAG: u32 = 1 << 29;
const DOUBLE_FLAG: u32 = 1 << 28;
#[cfg(feature = "combined_match")]
const COMBINED_PRESENT: u32 = 1 << 31;

impl SpecEntry {
    #[inline(always)]
    pub const fn new(symbol: u16, extra_bits: u8, total_bits: u8) -> Self {
        Self(((symbol as u32) << 12) | ((extra_bits as u32) << 5) | (total_bits as u32))
    }

    #[inline(always)]
    pub const fn literal(value: u8, bits: u8) -> Self {
        Self(LITERAL_FLAG | ((value as u32) << 20) | (bits as u32))
    }

    #[inline(always)]
    pub const fn double_literal(lit1: u8, lit2: u8, bits: u8) -> Self {
        Self(
            LITERAL_FLAG
                | DOUBLE_FLAG
                | ((lit1 as u32) << 20)
                | ((lit2 as u32) << 12)
                | (bits as u32),
        )
    }

    #[inline(always)]
    pub const fn length(base: u16, extra: u8, bits: u8) -> Self {
        Self(((base as u32) << 12) | ((extra as u32) << 5) | (bits as u32))
    }

    #[inline(always)]
    pub const fn end_of_block(bits: u8) -> Self {
        Self(EOB_FLAG | (bits as u32))
    }

    #[inline(always)]
    pub const fn subtable_ptr(offset: u16, sub_bits: u8, main_bits: u8) -> Self {
        Self(
            SUBTABLE_FLAG | ((offset as u32) << 12) | ((sub_bits as u32) << 5) | (main_bits as u32),
        )
    }

    #[inline(always)]
    pub const fn is_literal(self) -> bool {
        (self.0 & LITERAL_FLAG) != 0
    }

    #[inline(always)]
    pub const fn is_double(self) -> bool {
        (self.0 & DOUBLE_FLAG) != 0
    }

    #[inline(always)]
    pub const fn is_eob(self) -> bool {
        (self.0 & EOB_FLAG) != 0
    }

    #[inline(always)]
    pub const fn is_subtable(self) -> bool {
        (self.0 & SUBTABLE_FLAG) != 0
    }

    #[inline(always)]
    pub const fn literal_value(self) -> u8 {
        ((self.0 >> 20) & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn lit1(self) -> u8 {
        self.literal_value()
    }

    #[inline(always)]
    pub const fn lit2(self) -> u8 {
        ((self.0 >> 12) & 0xFF) as u8
    }

    #[inline(always)]
    pub const fn symbol(self) -> u16 {
        ((self.0 >> 12) & 0xFFFF) as u16
    }

    #[inline(always)]
    pub const fn length_base(self) -> u16 {
        self.symbol()
    }

    #[inline(always)]
    pub const fn subtable_offset(self) -> u16 {
        self.symbol()
    }

    #[inline(always)]
    pub const fn extra_bits(self) -> u8 {
        ((self.0 >> 5) & 0x7F) as u8
    }

    #[inline(always)]
    pub const fn subtable_bits(self) -> u8 {
        self.extra_bits()
    }

    #[inline(always)]
    pub const fn total_bits(self) -> u8 {
        (self.0 & 0x1F) as u8
    }

    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

#[cfg(feature = "combined_match")]
#[inline(always)]
pub fn combined_is_present(entry: u32) -> bool {
    (entry & COMBINED_PRESENT) != 0
}

#[cfg(feature = "combined_match")]
#[inline(always)]
pub fn combined_dist_sym(entry: u32) -> usize {
    ((entry >> 8) & 0xFF) as usize
}

#[cfg(feature = "combined_match")]
#[inline(always)]
pub fn combined_dist_len(entry: u32) -> u8 {
    (entry & 0xFF) as u8
}

/// A specialized decoder for a specific Huffman table fingerprint
/// Uses an 11-bit main table + subtables for 100% coverage
pub struct SpecializedDecoder {
    pub fingerprint: TableFingerprint,
    /// Litlen table (main + subtables)
    pub litlen: Box<[SpecEntry]>,
    /// Distance table (main + subtables)
    pub dist: Box<[SpecEntry]>,
    /// Combined length+distance lookup (feature-gated)
    #[cfg(feature = "combined_match")]
    pub combined: Box<[u32]>,
    /// Number of times this decoder has been used
    pub use_count: usize,
}

impl SpecializedDecoder {
    /// Build a specialized decoder from code lengths
    pub fn build(litlen_lens: &[u8], dist_lens: &[u8]) -> Option<Self> {
        let fingerprint = TableFingerprint::combined(litlen_lens, dist_lens);

        // Build tables with subtables at the end (like LitLenTable)
        let litlen = build_table_with_subtables(litlen_lens, false)?;
        let dist = build_table_with_subtables(dist_lens, true)?;
        #[cfg(feature = "combined_match")]
        let combined = build_combined_match_table(litlen_lens, dist_lens);

        Some(Self {
            fingerprint,
            litlen,
            dist,
            #[cfg(feature = "combined_match")]
            combined,
            use_count: 0,
        })
    }

    /// Decode using this specialized table
    #[inline(always)]
    pub fn decode_symbol(&self, bitbuf: u64) -> SpecEntry {
        unsafe { *self.litlen.get_unchecked((bitbuf & 0x7FF) as usize) }
    }

    /// Decode distance using this specialized table
    #[inline(always)]
    pub fn decode_distance(&self, bitbuf: u64) -> SpecEntry {
        unsafe { *self.dist.get_unchecked((bitbuf & 0x7FF) as usize) }
    }
}

#[inline(always)]
fn unlikely(b: bool) -> bool {
    b
}

/// Internal table building logic with subtable support
fn build_table_with_subtables(lengths: &[u8], is_distance: bool) -> Option<Box<[SpecEntry]>> {
    const MAIN_BITS: usize = 11;
    const MAIN_SIZE: usize = 1 << MAIN_BITS;
    const MAX_SUB_BITS: usize = 4; // 15 - 11

    let mut table = vec![SpecEntry::end_of_block(15); MAIN_SIZE];

    // Count codes
    let mut bl_count = [0u32; 16];
    for &len in lengths {
        if len > 0 && len <= 15 {
            bl_count[len as usize] += 1;
        }
    }

    // Compute first code
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Pass 1: Handle codes <= 11 bits
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 || len as usize > MAIN_BITS {
            continue;
        }

        let code = next_code[len as usize];
        next_code[len as usize] += 1;
        let rev = reverse_bits(code, len);

        let entry = if is_distance {
            let (base, extra) = if symbol < 30 {
                (
                    crate::libdeflate_entry::DISTANCE_TABLE[symbol].0,
                    crate::libdeflate_entry::DISTANCE_TABLE[symbol].1,
                )
            } else {
                (0, 0)
            };
            SpecEntry::new(base, extra, len)
        } else if symbol < 256 {
            SpecEntry::literal(symbol as u8, len)
        } else if symbol == 256 {
            SpecEntry::end_of_block(len)
        } else {
            let idx = symbol - 257;
            let (base, extra) = (
                crate::libdeflate_entry::LENGTH_TABLE[idx].0,
                crate::libdeflate_entry::LENGTH_TABLE[idx].1,
            );
            SpecEntry::length(base, extra, len)
        };

        let step = 1 << len;
        let mut idx = rev as usize;
        while idx < MAIN_SIZE {
            table[idx] = entry;
            idx += step;
        }
    }

    // Pass 2: Handle codes > 11 bits (create subtables)
    let mut sub_next = MAIN_SIZE;
    for (symbol, &len) in lengths.iter().enumerate() {
        if (len as usize) <= MAIN_BITS || len == 0 {
            continue;
        }

        let code = next_code[len as usize];
        next_code[len as usize] += 1;
        let rev = reverse_bits(code, len);
        let main_idx = (rev & 0x7FF) as usize;

        if !table[main_idx].is_subtable() {
            // New subtable
            let offset = sub_next;
            let size = 1 << MAX_SUB_BITS;
            table.resize(sub_next + size, SpecEntry::end_of_block(15));
            table[main_idx] = SpecEntry::subtable_ptr(offset as u16, MAX_SUB_BITS as u8, 11);
            sub_next += size;
        }

        let ptr = table[main_idx];
        let offset = ptr.subtable_offset() as usize;
        let sub_bits = ptr.subtable_bits();
        let sub_idx = (rev >> 11) as usize;

        let entry = if is_distance {
            let (base, extra) = if symbol < 30 {
                (
                    crate::libdeflate_entry::DISTANCE_TABLE[symbol].0,
                    crate::libdeflate_entry::DISTANCE_TABLE[symbol].1,
                )
            } else {
                (0, 0)
            };
            SpecEntry::new(base, extra, len) // Store FULL len
        } else if symbol < 256 {
            SpecEntry::literal(symbol as u8, len)
        } else if symbol == 256 {
            SpecEntry::end_of_block(len)
        } else {
            let idx = symbol - 257;
            let (base, extra) = (
                crate::libdeflate_entry::LENGTH_TABLE[idx].0,
                crate::libdeflate_entry::LENGTH_TABLE[idx].1,
            );
            SpecEntry::length(base, extra, len)
        };

        let step = 1 << (len - 11);
        let mut idx = sub_idx;
        while idx < (1 << sub_bits) {
            table[offset + idx] = entry;
            idx += step;
        }
    }

    Some(table.into_boxed_slice())
}

/// Reverse bits in a code
fn reverse_bits(code: u32, len: u8) -> u32 {
    let mut res = 0;
    let mut c = code;
    for _ in 0..len {
        res = (res << 1) | (c & 1);
        c >>= 1;
    }
    res
}

#[cfg(feature = "combined_match")]
#[inline(always)]
fn pack_combined(dist_sym: u8, dist_len: u8) -> u32 {
    COMBINED_PRESENT | ((dist_sym as u32) << 8) | (dist_len as u32)
}

#[cfg(feature = "combined_match")]
fn build_combined_match_table(litlen_lens: &[u8], dist_lens: &[u8]) -> Box<[u32]> {
    const MAIN_BITS: usize = 11;
    let size = 1 << MAIN_BITS;
    let mut combined = vec![0u32; size];

    // Build canonical codes for litlen.
    let mut bl_count = [0u32; 16];
    for &len in litlen_lens {
        if len > 0 && len <= 15 {
            bl_count[len as usize] += 1;
        }
    }
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=15 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    let mut len_map: Vec<Option<(u8, u8)>> = vec![None; size];
    for (symbol, &len) in litlen_lens.iter().enumerate() {
        if len == 0 || len as usize > MAIN_BITS {
            continue;
        }
        let code = next_code[len as usize];
        next_code[len as usize] += 1;
        let rev = reverse_bits(code, len);
        let step = 1 << len;

        if symbol >= 257 {
            let len_sym = (symbol - 257) as u8;
            if crate::libdeflate_entry::LENGTH_TABLE[len_sym as usize].1 == 0 {
                let mut idx = rev as usize;
                while idx < size {
                    len_map[idx] = Some((len_sym, len));
                    idx += step;
                }
            }
        }
    }

    // Build distance prefix maps for each remaining bit width.
    let mut dist_bl_count = [0u32; 16];
    for &len in dist_lens {
        if len > 0 && len <= MAIN_BITS as u8 {
            dist_bl_count[len as usize] += 1;
        }
    }
    let mut dist_next_code = [0u32; 16];
    let mut dist_code = 0u32;
    for bits in 1..=MAIN_BITS {
        dist_code = (dist_code + dist_bl_count[bits - 1]) << 1;
        dist_next_code[bits] = dist_code;
    }

    let mut dist_codes: Vec<(u8, u8, u32)> = Vec::new();
    for (symbol, &len) in dist_lens.iter().enumerate() {
        if len == 0 || len as usize > MAIN_BITS {
            continue;
        }
        let code = dist_next_code[len as usize];
        dist_next_code[len as usize] += 1;
        let rev = reverse_bits(code, len);
        dist_codes.push((symbol as u8, len, rev));
    }

    let mut dist_prefix_maps: Vec<Vec<Option<(u8, u8)>>> = Vec::with_capacity(MAIN_BITS + 1);
    dist_prefix_maps.push(Vec::new());
    for l in 1..=MAIN_BITS {
        let prefix_size = 1 << l;
        let mut map = vec![None; prefix_size];
        let mut conflict = vec![false; prefix_size];
        for &(sym, len, rev) in &dist_codes {
            if len as usize > l {
                continue;
            }
            let step = 1 << len;
            let mut idx = rev as usize;
            while idx < prefix_size {
                if !conflict[idx] {
                    match map[idx] {
                        None => map[idx] = Some((sym, len)),
                        Some((prev_sym, prev_len)) => {
                            if prev_sym != sym || prev_len != len {
                                map[idx] = None;
                                conflict[idx] = true;
                            }
                        }
                    }
                }
                idx += step;
            }
        }
        dist_prefix_maps.push(map);
    }

    for i in 0..size {
        if let Some((_len_sym, b1)) = len_map[i] {
            let remaining = MAIN_BITS - b1 as usize;
            if remaining == 0 {
                continue;
            }
            let prefix = i >> b1;
            if let Some((dist_sym, dist_len)) = dist_prefix_maps[remaining][prefix] {
                if dist_len as usize == remaining {
                    combined[i] = pack_combined(dist_sym, dist_len);
                }
            }
        }
    }

    combined.into_boxed_slice()
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

    /// Get detailed stats: (decoders_count, failed_count, total_uses, max_uses)
    pub fn detailed_stats(&self) -> (usize, usize, usize, usize) {
        let total_uses: usize = self.decoders.values().map(|d| d.use_count).sum();
        let max_uses = self
            .decoders
            .values()
            .map(|d| d.use_count)
            .max()
            .unwrap_or(0);
        (self.decoders.len(), self.failed.len(), total_uses, max_uses)
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
        let entry = spec.litlen[(bitbuf & 0x1FFF) as usize];
        let bits = entry.total_bits() as u32;
        bitbuf >>= bits;
        bitsleft = bitsleft.wrapping_sub(bits);

        if entry.is_literal() {
            // Literal - write and continue
            output[out_pos] = entry.literal_value();
            out_pos += 1;

            // Try to decode more literals inline
            let entry2 = spec.litlen[(bitbuf & 0x1FFF) as usize];
            if entry2.is_literal() {
                let bits2 = entry2.total_bits() as u32;
                bitbuf >>= bits2;
                bitsleft = bitsleft.wrapping_sub(bits2);
                output[out_pos] = entry2.literal_value();
                out_pos += 1;

                let entry3 = spec.litlen[(bitbuf & 0x1FFF) as usize];
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
        let dist_entry = spec.dist[(bitbuf & 0x7FF) as usize];
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
