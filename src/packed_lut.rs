//! Packed Lookup Table - libdeflate-style optimization
//!
//! This module implements the key libdeflate optimizations:
//! 1. Single u32 entry containing all decode info
//! 2. `bitsleft -= entry` instead of field extraction
//! 3. Bit testing instead of match statements
//!
//! Entry format (32 bits):
//! ```text
//! Literal:     [1][symbol:8][0000000][bits:8] (bit 31 = 1 means literal)
//! Length:      [0][length-3:8][dist_or_flag:15][bits:8] (bit 31 = 0)
//! EOB:         [0][00000000][111111111111111][bits:8] (dist = 0x7FFF)
//! SlowPath:    [0][length-3:8][111111111111110][bits:8] (dist = 0x7FFE)
//! ```
//!
//! The key insight: we test bit 31 first (literal), then bits 15-30 (dist_or_flag).

#![allow(dead_code)]

use std::io;

/// LUT size constants
pub const PACKED_LUT_BITS: usize = 12;
pub const PACKED_LUT_SIZE: usize = 1 << PACKED_LUT_BITS;
pub const PACKED_LUT_MASK: u64 = (PACKED_LUT_SIZE - 1) as u64;

/// Entry bit layout constants
const LITERAL_FLAG: u32 = 1 << 31;
const BITS_MASK: u32 = 0xFF;
const SYMBOL_SHIFT: u32 = 23;
const SYMBOL_MASK: u32 = 0xFF << SYMBOL_SHIFT;
const DIST_SHIFT: u32 = 8;
const DIST_MASK: u32 = 0x7FFF << DIST_SHIFT;

/// Special distance values (in the 15-bit field, shifted by DIST_SHIFT)
const DIST_EOB: u32 = 0x7FFF << DIST_SHIFT;
const DIST_SLOW: u32 = 0x7FFE << DIST_SHIFT;

/// Packed entry - fits in a register for fast operations
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct PackedEntry(pub u32);

impl PackedEntry {
    /// Create a literal entry
    /// Bit 31 set = literal, symbol in bits 23-30, bits to consume in bits 0-7
    #[inline(always)]
    pub const fn literal(bits: u8, symbol: u8) -> Self {
        Self(LITERAL_FLAG | ((symbol as u32) << SYMBOL_SHIFT) | (bits as u32))
    }

    /// Create an end-of-block entry
    #[inline(always)]
    pub const fn end_of_block(bits: u8) -> Self {
        Self(DIST_EOB | (bits as u32))
    }

    /// Create a slow-path length entry (distance decoded separately)
    #[inline(always)]
    pub const fn slow_path(bits: u8, length_minus_3: u8) -> Self {
        Self(DIST_SLOW | ((length_minus_3 as u32) << SYMBOL_SHIFT) | (bits as u32))
    }

    /// Create an LZ77 match entry with pre-computed distance
    #[inline(always)]
    pub const fn lz77(bits: u8, length_minus_3: u8, distance: u16) -> Self {
        // Distance is limited to 15 bits (max 32767), but deflate max is 32768
        // We use slow path for distance > 32767
        let dist_val = (distance as u32) << DIST_SHIFT;
        Self(dist_val | ((length_minus_3 as u32) << SYMBOL_SHIFT) | (bits as u32))
    }

    /// Get bits to consume (low 8 bits)
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.0 & BITS_MASK
    }

    /// Check if this is a literal (bit 31 set)
    #[inline(always)]
    pub const fn is_literal(self) -> bool {
        (self.0 & LITERAL_FLAG) != 0
    }

    /// Get literal symbol (assumes is_literal() is true)
    #[inline(always)]
    pub const fn symbol(self) -> u8 {
        ((self.0 & SYMBOL_MASK) >> SYMBOL_SHIFT) as u8
    }

    /// Check if this is end-of-block
    #[inline(always)]
    pub const fn is_eob(self) -> bool {
        (self.0 & DIST_MASK) == DIST_EOB
    }

    /// Check if this is slow-path (needs distance decode)
    #[inline(always)]
    pub const fn is_slow_path(self) -> bool {
        (self.0 & DIST_MASK) == DIST_SLOW
    }

    /// Get length (length_minus_3 + 3)
    #[inline(always)]
    pub const fn length(self) -> usize {
        (((self.0 & SYMBOL_MASK) >> SYMBOL_SHIFT) as usize) + 3
    }

    /// Get distance (assumes not literal, not EOB, not slow_path)
    #[inline(always)]
    pub const fn distance(self) -> usize {
        ((self.0 & DIST_MASK) >> DIST_SHIFT) as usize
    }

    /// Check if entry has valid bits (bits > 0)
    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        (self.0 & BITS_MASK) != 0
    }
}

/// Packed lookup table for fast decode
pub struct PackedLUT {
    pub table: Box<[PackedEntry; PACKED_LUT_SIZE]>,
}

impl PackedLUT {
    /// Build packed LUT from code lengths
    pub fn build(lit_len_lens: &[u8], _dist_lens: &[u8]) -> io::Result<Self> {
        // Allocate table
        let table_vec = vec![PackedEntry::default(); PACKED_LUT_SIZE];
        let table_ptr = Box::into_raw(table_vec.into_boxed_slice());
        let mut table = unsafe { Box::from_raw(table_ptr as *mut [PackedEntry; PACKED_LUT_SIZE]) };

        // Build Huffman codes
        let (codes, code_lens) = build_huffman_codes(lit_len_lens)?;

        for (symbol, &code_len) in code_lens.iter().enumerate() {
            if code_len == 0 || code_len > PACKED_LUT_BITS as u8 {
                continue;
            }

            let code = codes[symbol];
            let reversed_code = reverse_bits(code, code_len);

            if symbol < 256 {
                // Literal
                insert_entry(
                    &mut table,
                    reversed_code,
                    code_len,
                    PackedEntry::literal(code_len, symbol as u8),
                );
            } else if symbol == 256 {
                // End of block
                insert_entry(
                    &mut table,
                    reversed_code,
                    code_len,
                    PackedEntry::end_of_block(code_len),
                );
            } else if symbol <= 285 {
                // Length code
                use crate::inflate_tables::{LEN_EXTRA_BITS, LEN_START};

                let len_idx = symbol - 257;
                let len_extra = LEN_EXTRA_BITS[len_idx];
                let bits_for_length = code_len + len_extra;

                if bits_for_length <= PACKED_LUT_BITS as u8 {
                    // Enumerate length extra bits
                    let num_len_extras = 1u32 << len_extra;
                    for len_extra_val in 0..num_len_extras {
                        let length = LEN_START[len_idx] as usize + len_extra_val as usize;
                        let length_minus_3 = (length - 3) as u8;
                        let combined_code = reversed_code | ((len_extra_val as u16) << code_len);

                        insert_entry(
                            &mut table,
                            combined_code,
                            bits_for_length,
                            PackedEntry::slow_path(bits_for_length, length_minus_3),
                        );
                    }
                }
            }
        }

        Ok(Self { table })
    }

    /// Decode entry from bit buffer
    #[inline(always)]
    pub fn decode(&self, bits: u64) -> PackedEntry {
        self.table[(bits & PACKED_LUT_MASK) as usize]
    }
}

fn insert_entry(
    table: &mut [PackedEntry; PACKED_LUT_SIZE],
    reversed_code: u16,
    code_len: u8,
    entry: PackedEntry,
) {
    let filler_bits = PACKED_LUT_BITS as u8 - code_len;
    let num_slots = 1u32 << filler_bits;

    for i in 0..num_slots {
        let idx = (reversed_code as usize) | ((i as usize) << code_len);
        if idx < PACKED_LUT_SIZE {
            table[idx] = entry;
        }
    }
}

fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut result = 0u16;
    let mut code = code;
    for _ in 0..len {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

fn build_huffman_codes(lens: &[u8]) -> io::Result<(Vec<u16>, Vec<u8>)> {
    let max_code_len = *lens.iter().max().unwrap_or(&0) as usize;
    if max_code_len > 15 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Code too long"));
    }

    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..=max_code_len {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    let mut codes = vec![0u16; lens.len()];
    for (n, &len) in lens.iter().enumerate() {
        if len > 0 {
            codes[n] = next_code[len as usize];
            next_code[len as usize] += 1;
        }
    }

    Ok((codes, lens.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_entry_size() {
        assert_eq!(std::mem::size_of::<PackedEntry>(), 4);
    }

    #[test]
    fn test_packed_entry_literal() {
        let entry = PackedEntry::literal(8, b'A');
        assert!(entry.is_literal());
        assert!(!entry.is_eob());
        assert!(!entry.is_slow_path());
        assert_eq!(entry.bits(), 8);
        assert_eq!(entry.symbol(), b'A');
    }

    #[test]
    fn test_packed_entry_eob() {
        let entry = PackedEntry::end_of_block(7);
        assert!(!entry.is_literal());
        assert!(entry.is_eob());
        assert!(!entry.is_slow_path());
        assert_eq!(entry.bits(), 7);
    }

    #[test]
    fn test_packed_entry_slow_path() {
        let entry = PackedEntry::slow_path(10, 5); // length = 8
        assert!(!entry.is_literal());
        assert!(!entry.is_eob());
        assert!(entry.is_slow_path());
        assert_eq!(entry.bits(), 10);
        assert_eq!(entry.length(), 8);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_packed_lut_build() {
        // Build fixed Huffman tables (RFC 1951 fixed Huffman code lengths)
        let mut lit_len_lens = vec![0u8; 288];
        for i in 0..144 {
            lit_len_lens[i] = 8;
        }
        for i in 144..256 {
            lit_len_lens[i] = 9;
        }
        for i in 256..280 {
            lit_len_lens[i] = 7;
        }
        for i in 280..288 {
            lit_len_lens[i] = 8;
        }

        let dist_lens = vec![5u8; 32];
        let lut = PackedLUT::build(&lit_len_lens, &dist_lens).unwrap();

        // Count entry types
        let mut literals = 0;
        let mut eobs = 0;
        let mut slow_paths = 0;

        for entry in lut.table.iter() {
            if entry.is_valid() {
                if entry.is_literal() {
                    literals += 1;
                } else if entry.is_eob() {
                    eobs += 1;
                } else if entry.is_slow_path() {
                    slow_paths += 1;
                }
            }
        }

        eprintln!(
            "Literals: {}, EOBs: {}, SlowPaths: {}",
            literals, eobs, slow_paths
        );
        assert!(literals > 0);
        assert!(eobs > 0);
    }
}
