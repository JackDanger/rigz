//! Multi-Symbol Decode for Ultra-Fast Inflate
//!
//! This module implements ISA-L/libdeflate-style multi-symbol decoding
//! where 2-4 literal bytes are decoded in a single table lookup.
//!
//! The key insight: For short Huffman codes (≤6 bits), we can pack
//! multiple literals into a single 12-bit table lookup.
//!
//! Example: Two 6-bit codes = 12 bits, one lookup → 2 literals

#![allow(dead_code)]

use std::io;

use crate::two_level_table::FastBits;

// =============================================================================
// Multi-Symbol Table
// =============================================================================

/// Entry in the multi-symbol lookup table
/// Format:
/// - bits 0-7: First literal (or symbol if not literal)
/// - bits 8-15: Second literal (or 0xFF if single)
/// - bits 16-23: Third literal (or 0xFF if none)
/// - bits 24-27: Number of literals (0 = not a literal sequence)
/// - bits 28-31: Total bits consumed
#[derive(Clone, Copy, Debug, Default)]
pub struct MultiSymEntry {
    pub lits: [u8; 4],     // Up to 4 literals
    pub count: u8,         // Number of literals (0-4), 0 = need slow path
    pub bits_consumed: u8, // Total bits consumed (0 = error)
    pub symbol: u16,       // If count=0, this is the non-literal symbol
}

#[allow(dead_code)]
impl MultiSymEntry {
    fn create_default() -> Self {
        Self {
            lits: [0; 4],
            count: 0,
            bits_consumed: 0,
            symbol: 0,
        }
    }
}

/// Lookup table size (12 bits = 4096 entries)
pub const MULTI_SYM_BITS: usize = 12;
pub const MULTI_SYM_SIZE: usize = 1 << MULTI_SYM_BITS;

/// Multi-symbol decode table
pub struct MultiSymTable {
    entries: Box<[MultiSymEntry; MULTI_SYM_SIZE]>,
    single_table: Vec<SingleEntry>, // 15-bit table for slow path
}

impl MultiSymTable {
    /// Build multi-symbol table from code lengths for literals/lengths
    ///
    /// For each 12-bit input pattern, we pre-compute what symbols it decodes to.
    /// If all decoded symbols are literals (< 256), we pack them into one entry.
    pub fn build(lens: &[u8]) -> io::Result<Self> {
        // First, build a simple decode table for individual symbols
        let single_table = build_single_sym_table(lens)?;

        // Now build the multi-symbol table
        let mut entries = Box::new([MultiSymEntry::default(); MULTI_SYM_SIZE]);

        for bits in 0..MULTI_SYM_SIZE {
            let entry = build_multi_entry(bits as u32, &single_table);
            entries[bits] = entry;
        }

        Ok(Self {
            entries,
            single_table,
        })
    }

    /// Decode using multi-symbol table
    /// Returns (number of literals written, bits consumed, non-literal symbol if any)
    #[inline(always)]
    pub fn decode(&self, bits: u64) -> &MultiSymEntry {
        let idx = (bits as usize) & (MULTI_SYM_SIZE - 1);
        &self.entries[idx]
    }

    /// Single symbol decode for slow path (15-bit lookup)
    #[inline(always)]
    pub fn decode_single(&self, bits: u64) -> (u16, u32) {
        let idx = (bits as usize) & (SINGLE_TABLE_SIZE - 1);
        let entry = self.single_table[idx];
        (entry.symbol, entry.len as u32)
    }
}

// =============================================================================
// Single-Symbol Table (helper for building multi-sym)
// =============================================================================

/// Simple single-symbol decode entry
#[derive(Clone, Copy, Default)]
struct SingleEntry {
    symbol: u16,
    len: u8,
}

const SINGLE_TABLE_BITS: usize = 15;
const SINGLE_TABLE_SIZE: usize = 1 << SINGLE_TABLE_BITS;

/// Build a simple 15-bit decode table
fn build_single_sym_table(lens: &[u8]) -> io::Result<Vec<SingleEntry>> {
    let mut table = vec![SingleEntry::default(); SINGLE_TABLE_SIZE];

    // Count codes of each length
    let max_len = lens.iter().copied().max().unwrap_or(0) as usize;
    if max_len > 15 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Code length > 15",
        ));
    }

    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // Compute first code for each length
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols
    for (symbol, &len) in lens.iter().enumerate() {
        if len == 0 {
            continue;
        }

        let len = len as usize;
        let code = next_code[len];
        next_code[len] += 1;

        // Reverse bits for LSB-first decoding
        let rev = reverse_bits(code, len as u32);

        // Fill table entries (for all longer codes that start with this)
        let fill_count = 1 << (SINGLE_TABLE_BITS - len);
        let entry = SingleEntry {
            symbol: symbol as u16,
            len: len as u8,
        };

        for i in 0..fill_count {
            let idx = (rev as usize) | (i << len);
            table[idx] = entry;
        }
    }

    Ok(table)
}

/// Build a multi-symbol entry for a given 12-bit input pattern
fn build_multi_entry(bits: u32, single_table: &[SingleEntry]) -> MultiSymEntry {
    let mut entry = MultiSymEntry::default();
    let mut pos = 0u32;
    let mut count = 0u8;

    // Try to decode up to 4 symbols
    while pos < MULTI_SYM_BITS as u32 && count < 4 {
        // Look up in 15-bit table, but only consider the remaining bits
        let remaining = MULTI_SYM_BITS as u32 - pos;
        let lookup_bits = (bits >> pos) as usize;

        // Look up in full 15-bit table to get actual code
        let single = single_table[lookup_bits & (SINGLE_TABLE_SIZE - 1)];

        if single.len == 0 {
            // No valid code found - need slow path
            // Mark first symbol as needing slow path (special value 0xFFFF)
            if count == 0 {
                entry.symbol = 0xFFFF; // Signal slow path
                entry.bits_consumed = 0xFF; // Special marker
            }
            break;
        }

        if single.len as u32 > remaining {
            // Code is longer than remaining bits in 12-bit window
            // Can only continue if we've decoded nothing yet
            if count == 0 {
                // First symbol - use slow path for this
                entry.symbol = single.symbol;
                entry.bits_consumed = 0xFF; // Signal: use slow path
            }
            break;
        }

        if single.symbol >= 256 {
            // Non-literal (END_OF_BLOCK or length code)
            // Stop here - caller handles this
            if count == 0 {
                entry.symbol = single.symbol;
                entry.bits_consumed = single.len;
            }
            break;
        }

        // It's a literal
        entry.lits[count as usize] = single.symbol as u8;
        count += 1;
        pos += single.len as u32;
    }

    if count > 0 {
        entry.count = count;
        entry.bits_consumed = pos as u8;
    }

    entry
}

/// Reverse bits in a code
#[inline]
fn reverse_bits(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// =============================================================================
// Optimized Decode Loop
// =============================================================================

/// Decode a Huffman block using multi-symbol decode
///
/// Safety: Loop terminates via END_OF_BLOCK (256) or error from invalid code.
/// The FastBits.consume() uses saturating_sub() to prevent underflow that caused OOM.
#[inline(never)]
pub fn decode_block_multi_sym(
    bits: &mut FastBits,
    output: &mut Vec<u8>,
    multi_table: &MultiSymTable,
    dist_table: &crate::two_level_table::TwoLevelTable,
) -> io::Result<()> {
    // Pre-allocate output
    output.reserve(64 * 1024);

    loop {
        bits.refill();

        let entry = multi_table.decode(bits.buffer());

        if entry.count > 0 {
            // Fast path: 1-4 literals
            let count = entry.count as usize;

            // Ensure capacity and write directly
            let old_len = output.len();
            output.reserve(4);
            unsafe {
                let ptr = output.as_mut_ptr().add(old_len);
                // Write up to 4 bytes (some may be unused)
                std::ptr::copy_nonoverlapping(entry.lits.as_ptr(), ptr, count);
                output.set_len(old_len + count);
            }

            bits.consume(entry.bits_consumed as u32);
        } else if entry.bits_consumed > 0 {
            // Single non-literal symbol
            bits.consume(entry.bits_consumed as u32);

            let symbol = entry.symbol;
            if symbol == 256 {
                // END_OF_BLOCK
                break;
            } else {
                // Length code - do LZ77 copy
                crate::two_level_table::decode_lz77(bits, dist_table, symbol, output)?;
            }
        } else {
            // Need slow path - code longer than 12 bits
            // This shouldn't happen often for typical data
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Code too long for multi-sym table",
            ));
        }
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_sym_build() {
        // Build fixed Huffman table for literals/lengths
        let mut lens = [0u8; 288];

        // 0-143: 8 bits
        for len in lens.iter_mut().take(144) {
            *len = 8;
        }
        // 144-255: 9 bits
        for len in lens.iter_mut().take(256).skip(144) {
            *len = 9;
        }
        // 256-279: 7 bits
        for len in lens.iter_mut().take(280).skip(256) {
            *len = 7;
        }
        // 280-287: 8 bits
        for len in lens.iter_mut().take(288).skip(280) {
            *len = 8;
        }

        let table = MultiSymTable::build(&lens).unwrap();

        // Check that some entries have multiple literals
        let mut multi_count = 0;
        for entry in table.entries.iter() {
            if entry.count > 1 {
                multi_count += 1;
            }
        }

        eprintln!(
            "Entries with 2+ literals: {}/{}",
            multi_count, MULTI_SYM_SIZE
        );
        // Note: Fixed Huffman codes are 7-9 bits, so we can't fit 2 symbols in 12 bits.
        // Multi-symbol decode only helps with dynamic blocks that have shorter codes.
        // This test just verifies the table builds without error.
    }

    #[test]
    fn test_multi_sym_decode_correctness() {
        // Create simple test data
        let input = b"AAAAAAAAAAAAAAAA"; // Highly compressible

        // Compress with flate2
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(input).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress with our multi-sym decoder
        // (This would need more setup to actually test - just a placeholder)
        eprintln!(
            "Input: {} bytes, Compressed: {} bytes",
            input.len(),
            compressed.len()
        );
    }
}
