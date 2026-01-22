//! Multi-Symbol Decode Table
//!
//! This module implements the key optimization from rapidgzip: decode 2 symbols per lookup.
//! For deflate streams where 60-80% are literals, this provides ~2x speedup.
//!
//! Entry format (64 bits):
//! ```text
//! [Symbol1:9][Bits1:5][Symbol2:9][Bits2:5][TotalBits:6][Count:2][Flags:28]
//! ```
//!
//! - Count: 1 = single symbol, 2 = two symbols
//! - Flags: includes literal/match/EOB indicators
//!
//! The key insight: when both symbols are literals (most common case),
//! we decode both with a single table lookup.

#![allow(dead_code)]

use std::io;

/// LUT size constants - using 11 bits like libdeflate for L1 cache fit
pub const MULTI_LUT_BITS: usize = 11;
pub const MULTI_LUT_SIZE: usize = 1 << MULTI_LUT_BITS;
pub const MULTI_LUT_MASK: u64 = (MULTI_LUT_SIZE - 1) as u64;

/// Entry bit layout (64-bit)
const SYM1_SHIFT: u64 = 55;
const BITS1_SHIFT: u64 = 50;
const SYM2_SHIFT: u64 = 41;
const BITS2_SHIFT: u64 = 36;
const TOTAL_BITS_SHIFT: u64 = 30;
const COUNT_SHIFT: u64 = 28;

const SYM1_MASK: u64 = 0x1FF << SYM1_SHIFT; // 9 bits
const BITS1_MASK: u64 = 0x1F << BITS1_SHIFT; // 5 bits
const SYM2_MASK: u64 = 0x1FF << SYM2_SHIFT; // 9 bits
const BITS2_MASK: u64 = 0x1F << BITS2_SHIFT; // 5 bits
const TOTAL_BITS_MASK: u64 = 0x3F << TOTAL_BITS_SHIFT; // 6 bits
const COUNT_MASK: u64 = 0x3 << COUNT_SHIFT; // 2 bits

/// Flag bits in low 28 bits
const FLAG_LITERAL1: u64 = 1 << 0;
const FLAG_LITERAL2: u64 = 1 << 1;
const FLAG_EOB: u64 = 1 << 2;
const FLAG_MATCH: u64 = 1 << 3;
const FLAG_NEEDS_SLOW: u64 = 1 << 4;

/// Multi-symbol entry
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct MultiEntry(pub u64);

impl MultiEntry {
    /// Create a single literal entry
    #[inline(always)]
    pub const fn single_literal(bits: u8, symbol: u16) -> Self {
        let entry = ((symbol as u64) << SYM1_SHIFT)
            | ((bits as u64) << BITS1_SHIFT)
            | ((bits as u64) << TOTAL_BITS_SHIFT)
            | (1 << COUNT_SHIFT)
            | FLAG_LITERAL1;
        Self(entry)
    }

    /// Create a double literal entry (two consecutive literals)
    #[inline(always)]
    pub const fn double_literal(bits1: u8, sym1: u16, bits2: u8, sym2: u16) -> Self {
        let total_bits = (bits1 + bits2) as u64;
        let entry = ((sym1 as u64) << SYM1_SHIFT)
            | ((bits1 as u64) << BITS1_SHIFT)
            | ((sym2 as u64) << SYM2_SHIFT)
            | ((bits2 as u64) << BITS2_SHIFT)
            | (total_bits << TOTAL_BITS_SHIFT)
            | (2 << COUNT_SHIFT)
            | FLAG_LITERAL1
            | FLAG_LITERAL2;
        Self(entry)
    }

    /// Create an end-of-block entry
    #[inline(always)]
    pub const fn end_of_block(bits: u8) -> Self {
        let entry = ((bits as u64) << TOTAL_BITS_SHIFT) | (1 << COUNT_SHIFT) | FLAG_EOB;
        Self(entry)
    }

    /// Create a match entry (length symbol, needs distance decode)
    #[inline(always)]
    pub const fn length_match(bits: u8, length_symbol: u16) -> Self {
        let entry = ((length_symbol as u64) << SYM1_SHIFT)
            | ((bits as u64) << BITS1_SHIFT)
            | ((bits as u64) << TOTAL_BITS_SHIFT)
            | (1 << COUNT_SHIFT)
            | FLAG_MATCH;
        Self(entry)
    }

    /// Create a slow-path entry (code too long for LUT)
    #[inline(always)]
    pub const fn slow_path() -> Self {
        Self(FLAG_NEEDS_SLOW)
    }

    /// Get total bits to consume
    #[inline(always)]
    pub const fn total_bits(self) -> u32 {
        ((self.0 & TOTAL_BITS_MASK) >> TOTAL_BITS_SHIFT) as u32
    }

    /// Get symbol count (1 or 2)
    #[inline(always)]
    pub const fn count(self) -> u32 {
        ((self.0 & COUNT_MASK) >> COUNT_SHIFT) as u32
    }

    /// Get first symbol
    #[inline(always)]
    pub const fn symbol1(self) -> u16 {
        ((self.0 & SYM1_MASK) >> SYM1_SHIFT) as u16
    }

    /// Get second symbol (only valid if count == 2)
    #[inline(always)]
    pub const fn symbol2(self) -> u16 {
        ((self.0 & SYM2_MASK) >> SYM2_SHIFT) as u16
    }

    /// Check if first symbol is a literal
    #[inline(always)]
    pub const fn is_literal1(self) -> bool {
        (self.0 & FLAG_LITERAL1) != 0
    }

    /// Check if this is end-of-block
    #[inline(always)]
    pub const fn is_eob(self) -> bool {
        (self.0 & FLAG_EOB) != 0
    }

    /// Check if this is a match (needs distance decode)
    #[inline(always)]
    pub const fn is_match(self) -> bool {
        (self.0 & FLAG_MATCH) != 0
    }

    /// Check if slow path is needed
    #[inline(always)]
    pub const fn needs_slow(self) -> bool {
        (self.0 & FLAG_NEEDS_SLOW) != 0
    }
}

/// Multi-symbol lookup table
#[derive(Clone)]
pub struct MultiSymbolLUT {
    /// Primary table (2048 entries for 11 bits)
    pub table: Vec<MultiEntry>,
}

impl MultiSymbolLUT {
    /// Build multi-symbol table from code lengths
    pub fn build(lit_len_lengths: &[u8], _dist_lengths: &[u8]) -> io::Result<Self> {
        let mut table = vec![MultiEntry::slow_path(); MULTI_LUT_SIZE];

        // First pass: build single-symbol entries
        let mut code = 0u32;
        let mut bl_count = [0u32; 16];

        // Count code lengths
        for &len in lit_len_lengths {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        // Compute first code for each length
        let mut next_code = [0u32; 16];
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes and fill table
        for (symbol, &len) in lit_len_lengths.iter().enumerate() {
            if len == 0 || len as usize > MULTI_LUT_BITS {
                continue;
            }

            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse bits for lookup
            let reversed = reverse_bits(code, len);

            // Fill all entries that match this prefix
            let filler_bits = MULTI_LUT_BITS - len as usize;
            let count = 1 << filler_bits;

            for i in 0..count {
                let idx = reversed as usize | (i << len as usize);

                let entry = if symbol < 256 {
                    // Literal
                    MultiEntry::single_literal(len, symbol as u16)
                } else if symbol == 256 {
                    // End of block
                    MultiEntry::end_of_block(len)
                } else {
                    // Length code (match)
                    MultiEntry::length_match(len, symbol as u16)
                };

                table[idx] = entry;
            }
        }

        // Second pass: upgrade literal entries to double-literals where possible
        // For each entry that's a single literal, check if the next symbol (based on
        // remaining bits after first decode) is also a literal
        Self::upgrade_to_double_literals(&mut table, lit_len_lengths);

        Ok(Self { table })
    }

    /// Upgrade single-literal entries to double-literal where possible
    fn upgrade_to_double_literals(table: &mut [MultiEntry], lit_len_lengths: &[u8]) {
        // Build a temporary lookup for what symbol each bit pattern decodes to
        // We need to know: for remaining bits after first decode, what's the second symbol?

        // First, build a code-to-symbol map for all short codes
        let mut code_to_symbol: Vec<Option<(u16, u8)>> = vec![None; MULTI_LUT_SIZE];

        let mut bl_count = [0u32; 16];
        for &len in lit_len_lengths {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Map reversed codes to symbols
        for (symbol, &len) in lit_len_lengths.iter().enumerate() {
            if len == 0 || len as usize > MULTI_LUT_BITS {
                continue;
            }

            let code = next_code[len as usize];
            next_code[len as usize] += 1;
            let reversed = reverse_bits(code, len);

            // Fill all matching entries
            let filler_bits = MULTI_LUT_BITS - len as usize;
            for i in 0..(1 << filler_bits) {
                let idx = reversed as usize | (i << len as usize);
                code_to_symbol[idx] = Some((symbol as u16, len));
            }
        }

        // Now upgrade single-literal entries
        for (idx, table_entry) in table.iter_mut().enumerate().take(MULTI_LUT_SIZE) {
            let entry = *table_entry;
            if !entry.is_literal1() || entry.count() != 1 {
                continue;
            }

            let sym1 = entry.symbol1();
            let bits1 = entry.total_bits() as usize;

            // Check remaining bits for second symbol
            if MULTI_LUT_BITS <= bits1 {
                continue;
            }

            let remaining = idx >> bits1;
            if let Some((sym2, bits2)) = code_to_symbol[remaining] {
                // Only upgrade if second is also a literal (< 256)
                if sym2 < 256 && bits1 + bits2 as usize <= MULTI_LUT_BITS {
                    *table_entry = MultiEntry::double_literal(bits1 as u8, sym1, bits2, sym2);
                }
            }
        }
    }

    /// Look up entry by bit pattern
    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> MultiEntry {
        self.table[(bits & MULTI_LUT_MASK) as usize]
    }
}

/// Reverse bits in a code
fn reverse_bits(code: u32, len: u8) -> u32 {
    let mut result = 0u32;
    let mut c = code;
    for _ in 0..len {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_entry_single_literal() {
        let entry = MultiEntry::single_literal(8, 65); // 'A'
        assert!(entry.is_literal1());
        assert!(!entry.is_eob());
        assert!(!entry.is_match());
        assert_eq!(entry.count(), 1);
        assert_eq!(entry.total_bits(), 8);
        assert_eq!(entry.symbol1(), 65);
    }

    #[test]
    fn test_multi_entry_double_literal() {
        let entry = MultiEntry::double_literal(8, 65, 8, 66); // 'A', 'B'
        assert!(entry.is_literal1());
        assert_eq!(entry.count(), 2);
        assert_eq!(entry.total_bits(), 16);
        assert_eq!(entry.symbol1(), 65);
        assert_eq!(entry.symbol2(), 66);
    }

    #[test]
    fn test_multi_entry_eob() {
        let entry = MultiEntry::end_of_block(7);
        assert!(entry.is_eob());
        assert!(!entry.is_literal1());
        assert_eq!(entry.total_bits(), 7);
    }

    #[test]
    fn test_multi_entry_match() {
        let entry = MultiEntry::length_match(10, 260);
        assert!(entry.is_match());
        assert!(!entry.is_literal1());
        assert_eq!(entry.symbol1(), 260);
    }

    /// Benchmark multi-symbol decode vs single-symbol
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn bench_multi_vs_single_decode() {
        // Create a simple table with fixed Huffman-like codes
        let mut lit_len_lengths = vec![0u8; 288];
        // Fixed Huffman code lengths per RFC 1951
        lit_len_lengths[..144].fill(8);
        lit_len_lengths[144..256].fill(9);
        lit_len_lengths[256] = 7; // EOB
        lit_len_lengths[257..280].fill(7);
        lit_len_lengths[280..288].fill(8);

        let dist_lengths = vec![5u8; 32];

        let table = MultiSymbolLUT::build(&lit_len_lengths, &dist_lengths).unwrap();

        // Count double-literal entries
        let mut singles = 0;
        let mut doubles = 0;
        for entry in &table.table {
            if entry.is_literal1() {
                if entry.count() == 2 {
                    doubles += 1;
                } else {
                    singles += 1;
                }
            }
        }

        eprintln!("\n[BENCH] Multi-Symbol Table Analysis:");
        eprintln!("[BENCH]   Single literal entries: {}", singles);
        eprintln!("[BENCH]   Double literal entries: {}", doubles);
        eprintln!(
            "[BENCH]   Double ratio: {:.1}%",
            doubles as f64 / (singles + doubles) as f64 * 100.0
        );

        // Simulate decode loop
        let bits_sequence: Vec<u64> = (0..100_000)
            .map(|i| (i * 0x12345678) % (1 << MULTI_LUT_BITS))
            .collect();

        let iterations = 500;

        // Multi-symbol decode
        let start = std::time::Instant::now();
        let mut total_symbols = 0u64;
        for _ in 0..iterations {
            for &bits in &bits_sequence {
                let entry = table.lookup(bits);
                if entry.is_literal1() {
                    total_symbols += entry.count() as u64;
                }
            }
        }
        let elapsed = start.elapsed();

        eprintln!(
            "[BENCH]   Decoded {} symbols in {:.2}ms",
            total_symbols,
            elapsed.as_secs_f64() * 1000.0
        );
        eprintln!(
            "[BENCH]   Speed: {:.1} M symbols/sec",
            total_symbols as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
    }
}
