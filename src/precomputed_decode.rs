//! Precomputed Decode Sequences
//!
//! For fixed Huffman, we precompute: (16-bit input) → (decoded literals, bits consumed)
//! One table lookup decodes 1-2 complete symbols, eliminating per-symbol loop overhead.
//!
//! Fixed Huffman codes range from 7-9 bits for litlen. With 16-bit lookahead:
//! - Best case: 2 literals at 7 bits each = 14 bits, 2 symbols per lookup
//! - Worst case: 1 literal at 9 bits = can still decode 1 symbol
//!
//! Table size: 2^16 × 4 bytes = 256KB (fits in L2 cache)

#![allow(dead_code)]

use std::sync::OnceLock;

/// Lookup table bits - 16 bits gives good coverage while fitting in L2
pub const PRECOMPUTE_BITS: usize = 16;
pub const PRECOMPUTE_SIZE: usize = 1 << PRECOMPUTE_BITS;
pub const PRECOMPUTE_MASK: u64 = (PRECOMPUTE_SIZE - 1) as u64;

/// Precomputed entry format (32 bits):
/// - bytes[0..2]: decoded literal bytes (up to 2)
/// - count: number of symbols decoded (1 or 2)
/// - bits: total bits consumed
/// - flags: special handling needed (match, EOB, needs_more)
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct PrecomputeEntry {
    /// First decoded literal (or 0xFF if not literal)
    pub lit1: u8,
    /// Second decoded literal (or 0xFF if none/not literal)
    pub lit2: u8,
    /// Number of complete symbols decoded (0, 1, or 2)
    pub count: u8,
    /// Total bits consumed
    pub bits: u8,
}

impl PrecomputeEntry {
    const fn new(lit1: u8, lit2: u8, count: u8, bits: u8) -> Self {
        Self {
            lit1,
            lit2,
            count,
            bits,
        }
    }

    /// Entry indicates we decoded 2 literals
    #[inline(always)]
    pub fn is_double_literal(&self) -> bool {
        self.count == 2
    }

    /// Entry indicates we decoded 1 literal
    #[inline(always)]
    pub fn is_single_literal(&self) -> bool {
        self.count == 1
    }

    /// Entry indicates no literals (length code, EOB, or needs subtable)
    #[inline(always)]
    pub fn is_special(&self) -> bool {
        self.count == 0
    }
}

/// Static table for fixed Huffman - computed once at startup
static FIXED_PRECOMPUTE: OnceLock<Box<[PrecomputeEntry; PRECOMPUTE_SIZE]>> = OnceLock::new();

/// Get the fixed Huffman precomputed table
pub fn get_fixed_precompute() -> &'static [PrecomputeEntry; PRECOMPUTE_SIZE] {
    FIXED_PRECOMPUTE.get_or_init(build_fixed_precompute)
}

/// Build the precomputed table for fixed Huffman
fn build_fixed_precompute() -> Box<[PrecomputeEntry; PRECOMPUTE_SIZE]> {
    let mut table = vec![PrecomputeEntry::default(); PRECOMPUTE_SIZE];

    // Fixed Huffman litlen codes (RFC 1951):
    // Symbol 0-143:   8-bit codes 00110000-10111111 (reversed in bitstream)
    // Symbol 144-255: 9-bit codes 110010000-111111111
    // Symbol 256-279: 7-bit codes 0000000-0010111 (256=EOB)
    // Symbol 280-287: 8-bit codes 11000000-11000111

    // Build code-to-symbol mapping for codes up to 9 bits
    let mut code_to_sym: [(u16, u8); 512] = [(0xFFFF, 0); 512]; // (symbol, len)

    // Generate fixed Huffman codes
    for sym in 0u16..288 {
        let (code, len) = fixed_huffman_code(sym);
        if len <= 9 {
            // Reverse bits for deflate format
            let reversed = reverse_bits(code, len);
            // Fill all entries that start with this code
            let fill_bits = 9 - len as usize;
            for suffix in 0..(1 << fill_bits) {
                let idx = reversed as usize | (suffix << len as usize);
                if idx < 512 {
                    code_to_sym[idx] = (sym, len);
                }
            }
        }
    }

    // Now build precomputed table
    for (bits_pattern, table_entry) in table.iter_mut().enumerate() {
        let bits = bits_pattern as u64;

        // Decode first symbol (using 9-bit lookup)
        let idx1 = (bits & 0x1FF) as usize;
        let (sym1, len1) = code_to_sym[idx1];

        if sym1 >= 256 {
            // Not a literal (length code or EOB) - mark as special
            *table_entry = PrecomputeEntry::new(0xFF, 0xFF, 0, len1);
            continue;
        }

        // First symbol is a literal
        let lit1 = sym1 as u8;
        let remaining = bits >> len1;

        // Try to decode second symbol
        let idx2 = (remaining & 0x1FF) as usize;
        let (sym2, len2) = code_to_sym[idx2];

        if sym2 < 256 && len1 + len2 <= 16 {
            // Second symbol is also a literal and fits in our lookahead
            let lit2 = sym2 as u8;
            *table_entry = PrecomputeEntry::new(lit1, lit2, 2, len1 + len2);
        } else {
            // Only first literal decoded
            *table_entry = PrecomputeEntry::new(lit1, 0xFF, 1, len1);
        }
    }

    table.into_boxed_slice().try_into().unwrap()
}

/// Get fixed Huffman code for a symbol
fn fixed_huffman_code(sym: u16) -> (u16, u8) {
    match sym {
        0..=143 => (0b00110000 + sym, 8),
        144..=255 => (0b110010000 + (sym - 144), 9),
        256..=279 => (sym - 256, 7),
        280..=287 => (0b11000000 + (sym - 280), 8),
        _ => (0, 0),
    }
}

/// Reverse n bits
fn reverse_bits(code: u16, n: u8) -> u16 {
    let mut result = 0u16;
    let mut c = code;
    for _ in 0..n {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_precompute_table_builds() {
        let table = get_fixed_precompute();

        // Check some known patterns
        let mut double_count = 0;
        let mut single_count = 0;
        let mut special_count = 0;

        for entry in table.iter() {
            match entry.count {
                2 => double_count += 1,
                1 => single_count += 1,
                0 => special_count += 1,
                _ => panic!("Invalid count"),
            }
        }

        eprintln!("\nPrecompute table statistics:");
        eprintln!(
            "  Double literals: {} ({:.1}%)",
            double_count,
            100.0 * double_count as f64 / PRECOMPUTE_SIZE as f64
        );
        eprintln!(
            "  Single literals: {} ({:.1}%)",
            single_count,
            100.0 * single_count as f64 / PRECOMPUTE_SIZE as f64
        );
        eprintln!(
            "  Special (match/EOB): {} ({:.1}%)",
            special_count,
            100.0 * special_count as f64 / PRECOMPUTE_SIZE as f64
        );
    }

    #[test]
    fn bench_precompute_lookup() {
        let table = get_fixed_precompute();
        let iterations = 50_000_000u64;

        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..iterations {
            let entry = table[(i & PRECOMPUTE_MASK) as usize];
            sum = sum.wrapping_add(entry.lit1 as u64 + entry.count as u64);
        }
        let elapsed = start.elapsed();

        let lookups_per_sec = iterations as f64 / elapsed.as_secs_f64();
        eprintln!("\nPrecompute lookup benchmark:");
        eprintln!(
            "  {} lookups in {:.2}ms",
            iterations,
            elapsed.as_secs_f64() * 1000.0
        );
        eprintln!("  {:.1} M lookups/sec", lookups_per_sec / 1_000_000.0);
        eprintln!("  (sum: {} to prevent opt)", sum);
    }
}
