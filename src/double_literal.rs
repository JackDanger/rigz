//! Double-Literal Cache for DEFLATE decoding
//! 
//! Inspired by rapidgzip's HuffmanCodingDoubleLiteralCached.hpp
//! 
//! Key insight: Most DEFLATE streams have runs of literals. By caching
//! pairs of consecutive literals in a single lookup, we can decode
//! two symbols at once, roughly doubling literal throughput.
//!
//! Cache structure:
//! - 13-bit lookup (8KB cache)
//! - Each entry is 32 bits: [symbol1:8][length:8][symbol2:8][flags:8]
//! - If symbol2 == 0xFF, it's a single-symbol entry

use crate::libdeflate_entry::{LitLenTable, LitLenEntry};

/// Double-literal cache entry
/// 
/// Bit layout:
/// - bits 0-7: symbol1 (first literal value)
/// - bits 8-15: total_bits (bits consumed for both symbols)
/// - bits 16-23: symbol2 (second literal value, 0xFF = none)
/// - bits 24-31: flags (0x01 = has second symbol, 0x80 = not a literal)
#[derive(Clone, Copy, Debug)]
pub struct DoubleLitEntry(u32);

impl DoubleLitEntry {
    pub const NONE_SYMBOL: u8 = 0xFF;
    pub const FLAG_HAS_SECOND: u8 = 0x01;
    pub const FLAG_NOT_LITERAL: u8 = 0x80;
    
    /// Create a single-literal entry
    #[inline(always)]
    pub fn single(symbol: u8, bits: u8) -> Self {
        Self((symbol as u32) | ((bits as u32) << 8) | ((Self::NONE_SYMBOL as u32) << 16))
    }
    
    /// Create a double-literal entry
    #[inline(always)]
    pub fn double(symbol1: u8, symbol2: u8, total_bits: u8) -> Self {
        Self((symbol1 as u32) | ((total_bits as u32) << 8) | ((symbol2 as u32) << 16) | ((Self::FLAG_HAS_SECOND as u32) << 24))
    }
    
    /// Create a non-literal entry (fall back to regular decode)
    #[inline(always)]
    pub fn not_literal() -> Self {
        Self((Self::FLAG_NOT_LITERAL as u32) << 24)
    }
    
    /// Check if this is a literal (or double literal)
    #[inline(always)]
    pub fn is_literal(&self) -> bool {
        (self.0 >> 24) & Self::FLAG_NOT_LITERAL as u32 == 0
    }
    
    /// Check if this has a second symbol
    #[inline(always)]
    pub fn has_second(&self) -> bool {
        (self.0 >> 24) & Self::FLAG_HAS_SECOND as u32 != 0
    }
    
    /// Get first symbol
    #[inline(always)]
    pub fn symbol1(&self) -> u8 {
        self.0 as u8
    }
    
    /// Get second symbol (only valid if has_second())
    #[inline(always)]
    pub fn symbol2(&self) -> u8 {
        (self.0 >> 16) as u8
    }
    
    /// Get total bits consumed
    #[inline(always)]
    pub fn total_bits(&self) -> u8 {
        (self.0 >> 8) as u8
    }
}

/// Double-literal cache
/// 
/// 16-bit lookup = 65536 entries = 256KB
/// This fits two 8-bit literals plus allows for some filler bits
pub const DOUBLE_LIT_BITS: usize = 16;
pub const DOUBLE_LIT_SIZE: usize = 1 << DOUBLE_LIT_BITS;

pub struct DoubleLitCache {
    entries: Box<[DoubleLitEntry; DOUBLE_LIT_SIZE]>,
}

impl DoubleLitCache {
    /// Build a double-literal cache from a litlen table
    pub fn build(litlen_table: &LitLenTable) -> Self {
        let mut entries = vec![DoubleLitEntry::not_literal(); DOUBLE_LIT_SIZE];
        
        // For each possible 13-bit pattern, try to decode one or two literals
        for pattern in 0..DOUBLE_LIT_SIZE {
            let bits = pattern as u64;
            
            // Look up first symbol
            let e1 = litlen_table.resolve(bits);
            
            // If not a literal, mark as non-literal
            if !e1.is_literal() {
                entries[pattern] = DoubleLitEntry::not_literal();
                continue;
            }
            
            let sym1 = e1.literal_value();
            let bits1 = e1.codeword_bits();
            
            // If first symbol uses too many bits, just store single
            if bits1 >= DOUBLE_LIT_BITS as u8 {
                entries[pattern] = DoubleLitEntry::single(sym1, bits1);
                continue;
            }
            
            // Try to decode second symbol
            let remaining = bits >> bits1;
            let e2 = litlen_table.resolve(remaining);
            
            // If second is not a literal, store single
            if !e2.is_literal() {
                entries[pattern] = DoubleLitEntry::single(sym1, bits1);
                continue;
            }
            
            let sym2 = e2.literal_value();
            let bits2 = e2.codeword_bits();
            let total_bits = bits1 + bits2;
            
            // If total exceeds our cache bits, store single
            if total_bits as usize > DOUBLE_LIT_BITS {
                entries[pattern] = DoubleLitEntry::single(sym1, bits1);
                continue;
            }
            
            // Store double literal!
            entries[pattern] = DoubleLitEntry::double(sym1, sym2, total_bits);
        }
        
        Self {
            entries: entries.into_boxed_slice().try_into().unwrap(),
        }
    }
    
    /// Look up a pattern
    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> DoubleLitEntry {
        let idx = (bits as usize) & (DOUBLE_LIT_SIZE - 1);
        unsafe { *self.entries.get_unchecked(idx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_double_lit_entry() {
        let single = DoubleLitEntry::single(b'A', 8);
        assert!(single.is_literal());
        assert!(!single.has_second());
        assert_eq!(single.symbol1(), b'A');
        assert_eq!(single.total_bits(), 8);
        
        let double = DoubleLitEntry::double(b'A', b'B', 16);
        assert!(double.is_literal());
        assert!(double.has_second());
        assert_eq!(double.symbol1(), b'A');
        assert_eq!(double.symbol2(), b'B');
        assert_eq!(double.total_bits(), 16);
        
        let non_lit = DoubleLitEntry::not_literal();
        assert!(!non_lit.is_literal());
    }
    
    #[test]
    fn test_double_lit_cache_build() {
        // Build fixed litlen table
        let tables = crate::libdeflate_decode::get_fixed_tables();
        let cache = DoubleLitCache::build(&tables.0);
        
        // Debug: check a few entries
        for pattern in 0..5 {
            let bits = pattern as u64;
            let e1 = tables.0.resolve(bits);
            eprintln!("Pattern {}: is_literal={}, literal_value={}, codeword_bits={}", 
                pattern, e1.is_literal(), e1.literal_value(), e1.codeword_bits());
        }
        
        // Count different entry types
        let mut singles = 0;
        let mut doubles = 0;
        let mut non_lits = 0;
        
        for i in 0..DOUBLE_LIT_SIZE {
            let entry = cache.lookup(i as u64);
            if !entry.is_literal() {
                non_lits += 1;
            } else if entry.has_second() {
                doubles += 1;
            } else {
                singles += 1;
            }
        }
        
        eprintln!("Double-literal cache stats:");
        eprintln!("  Singles: {}", singles);
        eprintln!("  Doubles: {}", doubles);
        eprintln!("  Non-literals: {}", non_lits);
        
        // We expect a mix - the exact ratio depends on the fixed Huffman code structure
        assert!(singles + doubles > 0, "Expected some literal entries");
    }
    
    #[test]
    fn bench_double_literal_decode() {
        // Create test data: all ASCII letters
        let test_data: Vec<u8> = (0..100_000).map(|i| (b'A' + (i % 26) as u8)).collect();
        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut encoder = flate2::write::GzEncoder::new(&mut compressed, flate2::Compression::default());
            encoder.write_all(&test_data).unwrap();
            encoder.finish().unwrap();
        }
        
        // Parse gzip header
        let start = 10 + if (compressed[3] & 0x08) != 0 {
            compressed[10..].iter().position(|&b| b == 0).unwrap_or(0) + 1
        } else { 0 };
        let end = compressed.len() - 8;
        let deflate = &compressed[start..end];
        
        // Build tables
        let tables = crate::libdeflate_decode::get_fixed_tables();
        let double_cache = DoubleLitCache::build(&tables.0);
        
        // Benchmark: simulate decoding literals using double-literal cache
        let iters = 1000;
        let mut bits: u64 = 0;
        for i in 0..64.min(deflate.len()) {
            bits |= (deflate[i] as u64) << (i * 8);
        }
        
        let start_t = std::time::Instant::now();
        let mut total_symbols = 0u64;
        for _ in 0..iters {
            let mut local_bits = bits;
            for _ in 0..1000 {
                let entry = double_cache.lookup(local_bits);
                if entry.is_literal() {
                    if entry.has_second() {
                        total_symbols += 2;
                        local_bits >>= entry.total_bits();
                    } else {
                        total_symbols += 1;
                        local_bits >>= entry.total_bits();
                    }
                } else {
                    break;
                }
            }
        }
        let elapsed = start_t.elapsed();
        
        let symbols_per_sec = total_symbols as f64 / elapsed.as_secs_f64();
        eprintln!("\nDouble-literal cache benchmark:");
        eprintln!("  Total symbols: {}", total_symbols);
        eprintln!("  Time: {:.2}ms", elapsed.as_millis());
        eprintln!("  Throughput: {:.1}M symbols/sec", symbols_per_sec / 1e6);
    }
}
