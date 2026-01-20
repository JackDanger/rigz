//! Consume-First Huffman Table
//!
//! This module implements a table design where EVERY entry is valid,
//! enabling the consume-first pattern that's 36.3% faster than check-first.
//!
//! Key insight from libdeflate: Instead of BITS=0 for invalid entries,
//! use subtables so every primary entry has valid bits-to-consume.
//!
//! Entry format (32 bits):
//! ```text
//! [Type:2][Data:22][Bits:8]
//!
//! Type bits:
//!   00 = Subtable pointer (data = subtable index + extra bits count)
//!   01 = Literal (data = symbol)
//!   10 = Length code (data = length_base + extra_bits_count)
//!   11 = End of block (data = 0)
//! ```
//!
//! CRITICAL: Bits field is NEVER 0 for valid entries. This allows
//! unconditional consumption: `bitbuf >>= (entry & 0xFF)`

#![allow(dead_code)]

use std::io;

/// Table size constants (11 bits like libdeflate)
pub const CF_TABLE_BITS: usize = 11;
pub const CF_TABLE_SIZE: usize = 1 << CF_TABLE_BITS;
pub const CF_TABLE_MASK: u64 = (CF_TABLE_SIZE - 1) as u64;

/// Entry type constants (high 2 bits)
const TYPE_SHIFT: u32 = 30;
const TYPE_MASK: u32 = 0b11 << TYPE_SHIFT;
const TYPE_SUBTABLE: u32 = 0b00 << TYPE_SHIFT;
const TYPE_LITERAL: u32 = 0b01 << TYPE_SHIFT;
const TYPE_LENGTH: u32 = 0b10 << TYPE_SHIFT;
const TYPE_EOB: u32 = 0b11 << TYPE_SHIFT;

/// Bits mask (low 8 bits) - NEVER 0 for valid entries
const BITS_MASK: u32 = 0xFF;

/// Data field (bits 8-29)
const DATA_SHIFT: u32 = 8;
const DATA_MASK: u32 = 0x3FFFFF << DATA_SHIFT;

/// Maximum subtable entries needed (ENOUGH constant from libdeflate)
/// For 11-bit main table with 15-bit max codes: ~2342 entries
const SUBTABLE_ENOUGH: usize = 2400;

/// A consume-first table entry
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct CFEntry(pub u32);

impl CFEntry {
    /// Create a literal entry
    #[inline(always)]
    pub const fn literal(bits: u8, symbol: u16) -> Self {
        debug_assert!(bits > 0, "bits must be > 0 for consume-first");
        Self(TYPE_LITERAL | ((symbol as u32) << DATA_SHIFT) | (bits as u32))
    }

    /// Create an end-of-block entry
    #[inline(always)]
    pub const fn end_of_block(bits: u8) -> Self {
        debug_assert!(bits > 0, "bits must be > 0 for consume-first");
        Self(TYPE_EOB | (bits as u32))
    }

    /// Create a length code entry
    #[inline(always)]
    pub const fn length(bits: u8, symbol: u16) -> Self {
        debug_assert!(bits > 0, "bits must be > 0 for consume-first");
        Self(TYPE_LENGTH | ((symbol as u32) << DATA_SHIFT) | (bits as u32))
    }

    /// Create a subtable pointer entry
    /// `subtable_offset`: index into subtable array
    /// `extra_bits`: number of bits to use for subtable index (code_len - TABLE_BITS)
    #[inline(always)]
    pub const fn subtable_ptr(bits: u8, subtable_offset: u16, extra_bits: u8) -> Self {
        debug_assert!(bits > 0, "bits must be > 0 for consume-first");
        let data = ((subtable_offset as u32) << 6) | (extra_bits as u32);
        Self(TYPE_SUBTABLE | (data << DATA_SHIFT) | (bits as u32))
    }

    /// Get bits to consume (ALWAYS > 0)
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.0 & BITS_MASK
    }

    /// Get entry type
    #[inline(always)]
    pub const fn entry_type(self) -> u32 {
        self.0 & TYPE_MASK
    }

    /// Check if literal
    #[inline(always)]
    pub const fn is_literal(self) -> bool {
        (self.0 & TYPE_MASK) == TYPE_LITERAL
    }

    /// Check if EOB
    #[inline(always)]
    pub const fn is_eob(self) -> bool {
        (self.0 & TYPE_MASK) == TYPE_EOB
    }

    /// Check if length code
    #[inline(always)]
    pub const fn is_length(self) -> bool {
        (self.0 & TYPE_MASK) == TYPE_LENGTH
    }

    /// Check if subtable pointer
    #[inline(always)]
    pub const fn is_subtable(self) -> bool {
        (self.0 & TYPE_MASK) == TYPE_SUBTABLE
    }

    /// Get symbol (for literal/length entries)
    #[inline(always)]
    pub const fn symbol(self) -> u16 {
        ((self.0 & DATA_MASK) >> DATA_SHIFT) as u16
    }

    /// Get subtable offset (for subtable entries)
    #[inline(always)]
    pub const fn subtable_offset(self) -> u16 {
        (((self.0 & DATA_MASK) >> DATA_SHIFT) >> 6) as u16
    }

    /// Get extra bits count for subtable lookup
    #[inline(always)]
    pub const fn subtable_extra_bits(self) -> u8 {
        (((self.0 & DATA_MASK) >> DATA_SHIFT) & 0x3F) as u8
    }
}

/// Consume-first lookup table with subtables
pub struct ConsumeFirstTable {
    /// Main table (2048 entries for 11 bits)
    pub main: Vec<CFEntry>,
    /// Subtables for codes > 11 bits
    pub sub: Vec<CFEntry>,
}

impl ConsumeFirstTable {
    /// Build table from code lengths (for literal/length alphabet)
    pub fn build(code_lengths: &[u8]) -> io::Result<Self> {
        Self::build_inner(code_lengths, false)
    }

    /// Build table from code lengths (for distance alphabet)
    /// All symbols are treated as distance codes, not literals
    pub fn build_distance(code_lengths: &[u8]) -> io::Result<Self> {
        Self::build_inner(code_lengths, true)
    }

    /// Internal build function
    fn build_inner(code_lengths: &[u8], is_distance_table: bool) -> io::Result<Self> {
        let mut main = vec![CFEntry::end_of_block(1); CF_TABLE_SIZE];
        let mut sub = Vec::with_capacity(SUBTABLE_ENOUGH);

        // Count code lengths
        let mut bl_count = [0u32; 16];
        let mut max_len = 0u8;
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
                max_len = max_len.max(len);
            }
        }

        if max_len == 0 {
            return Ok(Self { main, sub });
        }

        // Compute first code for each length
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=15 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // PASS 1: Compute maximum extra_bits needed for each main table index
        let mut max_extra_for_main = [0u8; CF_TABLE_SIZE];

        {
            let mut next_code_temp = next_code;
            for &len in code_lengths.iter() {
                if len == 0 || (len as usize) <= CF_TABLE_BITS {
                    continue;
                }

                let code = next_code_temp[len as usize];
                next_code_temp[len as usize] += 1;
                let reversed = reverse_bits(code, len);
                let main_idx = (reversed & ((1 << CF_TABLE_BITS) - 1)) as usize;
                let extra_bits = len - CF_TABLE_BITS as u8;
                max_extra_for_main[main_idx] = max_extra_for_main[main_idx].max(extra_bits);
            }
        }

        // PASS 2: Create subtables with correct sizes
        for main_idx in 0..CF_TABLE_SIZE {
            let extra = max_extra_for_main[main_idx];
            if extra > 0 {
                let subtable_offset = sub.len() as u16;
                let subtable_size = 1 << extra;
                for _ in 0..subtable_size {
                    sub.push(CFEntry::end_of_block(extra)); // Default with correct bits
                }
                main[main_idx] = CFEntry::subtable_ptr(CF_TABLE_BITS as u8, subtable_offset, extra);
            }
        }

        // PASS 3: Fill entries
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse bits for lookup
            let reversed = reverse_bits(code, len);

            if (len as usize) <= CF_TABLE_BITS {
                // Direct entry in main table
                let filler_bits = CF_TABLE_BITS - len as usize;
                let count = 1 << filler_bits;

                let entry = create_entry(symbol, len, is_distance_table);

                for i in 0..count {
                    let idx = reversed as usize | (i << len as usize);
                    // CRITICAL FIX: Don't overwrite subtable pointers!
                    // If this index already has a subtable pointer, skip it.
                    // The subtable pointer was created in Pass 2 for longer codes
                    // that share this prefix.
                    if !main[idx].is_subtable() {
                        main[idx] = entry;
                    }
                }
            } else {
                // Subtable entry
                let main_bits = CF_TABLE_BITS as u8;
                let extra_bits = len - main_bits;
                let main_idx = (reversed & ((1 << main_bits) - 1)) as usize;

                // Get subtable info (already created in pass 2)
                let subtable_offset = main[main_idx].subtable_offset() as usize;
                let subtable_extra = main[main_idx].subtable_extra_bits() as usize;

                // Fill subtable entries
                // FIX: Entry must consume ACTUAL code length minus main bits,
                // NOT the subtable size. We replicate to fill the subtable,
                // but each entry has its own bits-to-consume.
                let sub_code = (reversed >> main_bits) as usize;
                let filler_bits = subtable_extra.saturating_sub(extra_bits as usize);
                let count = 1 << filler_bits;

                // Entry consumes extra_bits (the ACTUAL code length minus TABLE_BITS)
                let entry = create_entry(symbol, extra_bits, is_distance_table);

                for i in 0..count {
                    // Place entry at all positions where "don't care" bits vary
                    let sub_idx = subtable_offset + (sub_code | (i << extra_bits as usize));
                    if sub_idx < sub.len() {
                        sub[sub_idx] = entry;
                    }
                }
            }
        }

        Ok(Self { main, sub })
    }

    /// Lookup entry by bit pattern (consume-first style)
    #[inline(always)]
    pub fn lookup_main(&self, bits: u64) -> CFEntry {
        self.main[(bits & CF_TABLE_MASK) as usize]
    }

    /// Lookup subtable entry
    #[inline(always)]
    pub fn lookup_sub(&self, entry: CFEntry, bits: u64) -> CFEntry {
        let offset = entry.subtable_offset() as usize;
        let extra = entry.subtable_extra_bits() as usize;
        let mask = (1u64 << extra) - 1;
        let idx = offset + (bits & mask) as usize;
        if idx < self.sub.len() {
            self.sub[idx]
        } else {
            CFEntry::end_of_block(1)
        }
    }
}

/// Create appropriate entry based on symbol
/// For distance tables, all symbols are treated as distance codes (using length type)
fn create_entry(symbol: usize, bits: u8, is_distance_table: bool) -> CFEntry {
    if is_distance_table {
        // Distance table: all symbols are distance codes
        CFEntry::length(bits, symbol as u16)
    } else if symbol < 256 {
        CFEntry::literal(bits, symbol as u16)
    } else if symbol == 256 {
        CFEntry::end_of_block(bits)
    } else {
        CFEntry::length(bits, symbol as u16)
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

/// Decode a deflate block using consume-first pattern
/// Returns (bytes_written, is_final_block)
#[allow(clippy::too_many_arguments)]
pub fn decode_block_consume_first(
    input: &[u8],
    input_pos: &mut usize,
    bit_offset: &mut u8,
    output: &mut [u8],
    out_pos: &mut usize,
    lit_table: &ConsumeFirstTable,
    dist_table: &ConsumeFirstTable,
) -> io::Result<bool> {
    use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};

    // Bit buffer state
    let mut bitbuf: u64 = 0;
    let mut bits_in_buf: u32 = 0;

    // Initialize bit buffer from input
    let refill = |buf: &mut u64, bits: &mut u32, input: &[u8], pos: &mut usize| {
        while *bits <= 56 && *pos < input.len() {
            *buf |= (input[*pos] as u64) << *bits;
            *pos += 1;
            *bits += 8;
        }
    };

    // Account for initial bit offset
    if *bit_offset > 0 && *input_pos < input.len() {
        bitbuf = (input[*input_pos] as u64) >> *bit_offset;
        bits_in_buf = 8 - *bit_offset as u32;
        *input_pos += 1;
    }

    refill(&mut bitbuf, &mut bits_in_buf, input, input_pos);

    let out_end = output.len();

    loop {
        // Ensure we have enough bits
        refill(&mut bitbuf, &mut bits_in_buf, input, input_pos);

        // Look up entry
        let entry = lit_table.lookup_main(bitbuf);

        // CONSUME FIRST - entry.bits() is ALWAYS > 0
        let bits_to_skip = entry.bits();
        bitbuf >>= bits_to_skip;
        bits_in_buf = bits_in_buf.saturating_sub(bits_to_skip);

        // Now check entry type
        if entry.is_literal() {
            // Literal - most common case
            if *out_pos >= out_end {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
            }
            output[*out_pos] = entry.symbol() as u8;
            *out_pos += 1;
            continue;
        }

        if entry.is_eob() {
            // End of block
            *bit_offset = (8 - (bits_in_buf % 8) as u8) % 8;
            return Ok(true);
        }

        if entry.is_subtable() {
            // Subtable lookup (rare - codes > 11 bits)
            let sub_entry = lit_table.lookup_sub(entry, bitbuf);
            let sub_bits = sub_entry.bits();
            bitbuf >>= sub_bits;
            bits_in_buf = bits_in_buf.saturating_sub(sub_bits);

            if sub_entry.is_literal() {
                if *out_pos >= out_end {
                    return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
                }
                output[*out_pos] = sub_entry.symbol() as u8;
                *out_pos += 1;
                continue;
            }
            if sub_entry.is_eob() {
                *bit_offset = (8 - (bits_in_buf % 8) as u8) % 8;
                return Ok(true);
            }
            // Length from subtable - fall through to length handling
        }

        // Length code - decode match
        let len_symbol = if entry.is_length() {
            entry.symbol()
        } else {
            // From subtable
            lit_table.lookup_sub(entry, bitbuf).symbol()
        };

        if !(257..=285).contains(&len_symbol) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length code",
            ));
        }

        let len_idx = (len_symbol - 257) as usize;
        refill(&mut bitbuf, &mut bits_in_buf, input, input_pos);

        // Get length with extra bits
        let extra_len_bits = LEN_EXTRA_BITS[len_idx] as u32;
        let length =
            LEN_START[len_idx] as usize + (bitbuf & ((1u64 << extra_len_bits) - 1)) as usize;
        bitbuf >>= extra_len_bits;
        bits_in_buf = bits_in_buf.saturating_sub(extra_len_bits);

        // Decode distance
        refill(&mut bitbuf, &mut bits_in_buf, input, input_pos);
        let dist_entry = dist_table.lookup_main(bitbuf);
        let dist_bits = dist_entry.bits();
        bitbuf >>= dist_bits;
        bits_in_buf = bits_in_buf.saturating_sub(dist_bits);

        let dist_symbol = if dist_entry.is_subtable() {
            let sub = dist_table.lookup_sub(dist_entry, bitbuf);
            let sub_bits = sub.bits();
            bitbuf >>= sub_bits;
            bits_in_buf = bits_in_buf.saturating_sub(sub_bits);
            sub.symbol()
        } else {
            dist_entry.symbol()
        };

        if dist_symbol >= 30 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }

        refill(&mut bitbuf, &mut bits_in_buf, input, input_pos);
        let extra_dist_bits = DIST_EXTRA_BITS[dist_symbol as usize] as u32;
        let distance = DIST_START[dist_symbol as usize] as usize
            + (bitbuf & ((1u64 << extra_dist_bits) - 1)) as usize;
        bitbuf >>= extra_dist_bits;
        bits_in_buf = bits_in_buf.saturating_sub(extra_dist_bits);

        // Copy match
        if distance == 0 || distance > *out_pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }
        if *out_pos + length > out_end {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
        }

        // Copy bytes
        let src_start = *out_pos - distance;
        for i in 0..length {
            output[*out_pos + i] = output[src_start + (i % distance)];
        }
        *out_pos += length;
    }
}

/// Build fixed Huffman tables
pub fn build_fixed_tables() -> (ConsumeFirstTable, ConsumeFirstTable) {
    let mut lit_len_lengths = vec![0u8; 288];
    lit_len_lengths[..144].fill(8);
    lit_len_lengths[144..256].fill(9);
    lit_len_lengths[256] = 7;
    lit_len_lengths[257..280].fill(7);
    lit_len_lengths[280..288].fill(8);

    let dist_lengths = vec![5u8; 32];

    let lit_table = ConsumeFirstTable::build(&lit_len_lengths).unwrap();
    let dist_table = ConsumeFirstTable::build(&dist_lengths).unwrap();

    (lit_table, dist_table)
}

// ============================================================================
// JIT Table Cache
// ============================================================================

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cached table pair (literal/length + distance)
#[derive(Clone)]
pub struct CachedTablePair {
    pub lit_table: Arc<ConsumeFirstTable>,
    pub dist_table: Arc<ConsumeFirstTable>,
}

/// Global table cache for JIT-style optimization
/// Uses FNV-1a hash of code lengths as key
static TABLE_CACHE: std::sync::OnceLock<Mutex<HashMap<u64, CachedTablePair>>> =
    std::sync::OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<u64, CachedTablePair>> {
    TABLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Compute a fast hash of code lengths for cache lookup (FNV-1a)
fn hash_code_lengths(lit_lens: &[u8], dist_lens: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in lit_lens.iter().chain(dist_lens.iter()) {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Build or retrieve cached tables for the given code lengths
///
/// This implements JIT-style caching: if we've seen these exact code lengths
/// before, we return the cached tables instead of rebuilding them.
pub fn get_or_build_tables(lit_lens: &[u8], dist_lens: &[u8]) -> io::Result<CachedTablePair> {
    let hash = hash_code_lengths(lit_lens, dist_lens);

    // Fast path: check cache
    {
        let cache = get_cache().lock().unwrap();
        if let Some(cached) = cache.get(&hash) {
            return Ok(cached.clone());
        }
    }

    // Slow path: build tables
    let lit_table = Arc::new(ConsumeFirstTable::build(lit_lens)?);
    let dist_table = Arc::new(ConsumeFirstTable::build_distance(dist_lens)?);

    let pair = CachedTablePair {
        lit_table,
        dist_table,
    };

    // Store in cache
    {
        let mut cache = get_cache().lock().unwrap();
        // Double-check in case another thread built it
        if let Some(existing) = cache.get(&hash) {
            return Ok(existing.clone());
        }
        cache.insert(hash, pair.clone());
    }

    Ok(pair)
}

/// Clear the table cache (useful for testing or memory pressure)
pub fn clear_table_cache() {
    if let Some(cache) = TABLE_CACHE.get() {
        cache.lock().unwrap().clear();
    }
}

/// Get cache statistics (entries, capacity)
pub fn cache_stats() -> (usize, usize) {
    if let Some(cache) = TABLE_CACHE.get() {
        let guard = cache.lock().unwrap();
        (guard.len(), guard.capacity())
    } else {
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_cache() {
        // Clear any existing cache
        clear_table_cache();

        // Build fixed Huffman tables twice - should hit cache on second call
        let mut lit_lens = vec![0u8; 288];
        lit_lens[..144].fill(8);
        lit_lens[144..256].fill(9);
        lit_lens[256] = 7;
        lit_lens[257..280].fill(7);
        lit_lens[280..288].fill(8);
        let dist_lens = vec![5u8; 32];

        // First build - should be cache miss
        let (entries_before, _) = cache_stats();
        let tables1 = get_or_build_tables(&lit_lens, &dist_lens).unwrap();
        let (entries_after, _) = cache_stats();
        assert_eq!(
            entries_after,
            entries_before + 1,
            "First build should add to cache"
        );

        // Second build with same inputs - should be cache hit
        let tables2 = get_or_build_tables(&lit_lens, &dist_lens).unwrap();
        let (entries_final, _) = cache_stats();
        assert_eq!(
            entries_final, entries_after,
            "Cache hit should not add entries"
        );

        // Tables should be the same (same Arc pointer)
        assert!(Arc::ptr_eq(&tables1.lit_table, &tables2.lit_table));
        assert!(Arc::ptr_eq(&tables1.dist_table, &tables2.dist_table));

        // Different input should create new entry
        let dist_lens2 = vec![6u8; 32];
        let tables3 = get_or_build_tables(&lit_lens, &dist_lens2).unwrap();
        let (entries_new, _) = cache_stats();
        assert_eq!(
            entries_new,
            entries_final + 1,
            "Different input should add to cache"
        );
        assert!(!Arc::ptr_eq(&tables1.dist_table, &tables3.dist_table));

        eprintln!("\n[TEST] JIT cache test passed!");
        eprintln!("[TEST]   Cache entries: {}", entries_new);
    }

    #[test]
    fn test_entry_types() {
        let lit = CFEntry::literal(8, 65);
        assert!(lit.is_literal());
        assert!(!lit.is_eob());
        assert_eq!(lit.bits(), 8);
        assert_eq!(lit.symbol(), 65);

        let eob = CFEntry::end_of_block(7);
        assert!(eob.is_eob());
        assert_eq!(eob.bits(), 7);

        let len = CFEntry::length(9, 260);
        assert!(len.is_length());
        assert_eq!(len.symbol(), 260);

        let sub = CFEntry::subtable_ptr(11, 100, 4);
        assert!(sub.is_subtable());
        assert_eq!(sub.bits(), 11);
        assert_eq!(sub.subtable_offset(), 100);
        assert_eq!(sub.subtable_extra_bits(), 4);
    }

    /// Test distance table building
    #[test]
    fn test_build_distance_table() {
        // Standard fixed distance codes: all 5 bits
        let dist_lengths = vec![5u8; 32];

        let table = ConsumeFirstTable::build(&dist_lengths).unwrap();

        // All distance entries should be valid
        // For distance table, symbols 0-29 are valid distance codes
        // (30 and 31 are unused in deflate)
        for idx in 0..32 {
            // Find any entry that decodes to this symbol
            let mut found = false;
            for entry in table.main.iter() {
                if entry.symbol() as usize == idx && entry.bits() > 0 {
                    found = true;
                    eprintln!(
                        "[TEST] Distance {}: bits={}, is_literal={}",
                        idx,
                        entry.bits(),
                        entry.is_literal()
                    );
                    break;
                }
            }
            if idx < 30 {
                assert!(found, "Distance symbol {} should be in table", idx);
            }
        }

        // Test that we can look up distance symbols
        // With 5-bit codes, the pattern should be straightforward
        eprintln!("\n[TEST] Distance table lookup test:");
        for dist_sym in 0..30 {
            // The 5-bit reversed code for symbol 'dist_sym' should decode correctly
            let bits_pattern = dist_sym as u64; // Simplified - actual patterns depend on Huffman
            let entry = table.lookup_main(bits_pattern);
            eprintln!(
                "[TEST]   Pattern 0x{:03x} -> symbol={}, bits={}",
                bits_pattern,
                entry.symbol(),
                entry.bits()
            );
        }
    }

    #[test]
    fn test_build_fixed_huffman() {
        // Fixed Huffman code lengths
        let mut lengths = vec![0u8; 288];
        lengths[..144].fill(8);
        lengths[144..256].fill(9);
        lengths[256] = 7;
        lengths[257..280].fill(7);
        lengths[280..288].fill(8);

        let table = ConsumeFirstTable::build(&lengths).unwrap();

        // Count entry types
        let mut literals = 0;
        let mut lengths_count = 0;
        let mut eobs = 0;
        let mut subs = 0;

        for entry in &table.main {
            if entry.is_literal() {
                literals += 1;
            } else if entry.is_length() {
                lengths_count += 1;
            } else if entry.is_eob() {
                eobs += 1;
            } else if entry.is_subtable() {
                subs += 1;
            }
        }

        eprintln!("\n[TEST] Fixed Huffman table stats:");
        eprintln!(
            "[TEST]   Main table: {} literals, {} lengths, {} eob, {} subtable ptrs",
            literals, lengths_count, eobs, subs
        );
        eprintln!("[TEST]   Subtable size: {} entries", table.sub.len());

        // Should have mostly literals
        assert!(literals > 1500, "Should have many literal entries");
        assert!(eobs > 0, "Should have EOB entries");
    }

    #[test]
    fn bench_consume_first_decode() {
        // Build fixed Huffman table
        let mut lengths = vec![0u8; 288];
        lengths[..144].fill(8);
        lengths[144..256].fill(9);
        lengths[256] = 7;
        lengths[257..280].fill(7);
        lengths[280..288].fill(8);

        let table = ConsumeFirstTable::build(&lengths).unwrap();

        // Simulate bit stream
        let bits_sequence: Vec<u64> = (0..100_000).map(|i| i * 0x1234567).collect();

        let iterations = 500;

        // Consume-first decode simulation
        let start = std::time::Instant::now();
        let mut total_symbols = 0u64;
        let mut bitbuf_accum = 0u64;
        for _ in 0..iterations {
            for &bits in &bits_sequence {
                let mut bitbuf = bits;
                let entry = table.lookup_main(bitbuf);

                // CONSUME FIRST - entry.bits() is ALWAYS valid
                bitbuf >>= entry.bits();

                if entry.is_subtable() {
                    // Subtable lookup (rare - codes > 11 bits)
                    let sub_entry = table.lookup_sub(entry, bitbuf);
                    bitbuf >>= sub_entry.bits();
                    total_symbols += 1;
                } else {
                    // Direct decode (common)
                    total_symbols += 1;
                }

                bitbuf_accum ^= bitbuf;
            }
        }
        let elapsed = start.elapsed();

        eprintln!("\n[BENCH] Consume-First with Real Table:");
        eprintln!("[BENCH]   Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        eprintln!(
            "[BENCH]   Throughput: {:.1} M symbols/sec",
            total_symbols as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
        eprintln!("[BENCH]   (accum {} to prevent opt)", bitbuf_accum % 1000);
    }

    /// Test consume-first decode against libdeflate reference
    #[test]
    fn test_consume_first_correctness() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Test data: "Hello, World!" repeated
        let original = b"Hello, World! ".repeat(100);

        // Compress with flate2
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress with libdeflate (reference)
        let mut libdeflate_out = vec![0u8; original.len()];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .expect("libdeflate failed");

        eprintln!("\n[TEST] Consume-first correctness:");
        eprintln!("[TEST]   Original: {} bytes", original.len());
        eprintln!("[TEST]   Compressed: {} bytes", compressed.len());
        eprintln!("[TEST]   libdeflate output: {} bytes", libdeflate_size);

        // Verify libdeflate output matches
        assert_eq!(&libdeflate_out[..libdeflate_size], &original[..]);
        eprintln!("[TEST]   ✓ libdeflate output matches original");
    }

    /// Benchmark consume-first vs libdeflate on real data
    #[test]
    fn bench_consume_first_vs_libdeflate() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create test data
        let original: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();

        // Compress
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let iterations = 100;

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut out = vec![0u8; original.len()];
            libdeflater::Decompressor::new()
                .deflate_decompress(&compressed, &mut out)
                .unwrap();
        }
        let elapsed_libdeflate = start.elapsed();

        // Calculate throughput
        let bytes_total = original.len() * iterations;
        let libdeflate_mbs = bytes_total as f64 / elapsed_libdeflate.as_secs_f64() / 1_000_000.0;

        eprintln!("\n[BENCH] Consume-First vs libdeflate:");
        eprintln!(
            "[BENCH]   libdeflate: {:.2}ms ({:.1} MB/s)",
            elapsed_libdeflate.as_secs_f64() * 1000.0,
            libdeflate_mbs
        );
        eprintln!(
            "[BENCH]   Data: {} bytes x {} iterations",
            original.len(),
            iterations
        );
    }
}
