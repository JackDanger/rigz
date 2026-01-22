//! Libdeflate-compatible Huffman Table Entry Format
//!
//! This module implements the EXACT entry format used by libdeflate,
//! which is the key to matching or exceeding its performance.
//!
//! ## Entry Format (matching libdeflate/lib/deflate_decompress.c lines 439-471)
//!
//! ### Literal/Length Table Entries
//!
//! ```text
//! Literals:
//!   Bit 31:     1 (HUFFDEC_LITERAL)
//!   Bit 23-16:  literal value
//!   Bit 15:     0 (!HUFFDEC_EXCEPTIONAL)
//!   Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
//!   Bit 13:     0 (!HUFFDEC_END_OF_BLOCK)
//!   Bit 11-8:   remaining codeword length [not used]
//!   Bit 3-0:    remaining codeword length
//!
//! Lengths:
//!   Bit 31:     0 (!HUFFDEC_LITERAL)
//!   Bit 24-16:  length base value (3-258)
//!   Bit 15:     0 (!HUFFDEC_EXCEPTIONAL)
//!   Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
//!   Bit 13:     0 (!HUFFDEC_END_OF_BLOCK)
//!   Bit 11-8:   remaining codeword length
//!   Bit 4-0:    remaining codeword length + number of extra bits
//!
//! End of block:
//!   Bit 31:     0 (!HUFFDEC_LITERAL)
//!   Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
//!   Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
//!   Bit 13:     1 (HUFFDEC_END_OF_BLOCK)
//!   Bit 11-8:   remaining codeword length [not used]
//!   Bit 3-0:    remaining codeword length
//!
//! Subtable pointer:
//!   Bit 31:     0 (!HUFFDEC_LITERAL)
//!   Bit 30-16:  index of start of subtable
//!   Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
//!   Bit 14:     1 (HUFFDEC_SUBTABLE_POINTER)
//!   Bit 13:     0 (!HUFFDEC_END_OF_BLOCK)
//!   Bit 11-8:   number of subtable bits
//!   Bit 3-0:    number of main table bits
//! ```
//!
//! ### Distance Table Entries
//!
//! ```text
//! Distances:
//!   Bit 31-16:  offset base value (1-32768)
//!   Bit 15:     0 (!HUFFDEC_EXCEPTIONAL)
//!   Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
//!   Bit 11-8:   remaining codeword length
//!   Bit 4-0:    remaining codeword length + number of extra bits
//!
//! Subtable pointer:
//!   Bit 31-16:  index of start of subtable
//!   Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
//!   Bit 14:     1 (HUFFDEC_SUBTABLE_POINTER)
//!   Bit 11-8:   number of subtable bits
//!   Bit 3-0:    number of main table bits
//! ```

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

/// Flag: Entry is a literal (bit 31)
pub const HUFFDEC_LITERAL: u32 = 0x8000_0000;

/// Flag: Entry is exceptional (subtable pointer or EOB) (bit 15)
pub const HUFFDEC_EXCEPTIONAL: u32 = 0x0000_8000;

/// Flag: Entry is a subtable pointer (bit 14)
pub const HUFFDEC_SUBTABLE_POINTER: u32 = 0x0000_4000;

/// Flag: Entry is end-of-block (bit 13)
pub const HUFFDEC_END_OF_BLOCK: u32 = 0x0000_2000;

/// A literal/length table entry
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct LitLenEntry(u32);

impl LitLenEntry {
    /// Create a literal entry
    #[inline(always)]
    pub const fn literal(value: u8, codeword_bits: u8) -> Self {
        // Bit 31: LITERAL flag
        // Bit 23-16: literal value
        // Bit 3-0: codeword bits (for consumption)
        Self(HUFFDEC_LITERAL | ((value as u32) << 16) | (codeword_bits as u32))
    }

    /// Create a length entry
    #[inline(always)]
    pub const fn length(base: u16, extra_bits: u8, codeword_bits: u8) -> Self {
        // Bit 24-16: length base (9 bits, max 258)
        // Bit 11-8: remaining codeword length (for saved_bitbuf)
        // Bit 4-0: total bits = codeword_bits + extra_bits
        let total_bits = codeword_bits + extra_bits;
        Self(((base as u32) << 16) | ((codeword_bits as u32) << 8) | (total_bits as u32))
    }

    /// Create an end-of-block entry
    #[inline(always)]
    pub const fn end_of_block(codeword_bits: u8) -> Self {
        Self(HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK | (codeword_bits as u32))
    }

    /// Create a subtable pointer entry
    #[inline(always)]
    pub const fn subtable_ptr(subtable_start: u16, subtable_bits: u8, main_table_bits: u8) -> Self {
        Self(
            ((subtable_start as u32) << 16)
                | HUFFDEC_EXCEPTIONAL
                | HUFFDEC_SUBTABLE_POINTER
                | ((subtable_bits as u32) << 8)
                | (main_table_bits as u32),
        )
    }

    /// Check if this is a literal (bit 31 set)
    /// Uses signed comparison: (entry as i32) < 0
    #[inline(always)]
    pub const fn is_literal(self) -> bool {
        (self.0 as i32) < 0
    }

    /// Check if this is exceptional (subtable or EOB)
    #[inline(always)]
    pub const fn is_exceptional(self) -> bool {
        (self.0 & HUFFDEC_EXCEPTIONAL) != 0
    }

    /// Check if this is a subtable pointer
    #[inline(always)]
    pub const fn is_subtable_ptr(self) -> bool {
        (self.0 & HUFFDEC_SUBTABLE_POINTER) != 0
    }

    /// Check if this is end-of-block
    #[inline(always)]
    pub const fn is_end_of_block(self) -> bool {
        (self.0 & HUFFDEC_END_OF_BLOCK) != 0
    }

    /// Get the literal value (bits 23-16)
    #[inline(always)]
    pub const fn literal_value(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }

    /// Get the length base value (bits 24-16)
    #[inline(always)]
    pub const fn length_base(self) -> u16 {
        ((self.0 >> 16) & 0x1FF) as u16
    }

    /// Get the codeword bits (bits 11-8 for lengths, bits 3-0 for others)
    #[inline(always)]
    pub const fn codeword_bits(self) -> u8 {
        if self.is_literal() || self.is_exceptional() {
            // For literals and EOB, bits 3-0 contain codeword bits
            (self.0 & 0xF) as u8
        } else {
            // For lengths, bits 11-8 contain codeword bits
            ((self.0 >> 8) & 0xF) as u8
        }
    }

    /// Get total bits to consume (bits 4-0)
    /// For literals: just codeword bits
    /// For lengths: codeword + extra bits
    #[inline(always)]
    pub const fn total_bits(self) -> u8 {
        (self.0 & 0x1F) as u8
    }

    /// Get subtable start index (bits 30-16)
    #[inline(always)]
    pub const fn subtable_start(self) -> u16 {
        ((self.0 >> 16) & 0x7FFF) as u16
    }

    /// Get subtable bits (bits 11-8)
    #[inline(always)]
    pub const fn subtable_bits(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }

    /// Get main table bits (bits 3-0) for subtable entries
    #[inline(always)]
    pub const fn main_table_bits(self) -> u8 {
        (self.0 & 0xF) as u8
    }

    /// Get raw value for bit operations
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create from raw u32 value
    #[inline(always)]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Decode length value using saved_bitbuf
    /// Length = base + extra_bits_value
    /// Decode length from saved_bitbuf (branchless)
    #[inline(always)]
    pub fn decode_length(self, saved_bitbuf: u64) -> u32 {
        let base = self.length_base() as u32;
        let codeword_bits = self.codeword_bits();
        let total_bits = self.total_bits();
        let extra_bits = total_bits - codeword_bits;
        // Branchless: when extra_bits is 0, mask is 0 and extra_value is 0
        let extra_mask = (1u64 << extra_bits).wrapping_sub(1);
        let extra_value = (saved_bitbuf >> codeword_bits) & extra_mask;
        base + extra_value as u32
    }
}

/// A distance table entry
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct DistEntry(u32);

impl DistEntry {
    /// Create a distance entry
    #[inline(always)]
    pub const fn distance(base: u16, extra_bits: u8, codeword_bits: u8) -> Self {
        // Bit 31-16: offset base value
        // Bit 11-8: remaining codeword length
        // Bit 4-0: total bits = codeword_bits + extra_bits
        let total_bits = codeword_bits + extra_bits;
        Self(((base as u32) << 16) | ((codeword_bits as u32) << 8) | (total_bits as u32))
    }

    /// Create a subtable pointer entry
    #[inline(always)]
    pub const fn subtable_ptr(subtable_start: u16, subtable_bits: u8, main_table_bits: u8) -> Self {
        Self(
            ((subtable_start as u32) << 16)
                | HUFFDEC_EXCEPTIONAL
                | HUFFDEC_SUBTABLE_POINTER
                | ((subtable_bits as u32) << 8)
                | (main_table_bits as u32),
        )
    }

    /// Check if this is exceptional (subtable pointer)
    #[inline(always)]
    pub const fn is_exceptional(self) -> bool {
        (self.0 & HUFFDEC_EXCEPTIONAL) != 0
    }

    /// Check if this is a subtable pointer
    #[inline(always)]
    pub const fn is_subtable_ptr(self) -> bool {
        (self.0 & HUFFDEC_SUBTABLE_POINTER) != 0
    }

    /// Get the distance base value (bits 31-16)
    #[inline(always)]
    pub const fn distance_base(self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    /// Get the codeword bits (bits 11-8)
    #[inline(always)]
    pub const fn codeword_bits(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }

    /// Get total bits to consume (bits 4-0)
    #[inline(always)]
    pub const fn total_bits(self) -> u8 {
        (self.0 & 0x1F) as u8
    }

    /// Get subtable start index (bits 31-16)
    #[inline(always)]
    pub const fn subtable_start(self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    /// Get subtable bits (bits 11-8)
    #[inline(always)]
    pub const fn subtable_bits(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }

    /// Get main table bits (bits 3-0) for subtable entries
    #[inline(always)]
    pub const fn main_table_bits(self) -> u8 {
        (self.0 & 0xF) as u8
    }

    /// Get raw value for bit operations
    #[inline(always)]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Decode distance value using saved_bitbuf
    /// Distance = base + extra_bits_value
    /// Decode distance from saved_bitbuf (branchless, uses BMI2 when available)
    #[inline(always)]
    pub fn decode_distance(self, saved_bitbuf: u64) -> u32 {
        let base = self.distance_base() as u32;
        let codeword_bits = self.codeword_bits();
        let total_bits = self.total_bits();
        let extra_bits = total_bits - codeword_bits;
        // Use BMI2 _bzhi_u64 when available for faster bit extraction
        let extra_value =
            crate::bmi2::decode_extra_bits(saved_bitbuf, codeword_bits, extra_bits) as u32;
        base + extra_value
    }
}

/// Literal/Length decode table
#[derive(Clone)]
pub struct LitLenTable {
    /// Main table (size: 1 << table_bits)
    entries: Vec<LitLenEntry>,
    /// Number of bits for main table lookup
    table_bits: u8,
}

impl LitLenTable {
    /// Number of bits for main table (11 = 8KB, fits L1 cache)
    pub const TABLE_BITS: u8 = 11;
    /// Maximum number of subtable bits
    pub const MAX_SUBTABLE_BITS: u8 = 4; // 15 - 11

    /// Build a literal/length decode table from code lengths
    pub fn build(code_lengths: &[u8]) -> Option<Self> {
        let table_bits = Self::TABLE_BITS;
        let main_size = 1usize << table_bits;

        // Count codes of each length
        let mut count = [0u16; 16];
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                count[len as usize] += 1;
            }
        }

        // Compute first code for each length
        let mut first_code = [0u32; 16];
        let mut code = 0u32;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }

        // Allocate table with space for subtables
        let max_subtable_entries = (1usize << Self::MAX_SUBTABLE_BITS)
            * code_lengths.iter().filter(|&&l| l > table_bits).count();
        let mut entries = vec![LitLenEntry(0); main_size + max_subtable_entries];
        let mut subtable_next = main_size;

        // Assign codes to symbols
        let mut next_code = first_code;
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as usize;
            let codeword = next_code[len];
            next_code[len] += 1;

            // Reverse the codeword bits for table lookup
            let reversed = reverse_bits(codeword, len as u8);

            if len <= table_bits as usize {
                // Check if this is a length code that can be pre-expanded
                if let Some((base, extra_bits)) = get_length_info(symbol) {
                    let total_bits = len + extra_bits as usize;
                    if total_bits <= table_bits as usize && extra_bits > 0 {
                        // PRE-EXPAND: Create separate entries for each extra bit combination
                        // This eliminates runtime extra bit reading for these codes
                        let num_expansions = 1usize << extra_bits;
                        for extra_val in 0..num_expansions {
                            let final_length = base + extra_val as u16;
                            let entry =
                                create_preexpanded_length_entry(final_length, total_bits as u8);

                            // The expanded code is: reversed_codeword | (extra_val << codeword_len)
                            let expanded_code = (reversed as usize) | (extra_val << len);
                            let stride = 1usize << total_bits;
                            let mut idx = expanded_code;
                            while idx < main_size {
                                entries[idx] = entry;
                                idx += stride;
                            }
                        }
                        continue; // Skip the normal entry creation
                    }
                }

                // Direct entry in main table - replicate for all suffixes
                let entry = create_litlen_entry(symbol, len as u8);
                let stride = 1usize << len;
                let mut idx = reversed as usize;
                while idx < main_size {
                    entries[idx] = entry;
                    idx += stride;
                }
            } else {
                // Need subtable
                let main_idx = (reversed & ((1 << table_bits) - 1)) as usize;
                let extra_bits = len - table_bits as usize;

                // Check if subtable already exists
                if !entries[main_idx].is_subtable_ptr() {
                    // Create new subtable
                    let subtable_start = subtable_next as u16;
                    let subtable_bits = Self::MAX_SUBTABLE_BITS;
                    entries[main_idx] =
                        LitLenEntry::subtable_ptr(subtable_start, subtable_bits, table_bits);
                    subtable_next += 1usize << subtable_bits;
                }

                // Fill subtable entry
                let subtable_start = entries[main_idx].subtable_start() as usize;
                let subtable_bits = entries[main_idx].subtable_bits() as usize;
                let subtable_idx = (reversed >> table_bits) as usize;

                // For subtable entries, store (len - table_bits) not full len
                // This is the subtable portion only
                let subtable_len = (len - table_bits as usize) as u8;
                let entry = create_litlen_entry(symbol, subtable_len);
                let stride = 1usize << extra_bits;
                let mut idx = subtable_idx;
                while idx < (1usize << subtable_bits) {
                    entries[subtable_start + idx] = entry;
                    idx += stride;
                }
            }
        }

        entries.truncate(subtable_next);
        Some(Self {
            entries,
            table_bits,
        })
    }

    /// Look up an entry by bit pattern (unsafe unchecked for max speed)
    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> LitLenEntry {
        // Use const TABLE_BITS for compile-time optimization
        const MASK: usize = (1usize << LitLenTable::TABLE_BITS) - 1;
        let idx = (bits as usize) & MASK;
        // SAFETY: idx is masked to be within table_bits range,
        // and entries is always at least (1 << table_bits) in size
        unsafe { *self.entries.get_unchecked(idx) }
    }

    /// Get raw pointer to entries for even faster access
    #[inline(always)]
    pub fn entries_ptr(&self) -> *const LitLenEntry {
        self.entries.as_ptr()
    }

    /// Look up entry by direct index (for SIMD parallel decode)
    #[inline(always)]
    pub fn lookup_by_index(&self, idx: usize) -> LitLenEntry {
        // SAFETY: caller must ensure idx is within table bounds (0..2048 for main table)
        unsafe { *self.entries.get_unchecked(idx) }
    }

    /// Look up a subtable entry (unsafe unchecked for max speed)
    #[inline(always)]
    pub fn lookup_subtable(&self, entry: LitLenEntry, bits: u64) -> LitLenEntry {
        let subtable_start = entry.subtable_start() as usize;
        let subtable_bits = entry.subtable_bits();
        let main_bits = entry.main_table_bits();
        let idx = ((bits >> main_bits) as usize) & ((1usize << subtable_bits) - 1);
        // SAFETY: subtable entries are allocated during build
        unsafe { *self.entries.get_unchecked(subtable_start + idx) }
    }

    /// Resolve an entry (handle subtables)
    #[inline(always)]
    pub fn resolve(&self, bits: u64) -> LitLenEntry {
        let entry = self.lookup(bits);
        if entry.is_subtable_ptr() {
            self.lookup_subtable(entry, bits)
        } else {
            entry
        }
    }

    /// Resolve an entry and return total bits consumed (for double-literal cache)
    /// Returns (entry, total_bits) where total_bits includes TABLE_BITS for subtable entries
    #[inline(always)]
    pub fn resolve_with_total_bits(&self, bits: u64) -> (LitLenEntry, u8) {
        let entry = self.lookup(bits);
        if entry.is_subtable_ptr() {
            let subtable_entry = self.lookup_subtable(entry, bits);
            let total_bits = Self::TABLE_BITS + subtable_entry.codeword_bits();
            (subtable_entry, total_bits)
        } else {
            (entry, entry.codeword_bits())
        }
    }
}

/// Distance decode table
#[derive(Clone)]
pub struct DistTable {
    /// Main table (size: 1 << table_bits)
    entries: Vec<DistEntry>,
    /// Number of bits for main table lookup
    table_bits: u8,
}

impl DistTable {
    /// Number of bits for main table
    pub const TABLE_BITS: u8 = 8;
    /// Maximum number of subtable bits
    pub const MAX_SUBTABLE_BITS: u8 = 7; // 15 - 8

    /// Build a distance decode table from code lengths
    pub fn build(code_lengths: &[u8]) -> Option<Self> {
        let table_bits = Self::TABLE_BITS;
        let main_size = 1usize << table_bits;

        // Count codes of each length
        let mut count = [0u16; 16];
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                count[len as usize] += 1;
            }
        }

        // Compute first code for each length
        let mut first_code = [0u32; 16];
        let mut code = 0u32;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
        }

        // Allocate table with space for subtables
        let max_subtable_entries = (1usize << Self::MAX_SUBTABLE_BITS)
            * code_lengths.iter().filter(|&&l| l > table_bits).count();
        let mut entries = vec![DistEntry(0); main_size + max_subtable_entries];
        let mut subtable_next = main_size;

        // Assign codes to symbols
        let mut next_code = first_code;
        for (symbol, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as usize;
            let codeword = next_code[len];
            next_code[len] += 1;

            // Reverse the codeword bits for table lookup
            let reversed = reverse_bits(codeword, len as u8);

            if len <= table_bits as usize {
                // Direct entry in main table - replicate for all suffixes
                let entry = create_dist_entry(symbol, len as u8);
                let stride = 1usize << len;
                let mut idx = reversed as usize;
                while idx < main_size {
                    entries[idx] = entry;
                    idx += stride;
                }
            } else {
                // Need subtable
                let main_idx = (reversed & ((1 << table_bits) - 1)) as usize;
                let extra_bits = len - table_bits as usize;

                // Check if subtable already exists
                if !entries[main_idx].is_subtable_ptr() {
                    // Create new subtable
                    let subtable_start = subtable_next as u16;
                    let subtable_bits = Self::MAX_SUBTABLE_BITS;
                    entries[main_idx] =
                        DistEntry::subtable_ptr(subtable_start, subtable_bits, table_bits);
                    subtable_next += 1usize << subtable_bits;
                }

                // Fill subtable entry
                let subtable_start = entries[main_idx].subtable_start() as usize;
                let subtable_bits = entries[main_idx].subtable_bits() as usize;
                let subtable_idx = (reversed >> table_bits) as usize;

                // For subtable entries, store (len - table_bits) not full len
                let subtable_len = (len - table_bits as usize) as u8;
                let entry = create_dist_entry(symbol, subtable_len);
                let stride = 1usize << extra_bits;
                let mut idx = subtable_idx;
                while idx < (1usize << subtable_bits) {
                    entries[subtable_start + idx] = entry;
                    idx += stride;
                }
            }
        }

        entries.truncate(subtable_next);
        Some(Self {
            entries,
            table_bits,
        })
    }

    /// Look up an entry by bit pattern (unsafe unchecked for max speed)
    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> DistEntry {
        // Use const TABLE_BITS for compile-time optimization
        const MASK: usize = (1usize << DistTable::TABLE_BITS) - 1;
        let idx = (bits as usize) & MASK;
        // SAFETY: idx is masked to be within table_bits range
        unsafe { *self.entries.get_unchecked(idx) }
    }

    /// Look up a subtable entry (unsafe unchecked for max speed)
    #[inline(always)]
    pub fn lookup_subtable(&self, entry: DistEntry, bits: u64) -> DistEntry {
        let subtable_start = entry.subtable_start() as usize;
        let subtable_bits = entry.subtable_bits();
        let main_bits = entry.main_table_bits();
        let idx = ((bits >> main_bits) as usize) & ((1usize << subtable_bits) - 1);
        // SAFETY: subtable entries are allocated during build
        unsafe { *self.entries.get_unchecked(subtable_start + idx) }
    }

    /// Look up subtable entry from already-shifted bitbuf (libdeflate fastloop pattern)
    /// Use when bitbuf has already been shifted by main table bits
    #[inline(always)]
    pub fn lookup_subtable_direct(&self, entry: DistEntry, shifted_bits: u64) -> DistEntry {
        let subtable_start = entry.subtable_start() as usize;
        let subtable_bits = entry.subtable_bits();
        let idx = (shifted_bits as usize) & ((1usize << subtable_bits) - 1);
        // SAFETY: subtable entries are allocated during build
        unsafe { *self.entries.get_unchecked(subtable_start + idx) }
    }

    /// Resolve an entry (handle subtables)
    #[inline(always)]
    pub fn resolve(&self, bits: u64) -> DistEntry {
        let entry = self.lookup(bits);
        if entry.is_subtable_ptr() {
            self.lookup_subtable(entry, bits)
        } else {
            entry
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Reverse the bottom `n` bits of `code`
fn reverse_bits(code: u32, n: u8) -> u32 {
    let mut result = 0u32;
    let mut code = code;
    for _ in 0..n {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

/// Length base values and extra bits (RFC 1951)
pub const LENGTH_TABLE: [(u16, u8); 29] = [
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0), // 257-264
    (11, 1),
    (13, 1),
    (15, 1),
    (17, 1), // 265-268
    (19, 2),
    (23, 2),
    (27, 2),
    (31, 2), // 269-272
    (35, 3),
    (43, 3),
    (51, 3),
    (59, 3), // 273-276
    (67, 4),
    (83, 4),
    (99, 4),
    (115, 4), // 277-280
    (131, 5),
    (163, 5),
    (195, 5),
    (227, 5), // 281-284
    (258, 0), // 285
];

/// Distance base values and extra bits (RFC 1951)
pub const DISTANCE_TABLE: [(u16, u8); 30] = [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0), // 0-3
    (5, 1),
    (7, 1), // 4-5
    (9, 2),
    (13, 2), // 6-7
    (17, 3),
    (25, 3), // 8-9
    (33, 4),
    (49, 4), // 10-11
    (65, 5),
    (97, 5), // 12-13
    (129, 6),
    (193, 6), // 14-15
    (257, 7),
    (385, 7), // 16-17
    (513, 8),
    (769, 8), // 18-19
    (1025, 9),
    (1537, 9), // 20-21
    (2049, 10),
    (3073, 10), // 22-23
    (4097, 11),
    (6145, 11), // 24-25
    (8193, 12),
    (12289, 12), // 26-27
    (16385, 13),
    (24577, 13), // 28-29
];

/// Create a literal/length entry for a symbol
fn create_litlen_entry(symbol: usize, codeword_bits: u8) -> LitLenEntry {
    if symbol < 256 {
        // Literal
        LitLenEntry::literal(symbol as u8, codeword_bits)
    } else if symbol == 256 {
        // End of block
        LitLenEntry::end_of_block(codeword_bits)
    } else if symbol <= 285 {
        // Length code
        let idx = symbol - 257;
        let (base, extra) = LENGTH_TABLE[idx];
        LitLenEntry::length(base, extra, codeword_bits)
    } else {
        // Invalid symbol, create a dummy entry
        LitLenEntry(0)
    }
}

/// Create a pre-expanded length entry where length is already computed
/// Used when codeword + extra_bits fits in the table, avoiding runtime extra bit reading
#[inline(always)]
fn create_preexpanded_length_entry(length: u16, total_bits: u8) -> LitLenEntry {
    // For pre-expanded entries:
    // - Bit 24-16: final length value (not base, but actual length)
    // - Bit 11-8: total_bits (for saved_bitbuf, same as bits 4-0)
    // - Bit 4-0: total_bits (no extra bits to read at decode time)
    // The key insight: extra_bits field = 0, so decode_length just returns base
    LitLenEntry(((length as u32) << 16) | ((total_bits as u32) << 8) | (total_bits as u32))
}

/// Check if a length symbol can be pre-expanded and return the expansion info
/// Returns Some((base, extra_bits)) if symbol is a length code, None otherwise
fn get_length_info(symbol: usize) -> Option<(u16, u8)> {
    if (257..=285).contains(&symbol) {
        let idx = symbol - 257;
        Some(LENGTH_TABLE[idx])
    } else {
        None
    }
}

/// Create a distance entry for a symbol
fn create_dist_entry(symbol: usize, codeword_bits: u8) -> DistEntry {
    if symbol < 30 {
        let (base, extra) = DISTANCE_TABLE[symbol];
        DistEntry::distance(base, extra, codeword_bits)
    } else {
        // Invalid symbol, create a dummy entry
        DistEntry(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_entry() {
        let entry = LitLenEntry::literal(b'A', 8);
        assert!(entry.is_literal());
        assert!(!entry.is_exceptional());
        assert!(!entry.is_subtable_ptr());
        assert!(!entry.is_end_of_block());
        assert_eq!(entry.literal_value(), b'A');
        assert_eq!(entry.total_bits(), 8);
    }

    #[test]
    fn test_length_entry() {
        // Symbol 257 = length 3, 0 extra bits
        let entry = LitLenEntry::length(3, 0, 7);
        assert!(!entry.is_literal());
        assert!(!entry.is_exceptional());
        assert_eq!(entry.length_base(), 3);
        assert_eq!(entry.codeword_bits(), 7);
        assert_eq!(entry.total_bits(), 7);
    }

    #[test]
    fn test_length_with_extra() {
        // Symbol 265 = length 11-12, 1 extra bit
        let entry = LitLenEntry::length(11, 1, 7);
        assert_eq!(entry.length_base(), 11);
        assert_eq!(entry.codeword_bits(), 7);
        assert_eq!(entry.total_bits(), 8);

        // Decode with extra bit = 1 -> length = 11 + 1 = 12
        let length = entry.decode_length(0b1_0000000); // 7 codeword bits, then 1 extra bit
        assert_eq!(length, 12);
    }

    #[test]
    fn test_eob_entry() {
        let entry = LitLenEntry::end_of_block(7);
        assert!(!entry.is_literal());
        assert!(entry.is_exceptional());
        assert!(!entry.is_subtable_ptr());
        assert!(entry.is_end_of_block());
        assert_eq!(entry.total_bits(), 7);
    }

    #[test]
    fn test_subtable_ptr() {
        let entry = LitLenEntry::subtable_ptr(2048, 4, 11);
        assert!(!entry.is_literal());
        assert!(entry.is_exceptional());
        assert!(entry.is_subtable_ptr());
        assert!(!entry.is_end_of_block());
        assert_eq!(entry.subtable_start(), 2048);
        assert_eq!(entry.subtable_bits(), 4);
        assert_eq!(entry.main_table_bits(), 11);
    }

    #[test]
    fn test_distance_entry() {
        // Distance code 0 = distance 1, 0 extra bits
        let entry = DistEntry::distance(1, 0, 5);
        assert!(!entry.is_exceptional());
        assert_eq!(entry.distance_base(), 1);
        assert_eq!(entry.codeword_bits(), 5);
        assert_eq!(entry.total_bits(), 5);
    }

    #[test]
    fn test_distance_with_extra() {
        // Distance code 29 = distance 24577-32768, 13 extra bits
        let entry = DistEntry::distance(24577, 13, 5);
        assert_eq!(entry.distance_base(), 24577);
        assert_eq!(entry.codeword_bits(), 5);
        assert_eq!(entry.total_bits(), 18);

        // Decode with extra bits = 0x1FFF (max) -> distance = 24577 + 8191 = 32768
        let dist = entry.decode_distance(0b11_1111_1111_1110_0000); // 5 codeword bits, 13 extra
        assert_eq!(dist, 32768);
    }

    #[test]
    fn test_signed_literal_check() {
        // The key optimization: use (entry as i32) < 0 for literal check
        let literal = LitLenEntry::literal(b'X', 8);
        let length = LitLenEntry::length(3, 0, 7);
        let eob = LitLenEntry::end_of_block(7);

        // Signed comparison
        assert!((literal.raw() as i32) < 0);
        assert!((length.raw() as i32) >= 0);
        assert!((eob.raw() as i32) >= 0);
    }

    #[test]
    fn test_build_fixed_litlen_table() {
        // Build fixed Huffman table for lit/len
        let mut code_lengths = [0u8; 288];
        for i in 0..144 {
            code_lengths[i] = 8;
        }
        for i in 144..256 {
            code_lengths[i] = 9;
        }
        for i in 256..280 {
            code_lengths[i] = 7;
        }
        for i in 280..288 {
            code_lengths[i] = 8;
        }

        let table = LitLenTable::build(&code_lengths).expect("Failed to build table");

        // Test that we can look up entries
        // Check literal 'A' (65) which has 8-bit code
        // Fixed Huffman: symbols 0-143 get codes 00110000 + symbol (8 bits)
        // Symbol 65 = 0x41 + 0x30 = 0x71 = 01110001 binary
        // Reversed: 10001110 = 0x8E
        let entry = table.resolve(0x8E);
        assert!(
            entry.is_literal(),
            "Entry 0x8E should be literal, got {:08X}",
            entry.raw()
        );
        assert_eq!(entry.literal_value(), 65, "Should decode to 'A'");

        // Check end of block (symbol 256) which has 7-bit code
        // Fixed: symbols 256-279 get codes 0000000 + (symbol - 256) (7 bits)
        // Symbol 256 = 0000000
        // Reversed: 0000000 = 0x00
        let eob = table.resolve(0x00);
        assert!(
            eob.is_end_of_block(),
            "Entry 0x00 should be EOB, got {:08X}",
            eob.raw()
        );
    }

    #[test]
    fn test_build_fixed_dist_table() {
        // Build fixed Huffman table for distances (all 5-bit codes)
        let code_lengths = [5u8; 32];

        let table = DistTable::build(&code_lengths).expect("Failed to build table");

        // Test lookup for distance code 0
        let entry = table.resolve(0b00000); // 5-bit code for 0
        assert!(!entry.is_exceptional());
        assert_eq!(entry.distance_base(), 1);
        assert_eq!(entry.total_bits(), 5);
    }

    #[test]
    fn test_entry_format_matches_libdeflate() {
        // Verify our entry format exactly matches libdeflate's
        // From libdeflate: ENTRY(literal) = HUFFDEC_LITERAL | ((u32)literal << 16)

        let entry = LitLenEntry::literal(65, 8); // 'A'
        assert_eq!(entry.raw() & HUFFDEC_LITERAL, HUFFDEC_LITERAL);
        assert_eq!((entry.raw() >> 16) & 0xFF, 65);

        // EOB: HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK
        let eob = LitLenEntry::end_of_block(7);
        assert_eq!(
            eob.raw() & (HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK),
            HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK
        );
    }
}
