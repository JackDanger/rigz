//! Turbo Inflate - Ultra-fast deflate decompression
//!
//! Key optimizations from libdeflate and rapidgzip:
//! 1. Combined length+distance LUT - single lookup for full LZ77 match
//! 2. Packed entry format - flags, symbol, bits in single u32
//! 3. Multi-literal decode - up to 3 literals per loop iteration
//! 4. Minimal branching in hot path

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::io;

use crate::inflate_tables::{
    CODE_LENGTH_ORDER, DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START,
};

// =============================================================================
// Constants
// =============================================================================

/// Main table bits - 11 is optimal based on rapidgzip benchmarks
const TABLE_BITS: usize = 11;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: u64 = (TABLE_SIZE - 1) as u64;

/// Distance table bits - needs to be at least 10 for dynamic blocks
const DIST_TABLE_BITS: usize = 10;
const DIST_TABLE_SIZE: usize = 1 << DIST_TABLE_BITS;
const DIST_TABLE_MASK: u64 = (DIST_TABLE_SIZE - 1) as u64;

/// Entry type flags (in bits 8-9)
const TYPE_LITERAL: u32 = 0;
const TYPE_LENGTH: u32 = 1;
const TYPE_EOB: u32 = 2;
const TYPE_SUBTABLE: u32 = 3;

/// Marker values for combined entries
const DIST_EOB: u16 = 0xFFFF;
const DIST_SLOW: u16 = 0xFFFE;
const DIST_LITERAL: u16 = 0;

// =============================================================================
// Packed Entry Format (32 bits)
// =============================================================================
//
// For literals:
//   Bits 0-7:   Code length (1-15)
//   Bits 8-9:   TYPE_LITERAL (0)
//   Bits 16-23: Literal byte value
//
// For lengths (slow path):
//   Bits 0-7:   Code length
//   Bits 8-9:   TYPE_LENGTH (1)
//   Bits 10-13: Extra bits count
//   Bits 16-20: Length code index (0-28)
//
// For combined LZ77 (fast path):
//   Bits 0-7:   Total bits consumed (lit_len + extra + dist + dist_extra)
//   Bits 8-9:   TYPE_LENGTH (1)
//   Bits 10-15: Reserved
//   Bits 16-23: Length - 3 (0-255 maps to 3-258)
//   Combined with separate distance lookup
//
// For EOB:
//   Bits 0-7:   Code length
//   Bits 8-9:   TYPE_EOB (2)

#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct PackedEntry(pub u32);

impl PackedEntry {
    #[inline(always)]
    pub fn code_len(self) -> u32 {
        self.0 & 0xFF
    }

    #[inline(always)]
    pub fn entry_type(self) -> u32 {
        (self.0 >> 8) & 0x3
    }

    #[inline(always)]
    pub fn is_literal(self) -> bool {
        (self.0 >> 8) & 0x3 == TYPE_LITERAL
    }

    #[inline(always)]
    pub fn literal_byte(self) -> u8 {
        (self.0 >> 16) as u8
    }

    #[inline(always)]
    pub fn extra_bits(self) -> u32 {
        (self.0 >> 10) & 0xF
    }

    #[inline(always)]
    pub fn len_code_idx(self) -> usize {
        ((self.0 >> 16) & 0x1F) as usize
    }

    // Constructors
    #[inline(always)]
    fn literal(code_len: u8, byte: u8) -> Self {
        Self((code_len as u32) | (TYPE_LITERAL << 8) | ((byte as u32) << 16))
    }

    #[inline(always)]
    fn length(code_len: u8, extra_bits: u8, len_code_idx: u8) -> Self {
        Self(
            (code_len as u32)
                | (TYPE_LENGTH << 8)
                | ((extra_bits as u32) << 10)
                | ((len_code_idx as u32) << 16),
        )
    }

    #[inline(always)]
    fn eob(code_len: u8) -> Self {
        Self((code_len as u32) | (TYPE_EOB << 8))
    }
}

// =============================================================================
// Combined LUT Entry (for pre-decoded length+distance)
// =============================================================================

/// Combined entry for when we can pre-decode length and distance
#[derive(Clone, Copy, Default)]
pub struct CombinedEntry {
    pub bits_to_skip: u8,   // Total bits consumed
    pub length_minus_3: u8, // Length - 3 (so 0-255 = 3-258)
    pub distance: u16,      // Distance value (0xFFFF = EOB, 0xFFFE = slow path)
}

impl CombinedEntry {
    #[inline(always)]
    pub fn is_literal(self) -> bool {
        self.distance == DIST_LITERAL && self.bits_to_skip > 0
    }

    #[inline(always)]
    pub fn is_eob(self) -> bool {
        self.distance == DIST_EOB
    }

    #[inline(always)]
    pub fn is_slow_path(self) -> bool {
        self.distance == DIST_SLOW
    }

    #[inline(always)]
    pub fn length(self) -> usize {
        (self.length_minus_3 as usize) + 3
    }
}

// =============================================================================
// Distance Entry
// =============================================================================

#[derive(Clone, Copy, Default)]
pub struct DistEntry {
    pub code_len: u8,
    pub extra_bits: u8,
    pub base_dist: u16,
}

// =============================================================================
// Turbo Tables
// =============================================================================

/// Subtable entry for codes > TABLE_BITS
const SUBTABLE_BITS: usize = 4; // Max 4 extra bits (codes up to 15 bits)
const SUBTABLE_SIZE: usize = 1 << SUBTABLE_BITS;

pub struct TurboTables {
    /// Literal/length table with packed entries
    pub lit_len: Box<[PackedEntry; TABLE_SIZE]>,
    /// Subtables for codes > TABLE_BITS
    pub lit_len_subtables: Vec<[PackedEntry; SUBTABLE_SIZE]>,
    /// Combined entries for fast path (when length+distance fit)
    pub combined: Box<[CombinedEntry; TABLE_SIZE]>,
    /// Distance table
    pub dist: Box<[DistEntry; DIST_TABLE_SIZE]>,
    /// Whether combined table is valid
    pub has_combined: bool,
}

impl TurboTables {
    /// Build tables from fixed Huffman codes
    pub fn build_fixed() -> Self {
        let mut lit_len_lens = [0u8; 288];
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

        let dist_lens = [5u8; 32];

        Self::build(&lit_len_lens, &dist_lens).unwrap()
    }

    /// Build tables from dynamic Huffman codes
    pub fn build(lit_len_lens: &[u8], dist_lens: &[u8]) -> io::Result<Self> {
        let mut lit_len = Box::new([PackedEntry::default(); TABLE_SIZE]);
        let mut lit_len_subtables: Vec<[PackedEntry; SUBTABLE_SIZE]> = Vec::new();
        let mut combined = Box::new([CombinedEntry::default(); TABLE_SIZE]);
        let mut dist = Box::new([DistEntry::default(); DIST_TABLE_SIZE]);

        // Build Huffman codes
        let lit_len_codes = build_huffman_codes(lit_len_lens)?;
        let dist_codes = build_huffman_codes(dist_lens)?;

        // Build distance table first (needed for combined entries)
        build_dist_table(dist_lens, &dist_codes, &mut dist);

        // Build literal/length table with subtables for long codes
        build_lit_len_table_with_subtables(
            lit_len_lens,
            &lit_len_codes,
            &mut lit_len,
            &mut lit_len_subtables,
        );

        // Build combined table (pre-decoded length+distance)
        let has_combined = build_combined_table(
            lit_len_lens,
            &lit_len_codes,
            dist_lens,
            &dist_codes,
            &mut combined,
        );

        Ok(Self {
            lit_len,
            lit_len_subtables,
            combined,
            dist,
            has_combined,
        })
    }
}

// =============================================================================
// Table Building Functions
// =============================================================================

fn build_huffman_codes(lens: &[u8]) -> io::Result<Vec<u16>> {
    let max_len = *lens.iter().max().unwrap_or(&0) as usize;

    // Count codes of each length
    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 && (len as usize) < 16 {
            bl_count[len as usize] += 1;
        }
    }

    // Calculate starting codes
    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..=max_len.min(15) {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    // Assign codes
    let mut codes = vec![0u16; lens.len()];
    for (n, &len) in lens.iter().enumerate() {
        if len > 0 && (len as usize) < 16 {
            codes[n] = next_code[len as usize];
            next_code[len as usize] += 1;
        }
    }

    Ok(codes)
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

fn build_lit_len_table_with_subtables(
    lens: &[u8],
    codes: &[u16],
    table: &mut [PackedEntry; TABLE_SIZE],
    subtables: &mut Vec<[PackedEntry; SUBTABLE_SIZE]>,
) {
    // First pass: identify which prefixes need subtables
    let mut needs_subtable = [false; TABLE_SIZE];

    for (symbol, &code_len) in lens.iter().enumerate() {
        if code_len == 0 {
            continue;
        }

        if code_len as usize > TABLE_BITS {
            // This code needs a subtable
            let code = codes[symbol];
            let reversed = reverse_bits(code, code_len);
            let prefix = (reversed as usize) & ((1 << TABLE_BITS) - 1);
            needs_subtable[prefix] = true;
        }
    }

    // Allocate subtables and create pointer entries
    let mut subtable_indices = [0u16; TABLE_SIZE];
    for (prefix, &needs) in needs_subtable.iter().enumerate() {
        if needs {
            subtable_indices[prefix] = subtables.len() as u16;
            subtables.push([PackedEntry::default(); SUBTABLE_SIZE]);
            // Mark main table entry as subtable pointer
            table[prefix] = PackedEntry(
                (TABLE_BITS as u32)
                    | (TYPE_SUBTABLE << 8)
                    | ((subtable_indices[prefix] as u32) << 16),
            );
        }
    }

    // Second pass: fill tables
    for (symbol, &code_len) in lens.iter().enumerate() {
        if code_len == 0 {
            continue;
        }

        let code = codes[symbol];
        let reversed = reverse_bits(code, code_len);

        let entry = if symbol < 256 {
            PackedEntry::literal(code_len, symbol as u8)
        } else if symbol == 256 {
            PackedEntry::eob(code_len)
        } else if symbol <= 285 {
            let len_idx = (symbol - 257) as u8;
            let extra = if (len_idx as usize) < LEN_EXTRA_BITS.len() {
                LEN_EXTRA_BITS[len_idx as usize]
            } else {
                0
            };
            PackedEntry::length(code_len, extra, len_idx)
        } else {
            continue;
        };

        if code_len as usize <= TABLE_BITS {
            // Direct entry in main table
            let fill_count = 1usize << (TABLE_BITS - code_len as usize);
            for i in 0..fill_count {
                let idx = (reversed as usize) | (i << code_len);
                if idx < TABLE_SIZE && !needs_subtable[idx & ((1 << TABLE_BITS) - 1)] {
                    table[idx] = entry;
                }
            }
        } else {
            // Entry in subtable
            let prefix = (reversed as usize) & ((1 << TABLE_BITS) - 1);
            let subtable_idx = subtable_indices[prefix] as usize;
            let suffix_bits = code_len as usize - TABLE_BITS;
            let suffix = (reversed as usize) >> TABLE_BITS;

            let fill_count = 1usize << (SUBTABLE_BITS - suffix_bits);
            for i in 0..fill_count {
                let idx = suffix | (i << suffix_bits);
                if idx < SUBTABLE_SIZE {
                    // Store entry with adjusted code length (just the suffix bits)
                    let subtable_entry = PackedEntry((suffix_bits as u32) | (entry.0 & 0xFFFFFF00));
                    subtables[subtable_idx][idx] = subtable_entry;
                }
            }
        }
    }
}

fn build_dist_table(lens: &[u8], codes: &[u16], table: &mut [DistEntry; DIST_TABLE_SIZE]) {
    // First, fill the table with valid entries
    for (symbol, &code_len) in lens.iter().enumerate() {
        if code_len == 0 || symbol >= 30 {
            continue;
        }

        // Skip codes longer than our table can handle
        if code_len as usize > DIST_TABLE_BITS {
            continue;
        }

        let code = codes[symbol];
        let reversed = reverse_bits(code, code_len);

        let entry = DistEntry {
            code_len,
            extra_bits: DIST_EXTRA_BITS[symbol],
            base_dist: DIST_START[symbol] as u16,
        };

        let fill_count = 1usize << (DIST_TABLE_BITS - code_len as usize);
        for i in 0..fill_count {
            let idx = (reversed as usize) | (i << code_len);
            if idx < DIST_TABLE_SIZE {
                table[idx] = entry;
            }
        }
    }
}

/// Build combined table with pre-decoded length+distance
/// Returns true if any combined entries were created
fn build_combined_table(
    lit_len_lens: &[u8],
    lit_len_codes: &[u16],
    dist_lens: &[u8],
    dist_codes: &[u16],
    table: &mut [CombinedEntry; TABLE_SIZE],
) -> bool {
    let mut has_combined = false;

    for (symbol, &lit_code_len) in lit_len_lens.iter().enumerate() {
        if lit_code_len == 0 || lit_code_len as usize > TABLE_BITS {
            continue;
        }

        let code = lit_len_codes[symbol];
        let reversed = reverse_bits(code, lit_code_len);

        if symbol < 256 {
            // Literal - mark with DIST_LITERAL
            let entry = CombinedEntry {
                bits_to_skip: lit_code_len,
                length_minus_3: symbol as u8, // Store literal byte here
                distance: DIST_LITERAL,
            };

            let fill_count = 1usize << (TABLE_BITS - lit_code_len as usize);
            for i in 0..fill_count {
                let idx = (reversed as usize) | (i << lit_code_len);
                if idx < TABLE_SIZE {
                    table[idx] = entry;
                }
            }
        } else if symbol == 256 {
            // End of block
            let entry = CombinedEntry {
                bits_to_skip: lit_code_len,
                length_minus_3: 0,
                distance: DIST_EOB,
            };

            let fill_count = 1usize << (TABLE_BITS - lit_code_len as usize);
            for i in 0..fill_count {
                let idx = (reversed as usize) | (i << lit_code_len);
                if idx < TABLE_SIZE {
                    table[idx] = entry;
                }
            }
        } else if symbol <= 285 {
            // Length code - try to pre-decode distance
            let len_idx = symbol - 257;
            let len_extra_bits = if len_idx < LEN_EXTRA_BITS.len() {
                LEN_EXTRA_BITS[len_idx] as usize
            } else {
                0
            };

            let base_len = if len_idx < LEN_START.len() {
                LEN_START[len_idx]
            } else {
                258
            };

            // Check if we can fit length extra bits + distance code
            let remaining_bits = TABLE_BITS - lit_code_len as usize;

            if remaining_bits > len_extra_bits {
                // We might be able to include distance
                // For each possible length extra value
                for len_extra_val in 0..(1usize << len_extra_bits) {
                    let actual_len = (base_len as usize) + len_extra_val;
                    let bits_after_len = lit_code_len as usize + len_extra_bits;

                    // Try each distance code that fits
                    for (dist_sym, &dist_code_len) in dist_lens.iter().enumerate() {
                        if dist_code_len == 0 || dist_sym >= 30 {
                            continue;
                        }

                        let dist_extra_bits = DIST_EXTRA_BITS[dist_sym] as usize;
                        let total_bits = bits_after_len + dist_code_len as usize + dist_extra_bits;

                        if total_bits <= TABLE_BITS {
                            // Full match fits in table!
                            has_combined = true;

                            let dist_code = dist_codes[dist_sym];
                            let dist_reversed = reverse_bits(dist_code, dist_code_len);
                            let base_dist = DIST_START[dist_sym] as usize;

                            for dist_extra_val in 0..(1usize << dist_extra_bits) {
                                let actual_dist = base_dist + dist_extra_val;
                                if actual_dist > 32768 {
                                    continue;
                                }

                                let entry = CombinedEntry {
                                    bits_to_skip: total_bits as u8,
                                    length_minus_3: (actual_len - 3) as u8,
                                    distance: actual_dist as u16,
                                };

                                // Build the full table index
                                let base_idx = reversed as usize
                                    | (len_extra_val << lit_code_len as usize)
                                    | ((dist_reversed as usize) << bits_after_len)
                                    | (dist_extra_val << (bits_after_len + dist_code_len as usize));

                                if base_idx < TABLE_SIZE {
                                    table[base_idx] = entry;
                                }
                            }
                        }
                    }
                }
            }

            // Mark remaining slots as slow path
            let entry = CombinedEntry {
                bits_to_skip: lit_code_len,
                length_minus_3: (len_idx) as u8,
                distance: DIST_SLOW,
            };

            let fill_count = 1usize << (TABLE_BITS - lit_code_len as usize);
            for i in 0..fill_count {
                let idx = (reversed as usize) | (i << lit_code_len);
                if idx < TABLE_SIZE && table[idx].bits_to_skip == 0 {
                    table[idx] = entry;
                }
            }
        }
    }

    has_combined
}

// =============================================================================
// Fast Bit Reader
// =============================================================================

/// Maximum overread before we consider the stream corrupted.
const TURBO_MAX_OVERREAD: u32 = 8;

pub struct TurboBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
    /// Number of implicit zero bytes consumed (for detecting truncated streams)
    overread_count: u32,
}

impl<'a> TurboBits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
            overread_count: 0,
        };
        reader.refill();
        reader
    }

    #[inline(always)]
    pub fn refill(&mut self) {
        while self.bits <= 56 {
            if self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << self.bits;
                self.pos += 1;
            } else {
                // Track overread like libdeflate
                self.overread_count += 1;
            }
            self.bits += 8;
        }
    }

    /// Check if stream has been overread
    #[inline(always)]
    pub fn is_overread(&self) -> bool {
        self.overread_count > TURBO_MAX_OVERREAD
    }

    #[inline(always)]
    pub fn peek(&self) -> u64 {
        self.buf
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bits = self.bits.saturating_sub(n);
    }

    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u64 {
        let val = self.buf & ((1u64 << n) - 1);
        self.consume(n);
        val
    }

    #[inline(always)]
    pub fn ensure(&mut self, n: u32) {
        if self.bits < n {
            self.refill();
        }
    }

    #[inline(always)]
    pub fn bits_available(&self) -> u32 {
        self.bits
    }

    #[inline(always)]
    pub fn align(&mut self) {
        let discard = self.bits & 7;
        self.consume(discard);
    }
}

// =============================================================================
// Turbo Inflate - Main Decode Loop
// =============================================================================

/// Ultra-fast inflate using TurboBits + PackedLUT from bgzf
#[inline(never)]
pub fn inflate_turbo(data: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    // Ensure output has enough capacity
    let estimated = data.len().saturating_mul(4).max(1024 * 1024);
    output.resize(estimated, 0);

    // Use the optimized turbo path from bgzf
    match crate::bgzf::inflate_into_pub(data, output) {
        Ok(size) => {
            output.truncate(size);
            Ok(size)
        }
        Err(e) => Err(e),
    }
}

/// Experimental turbo inflate - still has issues with dynamic blocks
#[allow(dead_code)]
#[inline(never)]
fn inflate_turbo_experimental(data: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut bits = TurboBits::new(data);
    let fixed_tables = TurboTables::build_fixed();

    loop {
        bits.refill();
        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => decode_stored(&mut bits, output)?,
            1 => decode_huffman_turbo(&mut bits, output, &fixed_tables)?,
            2 => {
                let tables = read_dynamic_tables(&mut bits)?;
                decode_huffman_turbo(&mut bits, output, &tables)?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid block type",
                ))
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output.len())
}

fn decode_stored(bits: &mut TurboBits, output: &mut Vec<u8>) -> io::Result<()> {
    bits.align();
    bits.refill();

    let len = bits.read(16) as usize;
    let nlen = bits.read(16) as usize;

    if len != (!nlen & 0xFFFF) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored block length mismatch",
        ));
    }

    output.reserve(len);
    for _ in 0..len {
        if bits.bits_available() < 8 {
            bits.refill();
        }
        output.push(bits.read(8) as u8);
    }

    Ok(())
}

fn read_dynamic_tables(bits: &mut TurboBits) -> io::Result<TurboTables> {
    bits.refill();
    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;

    // Read code length code lengths
    let mut code_length_lens = [0u8; 19];
    for i in 0..hclen {
        if bits.bits_available() < 8 {
            bits.refill();
        }
        code_length_lens[CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    // Build code length table
    let cl_codes = build_huffman_codes(&code_length_lens)?;
    let cl_table = build_code_length_table(&code_length_lens, &cl_codes);

    // Decode literal/length and distance code lengths
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;

    while i < all_lens.len() {
        if bits.bits_available() < 16 {
            bits.refill();
        }

        let idx = (bits.peek() & 0x7F) as usize;
        let (code_len, symbol) = cl_table[idx];
        if code_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid code length",
            ));
        }
        bits.consume(code_len as u32);

        match symbol {
            0..=15 => {
                all_lens[i] = symbol;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid repeat"));
                }
                let repeat = 3 + bits.read(2) as usize;
                let prev = all_lens[i - 1];
                for _ in 0..repeat {
                    if i >= all_lens.len() {
                        break;
                    }
                    all_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + bits.read(3) as usize;
                for _ in 0..repeat {
                    if i >= all_lens.len() {
                        break;
                    }
                    all_lens[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = 11 + bits.read(7) as usize;
                for _ in 0..repeat {
                    if i >= all_lens.len() {
                        break;
                    }
                    all_lens[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length symbol",
                ))
            }
        }
    }

    let lit_lens = &all_lens[..hlit];
    let dist_lens = &all_lens[hlit..];

    TurboTables::build(lit_lens, dist_lens)
}

fn build_code_length_table(lens: &[u8; 19], codes: &[u16]) -> [(u8, u8); 128] {
    let mut table = [(0u8, 0u8); 128];

    for (symbol, &code_len) in lens.iter().enumerate() {
        if code_len == 0 || code_len > 7 {
            continue;
        }

        let code = codes[symbol];
        let reversed = reverse_bits(code, code_len);

        let fill_count = 1usize << (7 - code_len);
        for i in 0..fill_count {
            let idx = (reversed as usize) | (i << code_len);
            if idx < 128 {
                table[idx] = (code_len, symbol as u8);
            }
        }
    }

    table
}

/// Optimized Huffman decode loop with combined LUT and multi-literal
#[inline(never)]
fn decode_huffman_turbo(
    bits: &mut TurboBits,
    output: &mut Vec<u8>,
    tables: &TurboTables,
) -> io::Result<()> {
    output.reserve(256 * 1024);

    // Use combined table for fast path
    if tables.has_combined {
        return decode_with_combined_lut(bits, output, tables);
    }

    // Fallback to standard decode
    decode_standard(bits, output, tables)
}

/// Fast path using combined LUT
#[inline(never)]
fn decode_with_combined_lut(
    bits: &mut TurboBits,
    output: &mut Vec<u8>,
    tables: &TurboTables,
) -> io::Result<()> {
    loop {
        bits.ensure(48); // Enough for multiple operations

        let entry = tables.combined[(bits.peek() & TABLE_MASK) as usize];

        if entry.bits_to_skip == 0 {
            // Invalid - fall back to slow path
            return decode_standard(bits, output, tables);
        }

        if entry.distance == DIST_LITERAL {
            // Fast literal path with multi-literal decode
            bits.consume(entry.bits_to_skip as u32);
            output.push(entry.length_minus_3); // literal byte stored here

            // Try 2nd literal
            if bits.bits_available() >= 22 {
                let entry2 = tables.combined[(bits.peek() & TABLE_MASK) as usize];
                if entry2.distance == DIST_LITERAL && entry2.bits_to_skip > 0 {
                    bits.consume(entry2.bits_to_skip as u32);
                    output.push(entry2.length_minus_3);

                    // Try 3rd literal
                    if bits.bits_available() >= 11 {
                        let entry3 = tables.combined[(bits.peek() & TABLE_MASK) as usize];
                        if entry3.distance == DIST_LITERAL && entry3.bits_to_skip > 0 {
                            bits.consume(entry3.bits_to_skip as u32);
                            output.push(entry3.length_minus_3);
                        }
                    }
                }
            }
            continue;
        }

        if entry.distance == DIST_EOB {
            bits.consume(entry.bits_to_skip as u32);
            break;
        }

        if entry.distance == DIST_SLOW {
            // Slow path - decode length and distance separately
            let len_idx = entry.length_minus_3 as usize;
            bits.consume(entry.bits_to_skip as u32);

            if len_idx >= 29 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            bits.ensure(16);
            let extra = LEN_EXTRA_BITS[len_idx] as u32;
            let length = LEN_START[len_idx] as usize + bits.read(extra) as usize;

            // Decode distance
            let dist_entry = tables.dist[(bits.peek() & DIST_TABLE_MASK) as usize];
            if dist_entry.code_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance",
                ));
            }
            bits.consume(dist_entry.code_len as u32);

            bits.ensure(16);
            let distance =
                dist_entry.base_dist as usize + bits.read(dist_entry.extra_bits as u32) as usize;

            if distance > output.len() || distance == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance",
                ));
            }

            crate::simd_copy::lz77_copy_fast(output, distance, length);
            continue;
        }

        // Fast path - full LZ77 match pre-decoded!
        bits.consume(entry.bits_to_skip as u32);
        let length = entry.length();
        let distance = entry.distance as usize;

        if distance > output.len() || distance == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }

        crate::simd_copy::lz77_copy_fast(output, distance, length);
    }

    Ok(())
}

/// Slow distance decode that handles all cases
#[inline(never)]
fn decode_distance_slow(
    bits: &mut TurboBits,
    dist_table: &[DistEntry; DIST_TABLE_SIZE],
) -> io::Result<usize> {
    bits.ensure(16);

    let dist_entry = dist_table[(bits.peek() & DIST_TABLE_MASK) as usize];

    if dist_entry.code_len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Invalid distance code (peek={})",
                bits.peek() & DIST_TABLE_MASK
            ),
        ));
    }

    bits.consume(dist_entry.code_len as u32);

    bits.ensure(16);
    let distance = dist_entry.base_dist as usize + bits.read(dist_entry.extra_bits as u32) as usize;

    Ok(distance)
}

/// Standard decode without combined LUT
fn decode_standard(
    bits: &mut TurboBits,
    output: &mut Vec<u8>,
    tables: &TurboTables,
) -> io::Result<()> {
    loop {
        bits.ensure(32);

        let mut entry = tables.lit_len[(bits.peek() & TABLE_MASK) as usize];

        // Handle subtable if needed
        if entry.entry_type() == TYPE_SUBTABLE {
            let subtable_idx = (entry.0 >> 16) as usize;
            bits.consume(TABLE_BITS as u32);

            if subtable_idx >= tables.lit_len_subtables.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid subtable index",
                ));
            }

            let sub_idx = (bits.peek() & ((1 << SUBTABLE_BITS) - 1)) as usize;
            entry = tables.lit_len_subtables[subtable_idx][sub_idx];
        }

        let code_len = entry.code_len();

        if code_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code (code_len=0)",
            ));
        }

        bits.consume(code_len);

        match entry.entry_type() {
            TYPE_LITERAL => {
                output.push(entry.literal_byte());

                // Multi-literal: try 2 more (only for main table entries)
                if bits.bits_available() >= 22 {
                    let entry2 = tables.lit_len[(bits.peek() & TABLE_MASK) as usize];
                    if entry2.is_literal()
                        && entry2.code_len() > 0
                        && entry2.entry_type() != TYPE_SUBTABLE
                    {
                        bits.consume(entry2.code_len());
                        output.push(entry2.literal_byte());

                        if bits.bits_available() >= 11 {
                            let entry3 = tables.lit_len[(bits.peek() & TABLE_MASK) as usize];
                            if entry3.is_literal()
                                && entry3.code_len() > 0
                                && entry3.entry_type() != TYPE_SUBTABLE
                            {
                                bits.consume(entry3.code_len());
                                output.push(entry3.literal_byte());
                            }
                        }
                    }
                }
            }
            TYPE_EOB => break,
            TYPE_LENGTH => {
                let len_idx = entry.len_code_idx();
                if len_idx >= 29 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid length code",
                    ));
                }

                bits.ensure(16);
                let extra = entry.extra_bits();
                let length = LEN_START[len_idx] as usize + bits.read(extra) as usize;

                let distance = decode_distance_slow(bits, &tables.dist)?;

                if distance > output.len() || distance == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Invalid distance {} (output len={})",
                            distance,
                            output.len()
                        ),
                    ));
                }

                crate::simd_copy::lz77_copy_fast(output, distance, length);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid entry type: {}", entry.entry_type()),
                ))
            }
        }
    }

    Ok(())
}

/// Inflate gzip format using turbo decoder
pub fn inflate_gzip_turbo(data: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    // Skip gzip header
    let header_size = crate::marker_decode::skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    inflate_turbo(deflate_data, output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_tables_fixed() {
        let tables = TurboTables::build_fixed();

        // Count entry types
        let mut literals = 0;
        let mut lengths = 0;
        let mut eob = 0;

        for entry in tables.lit_len.iter() {
            if entry.code_len() > 0 {
                match entry.entry_type() {
                    TYPE_LITERAL => literals += 1,
                    TYPE_LENGTH => lengths += 1,
                    TYPE_EOB => eob += 1,
                    _ => {}
                }
            }
        }

        eprintln!(
            "Fixed table: {} literals, {} lengths, {} EOB",
            literals, lengths, eob
        );
        assert!(literals > 0);
        assert!(lengths > 0);
        assert!(eob > 0);
    }

    #[test]
    fn test_turbo_combined_table() {
        let tables = TurboTables::build_fixed();

        // Check combined entries
        let mut literal_count = 0;
        let mut lz77_count = 0;
        let mut slow_count = 0;
        let mut eob_count = 0;

        for entry in tables.combined.iter() {
            if entry.bits_to_skip > 0 {
                if entry.distance == DIST_LITERAL {
                    literal_count += 1;
                } else if entry.distance == DIST_EOB {
                    eob_count += 1;
                } else if entry.distance == DIST_SLOW {
                    slow_count += 1;
                } else {
                    lz77_count += 1;
                }
            }
        }

        eprintln!(
            "Combined: {} literal, {} LZ77, {} slow, {} EOB",
            literal_count, lz77_count, slow_count, eob_count
        );

        assert!(literal_count > 0);
        assert!(eob_count > 0);
        // LZ77 count may be 0 for some table configurations
    }

    #[test]
    fn test_turbo_inflate_simple() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        let original = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_turbo(&compressed, &mut output).unwrap();

        assert_eq!(output.len(), original.len());
        assert_slices_eq!(&output[..], &original[..]);
    }

    #[test]
    fn test_turbo_inflate_correctness() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => return,
        };

        // Decompress with turbo
        let mut output_turbo = Vec::new();
        inflate_gzip_turbo(&data, &mut output_turbo).unwrap();

        // Decompress with libdeflate for reference
        let mut output_ref = vec![0u8; 250_000_000];
        let size = libdeflater::Decompressor::new()
            .gzip_decompress(&data, &mut output_ref)
            .unwrap();
        output_ref.truncate(size);

        assert_slices_eq!(output_turbo, output_ref, "Turbo vs libdeflate mismatch");

        eprintln!(
            "Turbo inflate correctness verified: {} bytes match",
            output_turbo.len()
        );
    }

    #[test]
    fn benchmark_turbo_vs_all() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => return,
        };

        // Warm up
        let mut warmup = Vec::new();
        let _ = inflate_gzip_turbo(&data, &mut warmup);

        // Benchmark turbo
        let start = std::time::Instant::now();
        let mut output_turbo = Vec::new();
        let size = inflate_gzip_turbo(&data, &mut output_turbo).unwrap();
        let turbo_time = start.elapsed();

        eprintln!(
            "turbo_inflate: {} bytes in {:?} ({:.1} MB/s)",
            size,
            turbo_time,
            size as f64 / turbo_time.as_secs_f64() / 1_000_000.0
        );

        // Benchmark ultra_fast_inflate
        let start = std::time::Instant::now();
        let mut output_ultra = Vec::new();
        crate::ultra_fast_inflate::inflate_gzip_ultra_fast(&data, &mut output_ultra).unwrap();
        let ultra_time = start.elapsed();

        eprintln!(
            "ultra_fast_inflate: {} bytes in {:?} ({:.1} MB/s)",
            output_ultra.len(),
            ultra_time,
            output_ultra.len() as f64 / ultra_time.as_secs_f64() / 1_000_000.0
        );

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        let mut output_lib = vec![0u8; 250_000_000];
        let lib_size = libdeflater::Decompressor::new()
            .gzip_decompress(&data, &mut output_lib)
            .unwrap();
        let lib_time = start.elapsed();

        eprintln!(
            "libdeflate: {} bytes in {:?} ({:.1} MB/s)",
            lib_size,
            lib_time,
            lib_size as f64 / lib_time.as_secs_f64() / 1_000_000.0
        );

        eprintln!(
            "\nturbo vs libdeflate: {:.1}% of libdeflate speed",
            100.0 * lib_time.as_secs_f64() / turbo_time.as_secs_f64()
        );
        eprintln!(
            "turbo vs ultra: {:.1}x faster",
            ultra_time.as_secs_f64() / turbo_time.as_secs_f64()
        );
    }
}
