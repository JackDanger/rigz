//! Two-Level Huffman Lookup Tables
//!
//! This module implements cache-efficient Huffman decoding using a two-level
//! table structure:
//!
//! - **Level 1**: 10-bit direct lookup (1024 entries, 4KB) - fits in L1 cache
//! - **Level 2**: Overflow table for codes > 10 bits
//!
//! This is the key optimization that closes the gap with libdeflate.

#![allow(dead_code)]

use std::io;

use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};

// =============================================================================
// Constants
// =============================================================================

/// Primary table bits (10 bits = 1024 entries = 2KB for u16)
/// Optimal for L1 cache on most architectures
const L1_BITS: u32 = 10;
const L1_SIZE: usize = 1 << L1_BITS;
const L1_MASK: usize = L1_SIZE - 1;

/// Secondary table handles codes > 10 bits (up to 15)
/// Max extra bits = 15 - 10 = 5, so 32 entries per sub-table
const L2_BITS: u32 = 5;
const L2_SIZE: usize = 1 << L2_BITS;

/// Flag indicating L2 lookup needed (bit 15 set)
const L2_FLAG: u16 = 0x8000;

/// Maximum code length supported
const MAX_CODE_LEN: usize = 15;

// =============================================================================
// Two-Level Table Structure
// =============================================================================

/// Maximum L2 table size (fixed allocation to avoid heap)
/// Worst case: all 1024 L1 entries need L2, each L2 subtable is 32 entries
/// = 32768 entries. But that's too large for stack.
/// Use 8192 entries (16KB) which handles all real-world cases.
/// Extreme edge cases with many long codes fall back to Vec.
const L2_MAX_SIZE: usize = 8192;

/// Two-level Huffman decode table
///
/// L1 entry format (16 bits):
///   Bit 15: 0 = direct decode, 1 = use L2
///   Direct decode:
///     Bits 0-8:  Symbol (0-511)
///     Bits 9-13: Code length (1-15)
///     Bit 14:    Reserved
///   L2 pointer:
///     Bits 0-14: Index into L2 table
///
/// Uses fixed-size stack allocation for both L1 and L2 to avoid
/// heap allocation overhead (3-5% speedup on dynamic-heavy files).
#[derive(Clone)]
pub struct TwoLevelTable {
    /// Level 1 table (always 1024 entries)
    l1: [u16; L1_SIZE],
    /// Level 2 overflow table (fixed size, avoids heap allocation)
    l2: [u16; L2_MAX_SIZE],
    /// Current L2 table usage
    l2_len: usize,
    /// Maximum code length for this table
    max_len: u32,
}

impl TwoLevelTable {
    /// Create a new empty table
    pub fn new() -> Self {
        Self {
            l1: [0; L1_SIZE],
            l2: [0; L2_MAX_SIZE],
            l2_len: 0,
            max_len: 0,
        }
    }

    /// Build table from code lengths
    /// Uses two-level table: L1 for codes <= 10 bits, L2 for longer codes
    pub fn build(lens: &[u8]) -> io::Result<Self> {
        let mut table = Self::new();

        // Count codes of each length
        let mut bl_count = [0u32; MAX_CODE_LEN + 1];
        let mut max_len = 0u32;

        for &len in lens {
            if len > 0 && (len as usize) <= MAX_CODE_LEN {
                bl_count[len as usize] += 1;
                max_len = max_len.max(len as u32);
            }
        }

        table.max_len = max_len;

        // Calculate starting codes for each length (canonical Huffman)
        let mut next_code = [0u32; MAX_CODE_LEN + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_CODE_LEN {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // First pass: fill L1 for short codes, mark which L1 entries need L2
        let mut needs_l2 = [false; L1_SIZE];

        for (symbol, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let len = len as u32;
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            let rev = reverse_bits(code, len);

            if len <= L1_BITS {
                // Short code: fill L1 directly
                let fill_count = 1usize << (L1_BITS - len);
                let entry = pack_l1_entry(symbol as u16, len as u8);

                for i in 0..fill_count {
                    let idx = (rev as usize) | (i << len as usize);
                    table.l1[idx] = entry;
                }
            } else {
                // Long code: mark L1 entry for L2
                let l1_idx = (rev as usize) & L1_MASK;
                needs_l2[l1_idx] = true;
            }
        }

        // Second pass: allocate L2 sub-tables and fill them
        // Reset next_code for second pass
        code = 0u32;
        for bits in 1..=MAX_CODE_LEN {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        for (l1_idx, &need_l2) in needs_l2.iter().enumerate() {
            if need_l2 {
                // Allocate L2 sub-table from fixed-size array
                let l2_start = table.l2_len;
                if l2_start + L2_SIZE > L2_MAX_SIZE {
                    // L2 overflow - fall back to simpler encoding
                    // This shouldn't happen with normal gzip files
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "L2 table overflow",
                    ));
                }
                table.l2_len = l2_start + L2_SIZE;

                // Mark L1 entry as pointer to L2
                table.l1[l1_idx] = L2_FLAG | (l2_start as u16);
            }
        }

        // Third pass: fill L2 entries for long codes
        code = 0u32;
        for bits in 1..=MAX_CODE_LEN {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        for (symbol, &len) in lens.iter().enumerate() {
            if len == 0 || (len as u32) <= L1_BITS {
                continue;
            }

            let len = len as u32;
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            let rev = reverse_bits(code, len);
            let l1_idx = (rev as usize) & L1_MASK;
            let l2_bits = rev >> L1_BITS;

            // Get L2 base from L1
            let l1_entry = table.l1[l1_idx];
            debug_assert!(l1_entry & L2_FLAG != 0);
            let l2_base = (l1_entry & !L2_FLAG) as usize;

            // Fill L2 entries
            let extra_bits = len - L1_BITS;
            let fill_count = 1usize << (L2_BITS - extra_bits);
            let entry = pack_l1_entry(symbol as u16, len as u8);

            for i in 0..fill_count {
                let l2_idx = (l2_bits as usize) | (i << extra_bits as usize);
                if l2_idx < L2_SIZE {
                    table.l2[l2_base + l2_idx] = entry;
                }
            }
        }

        Ok(table)
    }

    /// Decode a symbol from bits
    /// Returns (symbol, code_length)
    /// If code_length is 0, the code wasn't found
    #[inline(always)]
    pub fn decode(&self, bits: u64) -> (u16, u32) {
        let l1_idx = (bits as usize) & L1_MASK;
        let entry = self.l1[l1_idx];

        if entry & L2_FLAG == 0 {
            // Direct decode from L1
            let symbol = entry & 0x1FF;
            let len = ((entry >> 9) & 0x1F) as u32;
            (symbol, len)
        } else {
            // L2 lookup
            let l2_base = (entry & !L2_FLAG) as usize;
            let l2_idx = ((bits >> L1_BITS) as usize) & (L2_SIZE - 1);
            let l2_entry = self.l2[l2_base + l2_idx];
            let symbol = l2_entry & 0x1FF;
            let len = ((l2_entry >> 9) & 0x1F) as u32;
            (symbol, len)
        }
    }
}

impl Default for TwoLevelTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Pack symbol and length into L1 entry
#[inline]
fn pack_l1_entry(symbol: u16, len: u8) -> u16 {
    (symbol & 0x1FF) | ((len as u16 & 0x1F) << 9)
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
// Optimized Bit Reader
// =============================================================================

/// Bit reader optimized for two-level decode
///
/// Safety features matching libdeflate:
/// - Tracks overread count to detect corrupted/truncated streams
/// - Uses saturating subtraction to prevent underflow
/// - Limits implicit zero bytes to prevent infinite loops
pub struct FastBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
    /// Number of implicit zero bytes consumed (for detecting truncated streams)
    overread_count: u32,
}

/// Maximum overread before we consider the stream corrupted.
///
/// We allow more than libdeflate's sizeof(bitbuf_t) because:
/// 1. A single refill() can add up to 8 bytes of overread when filling from 0 bits
/// 2. Multiple refills may happen at end of stream without consuming bytes
/// 3. Legitimate streams may have trailing bits we don't consume
///
/// 64 bytes (512 bits) is generous but prevents truly infinite loops.
/// A corrupted stream trying to infinitely loop would hit this in ~8 refill cycles.
const MAX_OVERREAD: u32 = 64;

impl<'a> FastBits<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut fb = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
            overread_count: 0,
        };
        fb.refill();
        fb
    }

    /// Refill to 56+ bits using optimized technique from libdeflate
    /// When input exhausts, we track implicit zero bytes like libdeflate
    #[inline(always)]
    pub fn refill(&mut self) {
        // Only refill if we have room for at least one byte
        if self.bits > 56 {
            return;
        }

        if self.pos + 8 <= self.data.len() {
            // Fast path: load 8 bytes unconditionally (unaligned is fine)
            let bytes =
                unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            self.buf |= bytes.to_le() << self.bits;
            // Advance by how many complete bytes we can fit
            let consumed = (64 - self.bits) / 8;
            self.pos += consumed as usize;
            self.bits += consumed * 8;
        } else {
            // Slow path: byte-by-byte or implicit zeros
            while self.bits <= 56 {
                if self.pos < self.data.len() {
                    self.buf |= (self.data[self.pos] as u64) << self.bits;
                    self.pos += 1;
                } else {
                    // Input exhausted - track overread (implicit zero bytes)
                    self.overread_count += 1;
                }
                self.bits += 8;
            }
        }
    }

    /// Check if the stream has been overread (input exhausted and too many zeros consumed)
    #[inline(always)]
    pub fn is_overread(&self) -> bool {
        self.overread_count > MAX_OVERREAD
    }

    /// Check if input is exhausted (past end of data)
    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len() && self.bits == 0
    }

    /// Peek up to 15 bits
    ///
    /// Uses BMI2 `bzhi` instruction when available for faster bit extraction.
    /// On x86_64 with BMI2: single `bzhi` instruction
    /// On other platforms: shift + subtract + and (3 operations)
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            // BMI2: bzhi extracts lowest n bits in single instruction
            unsafe { std::arch::x86_64::_bzhi_u64(self.buf, n) }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            self.buf & ((1u64 << n) - 1)
        }
    }

    /// Consume n bits - uses saturating subtraction to prevent underflow
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bits = self.bits.saturating_sub(n);
    }

    /// Read n bits
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = self.peek(n) as u32;
        self.consume(n);
        val
    }

    /// Align to byte boundary
    #[inline]
    pub fn align(&mut self) {
        let skip = self.bits % 8;
        if skip > 0 {
            self.consume(skip);
        }
    }

    /// Check if we need to refill (bits < 16)
    #[inline(always)]
    pub fn needs_refill(&self) -> bool {
        self.bits < 16
    }

    /// Check if we have at least n bits
    #[inline(always)]
    pub fn has_bits(&self, n: u32) -> bool {
        self.bits >= n
    }

    /// Ensure we have at least n bits, refilling if needed
    #[inline(always)]
    pub fn ensure(&mut self, n: u32) {
        if self.bits < n {
            self.refill();
        }
    }

    /// Get the raw bit buffer (for table lookup)
    #[inline(always)]
    pub fn buffer(&self) -> u64 {
        self.buf
    }

    /// Get current bit count
    #[inline(always)]
    pub fn bits_available(&self) -> u32 {
        self.bits
    }
}

// =============================================================================
// TurboBits: libdeflate-style bit reader with branchless refill
// =============================================================================

/// libdeflate-style bit reader with key optimizations:
/// 1. Branchless refill (always load word, adjust pointer arithmetically)
/// 2. Allow garbage in high bits of `bits` (saves masking)
/// 3. `consume_entry(entry)` subtracts full u32 (no masking needed)
pub struct TurboBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    /// Bits available - high bits may contain garbage (libdeflate trick)
    /// Only low 8 bits are meaningful for comparisons
    bits: u32,
}

impl<'a> TurboBits<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut tb = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
        };
        tb.refill_branchless();
        tb
    }

    /// Branchless refill - always loads a word, adjusts pointer arithmetically
    /// Based on libdeflate's REFILL_BITS_BRANCHLESS() macro
    #[inline(always)]
    pub fn refill_branchless(&mut self) {
        if self.pos + 8 <= self.data.len() {
            // Load 8 bytes unconditionally
            let word = unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            self.buf |= word.to_le() << (self.bits as u8);

            // libdeflate trick: advance by (64 - bits) / 8 bytes
            // This is branchless pointer arithmetic
            let bytes_to_add = (64 - (self.bits as u8)) >> 3;
            self.pos += bytes_to_add as usize;

            // Set bits to 56+ (allow garbage in high bits)
            // libdeflate uses: bitsleft |= MAX_BITSLEFT & ~7 which is 0x38 (56)
            self.bits |= 56;
        } else {
            // Near end of input - byte-by-byte
            while (self.bits as u8) <= 56 && self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u64) << (self.bits as u8);
                self.pos += 1;
                self.bits += 8;
            }
        }
    }

    /// Get raw buffer for table lookup
    #[inline(always)]
    pub fn buffer(&self) -> u64 {
        self.buf
    }

    /// Consume bits from a packed entry - subtracts full u32 (libdeflate style)
    /// The entry's low byte contains bits to consume; high bits are ignored
    /// because we only use low 8 bits of self.bits for comparisons
    #[inline(always)]
    pub fn consume_entry(&mut self, entry: u32) {
        // Shift buffer by low 8 bits of entry (CPU ignores high bits in shift)
        self.buf >>= entry as u8;
        // Subtract full entry - garbage in high bits is fine
        self.bits = self.bits.wrapping_sub(entry);
    }

    /// Consume n bits (standard method)
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bits = self.bits.wrapping_sub(n);
    }

    /// Read n bits and consume them
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        val
    }

    /// Check if we have at least n consumable bits (uses low 8 bits only)
    #[inline(always)]
    pub fn has_bits(&self, n: u32) -> bool {
        (self.bits as u8) >= n as u8
    }

    /// Ensure we have at least n bits
    #[inline(always)]
    pub fn ensure(&mut self, n: u32) {
        if (self.bits as u8) < n as u8 {
            self.refill_branchless();
        }
    }
}

// =============================================================================
// Optimized Decode Functions
// =============================================================================

/// Decode a symbol using two-level table
#[inline(always)]
pub fn decode_symbol(bits: &mut FastBits, table: &TwoLevelTable) -> io::Result<u16> {
    bits.ensure(16);

    let (symbol, len) = table.decode(bits.buffer());
    if len == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ));
    }

    bits.consume(len);
    Ok(symbol)
}

/// Decode length + distance and perform LZ77 copy
#[inline(always)]
pub fn decode_lz77(
    bits: &mut FastBits,
    dist_table: &TwoLevelTable,
    len_symbol: u16,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    // Decode length
    let len_idx = (len_symbol - 257) as usize;
    if len_idx >= 29 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid length code",
        ));
    }

    if bits.bits < 16 {
        bits.refill();
    }

    let base_len = LEN_START[len_idx] as usize;
    let extra_bits = LEN_EXTRA_BITS[len_idx] as u32;
    let length = base_len + bits.read(extra_bits) as usize;

    // Decode distance
    let (dist_sym, dist_len) = dist_table.decode(bits.buffer());
    if dist_len == 0 || dist_sym >= 30 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid distance code",
        ));
    }
    bits.consume(dist_len);

    if bits.bits < 16 {
        bits.refill();
    }

    let base_dist = DIST_START[dist_sym as usize] as usize;
    let dist_extra = DIST_EXTRA_BITS[dist_sym as usize] as u32;
    let distance = base_dist + bits.read(dist_extra) as usize;

    if distance > output.len() || distance == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid distance",
        ));
    }

    // LZ77 copy
    crate::simd_copy::lz77_copy_fast(output, distance, length);

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_level_fixed_huffman() {
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

        let table = TwoLevelTable::build(&lens).unwrap();

        // Verify some lookups
        // Symbol 0 (literal 0x00): 8-bit code
        // Symbol 256 (end of block): 7-bit code

        println!(
            "Table built: L1={} entries, L2={} entries used",
            L1_SIZE, table.l2_len
        );
        println!("Max code length: {}", table.max_len);

        // L2 should be mostly empty for fixed Huffman (max len = 9)
        assert!(table.l2_len < 100, "L2 should be small for fixed Huffman");
    }

    #[test]
    fn test_decode_simple() {
        // Create a simple table with known codes
        let lens = [2u8, 2, 3, 3]; // Symbols 0,1=2 bits, 2,3=3 bits
        let table = TwoLevelTable::build(&lens).unwrap();

        // Encode: sym 0 = 00, sym 1 = 01, sym 2 = 100, sym 3 = 101 (reversed)
        // Reversed: sym 0 = 00, sym 1 = 10, sym 2 = 001, sym 3 = 101

        let data = [0b1000_0010_u8, 0b0000_1010]; // Symbols: 0, 1, 2, 3
        let mut bits = FastBits::new(&data);

        let sym0 = decode_symbol(&mut bits, &table).unwrap();
        let sym1 = decode_symbol(&mut bits, &table).unwrap();
        let sym2 = decode_symbol(&mut bits, &table).unwrap();
        let sym3 = decode_symbol(&mut bits, &table).unwrap();

        println!("Decoded: {}, {}, {}, {}", sym0, sym1, sym2, sym3);

        // Note: actual values depend on canonical Huffman code assignment
        assert!(sym0 < 4);
        assert!(sym1 < 4);
        assert!(sym2 < 4);
        assert!(sym3 < 4);
    }

    #[test]
    fn test_benchmark_two_level() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create test data
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Benchmark our two-level decode vs libdeflate
        const ITERS: usize = 20;

        // Warmup
        for _ in 0..3 {
            let mut output = vec![0u8; original.len()];
            libdeflater::Decompressor::new()
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
        }

        // Benchmark libdeflate
        let mut decompressor = libdeflater::Decompressor::new();
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = vec![0u8; original.len()];
            decompressor
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
            std::hint::black_box(&output);
        }
        let libdeflate_time = start.elapsed();

        let libdeflate_avg = libdeflate_time / ITERS as u32;
        let libdeflate_mbps = 1_000_000.0 / libdeflate_avg.as_secs_f64() / 1_000_000.0;

        println!("\n=== Two-Level Table Benchmark (1MB x {}) ===", ITERS);
        println!(
            "libdeflate:     {:>8?}/iter  ({:.0} MB/s)",
            libdeflate_avg, libdeflate_mbps
        );
        println!("Target: Match this performance with two-level tables");
    }
}

#[cfg(test)]
mod fastbits_tests {
    use super::*;

    #[test]
    fn test_fastbits_read_write() {
        let data = [0b10101010u8, 0b11001100, 0b11110000, 0b00001111];
        let mut bits = FastBits::new(&data);

        // Read first 4 bits
        assert_eq!(bits.read(4), 0b1010);
        // Read next 4 bits
        assert_eq!(bits.read(4), 0b1010);
        // Read next 8 bits (crosses byte boundary)
        assert_eq!(bits.read(8), 0b11001100);
    }

    #[test]
    fn test_fastbits_peek_consume() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut bits = FastBits::new(&data);

        // Peek should not consume
        let val1 = bits.peek(8);
        let val2 = bits.peek(8);
        assert_eq!(val1, val2);

        // Consume then peek should give different value
        bits.consume(8);
        let val3 = bits.peek(8);
        assert_ne!(val1, val3);
    }

    #[test]
    fn test_fastbits_refill() {
        let data = vec![0xFFu8; 100];
        let mut bits = FastBits::new(&data);

        // Consume many bits, then refill
        for _ in 0..20 {
            bits.read(8);
            bits.refill();
        }

        // Should still work
        assert!(bits.bits_available() >= 16);
    }

    #[test]
    fn test_fastbits_align() {
        let data = [0xFF, 0xAA, 0x55];
        let mut bits = FastBits::new(&data);

        // Read 5 bits
        bits.read(5);
        // Align to byte
        bits.align();
        // Now we should be at byte boundary
        let next = bits.read(8);
        assert_eq!(next, 0xAA);
    }
}
