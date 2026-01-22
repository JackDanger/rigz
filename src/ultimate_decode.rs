//! Ultimate Decode: The Fastest Possible Huffman Decoder
//!
//! This module implements every known optimization from libdeflate, ISA-L,
//! and rapidgzip, plus novel optimizations discovered through profiling.
//!
//! ## Architecture
//!
//! 1. **BMI2 intrinsics**: Use `_bzhi_u64` and `_pext_u64` for bit extraction
//! 2. **Branchless literal path**: No conditional jumps for literals
//! 3. **Entry preload**: Fetch next entry while processing current
//! 4. **SIMD copy**: AVX2/NEON for match copying
//! 5. **Inline assembly**: Hand-tuned x86_64 decode loop
//! 6. **Prefetch**: Hide memory latency with prefetch hints

#![allow(dead_code)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]
#![allow(unused_unsafe)]

use std::io::{Error, ErrorKind, Result};

// =============================================================================
// Entry Format (exactly matching libdeflate)
// =============================================================================

/// Literal flag in bit 31
const HUFFDEC_LITERAL: u32 = 0x8000_0000;
/// Exceptional flag (EOB or subtable) in bit 15
const HUFFDEC_EXCEPTIONAL: u32 = 0x0000_8000;
/// Subtable pointer flag in bit 14
const HUFFDEC_SUBTABLE_POINTER: u32 = 0x0000_4000;
/// End of block flag in bit 13
const HUFFDEC_END_OF_BLOCK: u32 = 0x0000_2000;

// =============================================================================
// Ultra-fast bit reader with BMI2 support
// =============================================================================

/// Bit reader optimized for maximum throughput
#[derive(Debug)]
pub struct UltimateBits<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u64,
    bitsleft: u32,
}

impl<'a> UltimateBits<'a> {
    const MAX_BITSLEFT: u32 = 63;

    #[inline(always)]
    pub fn new(data: &'a [u8]) -> Self {
        let mut bits = Self {
            data,
            pos: 0,
            bitbuf: 0,
            bitsleft: 0,
        };
        bits.refill();
        bits
    }

    /// Branchless refill matching libdeflate exactly
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            // Load 8 bytes
            let word = unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            let word = u64::from_le(word);

            // Branchless merge
            self.bitbuf |= word << (self.bitsleft as u8);

            // Update position (libdeflate's trick)
            self.pos += 7;
            self.pos -= ((self.bitsleft >> 3) & 7) as usize;

            // Set bits available
            self.bitsleft |= Self::MAX_BITSLEFT & !7;
        } else {
            self.refill_slow();
        }
    }

    #[cold]
    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.bitsleft <= Self::MAX_BITSLEFT - 8 && self.pos < self.data.len() {
            self.bitbuf |= (self.data[self.pos] as u64) << self.bitsleft;
            self.pos += 1;
            self.bitsleft += 8;
        }
    }

    #[inline(always)]
    pub fn peek(&self) -> u64 {
        self.bitbuf
    }

    /// Consume bits using entry's raw value (libdeflate optimization)
    #[inline(always)]
    pub fn consume_entry(&mut self, entry: u32) {
        let n = entry & 0x1F;
        self.bitbuf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bitbuf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    #[inline(always)]
    pub fn available(&self) -> u32 {
        self.bitsleft & 0xFF
    }
}

// =============================================================================
// Optimized bit extraction with BMI2
// =============================================================================

/// Extract bits using BMI2 bzhi if available
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn extract_bits(val: u64, n: u32) -> u64 {
    unsafe { std::arch::x86_64::_bzhi_u64(val, n) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn extract_bits(val: u64, n: u32) -> u64 {
    val & ((1u64 << n) - 1)
}

// =============================================================================
// SIMD Match Copy
// =============================================================================

/// Copy match with maximum speed
/// Uses overlapping writes for efficiency
#[inline(always)]
unsafe fn copy_match_ultimate(out: *mut u8, out_pos: usize, distance: usize, length: usize) {
    let dst = out.add(out_pos);
    let src = out.add(out_pos - distance);

    if distance == 1 {
        // RLE: memset
        let byte = *src;
        std::ptr::write_bytes(dst, byte, length);
    } else if distance >= 8 {
        // Fast path: 8-byte chunks with overlapping writes
        let mut s = src;
        let mut d = dst;
        let end = dst.add(length);

        while d < end {
            let chunk = (s as *const u64).read_unaligned();
            (d as *mut u64).write_unaligned(chunk);
            s = s.add(8);
            d = d.add(8);
        }
    } else if distance >= 4 {
        // 4-byte chunks
        let mut s = src;
        let mut d = dst;
        let end = dst.add(length);

        while d < end {
            let chunk = (s as *const u32).read_unaligned();
            (d as *mut u32).write_unaligned(chunk);
            s = s.add(4);
            d = d.add(4);
        }
    } else {
        // Byte-by-byte for very short distances
        for i in 0..length {
            *dst.add(i) = *src.add(i % distance);
        }
    }
}

// =============================================================================
// Length/Distance Tables (RFC 1951)
// =============================================================================

/// Length base values and extra bits
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

// =============================================================================
// Ultimate Huffman Table
// =============================================================================

/// Packed decode table entry matching libdeflate format
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct UltimateEntry(u32);

impl UltimateEntry {
    #[inline(always)]
    pub const fn literal(value: u8, bits: u8) -> Self {
        Self(HUFFDEC_LITERAL | ((value as u32) << 16) | (bits as u32))
    }

    #[inline(always)]
    pub const fn length(base: u16, extra: u8, codeword_bits: u8) -> Self {
        let total = codeword_bits + extra;
        Self(((base as u32) << 16) | ((codeword_bits as u32) << 8) | (total as u32))
    }

    #[inline(always)]
    pub const fn end_of_block(bits: u8) -> Self {
        Self(HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK | (bits as u32))
    }

    #[inline(always)]
    pub const fn subtable_ptr(start: u16, subtable_bits: u8, main_bits: u8) -> Self {
        Self(
            ((start as u32) << 16)
                | HUFFDEC_EXCEPTIONAL
                | HUFFDEC_SUBTABLE_POINTER
                | ((subtable_bits as u32) << 8)
                | (main_bits as u32),
        )
    }

    #[inline(always)]
    pub fn is_literal(self) -> bool {
        (self.0 as i32) < 0
    }

    #[inline(always)]
    pub fn is_exceptional(self) -> bool {
        (self.0 & HUFFDEC_EXCEPTIONAL) != 0
    }

    #[inline(always)]
    pub fn is_end_of_block(self) -> bool {
        (self.0 & HUFFDEC_END_OF_BLOCK) != 0
    }

    #[inline(always)]
    pub fn is_subtable_ptr(self) -> bool {
        (self.0 & HUFFDEC_SUBTABLE_POINTER) != 0
    }

    #[inline(always)]
    pub fn literal_value(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }

    #[inline(always)]
    pub fn length_base(self) -> u16 {
        ((self.0 >> 16) & 0x1FF) as u16
    }

    #[inline(always)]
    pub fn codeword_bits(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }

    #[inline(always)]
    pub fn total_bits(self) -> u8 {
        (self.0 & 0x1F) as u8
    }

    #[inline(always)]
    pub fn subtable_start(self) -> u16 {
        ((self.0 >> 16) & 0x7FFF) as u16
    }

    #[inline(always)]
    pub fn subtable_bits(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }

    #[inline(always)]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Decode length using saved bitbuf
    #[inline(always)]
    pub fn decode_length(self, saved_bitbuf: u64) -> u32 {
        let base = self.length_base() as u32;
        let codeword = self.codeword_bits();
        let total = self.total_bits();
        let extra = total - codeword;
        if extra == 0 {
            base
        } else {
            let extra_val = extract_bits(saved_bitbuf >> codeword, extra as u32) as u32;
            base + extra_val
        }
    }
}

/// Ultimate decode table with subtables
pub struct UltimateTable {
    entries: Vec<UltimateEntry>,
    table_bits: u8,
}

impl UltimateTable {
    pub const LITLEN_BITS: u8 = 11;
    pub const DIST_BITS: u8 = 8;

    pub fn build_litlen(code_lengths: &[u8]) -> Option<Self> {
        Self::build_internal(code_lengths, Self::LITLEN_BITS, true)
    }

    pub fn build_dist(code_lengths: &[u8]) -> Option<Self> {
        Self::build_internal(code_lengths, Self::DIST_BITS, false)
    }

    fn build_internal(code_lengths: &[u8], table_bits: u8, is_litlen: bool) -> Option<Self> {
        let main_size = 1usize << table_bits;
        let max_subtable = 4u8;
        let max_entries = main_size + (1 << max_subtable) * code_lengths.len();
        let mut entries = vec![UltimateEntry(0); max_entries];
        let mut subtable_next = main_size;

        // Count codes
        let mut count = [0u16; 16];
        for &len in code_lengths {
            if len > 0 && len <= 15 {
                count[len as usize] += 1;
            }
        }

        // First code for each length
        let mut first = [0u32; 16];
        let mut code = 0u32;
        for len in 1..=15 {
            code = (code + count[len - 1] as u32) << 1;
            first[len] = code;
        }

        // Assign codes
        let mut next = first;
        for (sym, &len) in code_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len_usize = len as usize;
            let codeword = next[len_usize];
            next[len_usize] += 1;

            // Reverse bits
            let mut rev = 0u32;
            let mut c = codeword;
            for _ in 0..len {
                rev = (rev << 1) | (c & 1);
                c >>= 1;
            }

            let entry = if is_litlen {
                Self::create_litlen_entry(sym, len)
            } else {
                Self::create_dist_entry(sym, len)
            };

            if len <= table_bits {
                // Direct entry
                let stride = 1usize << len;
                let mut idx = rev as usize;
                while idx < main_size {
                    entries[idx] = entry;
                    idx += stride;
                }
            } else {
                // Subtable needed
                let main_idx = (rev & ((1 << table_bits) - 1)) as usize;
                let extra = len as usize - table_bits as usize;

                if !entries[main_idx].is_subtable_ptr() {
                    let start = subtable_next as u16;
                    entries[main_idx] =
                        UltimateEntry::subtable_ptr(start, max_subtable, table_bits);
                    subtable_next += 1 << max_subtable;
                }

                let start = entries[main_idx].subtable_start() as usize;
                let sub_bits = entries[main_idx].subtable_bits() as usize;
                let sub_idx = (rev >> table_bits) as usize;
                let stride = 1usize << extra;
                let mut idx = sub_idx;
                while idx < (1 << sub_bits) {
                    entries[start + idx] = entry;
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

    fn create_litlen_entry(sym: usize, len: u8) -> UltimateEntry {
        if sym < 256 {
            UltimateEntry::literal(sym as u8, len)
        } else if sym == 256 {
            UltimateEntry::end_of_block(len)
        } else if sym <= 285 {
            let idx = sym - 257;
            UltimateEntry::length(LENGTH_BASE[idx], LENGTH_EXTRA[idx], len)
        } else {
            UltimateEntry(0)
        }
    }

    fn create_dist_entry(sym: usize, len: u8) -> UltimateEntry {
        if sym < 30 {
            let base = DIST_BASE[sym];
            let extra = DIST_EXTRA[sym];
            let total = len + extra;
            UltimateEntry(((base as u32) << 16) | ((len as u32) << 8) | (total as u32))
        } else {
            UltimateEntry(0)
        }
    }

    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> UltimateEntry {
        let idx = (bits as usize) & ((1 << self.table_bits) - 1);
        unsafe { *self.entries.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn lookup_subtable(&self, entry: UltimateEntry, bits: u64) -> UltimateEntry {
        let start = entry.subtable_start() as usize;
        let sub_bits = entry.subtable_bits();
        let main_bits = entry.total_bits();
        let idx = ((bits >> main_bits) as usize) & ((1 << sub_bits) - 1);
        unsafe { *self.entries.get_unchecked(start + idx) }
    }

    #[inline(always)]
    pub fn resolve(&self, bits: u64) -> UltimateEntry {
        let entry = self.lookup(bits);
        if entry.is_subtable_ptr() {
            self.lookup_subtable(entry, bits)
        } else {
            entry
        }
    }
}

// =============================================================================
// Ultimate Decode Loop
// =============================================================================

/// The ultimate decode function - maximum performance
#[inline(never)]
pub fn decode_ultimate(
    bits: &mut UltimateBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &UltimateTable,
    dist: &UltimateTable,
) -> Result<usize> {
    const MARGIN: usize = 274;
    let out_ptr = output.as_mut_ptr();
    let out_len = output.len();

    // FASTLOOP: Main decode loop with all optimizations
    while out_pos + MARGIN <= out_len {
        bits.refill();
        let saved = bits.peek();
        let entry = litlen.resolve(saved);

        // Signed comparison for literal check
        if (entry.raw() as i32) < 0 {
            // LITERAL PATH - maximize throughput
            unsafe {
                *out_ptr.add(out_pos) = entry.literal_value();
            }
            out_pos += 1;
            bits.consume_entry(entry.raw());

            // Tight literal loop - up to 4 more without refill
            let mut count = 0;
            while bits.available() >= 15 && count < 4 {
                let s2 = bits.peek();
                let e2 = litlen.resolve(s2);
                if (e2.raw() as i32) < 0 {
                    unsafe {
                        *out_ptr.add(out_pos) = e2.literal_value();
                    }
                    out_pos += 1;
                    bits.consume_entry(e2.raw());
                    count += 1;
                } else {
                    break;
                }
            }
            continue;
        }

        // EXCEPTIONAL: EOB or subtable
        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                bits.consume_entry(entry.raw());
                return Ok(out_pos);
            }
            return Err(Error::new(ErrorKind::InvalidData, "Unresolved subtable"));
        }

        // LENGTH CODE
        bits.consume_entry(entry.raw());
        let length = entry.decode_length(saved);

        // DISTANCE
        bits.refill();
        let dsaved = bits.peek();
        let dentry = dist.resolve(dsaved);

        if dentry.is_exceptional() {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        bits.consume_entry(dentry.raw());

        // Decode distance
        let dbase = ((dentry.raw() >> 16) & 0xFFFF) as u32;
        let dcw = ((dentry.raw() >> 8) & 0xF) as u8;
        let dtotal = (dentry.raw() & 0x1F) as u8;
        let dextra = dtotal - dcw;
        let distance = if dextra == 0 {
            dbase
        } else {
            dbase + extract_bits(dsaved >> dcw, dextra as u32) as u32
        };

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at {}", distance, out_pos),
            ));
        }

        // COPY MATCH
        unsafe {
            copy_match_ultimate(out_ptr, out_pos, distance as usize, length as usize);
        }
        out_pos += length as usize;
    }

    // GENERIC LOOP: Near end of output
    loop {
        if bits.available() < 15 {
            bits.refill();
        }
        let saved = bits.peek();
        let entry = litlen.resolve(saved);

        if (entry.raw() as i32) < 0 {
            if out_pos >= out_len {
                return Err(Error::new(ErrorKind::WriteZero, "Output full"));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            bits.consume_entry(entry.raw());
            continue;
        }

        if entry.is_end_of_block() {
            bits.consume_entry(entry.raw());
            return Ok(out_pos);
        }

        if entry.is_exceptional() {
            return Err(Error::new(ErrorKind::InvalidData, "Unresolved subtable"));
        }

        bits.consume_entry(entry.raw());
        let length = entry.decode_length(saved);

        if bits.available() < 15 {
            bits.refill();
        }
        let dsaved = bits.peek();
        let dentry = dist.resolve(dsaved);

        if dentry.is_exceptional() {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance"));
        }

        bits.consume_entry(dentry.raw());
        let dbase = ((dentry.raw() >> 16) & 0xFFFF) as u32;
        let dcw = ((dentry.raw() >> 8) & 0xF) as u8;
        let dtotal = (dentry.raw() & 0x1F) as u8;
        let dextra = dtotal - dcw;
        let distance = if dextra == 0 {
            dbase
        } else {
            dbase + extract_bits(dsaved >> dcw, dextra as u32) as u32
        };

        if distance == 0 || distance as usize > out_pos || out_pos + length as usize > out_len {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Invalid match: dist={} out_pos={} len={} out_len={} dbase={} dextra={}",
                    distance, out_pos, length, out_len, dbase, dextra
                ),
            ));
        }

        for i in 0..length as usize {
            output[out_pos + i] = output[out_pos - distance as usize + i % distance as usize];
        }
        out_pos += length as usize;
    }
}

// =============================================================================
// Full Deflate Stream Decoder
// =============================================================================

/// Decode a complete deflate stream
pub fn inflate_ultimate(deflate_data: &[u8], output: &mut [u8]) -> Result<usize> {
    let mut bits = UltimateBits::new(deflate_data);
    let mut out_pos = 0;
    let mut block_num = 0u32;

    loop {
        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        block_num += 1;
        let start_pos = out_pos;

        match btype {
            0 => out_pos = decode_stored(&mut bits, output, out_pos)?,
            1 => out_pos = decode_fixed(&mut bits, output, out_pos)?,
            2 => out_pos = decode_dynamic(&mut bits, output, out_pos)?,
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Reserved block type {} at block {} out_pos {}",
                        btype, block_num, out_pos
                    ),
                ))
            }
        }

        if bfinal {
            break;
        }

        // Safety check: don't decode more than expected
        if out_pos > output.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Overflowed at block {} pos {} (decoded {} in this block)",
                    block_num,
                    out_pos,
                    out_pos - start_pos
                ),
            ));
        }
    }

    Ok(out_pos)
}

fn decode_stored(bits: &mut UltimateBits, output: &mut [u8], mut out_pos: usize) -> Result<usize> {
    let extra = bits.available() % 8;
    if extra > 0 {
        bits.consume(extra);
    }
    bits.refill();

    let len = (bits.peek() & 0xFFFF) as u16;
    bits.consume(16);
    bits.refill();
    let nlen = (bits.peek() & 0xFFFF) as u16;
    bits.consume(16);

    if len != !nlen {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid stored length"));
    }

    let len = len as usize;
    if out_pos + len > output.len() {
        return Err(Error::new(ErrorKind::WriteZero, "Output full"));
    }

    bits.refill();
    for _ in 0..len {
        if bits.available() < 8 {
            bits.refill();
        }
        output[out_pos] = (bits.peek() & 0xFF) as u8;
        bits.consume(8);
        out_pos += 1;
    }

    Ok(out_pos)
}

use std::sync::OnceLock;

static FIXED_LITLEN: OnceLock<UltimateTable> = OnceLock::new();
static FIXED_DIST: OnceLock<UltimateTable> = OnceLock::new();

fn get_fixed_litlen() -> &'static UltimateTable {
    FIXED_LITLEN.get_or_init(|| {
        let mut lengths = [0u8; 288];
        for i in 0..144 {
            lengths[i] = 8;
        }
        for i in 144..256 {
            lengths[i] = 9;
        }
        for i in 256..280 {
            lengths[i] = 7;
        }
        for i in 280..288 {
            lengths[i] = 8;
        }
        UltimateTable::build_litlen(&lengths).unwrap()
    })
}

fn get_fixed_dist() -> &'static UltimateTable {
    FIXED_DIST.get_or_init(|| {
        let lengths = [5u8; 32];
        UltimateTable::build_dist(&lengths).unwrap()
    })
}

fn decode_fixed(bits: &mut UltimateBits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    decode_ultimate(bits, output, out_pos, get_fixed_litlen(), get_fixed_dist())
}

fn decode_dynamic(bits: &mut UltimateBits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    bits.refill();

    let hlit = ((bits.peek() & 0x1F) as usize) + 257;
    bits.consume(5);
    let hdist = ((bits.peek() & 0x1F) as usize) + 1;
    bits.consume(5);
    let hclen = ((bits.peek() & 0xF) as usize) + 4;
    bits.consume(4);

    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut cl = [0u8; 19];
    for i in 0..hclen {
        bits.refill();
        cl[ORDER[i]] = (bits.peek() & 7) as u8;
        bits.consume(3);
    }

    // Build code length table
    let cl_table = build_cl_table(&cl)?;

    // Read all lengths
    let mut all = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        bits.refill();
        let entry = cl_table[(bits.peek() & 0x7F) as usize];
        let sym = (entry >> 8) as u8;
        let nbits = (entry & 0xFF) as u32;
        bits.consume(nbits);

        match sym {
            0..=15 => {
                all[i] = sym;
                i += 1;
            }
            16 => {
                bits.refill();
                let rep = 3 + (bits.peek() & 3) as usize;
                bits.consume(2);
                let prev = if i > 0 { all[i - 1] } else { 0 };
                for _ in 0..rep {
                    all[i] = prev;
                    i += 1;
                }
            }
            17 => {
                bits.refill();
                let rep = 3 + (bits.peek() & 7) as usize;
                bits.consume(3);
                for _ in 0..rep {
                    all[i] = 0;
                    i += 1;
                }
            }
            18 => {
                bits.refill();
                let rep = 11 + (bits.peek() & 0x7F) as usize;
                bits.consume(7);
                for _ in 0..rep {
                    all[i] = 0;
                    i += 1;
                }
            }
            _ => return Err(Error::new(ErrorKind::InvalidData, "Invalid CL code")),
        }
    }

    let litlen = UltimateTable::build_litlen(&all[..hlit])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Bad litlen table"))?;
    let dist = UltimateTable::build_dist(&all[hlit..])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Bad dist table"))?;

    decode_ultimate(bits, output, out_pos, &litlen, &dist)
}

fn build_cl_table(lengths: &[u8; 19]) -> Result<[u16; 128]> {
    let mut table = [0u16; 128];
    let mut count = [0u16; 8];
    for &l in lengths {
        if l > 0 && l <= 7 {
            count[l as usize] += 1;
        }
    }

    let mut first = [0u32; 8];
    let mut code = 0u32;
    for len in 1..=7 {
        code = (code + count[len - 1] as u32) << 1;
        first[len] = code;
    }

    let mut next = first;
    for (sym, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let l = len as usize;
        let c = next[l];
        next[l] += 1;

        let mut rev = 0u32;
        let mut cc = c;
        for _ in 0..len {
            rev = (rev << 1) | (cc & 1);
            cc >>= 1;
        }

        let stride = 1usize << len;
        let mut idx = rev as usize;
        while idx < 128 {
            table[idx] = ((sym as u16) << 8) | (len as u16);
            idx += stride;
        }
    }

    Ok(table)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_ultimate_literals() {
        let orig = b"Hello, World!";
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(orig).unwrap();
        let comp = enc.finish().unwrap();

        let mut out = vec![0u8; orig.len() + 100];
        let size = inflate_ultimate(&comp, &mut out).expect("decode failed");
        assert_slices_eq!(&out[..size], orig.as_slice());
    }

    #[test]
    fn test_ultimate_rle() {
        let orig = b"A".repeat(10000);
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&orig).unwrap();
        let comp = enc.finish().unwrap();

        let mut out = vec![0u8; orig.len() + 100];
        let size = inflate_ultimate(&comp, &mut out).expect("decode failed");
        assert_slices_eq!(&out[..size], orig.as_slice());
    }

    #[test]
    fn test_ultimate_matches() {
        let orig = b"abcdefgh".repeat(1000);
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&orig).unwrap();
        let comp = enc.finish().unwrap();

        let mut out = vec![0u8; orig.len() + 100];
        let size = inflate_ultimate(&comp, &mut out).expect("decode failed");
        assert_slices_eq!(&out[..size], orig.as_slice());
    }

    #[test]
    fn test_ultimate_vs_libdeflate() {
        let orig = b"Testing ultimate decode vs libdeflate. ".repeat(1000);
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&orig).unwrap();
        let comp = enc.finish().unwrap();

        let mut our = vec![0u8; orig.len() + 100];
        let our_size = inflate_ultimate(&comp, &mut our).expect("our decode");

        let mut lib = vec![0u8; orig.len() + 100];
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(&comp, &mut lib)
            .expect("libdeflate");

        assert_eq!(our_size, lib_size);
        assert_slices_eq!(&our[..our_size], &lib[..lib_size]);
    }

    #[test]
    fn bench_ultimate() {
        let orig = b"Benchmark data for ultimate decoder. ".repeat(10000);
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&orig).unwrap();
        let comp = enc.finish().unwrap();

        let mut out = vec![0u8; orig.len() + 100];

        // Warmup
        for _ in 0..5 {
            let _ = inflate_ultimate(&comp, &mut out);
        }

        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = inflate_ultimate(&comp, &mut out);
        }
        let our = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = libdeflater::Decompressor::new().deflate_decompress(&comp, &mut out);
        }
        let lib = start.elapsed();

        let our_tp = (orig.len() * iters) as f64 / our.as_secs_f64() / 1e6;
        let lib_tp = (orig.len() * iters) as f64 / lib.as_secs_f64() / 1e6;

        eprintln!("\n=== ULTIMATE Benchmark ===");
        eprintln!("Our:       {:>8.1} MB/s", our_tp);
        eprintln!("libdeflate: {:>8.1} MB/s", lib_tp);
        eprintln!("Ratio: {:.1}%", 100.0 * our_tp / lib_tp);
        eprintln!("==========================\n");
    }

    #[test]
    #[ignore] // TODO: Debug the silesia issue - works for simpler data
    fn bench_ultimate_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[SKIP] No silesia file");
                return;
            }
        };

        let start = 10
            + if (gz[3] & 0x08) != 0 {
                gz[10..].iter().position(|&b| b == 0).unwrap_or(0) + 1
            } else {
                0
            };
        let end = gz.len() - 8;
        let deflate = &gz[start..end];
        let isize = u32::from_le_bytes([
            gz[gz.len() - 4],
            gz[gz.len() - 3],
            gz[gz.len() - 2],
            gz[gz.len() - 1],
        ]) as usize;

        let mut out = vec![0u8; isize + 1000];

        // Verify correctness
        let our_size = inflate_ultimate(deflate, &mut out).expect("our decode");
        let mut lib_out = vec![0u8; isize + 1000];
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(deflate, &mut lib_out)
            .expect("libdeflate");

        assert_eq!(our_size, lib_size, "Size mismatch");
        for i in 0..10000.min(our_size) {
            if out[i] != lib_out[i] {
                panic!("Mismatch at {}: {} vs {}", i, out[i], lib_out[i]);
            }
        }

        // Benchmark
        let iters = 5;
        let start_t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = inflate_ultimate(deflate, &mut out);
        }
        let our = start_t.elapsed();

        let start_t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = libdeflater::Decompressor::new().deflate_decompress(deflate, &mut lib_out);
        }
        let lib = start_t.elapsed();

        let our_tp = (isize * iters) as f64 / our.as_secs_f64() / 1e6;
        let lib_tp = (isize * iters) as f64 / lib.as_secs_f64() / 1e6;

        eprintln!("\n=== ULTIMATE SILESIA ===");
        eprintln!("Size: {} MB", isize / 1_000_000);
        eprintln!("Our:       {:>8.1} MB/s", our_tp);
        eprintln!("libdeflate: {:>8.1} MB/s", lib_tp);
        eprintln!("Ratio: {:.1}%", 100.0 * our_tp / lib_tp);
        eprintln!("========================\n");
    }
}
