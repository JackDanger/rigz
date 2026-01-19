//! Packed Decode - libdeflate-style packed entry format
//!
//! Key insight from libdeflate:
//! - Pack ALL decode info into a single u32 entry
//! - `bitsleft -= entry` works because only low byte matters
//! - Extra bits position encoded in entry, extracted with shift
//!
//! Entry format (literal):
//!   Bits 0-7:   Code length (bits to consume)
//!   Bits 8-15:  (unused for literals)
//!   Bits 16-23: Literal value
//!   Bit 31:     LITERAL flag
//!
//! Entry format (length):
//!   Bits 0-7:   Code length + extra bits (total to consume)
//!   Bits 8-13:  Extra bits count
//!   Bits 16-31: Length base value
//!
//! Entry format (distance):
//!   Bits 0-7:   Code length + extra bits (total to consume)
//!   Bits 8-13:  Extra bits count
//!   Bits 14-15: (unused)
//!   Bits 16-31: Distance base value

#![allow(dead_code)]

use crate::inflate_tables::{
    CODE_LENGTH_ORDER, DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START,
};
use crate::two_level_table::TwoLevelTable;
use std::io;

// Entry flags
const HUFFDEC_LITERAL: u32 = 0x8000_0000;
const HUFFDEC_END_OF_BLOCK: u32 = 0x4000_0000;
const HUFFDEC_SUBTABLE: u32 = 0x2000_0000;

/// 12-bit primary table (4096 entries) - covers most codes without subtable
const PRIMARY_BITS: usize = 12;
const PRIMARY_SIZE: usize = 1 << PRIMARY_BITS;
const PRIMARY_MASK: u64 = (PRIMARY_SIZE - 1) as u64;

/// Packed decode table
#[derive(Clone)]
pub struct PackedTable {
    pub entries: Vec<u32>,
}

impl PackedTable {
    /// Build packed litlen table
    pub fn build_litlen(code_lens: &[u8]) -> io::Result<Self> {
        let mut entries = vec![0u32; PRIMARY_SIZE + 512]; // + overflow

        // Count codes per length
        let mut bl_count = [0u32; 16];
        for &len in code_lens {
            if len > 0 && len <= 15 {
                bl_count[len as usize] += 1;
            }
        }

        // Compute first code for each length
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..16 {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Build table
        for (sym, &len) in code_lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as usize;
            let code = next_code[len];
            next_code[len] += 1;

            // Reverse bits
            let reversed = reverse_bits(code, len);

            // Build entry
            let entry = if sym < 256 {
                // Literal: just code_len in low byte, value in bits 16-23
                HUFFDEC_LITERAL | ((sym as u32) << 16) | (len as u32)
            } else if sym == 256 {
                // End of block
                HUFFDEC_END_OF_BLOCK | (len as u32)
            } else {
                // Length code: pack base + extra bits info
                let len_idx = sym - 257;
                if len_idx >= 29 {
                    continue;
                }
                let base = LEN_START[len_idx] as u32;
                let extra = LEN_EXTRA_BITS[len_idx] as u32;
                // Total bits = code_len + extra_bits
                let total_bits = (len as u32) + extra;
                // Entry: base in high 16, extra count in bits 8-13, total in low 8
                (base << 16) | (extra << 8) | total_bits
            };

            if len <= PRIMARY_BITS {
                // Replicate in primary table
                let replicate = 1 << (PRIMARY_BITS - len);
                for i in 0..replicate {
                    let idx = reversed | (i << len);
                    entries[idx as usize] = entry;
                }
            } else {
                // Would need subtable - mark as exceptional
                // For now, just put in primary with subtable flag
                let primary_idx = reversed & ((1 << PRIMARY_BITS) - 1);
                entries[primary_idx as usize] = HUFFDEC_SUBTABLE | (len as u32);
            }
        }

        Ok(Self { entries })
    }

    /// Build packed distance table
    pub fn build_dist(code_lens: &[u8]) -> io::Result<Self> {
        let mut entries = vec![0u32; PRIMARY_SIZE];

        let mut bl_count = [0u32; 16];
        for &len in code_lens {
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

        for (sym, &len) in code_lens.iter().enumerate() {
            if len == 0 || sym >= 30 {
                continue;
            }
            let len = len as usize;
            let code = next_code[len];
            next_code[len] += 1;

            let reversed = reverse_bits(code, len);

            let base = DIST_START[sym];
            let extra = DIST_EXTRA_BITS[sym] as u32;
            let total_bits = (len as u32) + extra;
            let entry = (base << 16) | (extra << 8) | total_bits;

            if len <= PRIMARY_BITS {
                let replicate = 1 << (PRIMARY_BITS - len);
                for i in 0..replicate {
                    let idx = reversed | (i << len);
                    if idx < PRIMARY_SIZE as u32 {
                        entries[idx as usize] = entry;
                    }
                }
            }
        }

        Ok(Self { entries })
    }

    #[inline(always)]
    pub fn lookup(&self, bits: u64) -> u32 {
        self.entries[(bits & PRIMARY_MASK) as usize]
    }
}

/// Optimized bit buffer for packed decode
pub struct PackedBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bitsleft: u32,
}

impl<'a> PackedBits<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut pb = Self {
            data,
            pos: 0,
            buf: 0,
            bitsleft: 0,
        };
        pb.refill();
        pb
    }

    /// Refill to at least 56 bits
    #[inline(always)]
    pub fn refill(&mut self) {
        while self.bitsleft < 56 && self.pos < self.data.len() {
            self.buf |= (self.data[self.pos] as u64) << self.bitsleft;
            self.pos += 1;
            self.bitsleft += 8;
        }
    }

    #[inline(always)]
    pub fn buffer(&self) -> u64 {
        self.buf
    }

    /// Consume bits using entry (libdeflate's trick: bitsleft -= entry)
    #[inline(always)]
    pub fn consume_entry(&mut self, entry: u32) {
        let n = entry & 0xFF;
        self.buf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.buf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        val
    }

    #[inline(always)]
    pub fn align(&mut self) {
        let skip = self.bitsleft % 8;
        if skip > 0 {
            self.consume(skip);
        }
    }

    #[inline(always)]
    pub fn past_end(&self) -> bool {
        self.pos > self.data.len() + 8
    }
}

/// Decode using packed tables
pub fn packed_decode_into(input: &[u8], output: &mut [u8]) -> io::Result<usize> {
    let mut bits = PackedBits::new(input);
    let mut out_pos = 0;

    loop {
        bits.refill();

        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => out_pos = decode_stored(&mut bits, output, out_pos)?,
            1 => out_pos = decode_huffman_packed(&mut bits, output, out_pos, true)?,
            2 => out_pos = decode_huffman_packed(&mut bits, output, out_pos, false)?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Reserved block type",
                ))
            }
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(out_pos)
}

fn decode_stored(
    bits: &mut PackedBits,
    output: &mut [u8],
    mut out_pos: usize,
) -> io::Result<usize> {
    bits.align();
    bits.refill();

    let len = bits.read(16) as usize;
    let nlen = bits.read(16);

    if len != (!nlen & 0xFFFF) as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored block length mismatch",
        ));
    }

    for _ in 0..len {
        bits.refill();
        if out_pos >= output.len() {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
        }
        output[out_pos] = bits.read(8) as u8;
        out_pos += 1;
    }

    Ok(out_pos)
}

fn decode_huffman_packed(
    bits: &mut PackedBits,
    output: &mut [u8],
    mut out_pos: usize,
    is_fixed: bool,
) -> io::Result<usize> {
    let (litlen_table, dist_table, litlen_fallback, dist_fallback) = if is_fixed {
        let (p1, p2) = get_fixed_packed_tables();
        let (f1, f2) = get_fixed_fallback_tables();
        (p1, p2, f1, f2)
    } else {
        let (p1, p2, f1, f2) = build_dynamic_packed_tables_with_fallback(bits)?;
        (p1, p2, f1, f2)
    };

    let out_fastloop_end = output.len().saturating_sub(258 + 32);
    let output_ptr = output.as_mut_ptr();

    // ==========================================================================
    // FASTLOOP - libdeflate style with packed entries
    // ==========================================================================
    while out_pos < out_fastloop_end && !bits.past_end() {
        bits.refill();

        let saved_buf = bits.buffer();
        let entry = litlen_table.lookup(saved_buf);

        // Check flags
        if entry & HUFFDEC_LITERAL != 0 {
            // Literal - extract from bits 16-23
            let lit = (entry >> 16) as u8;
            bits.consume_entry(entry);
            unsafe {
                *output_ptr.add(out_pos) = lit;
            }
            out_pos += 1;

            // Try another literal (multi-literal optimization)
            if bits.bitsleft >= 12 {
                let entry2 = litlen_table.lookup(bits.buffer());
                if entry2 & HUFFDEC_LITERAL != 0 {
                    let lit2 = (entry2 >> 16) as u8;
                    bits.consume_entry(entry2);
                    unsafe {
                        *output_ptr.add(out_pos) = lit2;
                    }
                    out_pos += 1;
                }
            }
            continue;
        }

        if entry & HUFFDEC_END_OF_BLOCK != 0 {
            bits.consume_entry(entry);
            return Ok(out_pos);
        }

        if entry & HUFFDEC_SUBTABLE != 0 {
            // Long code - fall back to generic loop
            break;
        }

        // Length code - extract base and extra bits
        let extra_bits = (entry >> 8) & 0x3F;
        let base = (entry >> 16) as usize;
        let total_bits = entry & 0xFF;
        let code_only = total_bits.saturating_sub(extra_bits);

        let extra_val = if extra_bits > 0 {
            ((saved_buf >> code_only) & ((1u64 << extra_bits) - 1)) as usize
        } else {
            0
        };
        let length = base + extra_val;

        bits.consume(total_bits);

        // Distance decode
        bits.refill();
        let saved_buf2 = bits.buffer();
        let dist_entry = dist_table.lookup(saved_buf2);

        let dist_extra_bits = (dist_entry >> 8) & 0x3F;
        let dist_base = (dist_entry >> 16) as usize;
        let dist_total_bits = dist_entry & 0xFF;
        let dist_code_only = dist_total_bits.saturating_sub(dist_extra_bits);

        let dist_extra_val = if dist_extra_bits > 0 {
            ((saved_buf2 >> dist_code_only) & ((1u64 << dist_extra_bits) - 1)) as usize
        } else {
            0
        };
        let distance = dist_base + dist_extra_val;

        bits.consume(dist_total_bits);

        if distance == 0 || distance > out_pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }

        // Copy match
        unsafe {
            let dst = output_ptr.add(out_pos);
            let src = output_ptr.add(out_pos - distance);
            copy_match(dst, src, length, distance);
        }
        out_pos += length;
    }

    // ==========================================================================
    // GENERIC LOOP - bounds checking, handles subtables
    // ==========================================================================
    loop {
        if bits.past_end() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unexpected end"));
        }

        bits.refill();
        let saved_buf = bits.buffer();
        let entry = litlen_table.lookup(saved_buf);

        // Handle subtable (long codes) - use TwoLevelTable fallback
        if entry & HUFFDEC_SUBTABLE != 0 {
            let (symbol, code_len) = litlen_fallback.decode(saved_buf);
            if code_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Huffman code",
                ));
            }
            bits.consume(code_len);

            if symbol < 256 {
                if out_pos >= output.len() {
                    return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
                }
                output[out_pos] = symbol as u8;
                out_pos += 1;
                continue;
            }

            if symbol == 256 {
                return Ok(out_pos);
            }

            // Length code
            let len_idx = (symbol - 257) as usize;
            if len_idx >= 29 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            bits.refill();
            let length =
                LEN_START[len_idx] as usize + bits.read(LEN_EXTRA_BITS[len_idx] as u32) as usize;

            let (dist_sym, dist_len) = dist_fallback.decode(bits.buffer());
            if dist_len == 0 || dist_sym >= 30 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance code",
                ));
            }
            bits.consume(dist_len);

            bits.refill();
            let distance = DIST_START[dist_sym as usize] as usize
                + bits.read(DIST_EXTRA_BITS[dist_sym as usize] as u32) as usize;

            if distance == 0 || distance > out_pos || out_pos + length > output.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid match"));
            }

            let src_start = out_pos - distance;
            if distance >= length {
                output.copy_within(src_start..src_start + length, out_pos);
            } else {
                for i in 0..length {
                    output[out_pos + i] = output[src_start + (i % distance)];
                }
            }
            out_pos += length;
            continue;
        }

        if entry & HUFFDEC_LITERAL != 0 {
            if out_pos >= output.len() {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "Output full"));
            }
            output[out_pos] = (entry >> 16) as u8;
            bits.consume_entry(entry);
            out_pos += 1;
            continue;
        }

        if entry & HUFFDEC_END_OF_BLOCK != 0 {
            bits.consume_entry(entry);
            return Ok(out_pos);
        }

        // Length
        let total_bits = entry & 0xFF;
        let extra_bits = (entry >> 8) & 0x3F;
        let base = (entry >> 16) as usize;
        let code_only = total_bits.saturating_sub(extra_bits);

        // For subtable case, we already consumed primary bits, need fresh buffer
        bits.refill();
        let len_buf = bits.buffer();
        let extra_val = if extra_bits > 0 {
            ((len_buf >> code_only) & ((1u64 << extra_bits) - 1)) as usize
        } else {
            0
        };
        let length = base + extra_val;
        bits.consume(total_bits);

        bits.refill();
        let saved_buf2 = bits.buffer();
        let dist_entry = dist_table.lookup(saved_buf2);
        let dist_total_bits = dist_entry & 0xFF;
        let dist_extra_bits = (dist_entry >> 8) & 0x3F;
        let dist_base = (dist_entry >> 16) as usize;
        let dist_code_only = dist_total_bits.saturating_sub(dist_extra_bits);
        let dist_extra_val = if dist_extra_bits > 0 {
            ((saved_buf2 >> dist_code_only) & ((1u64 << dist_extra_bits) - 1)) as usize
        } else {
            0
        };
        let distance = dist_base + dist_extra_val;
        bits.consume(dist_total_bits);

        if distance == 0 || distance > out_pos || out_pos + length > output.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid match"));
        }

        let src_start = out_pos - distance;
        if distance >= length {
            output.copy_within(src_start..src_start + length, out_pos);
        } else {
            for i in 0..length {
                output[out_pos + i] = output[src_start + (i % distance)];
            }
        }
        out_pos += length;
    }
}

/// Fast match copy
#[inline(always)]
unsafe fn copy_match(dst: *mut u8, src: *const u8, len: usize, offset: usize) {
    if offset == 1 {
        // RLE
        let byte = *src;
        std::ptr::write_bytes(dst, byte, len);
    } else if offset >= 8 {
        // Non-overlapping chunks
        let mut d = dst;
        let mut s = src;
        let mut remaining = len;
        while remaining >= 8 {
            (d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
            d = d.add(8);
            s = s.add(8);
            remaining -= 8;
        }
        for i in 0..remaining {
            *d.add(i) = *s.add(i);
        }
    } else {
        // Overlapping - byte by byte
        for i in 0..len {
            *dst.add(i) = *src.add(i % offset);
        }
    }
}

// Fixed tables cache
use std::sync::OnceLock;

static FIXED_PACKED: OnceLock<(PackedTable, PackedTable)> = OnceLock::new();
static FIXED_FALLBACK: OnceLock<(TwoLevelTable, TwoLevelTable)> = OnceLock::new();

fn fixed_code_lengths() -> ([u8; 288], [u8; 32]) {
    let mut lit_lens = [0u8; 288];
    for len in lit_lens.iter_mut().take(144) {
        *len = 8;
    }
    for len in lit_lens.iter_mut().take(256).skip(144) {
        *len = 9;
    }
    for len in lit_lens.iter_mut().take(280).skip(256) {
        *len = 7;
    }
    for len in lit_lens.iter_mut().skip(280) {
        *len = 8;
    }
    let dist_lens = [5u8; 32];
    (lit_lens, dist_lens)
}

fn get_fixed_packed_tables() -> (PackedTable, PackedTable) {
    FIXED_PACKED
        .get_or_init(|| {
            let (lit_lens, dist_lens) = fixed_code_lengths();
            (
                PackedTable::build_litlen(&lit_lens).unwrap(),
                PackedTable::build_dist(&dist_lens).unwrap(),
            )
        })
        .clone()
}

fn get_fixed_fallback_tables() -> (TwoLevelTable, TwoLevelTable) {
    FIXED_FALLBACK
        .get_or_init(|| {
            let (lit_lens, dist_lens) = fixed_code_lengths();
            (
                TwoLevelTable::build(&lit_lens).unwrap(),
                TwoLevelTable::build(&dist_lens).unwrap(),
            )
        })
        .clone()
}

fn build_dynamic_packed_tables_with_fallback(
    bits: &mut PackedBits,
) -> io::Result<(PackedTable, PackedTable, TwoLevelTable, TwoLevelTable)> {
    bits.refill();
    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;

    let mut code_len_lens = [0u8; 19];
    for i in 0..hclen {
        bits.refill();
        code_len_lens[CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    // Build code length table (simple - max 7 bits)
    let cl_table = build_simple_table(&code_len_lens)?;

    let total = hlit + hdist;
    let mut code_lens = vec![0u8; total];
    let mut i = 0;

    while i < total {
        bits.refill();
        let entry = cl_table[(bits.buffer() & 0x7F) as usize];
        let sym = (entry >> 8) as usize;
        let len = (entry & 0xFF) as u32;
        bits.consume(len);

        match sym {
            0..=15 => {
                code_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                bits.refill();
                let repeat = bits.read(2) as usize + 3;
                let prev = if i > 0 { code_lens[i - 1] } else { 0 };
                for _ in 0..repeat.min(total - i) {
                    code_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                bits.refill();
                let repeat = bits.read(3) as usize + 3;
                i += repeat.min(total - i);
            }
            18 => {
                bits.refill();
                let repeat = bits.read(7) as usize + 11;
                i += repeat.min(total - i);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length",
                ))
            }
        }
    }

    let litlen_packed = PackedTable::build_litlen(&code_lens[..hlit])?;
    let dist_packed = PackedTable::build_dist(&code_lens[hlit..])?;
    let litlen_fallback = TwoLevelTable::build(&code_lens[..hlit])?;
    let dist_fallback = TwoLevelTable::build(&code_lens[hlit..])?;

    Ok((litlen_packed, dist_packed, litlen_fallback, dist_fallback))
}

fn build_simple_table(lens: &[u8]) -> io::Result<Vec<u32>> {
    let mut table = vec![0u32; 128];
    let mut bl_count = [0u32; 8];
    for &len in lens {
        if len > 0 && len < 8 {
            bl_count[len as usize] += 1;
        }
    }

    let mut next_code = [0u32; 8];
    let mut code = 0u32;
    for bits in 1..8 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for (sym, &len) in lens.iter().enumerate() {
        if len == 0 || len >= 8 {
            continue;
        }
        let len = len as usize;
        let code = next_code[len];
        next_code[len] += 1;
        let reversed = reverse_bits(code, len);
        let entry = ((sym as u32) << 8) | (len as u32);

        let replicate = 1 << (7 - len);
        for i in 0..replicate {
            let idx = reversed | (i << len);
            if idx < 128 {
                table[idx as usize] = entry;
            }
        }
    }

    Ok(table)
}

#[inline]
fn reverse_bits(code: u32, len: usize) -> u32 {
    let mut result = 0u32;
    let mut code = code;
    for _ in 0..len {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_decode_simple() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let original = b"Hello, World! This is a test of packed decode.";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Parse gzip header
        let deflate_start = 10; // minimum header
        let deflate_end = compressed.len() - 8;
        let deflate_data = &compressed[deflate_start..deflate_end];

        let mut output = vec![0u8; original.len() + 100];
        let written = packed_decode_into(deflate_data, &mut output).unwrap();
        output.truncate(written);

        assert_eq!(output, original);
    }

    #[test]
    fn test_packed_decode_medium() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Medium-sized test to catch issues
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let deflate_data = &compressed[10..compressed.len() - 8];

        let mut output = vec![0u8; original.len() + 100];
        let written = packed_decode_into(deflate_data, &mut output).unwrap();
        output.truncate(written);

        assert_eq!(output.len(), original.len());
        assert_eq!(output, original);
    }

    #[test]
    fn benchmark_packed_vs_libdeflate() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let deflate_data = &compressed[10..compressed.len() - 8];

        const ITERS: usize = 50;

        // Warmup
        for _ in 0..3 {
            let mut output = vec![0u8; original.len() + 1000];
            packed_decode_into(deflate_data, &mut output).unwrap();
        }

        // Benchmark packed
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = vec![0u8; original.len() + 1000];
            let _ = packed_decode_into(deflate_data, &mut output);
            std::hint::black_box(&output);
        }
        let packed_time = start.elapsed();

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = vec![0u8; original.len() + 1000];
            let mut dec = libdeflater::Decompressor::new();
            let _ = dec.deflate_decompress(deflate_data, &mut output);
            std::hint::black_box(&output);
        }
        let libdeflate_time = start.elapsed();

        let packed_avg = packed_time / ITERS as u32;
        let libdeflate_avg = libdeflate_time / ITERS as u32;

        let packed_speed = original.len() as f64 / packed_avg.as_secs_f64() / 1_000_000.0;
        let libdeflate_speed = original.len() as f64 / libdeflate_avg.as_secs_f64() / 1_000_000.0;

        eprintln!("\n=== PACKED DECODE vs LIBDEFLATE (1MB x {}) ===", ITERS);
        eprintln!(
            "Packed:     {:>8?}/iter  ({:.0} MB/s)",
            packed_avg, packed_speed
        );
        eprintln!(
            "libdeflate: {:>8?}/iter  ({:.0} MB/s)",
            libdeflate_avg, libdeflate_speed
        );
        eprintln!(
            "Ratio: {:.1}% of libdeflate",
            packed_speed / libdeflate_speed * 100.0
        );
    }

    #[test]
    fn benchmark_packed_on_silesia() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping - no silesia file");
                return;
            }
        };

        // Get expected size from flate2
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        decoder.read_to_end(&mut expected).unwrap();
        let expected_size = expected.len();

        eprintln!(
            "\n=== PACKED DECODE on SILESIA ({:.1} MB) ===",
            expected_size as f64 / 1_000_000.0
        );

        // Parse gzip header
        let flags = data[3];
        let mut pos = 10;
        if flags & 0x04 != 0 {
            let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2 + xlen;
        }
        if flags & 0x08 != 0 {
            while data[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if flags & 0x10 != 0 {
            while data[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if flags & 0x02 != 0 {
            pos += 2;
        }
        let deflate_data = &data[pos..data.len() - 8];

        // Benchmark packed (3 iterations)
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let mut output = vec![0u8; expected_size + 1000];
            packed_decode_into(deflate_data, &mut output).unwrap();
        }
        let packed_time = start.elapsed() / 3;

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let mut decompressor = libdeflater::Decompressor::new();
            let mut output = vec![0u8; expected_size + 1000];
            decompressor
                .deflate_decompress(deflate_data, &mut output)
                .unwrap();
        }
        let libdeflate_time = start.elapsed() / 3;

        let packed_speed = expected_size as f64 / packed_time.as_secs_f64() / 1_000_000.0;
        let libdeflate_speed = expected_size as f64 / libdeflate_time.as_secs_f64() / 1_000_000.0;

        eprintln!("Packed:     {:>8?} = {:.1} MB/s", packed_time, packed_speed);
        eprintln!(
            "libdeflate: {:>8?} = {:.1} MB/s",
            libdeflate_time, libdeflate_speed
        );
        eprintln!(
            "Ratio: {:.1}% of libdeflate",
            packed_speed / libdeflate_speed * 100.0
        );
    }
}
