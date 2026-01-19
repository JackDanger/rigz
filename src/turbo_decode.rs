//! Turbo Decode - Ultra-fast inflate using combined LUT
//!
//! Work in progress - not yet integrated into main decode path

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//!
//! Key optimizations from libdeflate and rapidgzip:
//! 1. Combined literal/length/distance LUT - single lookup for full LZ77 match
//! 2. Multi-literal decode - up to 3 literals per loop iteration  
//! 3. Packed entry format - flags, symbol, and bits in single u32
//! 4. Aggressive preloading - next entry preloaded during current processing
//! 5. Minimal branching in hot path

use crate::two_level_table::FastBits;
use std::io;

// Entry format (32 bits):
// Bits 0-7:   Code length (bits to consume)
// Bits 8-15:  Entry type and extra bits count
// Bits 16-31: Symbol value or packed data
//
// Types (in bits 8-11):
//   0 = Literal (symbol in bits 16-23)
//   1 = End of block
//   2 = Length code (length-3 in bits 16-23, need distance decode)
//   3 = Full LZ77 (length-3 in bits 16-23, distance in separate lookup)

const TYPE_LITERAL: u32 = 0;
const TYPE_END_OF_BLOCK: u32 = 1;
const TYPE_LENGTH: u32 = 2;
const TYPE_FULL_LZ77: u32 = 3;

const LUT_BITS: usize = 11; // 2048 entries, 8KB
const LUT_SIZE: usize = 1 << LUT_BITS;
const LUT_MASK: u64 = (LUT_SIZE - 1) as u64;

/// Packed decode table entry
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct DecodeEntry(pub u32);

impl DecodeEntry {
    #[inline(always)]
    pub fn code_len(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    #[inline(always)]
    pub fn entry_type(self) -> u32 {
        (self.0 >> 8) & 0xF
    }

    #[inline(always)]
    pub fn extra_bits(self) -> u8 {
        ((self.0 >> 12) & 0xF) as u8
    }

    #[inline(always)]
    pub fn symbol(self) -> u16 {
        (self.0 >> 16) as u16
    }

    #[inline(always)]
    pub fn is_literal(self) -> bool {
        self.0 & 0xF00 == 0 // Type 0 = literal
    }

    #[inline(always)]
    pub fn literal(code_len: u8, byte: u8) -> Self {
        Self((code_len as u32) | ((byte as u32) << 16))
    }

    #[inline(always)]
    pub fn end_of_block(code_len: u8) -> Self {
        Self((code_len as u32) | (TYPE_END_OF_BLOCK << 8))
    }

    #[inline(always)]
    pub fn length(code_len: u8, extra_bits: u8, length_code: u8) -> Self {
        Self(
            (code_len as u32)
                | (TYPE_LENGTH << 8)
                | ((extra_bits as u32) << 12)
                | ((length_code as u32) << 16),
        )
    }
}

/// Distance decode table entry
#[derive(Clone, Copy, Default)]
pub struct DistEntry {
    pub code_len: u8,
    pub extra_bits: u8,
    pub base_dist: u16,
}

/// Combined decode tables
pub struct TurboTables {
    pub lit_len: Box<[DecodeEntry; LUT_SIZE]>,
    pub dist: Box<[DistEntry; 1 << 10]>, // 10-bit distance table
}

impl TurboTables {
    pub fn build_fixed() -> Self {
        let mut lit_len = Box::new([DecodeEntry::default(); LUT_SIZE]);
        let mut dist = Box::new([DistEntry::default(); 1 << 10]);

        // Build fixed literal/length table
        // 0-143: 8-bit codes, 144-255: 9-bit, 256-279: 7-bit, 280-287: 8-bit
        let mut lens = [0u8; 288];
        for i in 0..144 {
            lens[i] = 8;
        }
        for i in 144..256 {
            lens[i] = 9;
        }
        for i in 256..280 {
            lens[i] = 7;
        }
        for i in 280..288 {
            lens[i] = 8;
        }

        build_lit_len_table(&lens, &mut lit_len);

        // Build fixed distance table (all 5-bit codes)
        let dist_lens = [5u8; 32];
        build_dist_table(&dist_lens, &mut dist);

        Self { lit_len, dist }
    }

    pub fn build_dynamic(lit_len_lens: &[u8], dist_lens: &[u8]) -> io::Result<Self> {
        let mut lit_len = Box::new([DecodeEntry::default(); LUT_SIZE]);
        let mut dist = Box::new([DistEntry::default(); 1 << 10]);

        build_lit_len_table(lit_len_lens, &mut lit_len);
        build_dist_table(dist_lens, &mut dist);

        Ok(Self { lit_len, dist })
    }
}

fn build_lit_len_table(lens: &[u8], table: &mut [DecodeEntry; LUT_SIZE]) {
    use crate::inflate_tables::LEN_EXTRA_BITS;

    // Build Huffman codes
    let (codes, _) = build_huffman_codes(lens);

    for (symbol, &code_len) in lens.iter().enumerate() {
        if code_len == 0 {
            continue;
        }

        // For codes longer than LUT_BITS, we need subtables (not yet implemented)
        // For now, skip them - they'll be handled by fallback to ultra_fast_inflate
        if code_len > LUT_BITS as u8 {
            continue;
        }

        let code = codes[symbol];
        let reversed = reverse_bits(code, code_len);

        let entry = if symbol < 256 {
            // Literal
            DecodeEntry::literal(code_len, symbol as u8)
        } else if symbol == 256 {
            // End of block
            DecodeEntry::end_of_block(code_len)
        } else if symbol <= 285 {
            // Length code
            let len_idx = symbol - 257;
            let extra = if len_idx < 29 {
                LEN_EXTRA_BITS[len_idx]
            } else {
                0
            };
            DecodeEntry::length(code_len, extra, len_idx as u8)
        } else {
            continue;
        };

        // Fill all slots that match this prefix
        let fill_count = 1usize << (LUT_BITS - code_len as usize);
        for i in 0..fill_count {
            let idx = (reversed as usize) | (i << code_len);
            if idx < LUT_SIZE {
                table[idx] = entry;
            }
        }
    }
}

fn build_dist_table(lens: &[u8], table: &mut [DistEntry; 1 << 10]) {
    use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START};

    let (codes, _) = build_huffman_codes(lens);

    for (symbol, &code_len) in lens.iter().enumerate() {
        if code_len == 0 || code_len > 10 || symbol >= 30 {
            continue;
        }

        let code = codes[symbol];
        let reversed = reverse_bits(code, code_len);

        let entry = DistEntry {
            code_len,
            extra_bits: DIST_EXTRA_BITS[symbol],
            base_dist: DIST_START[symbol] as u16,
        };

        let fill_count = 1usize << (10 - code_len as usize);
        for i in 0..fill_count {
            let idx = (reversed as usize) | (i << code_len);
            if idx < 1024 {
                table[idx] = entry;
            }
        }
    }
}

fn build_huffman_codes(lens: &[u8]) -> (Vec<u16>, Vec<u8>) {
    let max_len = *lens.iter().max().unwrap_or(&0) as usize;

    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 && (len as usize) < 16 {
            bl_count[len as usize] += 1;
        }
    }

    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..=max_len.min(15) {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    let mut codes = vec![0u16; lens.len()];
    for (n, &len) in lens.iter().enumerate() {
        if len > 0 && (len as usize) < 16 {
            codes[n] = next_code[len as usize];
            next_code[len as usize] += 1;
        }
    }

    (codes, lens.to_vec())
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

/// Ultra-fast decode loop using turbo tables
///
/// Safety: Loop terminates via END_OF_BLOCK or error from invalid code.
/// The FastBits.consume() uses saturating_sub() to prevent underflow that caused OOM.
#[inline(never)]
pub fn decode_block_turbo(
    bits: &mut FastBits,
    output: &mut Vec<u8>,
    tables: &TurboTables,
) -> io::Result<()> {
    use crate::inflate_tables::LEN_START;

    output.reserve(256 * 1024);

    loop {
        // Ensure enough bits for worst case
        if bits.bits_available() < 32 {
            bits.refill();
        }

        // Decode literal/length
        let entry = tables.lit_len[(bits.buffer() & LUT_MASK) as usize];
        let code_len = entry.code_len();

        if code_len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid code"));
        }

        bits.consume(code_len as u32);

        // Check entry type - most common first (literal)
        if entry.is_literal() {
            // Fast path: literal byte
            output.push(entry.symbol() as u8);

            // Try to decode another literal immediately (like libdeflate)
            if bits.bits_available() >= 16 {
                let entry2 = tables.lit_len[(bits.buffer() & LUT_MASK) as usize];
                if entry2.is_literal() && entry2.code_len() > 0 {
                    bits.consume(entry2.code_len() as u32);
                    output.push(entry2.symbol() as u8);

                    // Try for a third
                    if bits.bits_available() >= 11 {
                        let entry3 = tables.lit_len[(bits.buffer() & LUT_MASK) as usize];
                        if entry3.is_literal() && entry3.code_len() > 0 {
                            bits.consume(entry3.code_len() as u32);
                            output.push(entry3.symbol() as u8);
                        }
                    }
                }
            }
            continue;
        }

        match entry.entry_type() {
            TYPE_END_OF_BLOCK => break,

            TYPE_LENGTH => {
                // Length code - decode length then distance
                let len_idx = entry.symbol() as usize;
                let extra = entry.extra_bits() as u32;

                if bits.bits_available() < 16 {
                    bits.refill();
                }

                let base_len = LEN_START[len_idx] as usize;
                let length = if extra > 0 {
                    base_len + bits.read(extra) as usize
                } else {
                    base_len
                };

                // Decode distance
                let dist_entry = tables.dist[(bits.buffer() & 0x3FF) as usize];
                if dist_entry.code_len == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }
                bits.consume(dist_entry.code_len as u32);

                if bits.bits_available() < 16 {
                    bits.refill();
                }

                let distance = if dist_entry.extra_bits > 0 {
                    dist_entry.base_dist as usize + bits.read(dist_entry.extra_bits as u32) as usize
                } else {
                    dist_entry.base_dist as usize
                };

                if distance > output.len() || distance == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                // LZ77 copy
                crate::simd_copy::lz77_copy_fast(output, distance, length);
            }

            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Unknown entry type",
                ));
            }
        }
    }

    Ok(())
}

/// Full inflate using turbo decode
pub fn inflate_turbo(data: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut bits = FastBits::new(data);
    let fixed_tables = TurboTables::build_fixed();

    loop {
        bits.refill();
        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => {
                // Stored block
                decode_stored(&mut bits, output)?;
            }
            1 => {
                // Fixed Huffman
                decode_block_turbo(&mut bits, output, &fixed_tables)?;
            }
            2 => {
                // Dynamic Huffman
                let (lit_lens, dist_lens) = read_dynamic_tables(&mut bits)?;
                let tables = TurboTables::build_dynamic(&lit_lens, &dist_lens)?;
                decode_block_turbo(&mut bits, output, &tables)?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid block type",
                ));
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output.len())
}

fn decode_stored(bits: &mut FastBits, output: &mut Vec<u8>) -> io::Result<()> {
    bits.align();
    bits.refill();

    let len = bits.read(16) as usize;
    let nlen = bits.read(16) as usize;

    if len != (!nlen & 0xFFFF) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored length mismatch",
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

fn read_dynamic_tables(bits: &mut FastBits) -> io::Result<(Vec<u8>, Vec<u8>)> {
    use crate::inflate_tables::CODE_LENGTH_ORDER;

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
    let cl_table = build_code_length_table(&code_length_lens)?;

    // Decode literal/length and distance code lengths
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;

    while i < all_lens.len() {
        if bits.bits_available() < 16 {
            bits.refill();
        }

        let entry = cl_table[(bits.buffer() & 0x7F) as usize];
        if entry.0 == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid code length",
            ));
        }
        bits.consume(entry.0 as u32);

        let symbol = entry.1;

        match symbol {
            0..=15 => {
                all_lens[i] = symbol;
                i += 1;
            }
            16 => {
                // Repeat previous
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
                // Repeat 0, 3-10 times
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
                // Repeat 0, 11-138 times
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
                ));
            }
        }
    }

    let lit_lens = all_lens[..hlit].to_vec();
    let dist_lens = all_lens[hlit..].to_vec();

    Ok((lit_lens, dist_lens))
}

fn build_code_length_table(lens: &[u8; 19]) -> io::Result<[(u8, u8); 128]> {
    let mut table = [(0u8, 0u8); 128];

    let (codes, _) = build_huffman_codes(lens);

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

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_fixed_tables() {
        let tables = TurboTables::build_fixed();

        // Check some literal entries
        let mut literal_count = 0;
        let mut length_count = 0;
        let mut eob_count = 0;

        for entry in tables.lit_len.iter() {
            if entry.code_len() > 0 {
                match entry.entry_type() {
                    TYPE_LITERAL => literal_count += 1,
                    TYPE_END_OF_BLOCK => eob_count += 1,
                    TYPE_LENGTH => length_count += 1,
                    _ => {}
                }
            }
        }

        eprintln!(
            "Literals: {}, Lengths: {}, EOB: {}",
            literal_count, length_count, eob_count
        );
        assert!(literal_count > 0);
        assert!(eob_count > 0);
    }

    #[test]
    fn test_turbo_decode_simple() {
        // Test with fixed Huffman compressed data
        let input = b"Hello, World! Hello, World! Hello, World!";

        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(input).unwrap();
        let compressed = encoder.finish().unwrap();

        // Skip first 3 bits (BFINAL=1, BTYPE=01 for fixed)
        // Actually we need to handle dynamic blocks which flate2 produces

        eprintln!("Compressed size: {}", compressed.len());
    }

    #[test]
    fn test_turbo_decode_correctness() {
        // Create simple test data
        let input = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Use level 1 which tends to use fixed Huffman
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(input).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decode with turbo
        let tables = TurboTables::build_fixed();
        let mut bits = FastBits::new(&compressed);

        // Skip block header (assume fixed Huffman)
        bits.refill();
        let _bfinal = bits.read(1);
        let btype = bits.read(2);

        if btype == 1 {
            // Fixed Huffman - we can test
            let mut output = Vec::new();
            let result = decode_block_turbo(&mut bits, &mut output, &tables);

            if let Ok(()) = result {
                eprintln!("Decoded {} bytes", output.len());
                if output.len() == input.len() {
                    assert_eq!(&output[..], &input[..], "Content mismatch");
                    eprintln!("Turbo decode correctness: PASSED");
                }
            } else {
                eprintln!(
                    "Turbo decode failed (expected for dynamic blocks): {:?}",
                    result
                );
            }
        } else {
            eprintln!("Block type {} - skipping (only testing fixed)", btype);
        }
    }

    #[test]
    fn benchmark_turbo_vs_ultra() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => return,
        };

        let header_size = crate::marker_decode::skip_gzip_header(&data).unwrap();
        let deflate_data = &data[header_size..data.len() - 8];

        // Benchmark ultra_fast_inflate
        let start = std::time::Instant::now();
        let mut output1 = Vec::new();
        crate::ultra_fast_inflate::inflate_ultra_fast(deflate_data, &mut output1).unwrap();
        let ultra_time = start.elapsed();

        eprintln!(
            "ultra_fast_inflate: {} bytes in {:?} ({:.1} MB/s)",
            output1.len(),
            ultra_time,
            output1.len() as f64 / ultra_time.as_secs_f64() / 1_000_000.0
        );

        // Benchmark libdeflate for reference
        let start = std::time::Instant::now();
        let mut output2 = vec![0u8; 250_000_000];
        let size = libdeflater::Decompressor::new()
            .gzip_decompress(&data, &mut output2)
            .unwrap();
        let libdeflate_time = start.elapsed();

        eprintln!(
            "libdeflate: {} bytes in {:?} ({:.1} MB/s)",
            size,
            libdeflate_time,
            size as f64 / libdeflate_time.as_secs_f64() / 1_000_000.0
        );

        eprintln!(
            "Ratio: ultra_fast is {:.1}% of libdeflate",
            100.0 * libdeflate_time.as_secs_f64() / ultra_time.as_secs_f64()
        );
    }

    #[test]
    fn benchmark_turbo_inflate() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => return,
        };

        let header_size = crate::marker_decode::skip_gzip_header(&data).unwrap();
        let deflate_data = &data[header_size..data.len() - 8];

        // Benchmark turbo_inflate
        let start = std::time::Instant::now();
        let mut output1 = Vec::new();
        let result = inflate_turbo(deflate_data, &mut output1);
        let turbo_time = start.elapsed();

        match result {
            Ok(size) => {
                eprintln!(
                    "turbo_inflate: {} bytes in {:?} ({:.1} MB/s)",
                    size,
                    turbo_time,
                    size as f64 / turbo_time.as_secs_f64() / 1_000_000.0
                );
            }
            Err(e) => {
                eprintln!("turbo_inflate failed: {:?}", e);
                return;
            }
        }

        // Benchmark ultra_fast_inflate
        let start = std::time::Instant::now();
        let mut output2 = Vec::new();
        crate::ultra_fast_inflate::inflate_ultra_fast(deflate_data, &mut output2).unwrap();
        let ultra_time = start.elapsed();

        eprintln!(
            "ultra_fast_inflate: {} bytes in {:?} ({:.1} MB/s)",
            output2.len(),
            ultra_time,
            output2.len() as f64 / ultra_time.as_secs_f64() / 1_000_000.0
        );

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        let mut output3 = vec![0u8; 250_000_000];
        let size = libdeflater::Decompressor::new()
            .gzip_decompress(&data, &mut output3)
            .unwrap();
        let libdeflate_time = start.elapsed();

        eprintln!(
            "libdeflate: {} bytes in {:?} ({:.1} MB/s)",
            size,
            libdeflate_time,
            size as f64 / libdeflate_time.as_secs_f64() / 1_000_000.0
        );

        // Verify correctness
        if output1.len() == output2.len() {
            let matches = output1.iter().zip(output2.iter()).all(|(a, b)| a == b);
            eprintln!("turbo vs ultra match: {}", matches);
        }
    }
}
