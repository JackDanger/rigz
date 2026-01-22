//! Ultra-Fast Decoder - Exceeding libdeflate
//!
//! Key innovations beyond libdeflate:
//! 1. Double-literal decode (rapidgzip's key optimization)
//! 2. Consume-first pattern with optimal refill timing
//! 3. Branchless decode for length/distance
//!
//! Target: 130%+ of libdeflate single-threaded throughput

#![allow(dead_code)]

use crate::double_literal::DoubleLitCache;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::io::{Error, ErrorKind, Result};

// =============================================================================
// Bit Reader - Optimized for double-literal decode
// =============================================================================

struct UltraBits<'a> {
    data: &'a [u8],
    pos: usize,
    bitbuf: u64,
    bitsleft: u32,
}

impl<'a> UltraBits<'a> {
    #[inline(always)]
    fn new(data: &'a [u8]) -> Self {
        let mut bits = Self {
            data,
            pos: 0,
            bitbuf: 0,
            bitsleft: 0,
        };
        bits.refill();
        bits
    }

    #[inline(always)]
    fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let word = unsafe { (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned() };
            let word = u64::from_le(word);
            self.bitbuf |= word << (self.bitsleft as u8);
            self.pos += 7;
            self.pos -= ((self.bitsleft >> 3) & 0x7) as usize;
            self.bitsleft |= 56;
        } else {
            self.refill_slow();
        }
    }

    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.bitsleft <= 56 {
            if self.pos < self.data.len() {
                self.bitbuf |= (self.data[self.pos] as u64) << self.bitsleft;
                self.pos += 1;
                self.bitsleft += 8;
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn peek(&self) -> u64 {
        self.bitbuf
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.bitbuf >>= n as u8;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    #[inline(always)]
    fn available(&self) -> u32 {
        (self.bitsleft as u8) as u32
    }

    fn align_to_byte(&mut self) {
        let discard = (self.bitsleft as u8) & 7;
        self.consume(discard as u32);
    }

    fn read_u16(&mut self) -> u16 {
        self.align_to_byte();
        if self.available() >= 16 {
            let val = (self.bitbuf & 0xFFFF) as u16;
            self.consume(16);
            return val;
        }
        self.refill();
        let val = (self.bitbuf & 0xFFFF) as u16;
        self.consume(16);
        val
    }
}

// =============================================================================
// Match Copy - Same as consume_first_decode
// =============================================================================

#[inline(always)]
fn copy_match(output: &mut [u8], out_pos: usize, distance: u32, length: u32) -> usize {
    let dist = distance as usize;
    let len = length as usize;

    unsafe {
        let out_ptr = output.as_mut_ptr();
        let mut dst = out_ptr.add(out_pos);
        let mut src = out_ptr.add(out_pos - dist);
        let end = dst.add(len);

        if dist >= 8 {
            // Fast path: 5-word unconditional copy
            for _ in 0..5 {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(8);
                dst = dst.add(8);
            }
            while dst < end {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(8);
                dst = dst.add(8);
            }
        } else if dist == 1 {
            // RLE broadcast
            let v = 0x0101010101010101u64 * (*src as u64);
            for _ in 0..4 {
                (dst as *mut u64).write_unaligned(v);
                dst = dst.add(8);
            }
            while dst < end {
                (dst as *mut u64).write_unaligned(v);
                dst = dst.add(8);
            }
        } else {
            // Small distance stride copy
            for _ in 0..4 {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(dist);
                dst = dst.add(dist);
            }
            while dst < end {
                (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned());
                src = src.add(dist);
                dst = dst.add(dist);
            }
        }
    }

    out_pos + len
}

// =============================================================================
// Ultra Decode - Double-literal + consume-first hybrid
// =============================================================================

/// Decode Huffman block with double-literal optimization
fn decode_huffman_ultra(
    bits: &mut UltraBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
    double_cache: &DoubleLitCache,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 274;

    bits.refill();
    let out_ptr = output.as_mut_ptr();

    // FASTLOOP with double-literal optimization
    while out_pos + FASTLOOP_MARGIN <= output.len() {
        // Try double-literal cache first
        let double_entry = double_cache.lookup(bits.peek());

        if double_entry.is_literal() {
            if double_entry.has_second() {
                // DOUBLE LITERAL - write both at once!
                let lit1 = double_entry.symbol1();
                let lit2 = double_entry.symbol2();
                let total_bits = double_entry.total_bits() as u32;

                unsafe {
                    *out_ptr.add(out_pos) = lit1;
                    *out_ptr.add(out_pos + 1) = lit2;
                }
                out_pos += 2;
                bits.consume(total_bits);

                // Check if we need refill
                if bits.available() < 32 {
                    bits.refill();
                }
                continue;
            } else {
                // Single literal from cache
                let lit = double_entry.symbol1();
                let lit_bits = double_entry.total_bits() as u32;

                unsafe {
                    *out_ptr.add(out_pos) = lit;
                }
                out_pos += 1;
                bits.consume(lit_bits);

                if bits.available() < 32 {
                    bits.refill();
                }
                continue;
            }
        }

        // Not a literal - fall back to regular decode
        let saved_bitbuf = bits.peek();
        let mut entry = litlen.lookup(saved_bitbuf);

        // Handle subtable - consume TABLE_BITS first, then subtable entry bits
        // For subtable entries, we need saved_sub for decode_length (captured after main bits)
        let saved_for_decode;
        if entry.is_subtable_ptr() {
            bits.consume(LitLenTable::TABLE_BITS as u32);
            entry = litlen.lookup_subtable(entry, saved_bitbuf);
            saved_for_decode = bits.peek(); // Capture AFTER main bits consumed
            bits.consume(entry.total_bits() as u32);
        } else {
            bits.consume(entry.total_bits() as u32);
            saved_for_decode = saved_bitbuf; // Use original for non-subtable
        }

        // Check for literal
        if entry.is_literal() {
            unsafe {
                *out_ptr.add(out_pos) = entry.literal_value();
            }
            out_pos += 1;
            if bits.available() < 32 {
                bits.refill();
            }
            continue;
        }

        // Check for EOB
        if entry.is_end_of_block() {
            return Ok(out_pos);
        }

        // Length code - decode length using appropriate saved_bitbuf
        let length = entry.decode_length(saved_for_decode);

        // Decode distance
        bits.refill();
        let dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);

        if dist_entry.is_subtable_ptr() {
            bits.consume(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
        }

        let dist_extra_saved = bits.peek();
        bits.consume(dist_entry.total_bits() as u32);
        let distance = dist_entry.decode_distance(dist_extra_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        // Preload before copy
        bits.refill();

        out_pos = copy_match(output, out_pos, distance, length);
    }

    // GENERIC LOOP (near end)
    decode_huffman_generic(bits, output, out_pos, litlen, dist)
}

/// Generic loop for end of output buffer
fn decode_huffman_generic(
    bits: &mut UltraBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    loop {
        if bits.available() < 15 {
            bits.refill();
        }

        let saved_bitbuf = bits.peek();
        let mut entry = litlen.lookup(saved_bitbuf);

        if entry.is_subtable_ptr() {
            entry = litlen.lookup_subtable(entry, saved_bitbuf);
        }

        bits.consume(entry.total_bits() as u32);

        if entry.is_literal() {
            if out_pos >= output.len() {
                return Err(Error::new(ErrorKind::WriteZero, "Output full"));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            continue;
        }

        if entry.is_end_of_block() {
            return Ok(out_pos);
        }

        let length = entry.decode_length(saved_bitbuf);

        if bits.available() < 15 {
            bits.refill();
        }

        let dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);
        if dist_entry.is_subtable_ptr() {
            bits.consume(DistTable::TABLE_BITS as u32);
            dist_entry = dist.lookup_subtable(dist_entry, dist_saved);
        }
        let dist_extra_saved = bits.peek();
        bits.consume(dist_entry.total_bits() as u32);
        let distance = dist_entry.decode_distance(dist_extra_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        if out_pos + length as usize > output.len() {
            return Err(Error::new(ErrorKind::WriteZero, "Output full"));
        }

        out_pos = copy_match(output, out_pos, distance, length);
    }
}

// =============================================================================
// Block Handling
// =============================================================================

fn decode_stored(bits: &mut UltraBits, output: &mut [u8], mut out_pos: usize) -> Result<usize> {
    bits.align_to_byte();
    let len = bits.read_u16();
    let nlen = bits.read_u16();

    if len != !nlen {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid stored block"));
    }

    let len = len as usize;
    if len == 0 {
        return Ok(out_pos);
    }

    if out_pos + len > output.len() {
        return Err(Error::new(ErrorKind::WriteZero, "Output full"));
    }

    // Drain buffered bytes first
    let mut remaining = len;
    while remaining > 0 && bits.available() >= 8 {
        output[out_pos] = (bits.bitbuf & 0xFF) as u8;
        bits.consume(8);
        out_pos += 1;
        remaining -= 1;
    }

    // Copy remaining from input
    if remaining > 0 {
        if bits.pos + remaining > bits.data.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "Unexpected end of stored block",
            ));
        }
        output[out_pos..out_pos + remaining]
            .copy_from_slice(&bits.data[bits.pos..bits.pos + remaining]);
        bits.pos += remaining;
        out_pos += remaining;
    }

    bits.bitbuf = 0;
    bits.bitsleft = 0;
    Ok(out_pos)
}

fn decode_fixed(bits: &mut UltraBits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    // Get or build fixed tables and double cache
    static FIXED_TABLES: std::sync::OnceLock<(LitLenTable, DistTable, DoubleLitCache)> =
        std::sync::OnceLock::new();

    let (litlen, dist, double_cache) = FIXED_TABLES.get_or_init(|| {
        let tables = crate::libdeflate_decode::get_fixed_tables();
        let double = DoubleLitCache::build(&tables.0);
        (tables.0.clone(), tables.1.clone(), double)
    });

    decode_huffman_ultra(bits, output, out_pos, litlen, dist, double_cache)
}

fn decode_dynamic(bits: &mut UltraBits, output: &mut [u8], out_pos: usize) -> Result<usize> {
    // Read dynamic Huffman table header
    if bits.available() < 14 {
        bits.refill();
    }

    let hlit = (bits.peek() & 0x1F) as usize + 257;
    bits.consume(5);
    let hdist = (bits.peek() & 0x1F) as usize + 1;
    bits.consume(5);
    let hclen = (bits.peek() & 0xF) as usize + 4;
    bits.consume(4);

    // Read code length code lengths
    const CODE_LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    let mut code_length_lengths = [0u8; 19];
    for i in 0..hclen {
        if bits.available() < 3 {
            bits.refill();
        }
        code_length_lengths[CODE_LENGTH_ORDER[i]] = (bits.peek() & 0x7) as u8;
        bits.consume(3);
    }

    // Build code length table
    let cl_table = build_code_length_table(&code_length_lengths)?;

    // Read literal/length and distance code lengths
    let mut all_lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        if bits.available() < 15 {
            bits.refill();
        }

        let entry = cl_table[(bits.peek() & 0x7F) as usize];
        let symbol = (entry >> 8) as u8;
        let len = (entry & 0xFF) as u8;
        bits.consume(len as u32);

        match symbol {
            0..=15 => {
                all_lengths[i] = symbol;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(Error::new(ErrorKind::InvalidData, "Invalid repeat"));
                }
                let repeat = 3 + (bits.peek() & 0x3) as usize;
                bits.consume(2);
                let val = all_lengths[i - 1];
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = val;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + (bits.peek() & 0x7) as usize;
                bits.consume(3);
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = 11 + (bits.peek() & 0x7F) as usize;
                bits.consume(7);
                for _ in 0..repeat {
                    if i >= hlit + hdist {
                        break;
                    }
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Invalid code length symbol",
                ))
            }
        }
    }

    let litlen_lengths = &all_lengths[..hlit];
    let dist_lengths = &all_lengths[hlit..];

    let litlen_table = LitLenTable::build(litlen_lengths)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid litlen table"))?;
    let dist_table = DistTable::build(dist_lengths)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid dist table"))?;

    // Build double-literal cache for this block's table
    let double_cache = DoubleLitCache::build(&litlen_table);

    decode_huffman_ultra(
        bits,
        output,
        out_pos,
        &litlen_table,
        &dist_table,
        &double_cache,
    )
}

fn build_code_length_table(lengths: &[u8; 19]) -> Result<[u16; 128]> {
    let mut table = [0u16; 128];

    let mut count = [0u16; 8];
    for &len in lengths.iter() {
        if len > 0 && len <= 7 {
            count[len as usize] += 1;
        }
    }

    let mut code = 0u32;
    let mut first_code = [0u32; 8];
    for len in 1..=7 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }

    let mut next_code = first_code;
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len = len as usize;
        let codeword = next_code[len];
        next_code[len] += 1;

        let mut reversed = 0u32;
        let mut c = codeword;
        for _ in 0..len {
            reversed = (reversed << 1) | (c & 1);
            c >>= 1;
        }

        let stride = 1usize << len;
        let mut idx = reversed as usize;
        while idx < 128 {
            table[idx] = ((symbol as u16) << 8) | (len as u16);
            idx += stride;
        }
    }

    Ok(table)
}

// =============================================================================
// Public API
// =============================================================================

/// Ultra-fast decode with double-literal optimization
pub fn inflate_ultra(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let mut bits = UltraBits::new(input);
    let mut out_pos = 0;

    loop {
        if bits.available() < 3 {
            bits.refill();
        }

        let bfinal = (bits.peek() & 1) != 0;
        let btype = ((bits.peek() >> 1) & 3) as u8;
        bits.consume(3);

        // eprintln!("BLOCK: bfinal={} btype={}", bfinal, btype);
        match btype {
            0 => out_pos = decode_stored(&mut bits, output, out_pos)?,
            1 => out_pos = decode_fixed(&mut bits, output, out_pos)?,
            2 => out_pos = decode_dynamic(&mut bits, output, out_pos)?,
            3 => return Err(Error::new(ErrorKind::InvalidData, "Reserved block type")),
            _ => unreachable!(),
        }

        if bfinal {
            return Ok(out_pos);
        }
    }
}

// =============================================================================
// Tests and Benchmarks
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultra_simple() {
        let original = b"Hello, World! This is a test of ultra-fast decoding.";
        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::default());
            enc.write_all(original).unwrap();
            enc.finish().unwrap();
        }

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_ultra(&compressed, &mut output).expect("Decode failed");

        assert_eq!(size, original.len());
        assert_slices_eq!(&output[..size], original.as_slice());
    }

    #[test]
    fn test_ultra_rle() {
        let original = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let mut compressed = Vec::new();
        {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(&mut compressed, flate2::Compression::default());
            enc.write_all(original).unwrap();
            enc.finish().unwrap();
        }

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_ultra(&compressed, &mut output).expect("Decode failed");

        assert_eq!(size, original.len());
        assert_slices_eq!(&output[..size], original.as_slice());
    }

    #[test]
    fn bench_ultra_silesia() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping silesia benchmark - file not found");
                return;
            }
        };

        // Parse gzip header
        let mut pos = 10;
        let flg = gz[3];
        if (flg & 0x04) != 0 {
            let xlen = u16::from_le_bytes([gz[pos], gz[pos + 1]]) as usize;
            pos += 2 + xlen;
        }
        if (flg & 0x08) != 0 {
            while pos < gz.len() && gz[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if (flg & 0x10) != 0 {
            while pos < gz.len() && gz[pos] != 0 {
                pos += 1;
            }
            pos += 1;
        }
        if (flg & 0x02) != 0 {
            pos += 2;
        }

        let start = pos;
        let end = gz.len() - 8;
        let deflate = &gz[start..end];
        let isize = u32::from_le_bytes([
            gz[gz.len() - 4],
            gz[gz.len() - 3],
            gz[gz.len() - 2],
            gz[gz.len() - 1],
        ]) as usize;

        let mut output = vec![0u8; isize + 1000];
        let mut lib_output = vec![0u8; isize + 1000];

        // Verify with libdeflate first
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(deflate, &mut lib_output)
            .expect("libdeflate failed");

        // Test our decoder
        let our_size = inflate_ultra(deflate, &mut output).expect("Ultra decode failed");
        assert_eq!(our_size, lib_size, "Size mismatch");

        // Benchmark
        let iterations = 5;

        let start_t = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = inflate_ultra(deflate, &mut output);
        }
        let our_time = start_t.elapsed();

        let start_t = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = libdeflater::Decompressor::new().deflate_decompress(deflate, &mut lib_output);
        }
        let lib_time = start_t.elapsed();

        let our_throughput = (isize * iterations) as f64 / our_time.as_secs_f64() / 1e6;
        let lib_throughput = (isize * iterations) as f64 / lib_time.as_secs_f64() / 1e6;

        eprintln!("\n=== ULTRA DECODER (Double-Literal) ===");
        eprintln!("Data size: {} MB", isize / 1_000_000);
        eprintln!("Our throughput:       {:>8.1} MB/s", our_throughput);
        eprintln!("libdeflate throughput: {:>8.1} MB/s", lib_throughput);
        eprintln!("Ratio: {:.1}%", 100.0 * our_throughput / lib_throughput);
        eprintln!("======================================\n");
    }
}
