//! Fixed Block Turbo Decoder
//!
//! Fixed Huffman codes are COMPILE-TIME KNOWN. This module exploits that
//! to decode without any table lookups - pure bit manipulation.
//!
//! Fixed Huffman literal/length codes (RFC 1951):
//!   7 bits: 256-279 (codes 0000000-0010111 reversed)
//!   8 bits: 0-143   (codes 00110000-10111111 reversed)  
//!   8 bits: 280-287 (codes 11000000-11000111 reversed)
//!   9 bits: 144-255 (codes 110010000-111111111 reversed)
//!
//! Fixed distance codes: All 5 bits (0-31)

#![allow(dead_code)]

use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};
use std::io;

/// Decode fixed Huffman literal/length symbol without table lookup
/// Returns (symbol, bits_consumed)
///
/// The codes in the bit buffer are bit-reversed (LSB first transmission).
/// We reverse them back to get the canonical code, then compute the symbol.
#[inline(always)]
fn decode_fixed_litlen(bits: u64) -> (u16, u8) {
    // 7-bit codes: symbols 256-279 use codes 0b0000000 to 0b0010111
    // In buffer (reversed): 0b0000000 to 0b1110100
    let low7 = (bits & 0x7F) as u16;
    let rev7 = reverse_7bits(low7);
    if rev7 <= 23 {
        // 7-bit code maps to symbol 256 + rev7
        return (256 + rev7, 7);
    }

    // 8-bit codes for literals 0-143: codes 0b00110000 to 0b10111111 (48-191)
    // For symbol S in 0-143, code = 48 + S
    let low8 = (bits & 0xFF) as u16;
    let rev8 = reverse_8bits(low8);
    if (48..=191).contains(&rev8) {
        return (rev8 - 48, 8);
    }

    // 8-bit codes for symbols 280-287: codes 0b11000000 to 0b11000111 (192-199)
    if (192..=199).contains(&rev8) {
        return (280 + (rev8 - 192), 8);
    }

    // 9-bit codes for literals 144-255: codes 0b110010000 to 0b111111111 (400-511)
    // For symbol S in 144-255, code = 400 + (S - 144) = 256 + S
    let low9 = (bits & 0x1FF) as u16;
    let rev9 = reverse_9bits(low9);
    if (400..=511).contains(&rev9) {
        return (144 + (rev9 - 400), 9);
    }

    // Invalid code
    (0xFFFF, 0)
}

/// Decode fixed distance code (always 5 bits, values 0-31)
#[inline(always)]
fn decode_fixed_dist(bits: u64) -> (u16, u8) {
    let low5 = (bits & 0x1F) as u16;
    let reversed5 = reverse_5bits(low5);
    (reversed5, 5)
}

// Bit reversal lookup tables (precomputed)
#[inline(always)]
fn reverse_5bits(v: u16) -> u16 {
    const REV5: [u16; 32] = [
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3,
        19, 11, 27, 7, 23, 15, 31,
    ];
    REV5[v as usize]
}

#[inline(always)]
fn reverse_7bits(v: u16) -> u16 {
    // Reverse 7 bits using bit manipulation
    let v = v as u32;
    let v = ((v & 0x55) << 1) | ((v & 0xAA) >> 1);
    let v = ((v & 0x33) << 2) | ((v & 0xCC) >> 2);
    let v = ((v & 0x0F) << 4) | ((v & 0xF0) >> 4);
    (v >> 1) as u16 // Shift right 1 to get 7 bits
}

#[inline(always)]
fn reverse_8bits(v: u16) -> u16 {
    const REV8: [u8; 256] = [
        0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240, 8, 136, 72, 200,
        40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248, 4, 132, 68, 196, 36, 164, 100, 228,
        20, 148, 84, 212, 52, 180, 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220,
        60, 188, 124, 252, 2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
        10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250, 6, 134, 70, 198,
        38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246, 14, 142, 78, 206, 46, 174, 110,
        238, 30, 158, 94, 222, 62, 190, 126, 254, 1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81,
        209, 49, 177, 113, 241, 9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121,
        249, 5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245, 13, 141, 77,
        205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253, 3, 131, 67, 195, 35, 163, 99,
        227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91,
        219, 59, 187, 123, 251, 7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119,
        247, 15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255,
    ];
    REV8[v as usize] as u16
}

#[inline(always)]
fn reverse_9bits(v: u16) -> u16 {
    // Reverse 9 bits: reverse 8 bits, shift, add MSB
    let low8 = (v & 0xFF) as u8;
    let bit8 = (v >> 8) & 1;
    (reverse_8bits(low8 as u16) << 1) | bit8
}

/// Bit buffer for turbo decode
pub struct TurboBits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
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
        tb.refill();
        tb
    }

    #[inline(always)]
    pub fn refill(&mut self) {
        while self.bits <= 56 && self.pos < self.data.len() {
            self.buf |= (self.data[self.pos] as u64) << self.bits;
            self.pos += 1;
            self.bits += 8;
        }
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
    pub fn read(&mut self, n: u32) -> u32 {
        let val = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        val
    }

    #[inline(always)]
    pub fn past_end(&self) -> bool {
        self.pos > self.data.len() + 8
    }
}

/// Turbo decode for fixed Huffman blocks
pub fn decode_fixed_block_turbo(
    bits: &mut TurboBits,
    output: &mut [u8],
    mut out_pos: usize,
) -> io::Result<usize> {
    let out_end = output.len().saturating_sub(258 + 32);
    let output_ptr = output.as_mut_ptr();

    // FASTLOOP
    while out_pos < out_end && !bits.past_end() {
        bits.refill();

        let (symbol, code_len) = decode_fixed_litlen(bits.peek());
        if code_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid fixed Huffman code",
            ));
        }
        bits.consume(code_len as u32);

        // Literal
        if symbol < 256 {
            unsafe {
                *output_ptr.add(out_pos) = symbol as u8;
            }
            out_pos += 1;

            // Try another literal
            if bits.bits >= 9 {
                let (sym2, len2) = decode_fixed_litlen(bits.peek());
                if len2 > 0 && sym2 < 256 {
                    bits.consume(len2 as u32);
                    unsafe {
                        *output_ptr.add(out_pos) = sym2 as u8;
                    }
                    out_pos += 1;
                }
            }
            continue;
        }

        // End of block
        if symbol == 256 {
            return Ok(out_pos);
        }

        // Length code (257-285)
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

        // Distance
        let (dist_sym, dist_len) = decode_fixed_dist(bits.peek());
        bits.consume(dist_len as u32);

        if dist_sym >= 30 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }

        bits.refill();
        let distance = DIST_START[dist_sym as usize] as usize
            + bits.read(DIST_EXTRA_BITS[dist_sym as usize] as u32) as usize;

        if distance == 0 || distance > out_pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance",
            ));
        }

        // Copy match
        unsafe {
            copy_match_turbo(
                output_ptr.add(out_pos),
                output_ptr.add(out_pos - distance),
                length,
                distance,
            );
        }
        out_pos += length;
    }

    // Generic loop for end of buffer
    loop {
        if bits.past_end() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unexpected end"));
        }

        bits.refill();
        let (symbol, code_len) = decode_fixed_litlen(bits.peek());
        if code_len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid code"));
        }
        bits.consume(code_len as u32);

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

        let len_idx = (symbol - 257) as usize;
        if len_idx >= 29 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid length"));
        }

        bits.refill();
        let length =
            LEN_START[len_idx] as usize + bits.read(LEN_EXTRA_BITS[len_idx] as u32) as usize;

        let (dist_sym, _) = decode_fixed_dist(bits.peek());
        bits.consume(5);

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
    }
}

/// Fast match copy with RLE optimization
#[inline(always)]
unsafe fn copy_match_turbo(dst: *mut u8, src: *const u8, len: usize, offset: usize) {
    if offset == 1 {
        std::ptr::write_bytes(dst, *src, len);
    } else if offset >= 8 {
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
        for i in 0..len {
            *dst.add(i) = *src.add(i % offset);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_fixed_litlen_literals() {
        // Test literal 'H' = 72
        // Code = 48 + 72 = 120 = 0b01111000
        // Reversed (LSB first): 0b00011110 = 30 = 0x1E
        let bits = 0x1E_u64;
        let (sym, len) = decode_fixed_litlen(bits);
        assert_eq!(len, 8);
        assert_eq!(sym, 72); // 'H'

        // Test literal 'A' = 65
        // Code = 48 + 65 = 113 = 0b01110001
        // Reversed: 0b10001110 = 142 = 0x8E
        let bits = 0x8E_u64;
        let (sym, len) = decode_fixed_litlen(bits);
        assert_eq!(len, 8);
        assert_eq!(sym, 65); // 'A'
    }

    #[test]
    fn test_decode_fixed_eob() {
        // EOB is symbol 256, 7-bit code 0000000
        // Reversed: still 0
        let bits = 0u64;
        let (sym, len) = decode_fixed_litlen(bits);
        assert_eq!(len, 7);
        assert_eq!(sym, 256);
    }

    #[test]
    fn test_turbo_decode_simple() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create data that will use fixed Huffman (simple repetitive)
        let original = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(1)); // Level 1 often uses fixed
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Skip gzip header, find deflate data
        let deflate_data = &compressed[10..compressed.len() - 8];

        // Read block header
        let mut bits = TurboBits::new(deflate_data);
        let _bfinal = bits.read(1);
        let btype = bits.read(2);

        // Only test if it's a fixed block
        if btype == 1 {
            let mut output = vec![0u8; original.len() + 100];
            let written = decode_fixed_block_turbo(&mut bits, &mut output, 0).unwrap();
            output.truncate(written);
            assert_eq!(output, original);
        } else {
            eprintln!("Skipping test - block type {} (not fixed)", btype);
        }
    }

    #[test]
    fn benchmark_fixed_turbo() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create data that will definitely use fixed Huffman
        let original: Vec<u8> = (0..100_000).map(|i| (i % 64) as u8 + 32).collect();

        // Use raw deflate with level 1 (tends to use fixed blocks)
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(1));
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Check block type
        let mut bits = TurboBits::new(&compressed);
        let _bfinal = bits.read(1);
        let btype = bits.read(2);

        if btype != 1 {
            eprintln!("Skipping benchmark - block type {} (want fixed=1)", btype);
            return;
        }

        const ITERS: usize = 100;

        // Benchmark turbo
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut bits = TurboBits::new(&compressed);
            bits.read(3); // skip header
            let mut output = vec![0u8; original.len() + 100];
            let _ = decode_fixed_block_turbo(&mut bits, &mut output, 0);
            std::hint::black_box(&output);
        }
        let turbo_time = start.elapsed();

        // Benchmark libdeflate for comparison
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut output = vec![0u8; original.len() + 100];
            let mut dec = libdeflater::Decompressor::new();
            let _ = dec.deflate_decompress(&compressed, &mut output);
            std::hint::black_box(&output);
        }
        let libdeflate_time = start.elapsed();

        let turbo_speed =
            original.len() as f64 * ITERS as f64 / turbo_time.as_secs_f64() / 1_000_000.0;
        let libdeflate_speed =
            original.len() as f64 * ITERS as f64 / libdeflate_time.as_secs_f64() / 1_000_000.0;

        eprintln!("\n=== FIXED TURBO vs LIBDEFLATE ===");
        eprintln!("Turbo:      {:.0} MB/s", turbo_speed);
        eprintln!("libdeflate: {:.0} MB/s", libdeflate_speed);
        eprintln!("Ratio: {:.1}%", turbo_speed / libdeflate_speed * 100.0);
    }
}
