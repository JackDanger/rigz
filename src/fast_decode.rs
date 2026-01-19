//! Fast Decode - Combined Lit/Len/Dist Tables
//!
//! This implements libdeflate-style packed table entries where the
//! symbol type is encoded in the entry itself, allowing faster decode.
//!
//! Entry format (32 bits):
//! - Bits 0-15: Symbol or packed data
//! - Bits 16-19: Code length
//! - Bits 20-23: Entry type (0=lit, 1=len, 2=EOB, 3=L2)
//! - Bits 24-31: Extra bits value (for len/dist)

#![allow(dead_code)]

use std::io;

use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};
use crate::two_level_table::FastBits;

// Entry type flags
const TYPE_LITERAL: u32 = 0;
const TYPE_LENGTH: u32 = 1 << 20;
const TYPE_EOB: u32 = 2 << 20;
const TYPE_L2: u32 = 3 << 20;
const TYPE_MASK: u32 = 0xF << 20;

// Table size (11 bits for better coverage)
const TABLE_BITS: usize = 11;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: usize = TABLE_SIZE - 1;

/// Fast decode table with type-encoded entries
pub struct FastDecodeTable {
    entries: Box<[u32; TABLE_SIZE]>,
}

impl FastDecodeTable {
    /// Build from code lengths
    pub fn build(lens: &[u8]) -> io::Result<Self> {
        let mut entries = Box::new([0u32; TABLE_SIZE]);

        // Count codes of each length
        let max_len = lens.iter().copied().max().unwrap_or(0) as usize;
        if max_len > 15 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Code length > 15",
            ));
        }

        let mut bl_count = [0u32; 16];
        for &len in lens {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Compute first code for each length
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=max_len {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Fill table
        for (symbol, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }

            let len = len as usize;
            let code = next_code[len];
            next_code[len] += 1;

            // Reverse bits for LSB-first
            let rev = reverse_bits(code, len as u32);

            // Determine entry type
            let entry_type = if symbol < 256 {
                TYPE_LITERAL
            } else if symbol == 256 {
                TYPE_EOB
            } else {
                TYPE_LENGTH
            };

            // Pack entry: symbol(16) + len(4) + type(4)
            let entry = (symbol as u32) | ((len as u32) << 16) | entry_type;

            // Fill table entries
            if len <= TABLE_BITS {
                let fill_count = 1 << (TABLE_BITS - len);
                for i in 0..fill_count {
                    let idx = (rev as usize) | (i << len);
                    entries[idx] = entry;
                }
            }
            // Skip L2 for now - codes > TABLE_BITS are rare
        }

        Ok(Self { entries })
    }

    /// Decode - returns (entry, code_len)
    #[inline(always)]
    pub fn decode(&self, bits: u64) -> u32 {
        let idx = (bits as usize) & TABLE_MASK;
        self.entries[idx]
    }
}

/// Decode a Huffman block using fast combined table
///
/// Safety: Loop terminates via END_OF_BLOCK or error from invalid code.
/// The FastBits.consume() uses saturating_sub() to prevent underflow that caused OOM.
#[inline(never)]
pub fn decode_block_fast(
    bits: &mut FastBits,
    output: &mut Vec<u8>,
    lit_len_table: &FastDecodeTable,
    dist_table: &FastDecodeTable,
) -> io::Result<()> {
    output.reserve(128 * 1024);

    loop {
        bits.ensure(32);

        let entry = lit_len_table.decode(bits.buffer());
        let code_len = (entry >> 16) & 0xF;

        if code_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code",
            ));
        }

        bits.consume(code_len);

        match entry & TYPE_MASK {
            TYPE_LITERAL => {
                output.push((entry & 0xFF) as u8);
            }
            TYPE_EOB => {
                break;
            }
            TYPE_LENGTH => {
                // Length code
                let len_idx = ((entry & 0xFFFF) - 257) as usize;
                if len_idx >= 29 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid length code",
                    ));
                }

                let base_len = LEN_START[len_idx] as usize;
                let len_extra = LEN_EXTRA_BITS[len_idx] as u32;
                let length = base_len + bits.read(len_extra) as usize;

                bits.ensure(16);

                // Decode distance
                let dist_entry = dist_table.decode(bits.buffer());
                let dist_code_len = (dist_entry >> 16) & 0xF;
                if dist_code_len == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance code",
                    ));
                }
                bits.consume(dist_code_len);

                let dist_sym = (dist_entry & 0xFFFF) as usize;
                if dist_sym >= 30 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance symbol",
                    ));
                }

                let base_dist = DIST_START[dist_sym] as usize;
                let dist_extra = DIST_EXTRA_BITS[dist_sym] as u32;
                let distance = base_dist + bits.read(dist_extra) as usize;

                if distance > output.len() || distance == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                crate::simd_copy::lz77_copy_fast(output, distance, length);
            }
            _ => {
                // L2 lookup - not implemented yet
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "L2 lookup not implemented",
                ));
            }
        }
    }

    Ok(())
}

fn reverse_bits(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_decode_build() {
        // Fixed Huffman literal/length table
        let mut lens = [0u8; 288];
        for len in lens.iter_mut().take(144) {
            *len = 8;
        }
        for len in lens.iter_mut().take(256).skip(144) {
            *len = 9;
        }
        for len in lens.iter_mut().take(280).skip(256) {
            *len = 7;
        }
        for len in lens.iter_mut().take(288).skip(280) {
            *len = 8;
        }

        let table = FastDecodeTable::build(&lens).unwrap();

        // Check a few entries
        // Symbol 0 should have 8-bit code
        let mut found_literal = false;
        for i in 0..TABLE_SIZE {
            let entry = table.entries[i];
            if entry & TYPE_MASK == TYPE_LITERAL {
                found_literal = true;
                break;
            }
        }
        assert!(found_literal, "Should have literal entries");
    }
}

#[test]
fn benchmark_fast_decode_table() {
    // Fixed Huffman literal/length table
    let mut lens = [0u8; 288];
    for len in lens.iter_mut().take(144) {
        *len = 8;
    }
    for len in lens.iter_mut().take(256).skip(144) {
        *len = 9;
    }
    for len in lens.iter_mut().take(280).skip(256) {
        *len = 7;
    }
    for len in lens.iter_mut().take(288).skip(280) {
        *len = 8;
    }

    let table = FastDecodeTable::build(&lens).unwrap();

    // Benchmark lookup speed
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };
    let header_size = crate::marker_decode::skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    let start = std::time::Instant::now();
    let mut bits = FastBits::new(deflate_data);
    let mut sum = 0u64;
    for _ in 0..10_000_000 {
        bits.ensure(15);
        let entry = table.decode(bits.buffer());
        sum += entry as u64;
        let len = (entry >> 16) & 0xF;
        bits.consume(len);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "FastDecodeTable 10M ops: {:?} ({:.1} ns/op) sum={}",
        elapsed,
        elapsed.as_nanos() as f64 / 10_000_000.0,
        sum
    );
}
