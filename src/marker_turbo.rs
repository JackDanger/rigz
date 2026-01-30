//! Turbo Marker-Based Decoder
//!
//! This reuses the fast consume_first_decode hot path but outputs to a u16 buffer
//! with markers for unresolved back-references.
//!
//! ## Key Insight
//!
//! The existing MarkerDecoder reads bits one at a time (~70 MB/s).
//! The turbo_inflate uses 64-bit word buffering (~1400 MB/s).
//!
//! This module bridges the gap by using the turbo bit reading but outputting markers.
//!
//! ## Marker Format
//!
//! - Values 0-255: Literal bytes
//! - Values 256+: Markers encoding unresolved back-references
//!   `marker = MARKER_BASE + (distance - decoded_bytes - 1)`
//!
//! ## Performance Target
//!
//! For 8-thread parallel to beat single-thread:
//! - Single-thread turbo: ~1400 MB/s
//! - Needed: marker_speed * 8 > 1400 → marker_speed > 175 MB/s
//! - Target: 500+ MB/s

#![allow(dead_code)]

use std::io::{Error, ErrorKind, Result};

use crate::consume_first_decode::Bits;
use crate::libdeflate_decode::get_fixed_tables;
use crate::libdeflate_entry::{DistTable, LitLenTable};
// Note: DistEntry and LitLenEntry are internal to libdeflate_entry
use crate::marker_decode::{MARKER_BASE, WINDOW_SIZE};

/// Decode deflate stream with markers for unresolved back-references
///
/// Returns (output_u16, end_bit_position)
pub fn inflate_with_markers(deflate_data: &[u8], max_output: usize) -> Result<(Vec<u16>, usize)> {
    inflate_with_markers_at(deflate_data, 0, max_output)
}

/// Decode deflate stream starting at a specific bit position
///
/// This is the core function for parallel decompression - each chunk starts
/// at a speculative boundary and decodes with markers for unresolved back-refs.
///
/// Returns (output_u16, end_bit_position)
pub fn inflate_with_markers_at(
    deflate_data: &[u8],
    start_bit: usize,
    max_output: usize,
) -> Result<(Vec<u16>, usize)> {
    // Initialize bit reader at the specified position
    let start_byte = start_bit / 8;
    let skip_bits = (start_bit % 8) as u32;

    if start_byte >= deflate_data.len() {
        return Ok((Vec::new(), start_bit));
    }

    let mut bits = Bits::new(&deflate_data[start_byte..]);

    // Skip the fractional bits if starting mid-byte
    if skip_bits > 0 {
        bits.consume(skip_bits);
    }

    let mut output: Vec<u16> = Vec::with_capacity(max_output.min(256 * 1024));

    loop {
        // Read BFINAL (1 bit) and BTYPE (2 bits)
        if bits.available() < 3 {
            bits.refill();
        }

        let header = bits.peek() & 0x7;
        let bfinal = (header & 1) != 0;
        let btype = ((header >> 1) & 3) as u8;
        bits.consume(3);

        match btype {
            0 => decode_stored_markers(&mut bits, &mut output)?,
            1 => decode_fixed_markers(&mut bits, &mut output)?,
            2 => decode_dynamic_markers(&mut bits, &mut output)?,
            3 => return Err(Error::new(ErrorKind::InvalidData, "Invalid block type 3")),
            _ => unreachable!(),
        }

        if bfinal {
            break;
        }

        if output.len() >= max_output {
            break;
        }
    }

    // Calculate end bit position (relative to original data, not the slice)
    let consumed_bits = bits.pos * 8 - (bits.bitsleft as usize);
    let end_bit = start_bit + consumed_bits;

    Ok((output, end_bit))
}

/// Decode stored block (uncompressed) - just copy bytes as u16
fn decode_stored_markers(bits: &mut Bits, output: &mut Vec<u16>) -> Result<()> {
    // Align to byte boundary
    let skip = bits.bitsleft % 8;
    bits.consume(skip);

    // Read LEN and NLEN
    bits.refill();
    let len = (bits.peek() & 0xFFFF) as u16;
    bits.consume(16);
    bits.refill();
    let nlen = (bits.peek() & 0xFFFF) as u16;
    bits.consume(16);

    if len != !nlen {
        return Err(Error::new(ErrorKind::InvalidData, "LEN/NLEN mismatch"));
    }

    // Copy literal bytes
    for _ in 0..len {
        bits.refill();
        let byte = (bits.peek() & 0xFF) as u16;
        bits.consume(8);
        output.push(byte);
    }

    Ok(())
}

/// Decode fixed Huffman block with markers
fn decode_fixed_markers(bits: &mut Bits, output: &mut Vec<u16>) -> Result<()> {
    let (litlen, dist) = get_fixed_tables();
    decode_huffman_markers(bits, output, litlen, dist)
}

/// Decode dynamic Huffman block with markers
///
/// For now, use a simplified approach that builds tables inline.
fn decode_dynamic_markers(bits: &mut Bits, output: &mut Vec<u16>) -> Result<()> {
    use crate::inflate_tables::CODE_LENGTH_ORDER;

    bits.refill();

    // Read header
    let hlit = (bits.peek() & 0x1F) as usize + 257;
    bits.consume(5);
    let hdist = (bits.peek() & 0x1F) as usize + 1;
    bits.consume(5);
    let hclen = (bits.peek() & 0xF) as usize + 4;
    bits.consume(4);

    if hlit > 286 || hdist > 32 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid Huffman header"));
    }

    // Read code length code lengths
    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        if bits.available() < 3 {
            bits.refill();
        }
        cl_lengths[CODE_LENGTH_ORDER[i] as usize] = (bits.peek() & 0x7) as u8;
        bits.consume(3);
    }

    // Build code length table (simple 7-bit lookup)
    let mut cl_table = [0u16; 128]; // symbol | (bits << 8)
    build_simple_huffman_table(&cl_lengths, &mut cl_table)?;

    // Read literal/length and distance code lengths
    let mut lengths = [0u8; 320];
    let total_codes = hlit + hdist;
    let mut i = 0;

    while i < total_codes {
        if bits.available() < 16 {
            bits.refill();
        }

        let peek = bits.peek() as u16;
        let entry = cl_table[(peek & 0x7F) as usize];
        let symbol = entry & 0xFF;
        let code_bits = (entry >> 8) as u32;

        if code_bits == 0 {
            // Fallback for codes not in simple table
            return Err(Error::new(ErrorKind::InvalidData, "Complex dynamic block"));
        }

        bits.consume(code_bits);

        match symbol {
            0..=15 => {
                lengths[i] = symbol as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "Invalid repeat at start",
                    ));
                }
                let repeat = 3 + (bits.peek() & 3) as usize;
                bits.consume(2);
                let prev = lengths[i - 1];
                for _ in 0..repeat.min(total_codes - i) {
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + (bits.peek() & 7) as usize;
                bits.consume(3);
                for _ in 0..repeat.min(total_codes - i) {
                    lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = 11 + (bits.peek() & 0x7F) as usize;
                bits.consume(7);
                for _ in 0..repeat.min(total_codes - i) {
                    lengths[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Invalid code length symbol",
                ));
            }
        }
    }

    // Build literal/length and distance tables
    let litlen = LitLenTable::build(&lengths[..hlit])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid lit/len table"))?;
    let dist = DistTable::build(&lengths[hlit..hlit + hdist])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid distance table"))?;

    decode_huffman_markers(bits, output, &litlen, &dist)
}

/// Build a simple Huffman lookup table for code lengths (max 7 bits)
fn build_simple_huffman_table(lengths: &[u8; 19], table: &mut [u16; 128]) -> Result<()> {
    // Count codes of each length
    let mut count = [0u16; 8];
    for &len in lengths.iter() {
        if len > 0 && len <= 7 {
            count[len as usize] += 1;
        }
    }

    // Compute first code for each length
    let mut next_code = [0u16; 8];
    let mut code = 0u16;
    for bits in 1..=7 {
        code = (code + count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols
    for (symbol, &len) in lengths.iter().enumerate() {
        if len > 0 && len <= 7 {
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse bits for LSB-first lookup
            let mut rev_code = 0u16;
            for i in 0..len {
                if code & (1 << i) != 0 {
                    rev_code |= 1 << (len - 1 - i);
                }
            }

            // Fill table entries for all extensions of this code
            let fill_count = 1 << (7 - len);
            for i in 0..fill_count {
                let idx = rev_code as usize | ((i as usize) << len);
                if idx < 128 {
                    table[idx] = symbol as u16 | ((len as u16) << 8);
                }
            }
        }
    }

    Ok(())
}

/// Main Huffman decode loop with marker output
///
/// This is the hot path - must be as fast as possible while outputting markers.
fn decode_huffman_markers(
    bits: &mut Bits,
    output: &mut Vec<u16>,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<()> {
    loop {
        bits.refill();
        let saved_bitbuf = bits.peek();
        let entry = litlen.lookup(saved_bitbuf);
        bits.consume_entry(entry.raw());

        // Check for literal (most common case)
        if entry.is_literal() {
            output.push(entry.literal_value() as u16);

            // Try to decode more literals in a tight loop
            for _ in 0..7 {
                if bits.available() < 15 {
                    bits.refill();
                }

                let peek = bits.peek();
                let e = litlen.lookup(peek);

                if !e.is_literal() {
                    break;
                }

                bits.consume_entry(e.raw());
                output.push(e.literal_value() as u16);
            }

            continue;
        }

        // Check for end-of-block
        if entry.is_end_of_block() {
            return Ok(());
        }

        // Length code (257-285)
        // entry contains: length_base in bits 24-16, extra_bits encoded in total_bits
        let length = entry.decode_length(saved_bitbuf);

        // Read distance with subtable handling
        bits.refill();
        let mut dist_saved = bits.peek();
        let mut dist_entry = dist.lookup(dist_saved);

        // Handle distance subtables
        if dist_entry.is_subtable_ptr() {
            bits.consume(DistTable::TABLE_BITS as u32);
            bits.refill();
            dist_saved = bits.peek(); // Update saved for subtable decode
            dist_entry = dist.lookup_subtable_direct(dist_entry, dist_saved);
            bits.consume(dist_entry.total_bits() as u32);
        } else {
            bits.consume_entry(dist_entry.raw());
        }

        let distance = dist_entry.decode_distance(dist_saved) as usize;

        if distance == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "Zero distance"));
        }

        // Check if back-reference is within our total decoded data
        // Back-references can span across blocks, so use total output length
        if distance <= output.len() {
            // We can resolve this back-reference directly
            let src_pos = output.len() - distance;
            for i in 0..length as usize {
                let byte = output[src_pos + i];
                output.push(byte);
            }
        } else {
            // Back-reference goes into previous chunk's window - use markers
            // marker = MARKER_BASE + (distance - output.len() - 1)
            let marker_offset = distance - output.len() - 1;

            if marker_offset >= WINDOW_SIZE {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Distance {} exceeds window size", distance),
                ));
            }

            for _ in 0..length as usize {
                // Each byte in the match may have a different marker offset
                let current_len = output.len();
                let actual_distance = distance; // Distance is constant

                // Check if this position can now be resolved
                if actual_distance <= current_len {
                    // We've decoded enough - can resolve this byte
                    let src_pos = current_len - actual_distance;
                    let byte = output[src_pos];
                    output.push(byte);
                } else {
                    // Still need marker
                    let offset_in_window = actual_distance - current_len - 1;

                    if offset_in_window >= WINDOW_SIZE {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!(
                                "Distance {} exceeds window at pos {}",
                                distance, current_len
                            ),
                        ));
                    }

                    output.push(MARKER_BASE + offset_in_window as u16);
                }
            }
        }
    }
}

/// Replace markers with actual bytes using the provided window
pub fn replace_markers(data: &mut [u16], window: &[u8]) {
    for value in data.iter_mut() {
        if *value >= MARKER_BASE {
            let window_index = (*value - MARKER_BASE) as usize;
            if window_index < window.len() {
                *value = window[window_index] as u16;
            }
        }
    }
}

/// Convert u16 buffer to u8 buffer (after markers are resolved)
pub fn markers_to_bytes(data: &[u16]) -> Vec<u8> {
    data.iter().map(|&v| v as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_inflate_with_markers_simple() {
        // Create simple compressed data
        let original: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Skip gzip header (10 bytes) and trailer (8 bytes)
        let deflate_data = &compressed[10..compressed.len() - 8];

        let (output_u16, _end_bit) = inflate_with_markers(deflate_data, 1_000_000).unwrap();

        // Should have no markers since there's no previous chunk
        let mut marker_count = 0;
        let mut first_marker_pos = None;
        for (i, &v) in output_u16.iter().enumerate() {
            if v >= 256 {
                marker_count += 1;
                if first_marker_pos.is_none() {
                    first_marker_pos = Some((i, v));
                }
            }
        }
        if let Some((pos, val)) = first_marker_pos {
            eprintln!(
                "Found {} markers, first at pos {} with value {} (MARKER_BASE={})",
                marker_count,
                pos,
                val,
                crate::marker_decode::MARKER_BASE
            );
        }
        assert_eq!(marker_count, 0, "Unexpected markers in simple test");

        // Convert to bytes and verify
        let output_bytes: Vec<u8> = output_u16.iter().map(|&v| v as u8).collect();
        assert_eq!(output_bytes, original);
    }

    #[test]
    fn test_inflate_with_markers_stored() {
        // Test stored block
        let deflate_data: &[u8] = &[
            0x01, // BFINAL=1, BTYPE=00 (stored)
            0x05, 0x00, // LEN = 5
            0xFA, 0xFF, // NLEN = ~5
            b'H', b'e', b'l', b'l', b'o',
        ];

        let (output, _) = inflate_with_markers(deflate_data, 100).unwrap();
        assert_eq!(output.len(), 5);
        assert_eq!(markers_to_bytes(&output), b"Hello");
    }

    #[test]
    fn bench_marker_turbo() {
        use std::time::Instant;

        // Create compressible data
        let original: Vec<u8> = (0..1_000_000).map(|i| ((i * 7) % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();
        let deflate_data = &compressed[10..compressed.len() - 8];

        // Warm up
        let _ = inflate_with_markers(deflate_data, original.len() + 1024);

        // Benchmark
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = inflate_with_markers(deflate_data, original.len() + 1024).unwrap();
        }

        let elapsed = start.elapsed();
        let total_bytes = original.len() * iterations;
        let mb_per_sec = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!("\n=== MARKER TURBO BENCHMARK ===");
        eprintln!("Data size: {:.1} MB", original.len() as f64 / 1_000_000.0);
        eprintln!("Throughput: {:.1} MB/s", mb_per_sec);
        eprintln!("Target: 175+ MB/s (for 8-thread parallel to beat single-thread)");
        eprintln!("==============================\n");

        // Should be at least faster than the old 70 MB/s
        assert!(
            mb_per_sec > 70.0,
            "Marker turbo should be faster than old MarkerDecoder"
        );
    }
}
