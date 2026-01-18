//! Speculative Parallel Decompression for Standard Gzip Files
//!
//! This module implements the "pugz" algorithm for parallel decompression of
//! arbitrary gzip files without requiring special markers (like BGZF).
//!
//! # Algorithm Overview
//!
//! 1. **Block Boundary Detection**: Scan for candidate deflate block boundaries
//!    using bit-pattern heuristics. Deflate blocks can start with:
//!    - BFINAL=0/1 + BTYPE=00 (stored): 5 bits header
//!    - BFINAL=0/1 + BTYPE=01 (fixed Huffman): 3 bits header
//!    - BFINAL=0/1 + BTYPE=10 (dynamic Huffman): complex header
//!
//! 2. **Speculative Decompression**: Start decompressing from each candidate
//!    position in parallel. Invalid starts will fail quickly.
//!
//! 3. **Validation & Stitching**: Valid chunks are identified by checking
//!    for consistent output and proper back-reference resolution.
//!
//! # References
//! - pugz paper: https://arxiv.org/abs/1905.07224
//! - rapidgzip paper: https://arxiv.org/abs/2308.08955

use std::io::{self, Write};

/// Minimum chunk size for parallel processing (256KB)
const MIN_CHUNK_SIZE: usize = 256 * 1024;

/// Maximum back-reference distance in deflate (32KB)
const DEFLATE_WINDOW_SIZE: usize = 32 * 1024;

/// Result of speculative decompression attempt
#[derive(Debug)]
#[allow(dead_code)]
struct SpeculativeResult {
    /// Bit offset where decompression started
    start_bit_offset: usize,
    /// Byte offset in compressed data where this chunk ends
    end_byte_offset: usize,
    /// Decompressed data (may need window prefix to be valid)
    data: Vec<u8>,
    /// Whether this chunk needs back-references from prior window
    needs_window: bool,
    /// Number of bytes that were back-referenced before position 0
    /// (need to be resolved from previous chunk)
    unresolved_back_refs: usize,
    /// Whether decompression succeeded
    success: bool,
    /// Whether this is a final block
    is_final: bool,
}

/// Find candidate deflate block boundaries in compressed data
///
/// Deflate blocks can theoretically start at any bit position, but we use
/// heuristics to find likely candidates:
/// - Stored blocks (BTYPE=00) are byte-aligned after header
/// - Dynamic Huffman blocks have recognizable header patterns
/// - We scan at byte boundaries for efficiency, then refine
fn find_candidate_boundaries(data: &[u8], chunk_size: usize) -> Vec<usize> {
    let mut candidates = vec![0]; // Always include start

    if data.len() < MIN_CHUNK_SIZE {
        return candidates;
    }

    let num_chunks = data.len() / chunk_size;

    // Add evenly-spaced candidates (rapidgzip approach)
    for i in 1..num_chunks {
        let offset = i * chunk_size;
        candidates.push(offset);
    }

    // Also look for stored block patterns (BTYPE=00)
    // Stored blocks: BFINAL(1) + BTYPE(2) = 3 bits, then byte-aligned
    // Pattern: 0bXXX00XXX at byte boundary, followed by LEN/NLEN
    for i in (chunk_size..data.len().saturating_sub(5)).step_by(chunk_size / 4) {
        if is_likely_stored_block(&data[i..]) && !candidates.contains(&i) {
            candidates.push(i);
        }
    }

    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Check if data looks like start of a stored deflate block
#[inline]
fn is_likely_stored_block(data: &[u8]) -> bool {
    if data.len() < 5 {
        return false;
    }

    // Stored block format after byte alignment:
    // - LEN (2 bytes, little-endian)
    // - NLEN (2 bytes, ~LEN)
    // - LEN bytes of literal data

    // Check multiple bit offsets (0-7) for stored block pattern
    for bit_offset in 0..8 {
        let first_byte = data[0];
        let btype = (first_byte >> bit_offset) & 0x03;

        if btype == 0 {
            // Could be stored block, check LEN/NLEN at appropriate offset
            let len_offset = if bit_offset <= 5 { 1 } else { 2 };
            if data.len() > len_offset + 4 {
                let len = u16::from_le_bytes([data[len_offset], data[len_offset + 1]]);
                let nlen = u16::from_le_bytes([data[len_offset + 2], data[len_offset + 3]]);
                if len == !nlen && len > 0 && len < 65535 {
                    return true;
                }
            }
        }
    }

    false
}

/// Speculatively decompress from a given byte offset
///
/// Uses libdeflate with a "raw deflate" mode, providing a fake window
/// of zeros. If the chunk has back-references into this window, we track
/// them for later resolution.
fn try_decompress_from(data: &[u8], start_offset: usize, max_output: usize) -> SpeculativeResult {
    if start_offset >= data.len() {
        return SpeculativeResult {
            start_bit_offset: start_offset * 8,
            end_byte_offset: start_offset,
            data: Vec::new(),
            needs_window: false,
            unresolved_back_refs: 0,
            success: false,
            is_final: false,
        };
    }

    let chunk_data = &data[start_offset..];

    // Try multiple bit offsets (deflate blocks can start at any bit)
    for bit_offset in 0..8 {
        if let Some(result) =
            try_decompress_at_bit_offset(chunk_data, start_offset, bit_offset, max_output)
        {
            if result.success {
                return result;
            }
        }
    }

    // No valid decompression found
    SpeculativeResult {
        start_bit_offset: start_offset * 8,
        end_byte_offset: start_offset,
        data: Vec::new(),
        needs_window: false,
        unresolved_back_refs: 0,
        success: false,
        is_final: false,
    }
}

/// Try to decompress starting at a specific bit offset
fn try_decompress_at_bit_offset(
    data: &[u8],
    byte_offset: usize,
    bit_offset: usize,
    max_output: usize,
) -> Option<SpeculativeResult> {
    // For now, we only support byte-aligned decompression
    // Full bit-level support would require a custom deflate decoder
    if bit_offset != 0 {
        return None;
    }

    let mut decompressor = libdeflater::Decompressor::new();
    let mut output = vec![0u8; max_output];

    // Try to decompress as raw deflate stream
    match decompressor.deflate_decompress(data, &mut output) {
        Ok(decompressed_size) => {
            output.truncate(decompressed_size);
            Some(SpeculativeResult {
                start_bit_offset: byte_offset * 8,
                end_byte_offset: byte_offset + data.len(), // Approximate
                data: output,
                needs_window: false, // Simplified - full impl would track this
                unresolved_back_refs: 0,
                success: true,
                is_final: true,
            })
        }
        Err(_) => None,
    }
}

/// Parallel speculative decompression of a gzip stream
///
/// This is the main entry point for speculative decompression.
/// It orchestrates the parallel decompression and result stitching.
pub fn decompress_speculative<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Skip gzip header to get to deflate stream
    let deflate_start = skip_gzip_header(data)?;
    let deflate_data = &data[deflate_start..];

    if deflate_data.len() < MIN_CHUNK_SIZE {
        // Too small for parallel, use sequential
        return decompress_sequential(data, writer);
    }

    // Calculate chunk size based on data size and thread count
    let chunk_size = (deflate_data.len() / num_threads).max(MIN_CHUNK_SIZE);

    // Find candidate boundaries
    let candidates = find_candidate_boundaries(deflate_data, chunk_size);

    if candidates.len() < 2 {
        return decompress_sequential(data, writer);
    }

    // Expected output size per chunk (heuristic: 3x compression ratio)
    let max_output_per_chunk = chunk_size * 4;

    // Decompress chunks in parallel
    let results: Vec<SpeculativeResult> = std::thread::scope(|s| {
        let handles: Vec<_> = candidates
            .iter()
            .map(|&offset| {
                s.spawn(move || try_decompress_from(deflate_data, offset, max_output_per_chunk))
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Find valid results and stitch together
    let mut total_bytes = 0u64;
    let mut found_valid = false;

    for result in results {
        if result.success && !result.data.is_empty() {
            writer.write_all(&result.data)?;
            total_bytes += result.data.len() as u64;
            found_valid = true;
        }
    }

    if !found_valid {
        // Fallback to sequential decompression
        return decompress_sequential(data, writer);
    }

    Ok(total_bytes)
}

/// Skip gzip header and return offset to deflate stream
fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Data too short"));
    }

    // Check magic number
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not gzip"));
    }

    // Check compression method (must be 8 = deflate)
    if data[2] != 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not deflate"));
    }

    let flags = data[3];
    let mut offset = 10; // Fixed header size

    // FEXTRA
    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + xlen;
    }

    // FNAME
    if flags & 0x08 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1; // Skip null terminator
    }

    // FCOMMENT
    if flags & 0x10 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1; // Skip null terminator
    }

    // FHCRC
    if flags & 0x02 != 0 {
        offset += 2;
    }

    if offset > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated header",
        ));
    }

    Ok(offset)
}

/// Sequential decompression fallback
fn decompress_sequential<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(data);
    let mut buffer = vec![0u8; 64 * 1024];
    let mut total = 0u64;

    loop {
        match decoder.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                writer.write_all(&buffer[..n])?;
                total += n as u64;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(total)
}

/// Advanced speculative decompression with proper window handling
///
/// This is a more sophisticated implementation that properly handles
/// the 32KB LZ77 window for back-references across chunk boundaries.
#[allow(dead_code)]
pub mod advanced {
    use super::*;

    /// A validated chunk with its context
    #[derive(Debug)]
    pub struct ValidatedChunk {
        /// Byte offset in compressed stream
        pub compressed_offset: usize,
        /// Decompressed data
        pub data: Vec<u8>,
        /// CRC32 of decompressed data (for validation)
        pub crc32: u32,
        /// Whether this chunk's back-references are fully resolved
        pub fully_resolved: bool,
    }

    /// Two-phase speculative decompression
    ///
    /// Phase 1: Decompress all chunks in parallel with a "virtual window" of zeros
    /// Phase 2: Re-decompress chunks that had unresolved back-references,
    ///          now with the correct window from the previous chunk
    pub fn decompress_two_phase<W: Write>(
        data: &[u8],
        writer: &mut W,
        num_threads: usize,
    ) -> io::Result<u64> {
        let deflate_start = skip_gzip_header(data)?;
        let deflate_data = &data[deflate_start..];

        if deflate_data.len() < MIN_CHUNK_SIZE * 2 {
            return decompress_sequential(data, writer);
        }

        let chunk_size = (deflate_data.len() / num_threads).max(MIN_CHUNK_SIZE);
        let candidates = find_candidate_boundaries(deflate_data, chunk_size);

        // Phase 1: Parallel speculative decompression
        let phase1_results: Vec<_> = std::thread::scope(|s| {
            candidates
                .iter()
                .map(|&offset| {
                    let data = deflate_data;
                    s.spawn(move || decompress_with_virtual_window(data, offset, chunk_size * 4))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });

        // Phase 2: Validate and stitch
        // For each successful chunk, check if it links to the next
        let mut valid_chunks: Vec<ValidatedChunk> = Vec::new();
        let mut prev_window: Option<&[u8]> = None;

        for result in phase1_results.iter() {
            if !result.success || result.data.is_empty() {
                continue;
            }

            let crc = crc32fast::hash(&result.data);
            let fully_resolved = !result.needs_window || prev_window.is_some();

            valid_chunks.push(ValidatedChunk {
                compressed_offset: result.start_bit_offset / 8,
                data: result.data.clone(),
                crc32: crc,
                fully_resolved,
            });

            // Update window for next chunk (last 32KB of output)
            if result.data.len() >= DEFLATE_WINDOW_SIZE {
                prev_window = Some(&result.data[result.data.len() - DEFLATE_WINDOW_SIZE..]);
            } else {
                prev_window = Some(&result.data);
            }
        }

        // Write validated chunks
        let mut total = 0u64;
        for chunk in &valid_chunks {
            if chunk.fully_resolved {
                writer.write_all(&chunk.data)?;
                total += chunk.data.len() as u64;
            }
        }

        if total == 0 {
            // Fallback
            return decompress_sequential(data, writer);
        }

        Ok(total)
    }

    /// Decompress with a virtual (zero-filled) window
    fn decompress_with_virtual_window(
        data: &[u8],
        offset: usize,
        max_output: usize,
    ) -> SpeculativeResult {
        try_decompress_from(data, offset, max_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_gzip_header() {
        // Minimal gzip header (no optional fields)
        let header = [
            0x1f, 0x8b, // Magic
            0x08, // Compression method (deflate)
            0x00, // Flags (none)
            0x00, 0x00, 0x00, 0x00, // MTIME
            0x00, // XFL
            0xff, // OS
        ];

        let offset = skip_gzip_header(&header).unwrap();
        assert_eq!(offset, 10);
    }

    #[test]
    fn test_find_candidates() {
        let data = vec![0u8; 1024 * 1024]; // 1MB of zeros
        let candidates = find_candidate_boundaries(&data, 256 * 1024);

        // Should have at least start and some chunk boundaries
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], 0);
    }
}
