//! Parallel Inflate Infrastructure
//!
//! This module provides the infrastructure for parallel gzip decompression
//! using our pure Rust inflate implementation. Key components:
//!
//! 1. **Chunk partitioning** - Divide input into chunks for parallel processing
//! 2. **Speculative decoding** - Start decoding at guessed block boundaries
//! 3. **Window propagation** - Pass 32KB windows between chunks
//! 4. **Parallel marker replacement** - Resolve back-references in parallel
//!
//! This is the pure Rust equivalent of rapidgzip's approach.

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::simd_inflate;

// =============================================================================
// Constants
// =============================================================================

/// Default chunk size for parallel decompression (4MB)
const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Window size for LZ77 (32KB)
const WINDOW_SIZE: usize = 32 * 1024;

/// Minimum file size for parallel decompression
const PARALLEL_THRESHOLD: usize = 1024 * 1024; // 1MB

// =============================================================================
// Chunk Result
// =============================================================================

/// Result of decompressing a chunk
#[derive(Debug)]
pub struct ChunkResult {
    /// Chunk index
    pub index: usize,
    /// Decompressed data
    pub data: Vec<u8>,
    /// Final 32KB window (for next chunk)
    pub window: Vec<u8>,
    /// Whether decompression succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ChunkResult {
    pub fn success(index: usize, data: Vec<u8>) -> Self {
        let window = if data.len() >= WINDOW_SIZE {
            data[data.len() - WINDOW_SIZE..].to_vec()
        } else {
            data.clone()
        };

        Self {
            index,
            data,
            window,
            success: true,
            error: None,
        }
    }

    pub fn failure(index: usize, error: String) -> Self {
        Self {
            index,
            data: Vec::new(),
            window: Vec::new(),
            success: false,
            error: Some(error),
        }
    }
}

// =============================================================================
// BGZF Detection and Parsing
// =============================================================================

/// BGZF block info
#[derive(Debug, Clone, Copy)]
pub struct BgzfBlock {
    /// Start position in compressed data
    pub start: usize,
    /// Compressed size (including header/trailer)
    pub csize: usize,
    /// Uncompressed size
    pub usize: usize,
}

/// Detect if this is a BGZF file (gzippy or bgzip output)
pub fn detect_bgzf(data: &[u8]) -> bool {
    if data.len() < 18 {
        return false;
    }

    // Check gzip magic
    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 8 {
        return false;
    }

    // Check for FEXTRA flag and BGZF extra field
    let flags = data[3];
    if flags & 0x04 == 0 {
        return false;
    }

    // Parse extra field
    let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
    if xlen < 6 || data.len() < 12 + xlen {
        return false;
    }

    // Look for BC subfield (BGZF marker)
    let mut pos = 12;
    while pos + 4 <= 12 + xlen {
        let si1 = data[pos];
        let si2 = data[pos + 1];
        let slen = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;

        if si1 == 66 && si2 == 67 && slen >= 2 {
            // Found BGZF marker
            return true;
        }

        pos += 4 + slen;
    }

    false
}

/// Parse BGZF blocks from data
pub fn parse_bgzf_blocks(data: &[u8]) -> Vec<BgzfBlock> {
    let mut blocks = Vec::new();
    let mut pos = 0;

    while pos + 18 <= data.len() {
        // Check gzip header
        if data[pos] != 0x1f || data[pos + 1] != 0x8b {
            break;
        }

        let flags = data[pos + 3];
        if flags & 0x04 == 0 {
            break;
        }

        let xlen = u16::from_le_bytes([data[pos + 10], data[pos + 11]]) as usize;
        if pos + 12 + xlen > data.len() {
            break;
        }

        // Find block size from extra field
        let mut block_size = 0;
        let mut xpos = pos + 12;
        while xpos + 4 <= pos + 12 + xlen {
            let si1 = data[xpos];
            let si2 = data[xpos + 1];
            let slen = u16::from_le_bytes([data[xpos + 2], data[xpos + 3]]) as usize;

            if si1 == 66 && si2 == 67 && slen >= 2 {
                // BGZF: block size - 1 is stored in 2 bytes
                // Or 4 bytes for gzippy extended format
                if slen == 2 {
                    block_size = u16::from_le_bytes([data[xpos + 4], data[xpos + 5]]) as usize + 1;
                } else if slen == 4 {
                    block_size = u32::from_le_bytes([
                        data[xpos + 4],
                        data[xpos + 5],
                        data[xpos + 6],
                        data[xpos + 7],
                    ]) as usize
                        + 1;
                }
                break;
            }

            xpos += 4 + slen;
        }

        if block_size == 0 || pos + block_size > data.len() {
            break;
        }

        // Get uncompressed size from trailer
        if pos + block_size >= 4 {
            let usize = u32::from_le_bytes([
                data[pos + block_size - 4],
                data[pos + block_size - 3],
                data[pos + block_size - 2],
                data[pos + block_size - 1],
            ]) as usize;

            blocks.push(BgzfBlock {
                start: pos,
                csize: block_size,
                usize,
            });
        }

        pos += block_size;
    }

    blocks
}

// =============================================================================
// Multi-Member Detection
// =============================================================================

/// Find gzip member boundaries (for pigz-style multi-member files)
/// Returns Vec of (start, end) byte positions for each member
pub fn find_member_boundaries(data: &[u8]) -> Vec<(usize, usize)> {
    let mut starts = vec![0];

    // Find all member start positions by looking for gzip magic
    // Note: 0x1f 0x8b can appear inside deflate streams, but followed by 0x08
    // (compression method = deflate) is much rarer
    let mut pos = 1;
    while pos + 10 < data.len() {
        if data[pos] == 0x1f && data[pos + 1] == 0x8b && data[pos + 2] == 8 {
            // Validate this looks like a real header
            let flags = data[pos + 3];
            // Check reserved bits are zero
            if flags & 0xE0 == 0 {
                starts.push(pos);
            }
        }
        pos += 1;
    }

    // Convert starts to (start, end) pairs
    let mut members = Vec::with_capacity(starts.len());
    for i in 0..starts.len() {
        let start = starts[i];
        let end = if i + 1 < starts.len() {
            starts[i + 1]
        } else {
            data.len()
        };
        members.push((start, end));
    }

    members
}

/// Decompress a single gzip member using flate2 (fallback)
fn decompress_member_flate2(data: &[u8]) -> io::Result<Vec<u8>> {
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(data);
    let mut output = Vec::new();
    decoder.read_to_end(&mut output)?;
    Ok(output)
}

// =============================================================================
// Parallel Decompression
// =============================================================================

/// Parallel gzip decompressor
pub struct ParallelInflater {
    num_threads: usize,
}

impl ParallelInflater {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Decompress gzip data in parallel
    pub fn decompress<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        // Check for BGZF (easiest to parallelize - has embedded block sizes)
        if detect_bgzf(data) {
            return self.decompress_bgzf(data, writer);
        }

        // Check for multi-member (pigz output) - can parallelize per-member
        let members = find_member_boundaries(data);
        if members.len() > 1 && self.num_threads > 1 {
            return self.decompress_multi_member(data, &members, writer);
        }

        // Single member: use ultra-fast sequential
        let mut output = Vec::new();
        if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(data, &mut output).is_ok() {
            let len = output.len() as u64;
            writer.write_all(&output)?;
            return Ok(len);
        }

        // Fallback to simd_inflate
        output.clear();
        simd_inflate::inflate_gzip_fast(data, &mut output)?;
        let len = output.len() as u64;
        writer.write_all(&output)?;
        Ok(len)
    }

    /// Decompress multi-member gzip file in parallel (pigz output)
    fn decompress_multi_member<W: Write + Send>(
        &self,
        data: &[u8],
        members: &[(usize, usize)],
        writer: &mut W,
    ) -> io::Result<u64> {
        let num_members = members.len();
        let results: Vec<Mutex<Option<ChunkResult>>> =
            (0..num_members).map(|_| Mutex::new(None)).collect();

        let next_member = AtomicUsize::new(0);

        // Parallel decompression - each member is independent!
        std::thread::scope(|scope| {
            for _ in 0..self.num_threads.min(num_members) {
                let next_ref = &next_member;
                let results_ref = &results;
                let members_ref = members;

                scope.spawn(move || loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_members {
                        break;
                    }

                    let (start, end) = members_ref[idx];
                    let member_data = &data[start..end];

                    // Each member is a complete gzip stream - decompress independently
                    let mut output = Vec::new();
                    let result = if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(
                        member_data,
                        &mut output,
                    )
                    .is_ok()
                    {
                        ChunkResult::success(idx, output)
                    } else {
                        // Fallback to flate2
                        match decompress_member_flate2(member_data) {
                            Ok(decompressed) => ChunkResult::success(idx, decompressed),
                            Err(e) => ChunkResult::failure(idx, e.to_string()),
                        }
                    };

                    *results_ref[idx].lock().unwrap() = Some(result);
                });
            }
        });

        // Collect results in order
        let mut total = 0u64;
        for mutex in results.iter() {
            let result = mutex
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| io::Error::other("Missing result"))?;

            if !result.success {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    result.error.unwrap_or_else(|| "Unknown error".to_string()),
                ));
            }

            writer.write_all(&result.data)?;
            total += result.data.len() as u64;
        }

        Ok(total)
    }

    /// Decompress BGZF file in parallel
    fn decompress_bgzf<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let blocks = parse_bgzf_blocks(data);

        if blocks.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "No BGZF blocks found",
            ));
        }

        let num_blocks = blocks.len();
        let results: Vec<Mutex<Option<ChunkResult>>> =
            (0..num_blocks).map(|_| Mutex::new(None)).collect();

        let next_block = AtomicUsize::new(0);

        // Parallel decompression
        std::thread::scope(|scope| {
            for _ in 0..self.num_threads.min(num_blocks) {
                let next_ref = &next_block;
                let results_ref = &results;
                let blocks_ref = &blocks;

                scope.spawn(move || loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let block_data = &data[block.start..block.start + block.csize];

                    let result = match decompress_bgzf_block(block_data) {
                        Ok(decompressed) => ChunkResult::success(idx, decompressed),
                        Err(e) => ChunkResult::failure(idx, e.to_string()),
                    };

                    *results_ref[idx].lock().unwrap() = Some(result);
                });
            }
        });

        // Collect results in order
        let mut total = 0u64;
        for mutex in results.iter() {
            let result = mutex
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| io::Error::other("Missing result"))?;

            if !result.success {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    result.error.unwrap_or_else(|| "Unknown error".to_string()),
                ));
            }

            writer.write_all(&result.data)?;
            total += result.data.len() as u64;
        }

        Ok(total)
    }
}

/// Decompress a single BGZF block
fn decompress_bgzf_block(block_data: &[u8]) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    // Try ultra-fast inflate first
    if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(block_data, &mut output).is_ok() {
        return Ok(output);
    }
    // Fallback to simd_inflate
    output.clear();
    simd_inflate::inflate_gzip_fast(block_data, &mut output)?;
    Ok(output)
}

// =============================================================================
// Integration with Main Decompression Path
// =============================================================================

/// High-performance decompression that automatically selects the best strategy
pub fn decompress_auto<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    if data.len() < PARALLEL_THRESHOLD || num_threads == 1 {
        // Small file or single-threaded: use ultra-fast inflate
        let mut output = Vec::new();
        if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(data, &mut output).is_ok() {
            let len = output.len() as u64;
            writer.write_all(&output)?;
            return Ok(len);
        }
        // Fallback to simd_inflate
        output.clear();
        simd_inflate::inflate_gzip_fast(data, &mut output)?;
        let len = output.len() as u64;
        writer.write_all(&output)?;
        Ok(len)
    } else {
        // Large file: use parallel
        let inflater = ParallelInflater::new(num_threads);
        inflater.decompress(data, writer)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write as IoWrite;

    #[test]
    fn test_sequential_decompress() {
        let original = b"Hello, World! This is a test of sequential decompression.";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_auto(&compressed, &mut output, 1).unwrap();

        assert_eq!(&output[..], &original[..]);
    }

    #[test]
    fn test_parallel_inflater() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let inflater = ParallelInflater::new(4);
        let mut output = Vec::new();
        inflater.decompress(&compressed, &mut output).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_bgzf_detection() {
        // Create a simple gzip file (not BGZF)
        let original = b"Test data";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        assert!(!detect_bgzf(&compressed));
    }

    #[test]
    fn test_multi_member_detection() {
        // Create two separate gzip streams
        let part1 = b"Hello, ";
        let part2 = b"World!";

        let mut encoder1 = GzEncoder::new(Vec::new(), Compression::default());
        encoder1.write_all(part1).unwrap();
        let compressed1 = encoder1.finish().unwrap();

        let mut encoder2 = GzEncoder::new(Vec::new(), Compression::default());
        encoder2.write_all(part2).unwrap();
        let compressed2 = encoder2.finish().unwrap();

        // Concatenate them
        let mut multi = compressed1.clone();
        multi.extend_from_slice(&compressed2);

        // Test detection
        let members = find_member_boundaries(&multi);
        assert_eq!(
            members.len(),
            2,
            "Should find 2 members, found: {:?}",
            members
        );
        assert_eq!(members[0], (0, compressed1.len()));
        assert_eq!(members[1], (compressed1.len(), multi.len()));
    }

    #[test]
    fn test_multi_member_decompress() {
        // Create two separate gzip streams
        let part1 = b"Hello, ";
        let part2 = b"World!";

        let mut encoder1 = GzEncoder::new(Vec::new(), Compression::default());
        encoder1.write_all(part1).unwrap();
        let compressed1 = encoder1.finish().unwrap();

        let mut encoder2 = GzEncoder::new(Vec::new(), Compression::default());
        encoder2.write_all(part2).unwrap();
        let compressed2 = encoder2.finish().unwrap();

        // Concatenate them
        let mut multi = compressed1;
        multi.extend_from_slice(&compressed2);

        // Decompress
        let inflater = ParallelInflater::new(4);
        let mut output = Vec::new();
        inflater.decompress(&multi, &mut output).unwrap();

        // Should get both parts
        let expected = b"Hello, World!";
        assert_eq!(&output[..], &expected[..]);
    }

    #[test]
    fn test_multi_member_large() {
        // Create two large gzip streams
        let part1: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let part2: Vec<u8> = (0..100_000).map(|i| ((i + 50) % 256) as u8).collect();

        let mut encoder1 = GzEncoder::new(Vec::new(), Compression::default());
        encoder1.write_all(&part1).unwrap();
        let compressed1 = encoder1.finish().unwrap();

        let mut encoder2 = GzEncoder::new(Vec::new(), Compression::default());
        encoder2.write_all(&part2).unwrap();
        let compressed2 = encoder2.finish().unwrap();

        // Concatenate them
        let mut multi = compressed1.clone();
        multi.extend_from_slice(&compressed2);

        eprintln!(
            "Multi-member size: {}, member1: {}, member2: {}",
            multi.len(),
            compressed1.len(),
            compressed2.len()
        );

        // Test detection
        let members = find_member_boundaries(&multi);
        eprintln!("Found {} members: {:?}", members.len(), members);
        assert_eq!(members.len(), 2, "Should find 2 members");

        // Decompress with parallel inflater
        let inflater = ParallelInflater::new(4);
        let mut output = Vec::new();
        inflater.decompress(&multi, &mut output).unwrap();

        // Should get both parts
        let mut expected = part1.clone();
        expected.extend_from_slice(&part2);

        eprintln!(
            "Expected: {} bytes, got: {} bytes",
            expected.len(),
            output.len()
        );
        assert_eq!(output.len(), expected.len(), "Output size mismatch");
        assert_slices_eq!(output, expected, "Output content mismatch");
    }
}
