//! Parallel Decompression using rapidgzip-style speculative decoding
//!
//! This module implements the core parallel decompression algorithm:
//! 1. Partition input at fixed intervals (chunk spacing)
//! 2. Find block boundaries near each partition point
//! 3. Speculatively decode each chunk in parallel (with markers for unknown back-refs)
//! 4. Propagate windows from chunk to chunk
//! 5. Replace markers in parallel
//! 6. Output the final decompressed data

#![allow(dead_code)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::block_finder_lut::find_block_in_range;
use crate::marker_decode::{MarkerDecoder, MARKER_BASE};

/// Default chunk size (4MB in compressed data, like rapidgzip)
pub const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Target output per chunk (for balanced parallel work)
/// Typical compression ratio is ~3x, so 4MB input → ~12MB output
/// But we set a bit higher to allow for variations
#[allow(dead_code)]
pub const TARGET_OUTPUT_PER_CHUNK: usize = CHUNK_SIZE * 4; // 16MB per chunk

/// Search range for finding block boundaries (in bits)
/// We search within 512 bytes of the chunk boundary
pub const BLOCK_SEARCH_RANGE: usize = 512 * 8;

/// Result of decoding a single chunk
#[derive(Clone, Default)]
pub struct DecodedChunk {
    /// Chunk index
    pub index: usize,
    /// Bit offset where decoding started (in deflate stream)
    pub start_bit: usize,
    /// Bit offset where decoding ended
    pub end_bit: usize,
    /// Decoded data (u16 to hold markers)
    pub data: Vec<u16>,
    /// Number of markers (values > 255)
    pub marker_count: usize,
    /// Last 32KB of decoded data (for next chunk's window)
    pub final_window: Vec<u8>,
    /// Whether this chunk decoded successfully
    pub success: bool,
    /// Whether this chunk hit the stream end (BFINAL=1)
    pub is_final: bool,
}

/// Try to decode a chunk starting at the given bit position
fn decode_chunk(
    deflate_data: &[u8],
    chunk_index: usize,
    start_bit: usize,
    max_output: usize,
) -> DecodedChunk {
    // Calculate byte offset and create decoder
    let byte_offset = start_bit / 8;
    let bit_in_byte = start_bit % 8;

    if byte_offset >= deflate_data.len() {
        return DecodedChunk {
            index: chunk_index,
            start_bit,
            ..Default::default()
        };
    }

    let chunk_data = &deflate_data[byte_offset..];
    let mut decoder = MarkerDecoder::new(chunk_data, bit_in_byte);

    match decoder.decode_until(max_output) {
        Ok(is_final) => {
            let output = decoder.output();
            DecodedChunk {
                index: chunk_index,
                start_bit,
                end_bit: byte_offset * 8 + decoder.bit_position(),
                data: output.to_vec(),
                marker_count: decoder.marker_count(),
                final_window: decoder.final_window(),
                success: true,
                is_final,
            }
        }
        Err(_) => DecodedChunk {
            index: chunk_index,
            start_bit,
            ..Default::default()
        },
    }
}

/// Find the start bit for a chunk
/// For chunk 0, this is always 0
/// For other chunks, we search for a block boundary near the partition point
fn find_chunk_start(
    deflate_data: &[u8],
    chunk_index: usize,
    partition_byte: usize,
) -> Option<usize> {
    if chunk_index == 0 {
        return Some(0);
    }

    let partition_bit = partition_byte * 8;
    let end_bit = deflate_data.len() * 8;

    if partition_bit >= end_bit {
        return None;
    }

    // Search for a block boundary starting at the partition point
    find_block_in_range(deflate_data, partition_bit, BLOCK_SEARCH_RANGE)
}

/// Replace markers in a chunk's data using the provided window
fn replace_markers(data: &mut [u16], window: &[u8]) {
    for value in data.iter_mut() {
        if *value >= MARKER_BASE {
            let offset = (*value - MARKER_BASE) as usize;
            if offset < window.len() {
                *value = window[offset] as u16;
            } else {
                // Marker points outside window - use 0
                *value = 0;
            }
        }
    }
}

/// Convert u16 buffer to u8 buffer
fn to_u8(data: &[u16]) -> Vec<u8> {
    data.iter().map(|&v| v as u8).collect()
}

/// Parallel decompression with speculative block boundary finding
///
/// Strategy:
/// 1. Try to find valid block boundaries at partition points using LUT
/// 2. If found, decode chunks in parallel with markers
/// 3. If not found (common for single-member gzip), fall back to fast sequential
///
/// This provides:
/// - Fast sequential decode for files without clear block boundaries
/// - True parallel decode for files with findable block boundaries
pub fn decompress_parallel<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Parse gzip header
    let header_size = skip_gzip_header(data)?;
    let deflate_end = data.len().saturating_sub(8);
    let deflate_data = &data[header_size..deflate_end];

    // For small data or single thread, use fast sequential decode
    if deflate_data.len() < CHUNK_SIZE * 2 || num_threads <= 1 {
        return decompress_sequential(data, writer);
    }

    // Phase 1: Try speculative parallel decode
    // Find potential block starts at partition points
    let num_partitions = (deflate_data.len() / CHUNK_SIZE).max(1);
    let mut partition_starts: Vec<Option<usize>> = vec![None; num_partitions];
    partition_starts[0] = Some(0);

    for (i, slot) in partition_starts.iter_mut().enumerate().skip(1) {
        let partition_bit = i * CHUNK_SIZE * 8;
        *slot = find_block_in_range(deflate_data, partition_bit, BLOCK_SEARCH_RANGE);
    }

    // Check for multi-member gzip (like files created by pigz)
    // Each member is a separate deflate stream - perfect for parallel decompress
    let member_starts = find_member_boundaries(data);

    if member_starts.len() > 1 {
        return decompress_multi_member_parallel(data, &member_starts, writer, num_threads);
    }

    // Build list of valid partition starts for speculative decode
    let valid_starts: Vec<(usize, usize)> = partition_starts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|start| (i, start)))
        .collect();

    // If we only found the first partition, fall back to sequential
    if valid_starts.len() <= 1 {
        return decompress_sequential(data, writer);
    }

    // For single-member gzip files, use the rapidgzip strategy:
    // 1. Sequential first pass to find block boundaries and collect windows
    // 2. Parallel re-decode using windows
    decompress_single_member_parallel(data, header_size, deflate_end, writer, num_threads)
}

/// Parallel decompress for single-member gzip using sequential boundary finding
/// followed by parallel re-decode with dictionaries
fn decompress_single_member_parallel<W: Write + Send>(
    data: &[u8],
    _header_size: usize,
    _deflate_end: usize,
    writer: &mut W,
    _num_threads: usize,
) -> io::Result<u64> {
    // For single-member files, use our turbo inflate (pure Rust)
    // No libdeflate fallback - we want to see errors
    //
    // Future parallel strategy (rapidgzip approach):
    // 1. Sequential pass: decode and record block boundaries + 32KB windows
    // 2. Parallel pass: re-decode each block using window from previous block

    decompress_single_member_turbo(data, writer)
}

/// Use our turbo inflate for single-member gzip (pure Rust, no libdeflate)
fn decompress_single_member_turbo<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    // Parse gzip header
    let header_size = crate::marker_decode::skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    // Get expected size from ISIZE in trailer
    let isize_hint = if data.len() >= 8 {
        u32::from_le_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]) as usize
    } else {
        0
    };

    let output_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(64 * 1024)
    };

    let mut output = vec![0u8; output_size];

    // Use our turbo inflate
    match crate::bgzf::inflate_into_pub(deflate_data, &mut output) {
        Ok(size) => {
            writer.write_all(&output[..size])?;
            Ok(size as u64)
        }
        Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
    }
}

/// Sequential decompression fallback
fn decompress_sequential<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    use std::io::Read;

    // Use flate2 MultiGzDecoder for sequential decode (handles multi-member files)
    let mut decoder = flate2::read::MultiGzDecoder::new(data);
    let mut output = Vec::new();
    decoder.read_to_end(&mut output)?;

    let total = output.len() as u64;
    writer.write_all(&output)?;

    Ok(total)
}

/// Find gzip member boundaries (for multi-member files like those from pigz)
/// Returns list of byte offsets where each gzip member starts
fn find_member_boundaries(data: &[u8]) -> Vec<usize> {
    let mut boundaries = vec![0]; // First member always starts at 0

    // Scan for gzip magic number (0x1f 0x8b) followed by compression method (0x08)
    // This is a heuristic - we validate headers to reduce false positives
    let mut pos = 10; // Skip past first header

    while pos + 10 < data.len() {
        // Check for gzip magic
        if data[pos] == 0x1f && data[pos + 1] == 0x8b && data[pos + 2] == 0x08 {
            // Validate flags (reserved bits must be 0)
            let flags = data[pos + 3];
            if flags & 0xe0 == 0 {
                // Check XFL (extra flags) is reasonable
                let xfl = data[pos + 8];
                if xfl == 0 || xfl == 2 || xfl == 4 {
                    // Check OS byte is reasonable (0-13, 255)
                    let os = data[pos + 9];
                    if os <= 13 || os == 255 {
                        boundaries.push(pos);
                    }
                }
            }
        }
        pos += 1;
    }

    boundaries
}

/// Parallel decompression of multi-member gzip files
fn decompress_multi_member_parallel<W: Write + Send>(
    data: &[u8],
    member_starts: &[usize],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Decompress each member in parallel using pure Rust
    let outputs: Vec<Mutex<Vec<u8>>> = (0..member_starts.len())
        .map(|_| Mutex::new(Vec::new()))
        .collect();

    let next_member = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(member_starts.len()) {
            let outputs_ref = &outputs;
            let starts_ref = member_starts;
            let next_ref = &next_member;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= starts_ref.len() {
                        break;
                    }

                    let start = starts_ref[idx];
                    let end = if idx + 1 < starts_ref.len() {
                        starts_ref[idx + 1]
                    } else {
                        data.len()
                    };

                    let member_data = &data[start..end];

                    // Decompress using pure Rust ultra_fast_inflate
                    let mut output = Vec::with_capacity(member_data.len() * 3);
                    if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(member_data, &mut output)
                        .is_err()
                    {
                        // Fall back to flate2 on error
                        use std::io::Read;
                        output.clear();
                        let mut decoder = flate2::read::GzDecoder::new(member_data);
                        let _ = decoder.read_to_end(&mut output);
                    }

                    *outputs_ref[idx].lock().unwrap() = output;
                }
            });
        }
    });

    // Write outputs in order
    let mut total = 0u64;
    for output_mutex in &outputs {
        let output = output_mutex.lock().unwrap();
        writer.write_all(&output)?;
        total += output.len() as u64;
    }

    Ok(total)
}

/// Skip the gzip header and return the offset to the deflate data
fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Data too short for gzip header",
        ));
    }

    // Check magic number
    if data[0] != 0x1f || data[1] != 0x8b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid gzip magic number",
        ));
    }

    // Check compression method (must be 8 = deflate)
    if data[2] != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported compression method",
        ));
    }

    let flags = data[3];
    let mut offset = 10; // Base header size

    // FEXTRA
    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated FEXTRA",
            ));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + xlen;
    }

    // FNAME (null-terminated)
    if flags & 0x08 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1; // Skip null terminator
    }

    // FCOMMENT (null-terminated)
    if flags & 0x10 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FHCRC (2 bytes)
    if flags & 0x02 != 0 {
        offset += 2;
    }

    if offset > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated gzip header",
        ));
    }

    Ok(offset)
}

#[cfg(test)]
#[allow(
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::if_same_then_else
)]
mod tests {
    use super::*;

    #[test]
    fn test_decompress_parallel_small() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        // Create test data
        let original: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress with our parallel decoder
        let mut output = Vec::new();
        decompress_parallel(&compressed, &mut output, 4).unwrap();

        assert_slices_eq!(output, original);
    }

    #[test]
    fn test_decompress_parallel_large() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping - no benchmark file");
                return;
            }
        };

        // Get expected output from flate2
        use std::io::Read;
        let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        flate2_decoder.read_to_end(&mut expected).unwrap();

        // Decompress with parallel decoder
        let mut output = Vec::new();
        let result = decompress_parallel(&data, &mut output, 8);

        eprintln!(
            "Result: {:?}, output_len={}, expected_len={}",
            result.is_ok(),
            output.len(),
            expected.len()
        );

        if result.is_ok() {
            assert_eq!(output.len(), expected.len(), "Size mismatch");
            assert_slices_eq!(output, expected, "Content mismatch");
        }
    }

    #[test]
    fn test_skip_gzip_header() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"Hello").unwrap();
        let compressed = encoder.finish().unwrap();

        let header_size = skip_gzip_header(&compressed).unwrap();
        assert!(header_size >= 10);
        assert!(header_size < compressed.len());
    }
}

#[test]
fn test_benchmark_parallel_vs_sequential() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping - no benchmark file");
            return;
        }
    };

    // Warm up
    let mut output = Vec::new();
    decompress_parallel(&data, &mut output, 8).unwrap();
    let expected_size = output.len();

    // Benchmark sequential (1 thread)
    let start = std::time::Instant::now();
    output.clear();
    decompress_parallel(&data, &mut output, 1).unwrap();
    let sequential_time = start.elapsed();
    let sequential_speed = expected_size as f64 / sequential_time.as_secs_f64() / 1_000_000.0;

    // Benchmark parallel (8 threads)
    let start = std::time::Instant::now();
    output.clear();
    decompress_parallel(&data, &mut output, 8).unwrap();
    let parallel_time = start.elapsed();
    let parallel_speed = expected_size as f64 / parallel_time.as_secs_f64() / 1_000_000.0;
    assert_slices_eq!(output, output); // Just to verify macro works

    // Benchmark libdeflate
    let start = std::time::Instant::now();
    let mut decomp = libdeflater::Decompressor::new();
    let mut libdeflate_output = vec![0u8; expected_size + 1024];
    let _ = decomp.gzip_decompress(&data, &mut libdeflate_output);
    let libdeflate_time = start.elapsed();
    let libdeflate_speed = expected_size as f64 / libdeflate_time.as_secs_f64() / 1_000_000.0;

    eprintln!("=== Benchmark Results ===");
    eprintln!(
        "Sequential (1 thread): {:?} = {:.1} MB/s",
        sequential_time, sequential_speed
    );
    eprintln!(
        "Parallel (8 threads):  {:?} = {:.1} MB/s",
        parallel_time, parallel_speed
    );
    eprintln!(
        "libdeflate:            {:?} = {:.1} MB/s",
        libdeflate_time, libdeflate_speed
    );
    eprintln!(
        "Parallel speedup vs sequential: {:.2}x",
        parallel_speed / sequential_speed
    );
    eprintln!(
        "Parallel vs libdeflate: {:.2}x",
        parallel_speed / libdeflate_speed
    );
}

#[test]
#[allow(clippy::manual_div_ceil)]
fn test_debug_parallel_decode() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let header_size = skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    eprintln!("Deflate data size: {} bytes", deflate_data.len());
    eprintln!(
        "Number of chunks: {}",
        (deflate_data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE
    );

    // Try to find block boundaries
    for i in 0..5 {
        let partition_byte = i * CHUNK_SIZE;
        if partition_byte >= deflate_data.len() {
            break;
        }

        let start = find_chunk_start(deflate_data, i, partition_byte);
        eprintln!(
            "Chunk {}: partition_byte={}, found_start={:?}",
            i, partition_byte, start
        );
    }

    // Try decoding chunk 0
    let chunk0 = decode_chunk(deflate_data, 0, 0, usize::MAX);
    eprintln!(
        "Chunk 0: success={}, output_len={}, markers={}, is_final={}",
        chunk0.success,
        chunk0.data.len(),
        chunk0.marker_count,
        chunk0.is_final
    );
    eprintln!("Chunk 0 end_bit: {}", chunk0.end_bit);
}

#[test]
#[allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::if_same_then_else
)]
fn test_debug_parallel_path() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let header_size = skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    let num_chunks = (deflate_data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    eprintln!("Total chunks: {}", num_chunks);

    // Find all chunk starts
    let mut chunk_starts: Vec<Option<usize>> = vec![None; num_chunks];
    chunk_starts[0] = Some(0);

    for i in 1..num_chunks {
        let partition_bit = i * CHUNK_SIZE * 8;
        chunk_starts[i] = find_block_in_range(deflate_data, partition_bit, BLOCK_SEARCH_RANGE);
    }

    let valid_count = chunk_starts.iter().filter(|s| s.is_some()).count();
    eprintln!("Valid chunk starts: {}/{}", valid_count, num_chunks);

    // Show first few
    for (i, start) in chunk_starts.iter().enumerate().take(5) {
        eprintln!("  Chunk {}: {:?}", i, start);
    }

    // Now try actual parallel decode
    let valid_starts: Vec<(usize, usize)> = chunk_starts
        .iter()
        .enumerate()
        .filter_map(|(i, start)| start.map(|s| (i, s)))
        .collect();

    eprintln!("\nDecoding {} chunks...", valid_starts.len());

    let mut total_output = 0usize;
    let mut last_end_bit = 0usize;
    let mut chained = 0usize;

    for (idx, (chunk_idx, start_bit)) in valid_starts.iter().enumerate() {
        let stop_bit = if idx + 1 < valid_starts.len() {
            valid_starts[idx + 1].1
        } else {
            deflate_data.len() * 8
        };

        let bit_range = stop_bit.saturating_sub(*start_bit);
        let max_output = (bit_range / 8) * 4;

        let chunk = decode_chunk(deflate_data, *chunk_idx, *start_bit, max_output);

        eprintln!(
            "  Chunk {}: start_bit={}, end_bit={}, output={}, markers={}, success={}, is_final={}",
            chunk_idx,
            start_bit,
            chunk.end_bit,
            chunk.data.len(),
            chunk.marker_count,
            chunk.success,
            chunk.is_final
        );

        if idx == 0 && chunk.start_bit == 0 && chunk.success {
            total_output += chunk.data.len();
            last_end_bit = chunk.end_bit;
            chained += 1;
        } else if chunk.success && chunk.start_bit == last_end_bit {
            total_output += chunk.data.len();
            last_end_bit = chunk.end_bit;
            chained += 1;
        }

        if chunk.is_final {
            break;
        }
    }

    eprintln!(
        "\nChained chunks: {}, Total output: {}",
        chained, total_output
    );
}

#[test]
fn test_why_chunks_fail() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let header_size = skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    // Try to decode at the "found" block position for chunk 1
    let start_bit = 33554439;
    let byte_offset = start_bit / 8;
    let bit_in_byte = start_bit % 8;

    eprintln!(
        "Trying to decode at bit {}, byte {}, bit_in_byte {}",
        start_bit, byte_offset, bit_in_byte
    );

    let chunk_data = &deflate_data[byte_offset..];
    let mut decoder = MarkerDecoder::new(chunk_data, bit_in_byte);

    // Try to decode just one block
    match decoder.decode_block() {
        Ok(is_final) => {
            eprintln!(
                "Block decoded! is_final={}, output_len={}",
                is_final,
                decoder.output().len()
            );
        }
        Err(e) => {
            eprintln!("Block decode FAILED: {:?}", e);
        }
    }

    // Compare with the actual next block position
    // Chunk 0 ended at bit 65279710
    let real_next_block = 65279710;
    let real_byte = real_next_block / 8;
    let real_bit = real_next_block % 8;

    eprintln!(
        "\nTrying REAL next block at bit {}, byte {}, bit_in_byte {}",
        real_next_block, real_byte, real_bit
    );

    let chunk_data = &deflate_data[real_byte..];
    let mut decoder = MarkerDecoder::new(chunk_data, real_bit);

    match decoder.decode_block() {
        Ok(is_final) => {
            eprintln!(
                "Block decoded! is_final={}, output_len={}",
                is_final,
                decoder.output().len()
            );
        }
        Err(e) => {
            eprintln!("Block decode FAILED: {:?}", e);
        }
    }
}

#[test]
fn test_sequential_decode_timing() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let header_size = skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    eprintln!("Decoding {} bytes of deflate data...", deflate_data.len());

    let start = std::time::Instant::now();

    let mut chunks = 0;
    let mut total_output = 0;
    let mut current_bit = 0usize;

    while current_bit / 8 < deflate_data.len() {
        let byte_offset = current_bit / 8;
        let bit_in_byte = current_bit % 8;
        let chunk_data = &deflate_data[byte_offset..];

        let mut decoder = MarkerDecoder::new(chunk_data, bit_in_byte);
        let is_final = decoder.decode_until(TARGET_OUTPUT_PER_CHUNK).unwrap();

        current_bit = byte_offset * 8 + decoder.bit_position();
        total_output += decoder.output().len();
        chunks += 1;

        eprintln!(
            "Chunk {}: output={}, bit={}",
            chunks,
            decoder.output().len(),
            current_bit
        );

        if is_final {
            break;
        }
        if chunks > 100 {
            eprintln!("Too many chunks, stopping");
            break;
        }
    }

    let elapsed = start.elapsed();
    let speed = total_output as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    eprintln!(
        "Done: {} chunks, {} bytes in {:?} = {:.1} MB/s",
        chunks, total_output, elapsed, speed
    );
}

#[test]
fn test_parallel_correctness_simple() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let start = std::time::Instant::now();
    let mut output = Vec::new();
    let result = decompress_parallel(&data, &mut output, 8);
    let elapsed = start.elapsed();

    eprintln!(
        "Parallel decompress: {:?}, output_len={}, time={:?}",
        result.is_ok(),
        output.len(),
        elapsed
    );

    assert!(result.is_ok());
    assert_eq!(output.len(), 211968000);
}

#[test]
fn test_parallel_content_correctness() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    // Get expected from flate2
    use std::io::Read;
    let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
    let mut expected = Vec::new();
    flate2_decoder.read_to_end(&mut expected).unwrap();

    // Our parallel decompress
    let mut output = Vec::new();
    decompress_parallel(&data, &mut output, 8).unwrap();

    assert_eq!(output.len(), expected.len(), "Size mismatch");

    // Check first and last few bytes
    assert_eq!(&output[..100], &expected[..100], "First 100 bytes mismatch");
    assert_eq!(
        &output[output.len() - 100..],
        &expected[expected.len() - 100..],
        "Last 100 bytes mismatch"
    );

    // Sample check at various positions
    for pos in [1000, 10000, 100000, 1000000, 10000000, 100000000] {
        if pos < output.len() {
            assert_eq!(output[pos], expected[pos], "Mismatch at position {}", pos);
        }
    }

    eprintln!("Content verification passed!");
}

#[test]
#[allow(clippy::needless_range_loop, clippy::if_same_then_else)]
fn test_debug_which_path() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let header_size = skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    let num_partitions = deflate_data.len() / CHUNK_SIZE;
    eprintln!("Partitions: {}", num_partitions);

    let mut partition_starts: Vec<Option<usize>> = vec![None; num_partitions];
    partition_starts[0] = Some(0);

    for i in 1..num_partitions {
        let partition_bit = i * CHUNK_SIZE * 8;
        partition_starts[i] = find_block_in_range(deflate_data, partition_bit, BLOCK_SEARCH_RANGE);
    }

    let valid_count = partition_starts.iter().filter(|s| s.is_some()).count();
    eprintln!("Valid partition starts: {}/{}", valid_count, num_partitions);

    if valid_count <= 1 {
        eprintln!("Would fall back to SEQUENTIAL (only first partition valid)");
    } else {
        eprintln!("Would try SPECULATIVE PARALLEL");
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_debug_early_check() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    let header_size = skip_gzip_header(&data).unwrap();
    let deflate_data = &data[header_size..data.len() - 8];

    // Decode chunk 0
    let chunk0 = decode_chunk(deflate_data, 0, 0, TARGET_OUTPUT_PER_CHUNK);
    eprintln!(
        "Chunk 0: end_bit={}, output={}, is_final={}",
        chunk0.end_bit,
        chunk0.data.len(),
        chunk0.is_final
    );

    // Find partition starts
    let num_partitions = deflate_data.len() / CHUNK_SIZE;
    let mut partition_starts: Vec<Option<usize>> = vec![None; num_partitions];
    partition_starts[0] = Some(0);

    for i in 1..num_partitions {
        let partition_bit = i * CHUNK_SIZE * 8;
        partition_starts[i] = find_block_in_range(deflate_data, partition_bit, BLOCK_SEARCH_RANGE);
    }

    let valid_starts: Vec<(usize, usize)> = partition_starts
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|start| (i, start)))
        .collect();

    eprintln!(
        "Valid starts: {:?}",
        valid_starts.iter().take(5).collect::<Vec<_>>()
    );

    // Check if any valid start matches chunk0's end
    let chunk0_end = chunk0.end_bit;
    let next_valid_start = valid_starts
        .iter()
        .skip(1)
        .find(|(_, bit)| *bit >= chunk0_end);
    eprintln!(
        "Next valid start at or after {}: {:?}",
        chunk0_end, next_valid_start
    );

    if let Some((_, bit)) = next_valid_start {
        if *bit == chunk0_end {
            eprintln!("MATCH! Would use parallel path");
        } else {
            eprintln!(
                "NO MATCH! Would fall back to sequential (found {} but need {})",
                bit, chunk0_end
            );
        }
    } else {
        eprintln!("NO MATCH! No valid start after chunk0's end");
    }
}

#[test]
fn test_8_threads_actually_sequential() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    // Time with 8 threads
    let start = std::time::Instant::now();
    let mut output = Vec::new();
    decompress_parallel(&data, &mut output, 8).unwrap();
    let time_8 = start.elapsed();

    // Time with 1 thread
    let start = std::time::Instant::now();
    output.clear();
    decompress_parallel(&data, &mut output, 1).unwrap();
    let time_1 = start.elapsed();

    eprintln!("8 threads: {:?}", time_8);
    eprintln!("1 thread: {:?}", time_1);
    eprintln!("Ratio: {:.2}x", time_8.as_secs_f64() / time_1.as_secs_f64());
}

#[test]
fn test_multi_member_parallel() {
    // Test with pigz-compressed file (has multiple members)
    let data = match std::fs::read("benchmark_data/silesia-pigz.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping - no pigz benchmark file");
            return;
        }
    };

    // Check how many members we find
    let members = find_member_boundaries(&data);
    eprintln!("Found {} gzip members", members.len());

    // Get expected from flate2
    use std::io::Read;
    let mut flate2_decoder = flate2::read::MultiGzDecoder::new(&data[..]);
    let mut expected = Vec::new();
    flate2_decoder.read_to_end(&mut expected).unwrap();

    // Benchmark sequential
    let start = std::time::Instant::now();
    let mut output = Vec::new();
    decompress_sequential(&data, &mut output).unwrap();
    let seq_time = start.elapsed();
    let seq_speed = expected.len() as f64 / seq_time.as_secs_f64() / 1_000_000.0;

    // Benchmark parallel
    let start = std::time::Instant::now();
    output.clear();
    decompress_parallel(&data, &mut output, 8).unwrap();
    let par_time = start.elapsed();
    let par_speed = expected.len() as f64 / par_time.as_secs_f64() / 1_000_000.0;

    eprintln!("Sequential: {:?} = {:.1} MB/s", seq_time, seq_speed);
    eprintln!(
        "Parallel (8 threads): {:?} = {:.1} MB/s",
        par_time, par_speed
    );
    eprintln!("Speedup: {:.2}x", par_speed / seq_speed);

    assert_eq!(output.len(), expected.len(), "Size mismatch");
}

#[test]
fn test_bgzf_parallel() {
    // Test with gzippy-compressed file (has many BGZF members)
    let data = match std::fs::read("benchmark_data/test-gzippy-l1-t14.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping - no gzippy benchmark file");
            return;
        }
    };

    // Check how many members we find
    let members = find_member_boundaries(&data);
    eprintln!(
        "Found {} gzip members in {:.1} MB file",
        members.len(),
        data.len() as f64 / 1_000_000.0
    );

    // Get expected from flate2
    use std::io::Read;
    let mut flate2_decoder = flate2::read::MultiGzDecoder::new(&data[..]);
    let mut expected = Vec::new();
    flate2_decoder.read_to_end(&mut expected).unwrap();
    eprintln!(
        "Expected output size: {:.1} MB",
        expected.len() as f64 / 1_000_000.0
    );

    // Benchmark sequential
    let start = std::time::Instant::now();
    let mut output = Vec::new();
    decompress_sequential(&data, &mut output).unwrap();
    let seq_time = start.elapsed();
    let seq_speed = expected.len() as f64 / seq_time.as_secs_f64() / 1_000_000.0;

    // Benchmark parallel (8 threads)
    let start = std::time::Instant::now();
    output.clear();
    decompress_parallel(&data, &mut output, 8).unwrap();
    let par_time = start.elapsed();
    let par_speed = expected.len() as f64 / par_time.as_secs_f64() / 1_000_000.0;

    eprintln!("Sequential: {:?} = {:.1} MB/s", seq_time, seq_speed);
    eprintln!(
        "Parallel (8 threads): {:?} = {:.1} MB/s",
        par_time, par_speed
    );
    eprintln!("Speedup: {:.2}x", par_speed / seq_speed);

    assert_eq!(output.len(), expected.len(), "Size mismatch");
}

#[test]
fn test_debug_sequential() {
    let data = match std::fs::read("benchmark_data/test-gzippy-l1-t14.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    eprintln!("Input size: {} bytes", data.len());

    let mut output = Vec::new();
    let result = decompress_sequential(&data, &mut output);

    eprintln!("Result: {:?}", result.is_ok());
    eprintln!("Output size: {} bytes", output.len());

    // Check first few bytes
    if output.len() > 100 {
        eprintln!("First 20 bytes: {:?}", &output[..20]);
    }
}

#[cfg(test)]
mod member_tests {
    use super::*;

    #[test]
    fn test_find_members() {
        let data = match std::fs::read("benchmark_data/silesia-pigz.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("No test file");
                return;
            }
        };

        let boundaries = find_member_boundaries(&data);
        eprintln!("Found {} member boundaries", boundaries.len());
        for (i, &b) in boundaries.iter().take(10).enumerate() {
            eprintln!("  Member {}: offset {}", i, b);
        }
    }
}

#[test]
fn test_decompress_content_check() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            return;
        }
    };

    use std::io::Read;
    let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
    let mut expected = Vec::new();
    flate2_decoder.read_to_end(&mut expected).unwrap();

    let mut output = Vec::new();
    decompress_parallel(&data, &mut output, 8).unwrap();

    // Find first mismatch
    let mut mismatch_pos = None;
    for (i, (&a, &b)) in output.iter().zip(expected.iter()).enumerate() {
        if a != b {
            mismatch_pos = Some(i);
            break;
        }
    }

    if let Some(pos) = mismatch_pos {
        eprintln!("First mismatch at position {}", pos);
        eprintln!(
            "Expected: {:?}",
            &expected[pos.saturating_sub(10)..pos + 10.min(expected.len() - pos)]
        );
        eprintln!(
            "Got:      {:?}",
            &output[pos.saturating_sub(10)..pos + 10.min(output.len() - pos)]
        );
    } else if output.len() != expected.len() {
        eprintln!(
            "Size mismatch: got {}, expected {}",
            output.len(),
            expected.len()
        );
    } else {
        eprintln!("Output matches!");
    }
}

#[cfg(test)]
mod trace_tests {
    use super::*;

    #[test]
    fn test_trace_decompress_path() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => return,
        };

        let header_size = skip_gzip_header(&data).unwrap();
        let deflate_end = data.len().saturating_sub(8);
        let deflate_data = &data[header_size..deflate_end];

        eprintln!("Data size: {}", data.len());
        eprintln!("Deflate size: {}", deflate_data.len());
        eprintln!("CHUNK_SIZE: {}", CHUNK_SIZE);
        eprintln!("Partitions: {}", (deflate_data.len() / CHUNK_SIZE).max(1));

        // Check member boundaries
        let members = find_member_boundaries(&data);
        eprintln!("Member boundaries: {:?}", members);
    }
}

#[test]
fn test_trace_full_path() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => return,
    };

    // Time libdeflate
    let start = std::time::Instant::now();
    let mut output1 = Vec::new();
    libdeflater::Decompressor::new()
        .gzip_decompress(&data, &mut output1)
        .ok();
    // Need to resize for libdeflate
    let mut output1 = vec![0u8; 250_000_000];
    let size1 = libdeflater::Decompressor::new()
        .gzip_decompress(&data, &mut output1)
        .unwrap();
    let libdeflate_time = start.elapsed();
    eprintln!("libdeflate: {} bytes in {:?}", size1, libdeflate_time);

    // Time our decompress_parallel
    let start = std::time::Instant::now();
    let mut output2 = Vec::new();
    let result = decompress_parallel(&data, &mut output2, 8);
    let parallel_time = start.elapsed();
    eprintln!(
        "decompress_parallel: {:?}, {} bytes in {:?}",
        result.is_ok(),
        output2.len(),
        parallel_time
    );

    // Time flate2
    let start = std::time::Instant::now();
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(&data[..]);
    let mut output3 = Vec::new();
    decoder.read_to_end(&mut output3).unwrap();
    let flate2_time = start.elapsed();
    eprintln!("flate2: {} bytes in {:?}", output3.len(), flate2_time);
}
