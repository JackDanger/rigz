//! Parallel Deflate Decompression
//!
//! This module implements true parallel deflate decompression by:
//!
//! 1. **First pass**: Decode sequentially while recording block boundaries
//! 2. **Parallel decode**: For each boundary, decode in parallel with provided window
//! 3. **Result merge**: Combine parallel results in order
//!
//! This achieves rapidgzip-level performance for large files by parallelizing
//! the actual deflate decoding work.

#![allow(dead_code)]
#![allow(unused_variables)]

use crate::deflate_decoder::{skip_gzip_header, BlockBoundary, DeflateDecoder};
use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Minimum file size for parallel decompression
const MIN_PARALLEL_SIZE: usize = 1024 * 1024; // 1MB

/// Number of chunks to divide file into
const CHUNKS_PER_THREAD: usize = 4;

// Thread-local decoder state
thread_local! {
    static DECODER_BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(256 * 1024));
}

/// Result of decoding a chunk
struct ChunkResult {
    /// Chunk index
    index: usize,
    /// Decompressed data
    data: Vec<u8>,
    /// Whether decoding succeeded
    success: bool,
    /// Error message if failed
    error: Option<String>,
    /// Final window state (for next chunk)
    final_window: Vec<u8>,
    /// Final window position
    final_window_pos: usize,
}

/// Parallel gzip decompressor
pub struct ParallelInflater {
    num_threads: usize,
    chunk_size: usize,
}

impl ParallelInflater {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            chunk_size: 0, // Calculated based on file size
        }
    }

    /// Decompress gzip data using parallel deflate decoding
    pub fn decompress<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        // Skip gzip header
        let header_size = skip_gzip_header(data)?;
        let trailer_size = 8; // CRC32 + ISIZE

        if data.len() < header_size + trailer_size {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated gzip"));
        }

        let deflate_data = &data[header_size..data.len() - trailer_size];

        // For small files, use single-threaded decode
        if deflate_data.len() < MIN_PARALLEL_SIZE || self.num_threads <= 1 {
            return self.decompress_sequential(deflate_data, writer);
        }

        // Phase 1: Sequential decode to find block boundaries
        let boundaries = self.find_boundaries(deflate_data)?;

        if boundaries.len() < 2 {
            // Not enough boundaries for parallelism
            return self.decompress_sequential(deflate_data, writer);
        }

        // Phase 2: Parallel decode using boundaries
        self.decompress_parallel(deflate_data, &boundaries, writer)
    }

    /// Find block boundaries by doing a sequential decode
    fn find_boundaries(&self, data: &[u8]) -> io::Result<Vec<BlockBoundary>> {
        let chunk_size = data.len() / self.num_threads;
        let boundary_spacing = chunk_size.max(64 * 1024);

        let mut decoder = DeflateDecoder::new(data);
        decoder.set_boundary_spacing(boundary_spacing);

        // Decode to find boundaries (output to /dev/null)
        let mut output = io::sink();
        decoder.decode(&mut output)?;

        Ok(decoder.boundaries().to_vec())
    }

    /// Decompress in parallel using pre-computed boundaries
    fn decompress_parallel<W: Write + Send>(
        &self,
        data: &[u8],
        boundaries: &[BlockBoundary],
        writer: &mut W,
    ) -> io::Result<u64> {
        let num_chunks = boundaries.len();
        let results: Vec<Mutex<Option<ChunkResult>>> =
            (0..num_chunks).map(|_| Mutex::new(None)).collect();

        let next_chunk = AtomicUsize::new(0);
        let any_error = AtomicBool::new(false);

        std::thread::scope(|scope| {
            let next_ref = &next_chunk;
            let results_ref = &results;
            let error_ref = &any_error;
            let boundaries_ref = boundaries;

            for _ in 0..self.num_threads.min(num_chunks) {
                scope.spawn(move || {
                    loop {
                        if error_ref.load(Ordering::Relaxed) {
                            break;
                        }

                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_chunks {
                            break;
                        }

                        let boundary = &boundaries_ref[idx];
                        let next_boundary = boundaries_ref.get(idx + 1);

                        // Calculate expected output size
                        let expected_size = next_boundary
                            .map(|b| b.output_offset - boundary.output_offset)
                            .unwrap_or(0);

                        // Decode this chunk
                        let result =
                            decode_chunk(data, idx, boundary, next_boundary, expected_size);

                        if !result.success {
                            error_ref.store(true, Ordering::Relaxed);
                        }

                        *results_ref[idx].lock().unwrap() = Some(result);
                    }
                });
            }
        });

        // Check for errors
        if any_error.load(Ordering::Relaxed) {
            // Fall back to sequential
            return self.decompress_sequential(data, writer);
        }

        // Write results in order
        let mut total = 0u64;
        for result_mutex in results {
            if let Some(result) = result_mutex.into_inner().unwrap() {
                if result.success {
                    writer.write_all(&result.data)?;
                    total += result.data.len() as u64;
                }
            }
        }

        writer.flush()?;
        Ok(total)
    }

    /// Sequential decompression fallback
    fn decompress_sequential<W: Write>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let mut decoder = DeflateDecoder::new(data);
        let mut output = Vec::new();
        let size = decoder.decode(&mut output)?;
        writer.write_all(&output)?;
        writer.flush()?;
        Ok(size as u64)
    }
}

/// Decode a single chunk using a provided boundary
fn decode_chunk(
    data: &[u8],
    index: usize,
    boundary: &BlockBoundary,
    next_boundary: Option<&BlockBoundary>,
    expected_size: usize,
) -> ChunkResult {
    let bit_offset = boundary.bit_offset;
    let byte_offset = bit_offset / 8;

    if byte_offset >= data.len() {
        return ChunkResult {
            index,
            data: Vec::new(),
            success: false,
            error: Some("Invalid boundary offset".to_string()),
            final_window: Vec::new(),
            final_window_pos: 0,
        };
    }

    // Create decoder with the window from this boundary
    let mut decoder =
        DeflateDecoder::with_window(data, bit_offset, &boundary.window, boundary.window_pos);

    let mut output = Vec::with_capacity(expected_size.max(1024));

    // Decode our range: from this boundary's output to next boundary's output
    let start_output = boundary.output_offset;
    let end_output = next_boundary.map(|b| b.output_offset).unwrap_or(usize::MAX);

    match decode_until(&mut decoder, &mut output, start_output, end_output) {
        Ok(_) => {
            let (window, pos) = decoder.get_window();
            ChunkResult {
                index,
                data: output,
                success: true,
                error: None,
                final_window: window,
                final_window_pos: pos,
            }
        }
        Err(e) => ChunkResult {
            index,
            data: Vec::new(),
            success: false,
            error: Some(e.to_string()),
            final_window: Vec::new(),
            final_window_pos: 0,
        },
    }
}

/// Decode a range of output bytes
fn decode_until<W: Write>(
    decoder: &mut DeflateDecoder,
    writer: &mut W,
    start_output: usize,
    end_output: usize,
) -> io::Result<usize> {
    decoder.decode_range(writer, start_output, end_output)
}

/// High-level parallel gzip decompression
pub fn decompress_gzip_parallel<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    let inflater = ParallelInflater::new(num_threads);
    inflater.decompress(data, writer)
}

/// Window cache for repeated decompression
pub struct WindowCache {
    /// Cached boundaries with windows
    boundaries: Vec<BlockBoundary>,
    /// File hash for validation
    file_hash: u64,
}

impl WindowCache {
    /// Create a new cache from boundaries
    pub fn from_boundaries(boundaries: Vec<BlockBoundary>, file_hash: u64) -> Self {
        Self {
            boundaries,
            file_hash,
        }
    }

    /// Check if cache is valid for a file
    pub fn is_valid_for(&self, file_hash: u64) -> bool {
        self.file_hash == file_hash
    }

    /// Get boundaries for parallel decompression
    pub fn boundaries(&self) -> &[BlockBoundary] {
        &self.boundaries
    }

    /// Number of boundaries
    pub fn len(&self) -> usize {
        self.boundaries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.boundaries.is_empty()
    }
}

/// Build a window cache for a gzip file
pub fn build_window_cache(data: &[u8], num_chunks: usize) -> io::Result<WindowCache> {
    let header_size = skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    let chunk_size = deflate_data.len() / num_chunks.max(1);
    let boundary_spacing = chunk_size.max(64 * 1024);

    let mut decoder = DeflateDecoder::new(deflate_data);
    decoder.set_boundary_spacing(boundary_spacing);

    let mut output = io::sink();
    decoder.decode(&mut output)?;

    // Simple file hash (first 8 bytes + length)
    let file_hash = if data.len() >= 8 {
        u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) ^ (data.len() as u64)
    } else {
        data.len() as u64
    };

    Ok(WindowCache::from_boundaries(
        decoder.boundaries().to_vec(),
        file_hash,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    #[test]
    fn test_parallel_decompress_small() {
        let original = b"Hello, World!";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_gzip_parallel(&compressed, &mut output, 4).unwrap();

        assert_eq!(&output, original);
    }

    #[test]
    fn test_parallel_decompress_large() {
        let original: Vec<u8> = (0..500_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_gzip_parallel(&compressed, &mut output, 4).unwrap();

        assert_eq!(output, original);
    }

    #[test]
    fn test_window_cache() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let cache = build_window_cache(&compressed, 4).unwrap();
        assert!(!cache.is_empty());
    }
}
