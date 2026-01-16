//! High-performance parallel gzip compression
//!
//! This module implements parallel compression using rayon and memory-mapped I/O.
//! For large files, we use block-based parallel compression where each block
//! is a complete gzip member that can be concatenated.
//!
//! Key optimizations:
//! - Memory-mapped files for zero-copy access (no read_to_end latency)
//! - Global thread pool to avoid per-call initialization
//! - Thread-local buffer reuse to minimize allocations
//! - 128KB fixed blocks (matches pigz default)
//! - Level adjustment for zlib-ng (L1→L2 for better compression ratio)
//! - BGZF-style block size markers in FEXTRA for fast parallel decompression

use flate2::write::GzEncoder;
use flate2::Compression;
use memmap2::Mmap;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::OnceLock;

/// BGZF-style subfield ID for block size markers
/// Using "RZ" to identify rigz-compressed blocks
pub const RIGZ_SUBFIELD_ID: [u8; 2] = [b'R', b'Z'];

/// Adjust compression level for zlib-ng compatibility
///
/// zlib-ng's level 1 uses a faster but much less effective strategy than standard zlib.
/// This produces files 2-5x larger than expected for repetitive data.
/// We map level 1 to level 2 which gives similar speed but much better ratios.
#[inline]
fn adjust_compression_level(level: u32) -> u32 {
    if level == 1 {
        2
    } else {
        level
    }
}

// Thread-local compression buffer to avoid per-block allocation
thread_local! {
    static COMPRESS_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(256 * 1024));
}

/// Default block size for parallel compression (128KB like pigz)
const DEFAULT_BLOCK_SIZE: usize = 128 * 1024;

/// Global thread pool to avoid per-call initialization overhead
static THREAD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn get_thread_pool(num_threads: usize) -> &'static rayon::ThreadPool {
    THREAD_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool")
    })
}

/// Parallel gzip compression using rayon
pub struct ParallelGzEncoder {
    compression_level: u32,
    num_threads: usize,
}

impl ParallelGzEncoder {
    pub fn new(compression_level: u32, num_threads: usize) -> Self {
        Self {
            compression_level,
            num_threads,
        }
    }

    /// Compress data in parallel and write to output
    pub fn compress<R: Read, W: Write>(&self, mut reader: R, mut writer: W) -> io::Result<u64> {
        // Read all input data
        let mut input_data = Vec::new();
        let bytes_read = reader.read_to_end(&mut input_data)? as u64;

        if input_data.is_empty() {
            // Write empty gzip file
            let encoder = GzEncoder::new(
                &mut writer,
                Compression::new(adjust_compression_level(self.compression_level)),
            );
            encoder.finish()?;
            return Ok(0);
        }

        // Calculate optimal block size:
        // - Larger blocks = better compression ratio, less overhead
        // - Smaller blocks = better parallelization for small files
        // - Target: each thread gets at least 2-4 blocks for good load balancing
        let block_size = self.calculate_block_size(input_data.len());

        // For small files or single thread, use simple streaming compression
        if input_data.len() <= block_size || self.num_threads == 1 {
            let mut encoder = GzEncoder::new(
                &mut writer,
                Compression::new(adjust_compression_level(self.compression_level)),
            );
            encoder.write_all(&input_data)?;
            encoder.finish()?;
            return Ok(bytes_read);
        }

        // Large file with multiple threads: compress blocks in parallel
        // Each block becomes a complete gzip member (gzip allows concatenation)
        let blocks: Vec<&[u8]> = input_data.chunks(block_size).collect();

        // Use global thread pool to avoid per-call initialization
        let pool = get_thread_pool(self.num_threads);
        let compression_level = adjust_compression_level(self.compression_level);

        // Compress blocks in parallel using thread-local buffers
        let compressed_blocks: Vec<Vec<u8>> = pool.install(|| {
            blocks
                .par_iter()
                .map(|block| compress_block_with_reuse(block, compression_level))
                .collect()
        });

        // Concatenate all gzip members (valid per RFC 1952)
        for block in &compressed_blocks {
            writer.write_all(block)?;
        }

        Ok(bytes_read)
    }

    /// Calculate optimal block size based on file size and thread count
    fn calculate_block_size(&self, _file_size: usize) -> usize {
        // Use fixed 128KB blocks like pigz
        // This provides consistent parallelism and compression characteristics
        // regardless of file size or thread count
        DEFAULT_BLOCK_SIZE
    }

    /// Compress a file using memory-mapped I/O for zero-copy access
    /// This eliminates the latency of reading the file into memory before compression
    pub fn compress_file<P: AsRef<Path>, W: Write>(
        &self,
        path: P,
        mut writer: W,
    ) -> io::Result<u64> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;

        if file_len == 0 {
            // Write empty gzip file
            let encoder = GzEncoder::new(
                &mut writer,
                Compression::new(adjust_compression_level(self.compression_level)),
            );
            encoder.finish()?;
            return Ok(0);
        }

        // Memory-map the file for zero-copy access
        // This is safe because we only read the file, and it's opened with read-only access
        let mmap = unsafe { Mmap::map(&file)? };

        // For small files or single thread, use simple streaming compression
        let block_size = self.calculate_block_size(file_len);
        if file_len <= block_size || self.num_threads == 1 {
            let mut encoder = GzEncoder::new(
                &mut writer,
                Compression::new(adjust_compression_level(self.compression_level)),
            );
            encoder.write_all(&mmap)?;
            encoder.finish()?;
            return Ok(file_len as u64);
        }

        // Large file with multiple threads: compress blocks in parallel
        // Each block becomes a complete gzip member (gzip allows concatenation)
        let blocks: Vec<&[u8]> = mmap.chunks(block_size).collect();

        // Use global thread pool to avoid per-call initialization
        let pool = get_thread_pool(self.num_threads);
        let compression_level = adjust_compression_level(self.compression_level);

        // Compress blocks in parallel using thread-local buffers
        let compressed_blocks: Vec<Vec<u8>> = pool.install(|| {
            blocks
                .par_iter()
                .map(|block| compress_block_with_reuse(block, compression_level))
                .collect()
        });

        // Write all gzip members using vectorized I/O when possible
        // This reduces system calls by writing multiple blocks at once
        write_compressed_blocks(&compressed_blocks, &mut writer)?;

        Ok(file_len as u64)
    }
}

/// Write compressed blocks efficiently
/// Uses vectorized I/O (write_all_vectored) when available to reduce syscalls
#[inline]
fn write_compressed_blocks<W: Write>(blocks: &[Vec<u8>], writer: &mut W) -> io::Result<()> {
    use std::io::IoSlice;

    // For small number of blocks, just write sequentially
    if blocks.len() <= 4 {
        for block in blocks {
            writer.write_all(block)?;
        }
        return Ok(());
    }

    // Use vectorized write for many blocks (reduces syscalls)
    // Process in batches of up to 64 IoSlices (typical OS limit)
    const MAX_IOVECS: usize = 64;

    for chunk in blocks.chunks(MAX_IOVECS) {
        let slices: Vec<IoSlice<'_>> = chunk.iter().map(|b| IoSlice::new(b)).collect();

        // write_all_vectored handles partial writes
        let mut remaining = &slices[..];
        while !remaining.is_empty() {
            let written = writer.write_vectored(remaining)?;
            if written == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write compressed data",
                ));
            }

            // Advance past written data
            let mut bytes_left = written;
            let mut consumed = 0;
            for slice in remaining.iter() {
                if bytes_left >= slice.len() {
                    bytes_left -= slice.len();
                    consumed += 1;
                } else {
                    break;
                }
            }
            remaining = &remaining[consumed..];

            // If we didn't consume complete slices, fall back to sequential
            if bytes_left > 0 && !remaining.is_empty() {
                // Partial write - finish the rest of this slice
                writer.write_all(&remaining[0][bytes_left..])?;
                remaining = &remaining[1..];
                // Continue with remaining complete slices
                for slice in remaining {
                    writer.write_all(slice)?;
                }
                break;
            }
        }
    }

    Ok(())
}

/// Compress a single block using thread-local buffer to minimize allocations
/// Writes BGZF-style header with block size in FEXTRA for fast parallel decompression
#[inline]
fn compress_block_with_reuse(block: &[u8], compression_level: u32) -> Vec<u8> {
    COMPRESS_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();

        // Compress with BGZF-style header (includes block size marker)
        compress_block_bgzf(&mut *buf, block, compression_level);

        // Return a copy (buffer stays allocated for next use)
        buf.clone()
    })
}

/// Compress a block with BGZF-style gzip header containing block size
///
/// The header includes:
/// - Standard gzip magic (0x1f 0x8b)
/// - FEXTRA flag set (0x04)
/// - "RZ" subfield with compressed block size (allows parallel decompression)
///
/// This is compatible with all gzip decompressors (they ignore unknown subfields)
/// but enables rigz to find block boundaries without inflating.
fn compress_block_bgzf(output: &mut Vec<u8>, block: &[u8], compression_level: u32) {
    use crc32fast::Hasher;
    use flate2::Compress;
    use flate2::FlushCompress;
    use flate2::Status;

    // Reserve space for header (we'll write block size later)
    let header_start = output.len();

    // Write gzip header with FEXTRA flag
    // 10 bytes base header + 6 bytes XLEN + subfield
    output.extend_from_slice(&[
        0x1f, 0x8b, // Magic
        0x08,       // Compression method (deflate)
        0x04,       // Flags: FEXTRA
        0, 0, 0, 0, // MTIME (zero)
        0x00,       // XFL (no extra flags)
        0xff,       // OS (unknown)
    ]);

    // XLEN: 6 bytes (2 byte ID + 2 byte len + 2 byte block size)
    output.extend_from_slice(&[6, 0]);

    // Subfield: "RZ" + 2 bytes len + 2 bytes block size (placeholder)
    output.extend_from_slice(&RIGZ_SUBFIELD_ID);
    output.extend_from_slice(&[2, 0]); // Subfield data length
    let block_size_offset = output.len();
    output.extend_from_slice(&[0, 0]); // Placeholder for block size

    // Compress the data
    let mut compress = Compress::new(Compression::new(compression_level), false);
    let deflate_start = output.len();

    // Reserve space for compressed data (worst case: slightly larger than input)
    let max_compressed_size = block.len() + (block.len() >> 12) + (block.len() >> 14) + 11 + 1024;
    output.resize(deflate_start + max_compressed_size, 0);

    let status = compress
        .compress(block, &mut output[deflate_start..], FlushCompress::Finish)
        .expect("compression failed");

    if status != Status::StreamEnd {
        // Shouldn't happen with Finish flush, but handle gracefully
        panic!("Unexpected compression status: {:?}", status);
    }

    let compressed_len = compress.total_out() as usize;
    output.truncate(deflate_start + compressed_len);

    // Compute CRC32 of uncompressed data
    let mut hasher = Hasher::new();
    hasher.update(block);
    let crc32 = hasher.finalize();

    // Write gzip trailer: CRC32 + ISIZE (uncompressed size mod 2^32)
    output.extend_from_slice(&crc32.to_le_bytes());
    output.extend_from_slice(&(block.len() as u32).to_le_bytes());

    // Now write the total block size (including header and trailer)
    let total_block_size = output.len() - header_start;

    // Block size is stored as (size - 1) to match BGZF convention
    // This allows block sizes up to 65536 to fit in 16 bits
    let block_size_minus_1 = (total_block_size - 1) as u16;
    output[block_size_offset..block_size_offset + 2]
        .copy_from_slice(&block_size_minus_1.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parallel_compress_small() {
        let data = b"Hello, world!";
        let encoder = ParallelGzEncoder::new(6, 4);

        let mut output = Vec::new();
        encoder
            .compress(Cursor::new(&data[..]), &mut output)
            .unwrap();

        // Verify output is valid gzip by decompressing
        let mut decoder = flate2::read::GzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_parallel_compress_large() {
        let data = b"Hello, world! ".repeat(100000); // ~1.4MB
        let encoder = ParallelGzEncoder::new(6, 4);

        let mut output = Vec::new();
        encoder.compress(Cursor::new(&data), &mut output).unwrap();

        // Verify output is valid gzip by decompressing
        // Note: flate2's GzDecoder handles concatenated gzip members
        let mut decoder = flate2::read::MultiGzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }
}
