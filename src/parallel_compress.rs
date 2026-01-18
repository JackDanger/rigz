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

/// Adjust compression level for backend compatibility
///
/// For L1-L6, we use libdeflate which doesn't have zlib-ng's L1 RLE issue.
/// For L7-L9, we use zlib-ng which needs L1→L2 mapping.
///
/// However, since L7-L9 never uses L1, this mapping is only relevant for
/// the pipelined compressor which uses zlib-ng directly.
#[inline]
pub fn adjust_compression_level(level: u32) -> u32 {
    // Only needed for zlib-ng (pipelined compressor at L7-L9)
    // Parallel compressor uses libdeflate for L1-L6, no adjustment needed
    if level == 1 {
        2 // Map L1→L2 for zlib-ng (RLE produces 33% larger files)
    } else {
        level
    }
}

/// Get optimal block size based on compression level and file size
///
/// At lower levels (L1-L2), we use larger blocks to reduce per-block overhead
/// since compression is fast and synchronization becomes the bottleneck.
/// At higher levels (L3-L6), we use smaller blocks for better parallelization
/// since compression takes longer and can better utilize the parallelism.
///
/// Block size is also scaled based on file size to ensure enough blocks for
/// parallelism on small files while reducing overhead on large files.
#[inline]
pub fn get_block_size_for_level(level: u32) -> usize {
    match level {
        // L1-L2: Use 128KB blocks as baseline
        // This gives enough parallelism for small files
        // Note: 128KB > BGZF u16 limit, so BGZF markers will be disabled
        1 | 2 => 128 * 1024,
        // L3-L6: Use 64KB blocks - fits BGZF, enables parallel decompression
        _ => DEFAULT_BLOCK_SIZE,
    }
}

/// Get optimal block size considering both level and file size
/// This ensures we have enough blocks for parallelism (minimum 4*num_threads)
/// while not having too many blocks that synchronization overhead dominates.
#[inline]
pub fn get_optimal_block_size(level: u32, file_size: usize, num_threads: usize) -> usize {
    let base_block_size = get_block_size_for_level(level);
    
    // For L1-L2, dynamically size blocks based on file size
    // Goal: ~8 blocks per thread for good load balancing, max 256KB blocks
    if level <= 2 {
        let target_blocks = num_threads * 8;
        let dynamic_size = file_size / target_blocks;
        // Clamp between 64KB (minimum for efficiency) and 256KB (maximum for L1-L2)
        let clamped = dynamic_size.max(64 * 1024).min(256 * 1024);
        // Round up to 64KB boundary for alignment
        (clamped + 65535) & !65535
    } else {
        base_block_size
    }
}

// Thread-local compression buffer to avoid per-block allocation
thread_local! {
    static COMPRESS_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(256 * 1024));
    // Cache libdeflate Compressor by level to avoid per-block allocation
    // Tuple is (level, compressor) - we only cache one level per thread
    static LIBDEFLATE_COMPRESSOR: RefCell<Option<(i32, libdeflater::Compressor)>> = RefCell::new(None);
}

/// Default block size for parallel compression (128KB like pigz)
/// Block size for parallel compression
/// BGZF format stores block size as u16, so max is 65535 bytes
/// We use 64KB to stay within this limit while maximizing parallelism
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

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
        // Don't adjust level here - libdeflate (L1-L6) handles L1 correctly
        // The adjust_compression_level is only for zlib-ng paths (single-threaded, empty)
        let compression_level = self.compression_level;

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
    fn calculate_block_size(&self, file_size: usize) -> usize {
        // Use level and file-size aware block sizing:
        // - L1-L2: Dynamic (64KB-256KB based on file size)
        // - L3-L6: 64KB (fits BGZF, enables parallel decompression)
        get_optimal_block_size(self.compression_level, file_size, self.num_threads)
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
        // Don't adjust level here - libdeflate (L1-L6) handles L1 correctly
        // The adjust_compression_level is only for zlib-ng paths (single-threaded, empty)
        let compression_level = self.compression_level;

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
        compress_block_bgzf(&mut buf, block, compression_level);

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
///
/// Uses libdeflate for L1-L6 (faster, no dictionary needed).
/// L7-L9 use pipelined compression (separate module) with zlib-ng for dictionary support.
fn compress_block_bgzf(output: &mut Vec<u8>, block: &[u8], compression_level: u32) {
    // libdeflate is faster than zlib-ng for L1-L6
    // L7-L9 shouldn't reach here - they use pipelined compression
    compress_block_bgzf_libdeflate(output, block, compression_level);
}

/// Compress using libdeflate (faster for L1-L6)
/// Uses thread-local Compressor cache to avoid per-block allocation.
fn compress_block_bgzf_libdeflate(output: &mut Vec<u8>, block: &[u8], compression_level: u32) {
    use libdeflater::{CompressionLvl, Compressor};

    // Reserve space for header (we'll write block size later)
    let header_start = output.len();

    // Write gzip header with FEXTRA flag
    output.extend_from_slice(&[
        0x1f, 0x8b, // Magic
        0x08, // Compression method (deflate)
        0x04, // Flags: FEXTRA
        0, 0, 0, 0,    // MTIME (zero)
        0x00, // XFL (no extra flags)
        0xff, // OS (unknown)
    ]);

    // XLEN: 6 bytes (2 byte ID + 2 byte len + 2 byte block size)
    output.extend_from_slice(&[6, 0]);

    // Subfield: "RZ" + 2 bytes len + 2 bytes block size (placeholder)
    output.extend_from_slice(&RIGZ_SUBFIELD_ID);
    output.extend_from_slice(&[2, 0]); // Subfield data length
    let block_size_offset = output.len();
    output.extend_from_slice(&[0, 0]); // Placeholder for block size

    // Get or create compressor from thread-local cache
    let level = compression_level as i32;
    let (max_compressed_size, compressed_len) = LIBDEFLATE_COMPRESSOR.with(|cache| {
        let mut cache = cache.borrow_mut();
        
        // Check if cached compressor matches our level
        let compressor = match cache.as_mut() {
            Some((cached_level, comp)) if *cached_level == level => comp,
            _ => {
                // Create new compressor for this level
                let lvl = CompressionLvl::new(level).unwrap_or(CompressionLvl::default());
                *cache = Some((level, Compressor::new(lvl)));
                &mut cache.as_mut().unwrap().1
            }
        };
        
        let max_size = compressor.deflate_compress_bound(block.len());
        (max_size, max_size) // Return max_size twice to use outside closure
    });
    
    // Resize output buffer
    let deflate_start = output.len();
    output.resize(deflate_start + max_compressed_size, 0);
    
    // Do the actual compression (need to access compressor again)
    let actual_len = LIBDEFLATE_COMPRESSOR.with(|cache| {
        let mut cache = cache.borrow_mut();
        let compressor = &mut cache.as_mut().unwrap().1;
        compressor
            .deflate_compress(block, &mut output[deflate_start..])
            .expect("libdeflate compression failed")
    });

    output.truncate(deflate_start + actual_len);

    // Compute CRC32 of uncompressed data
    let crc32 = crc32fast::hash(block);

    // Write gzip trailer: CRC32 + ISIZE (uncompressed size mod 2^32)
    output.extend_from_slice(&crc32.to_le_bytes());
    output.extend_from_slice(&(block.len() as u32).to_le_bytes());

    // Now write the total block size (including header and trailer)
    let total_block_size = output.len() - header_start;

    // Block size is stored as (size - 1) to match BGZF convention
    let block_size_minus_1 = if total_block_size <= 65536 {
        (total_block_size - 1) as u16
    } else {
        0 // Overflow marker - decompressor will fall back to sequential
    };
    output[block_size_offset..block_size_offset + 2]
        .copy_from_slice(&block_size_minus_1.to_le_bytes());
}

/// Compress using flate2/zlib-ng (better for L7+)
fn compress_block_bgzf_flate2(output: &mut Vec<u8>, block: &[u8], compression_level: u32) {
    use flate2::Compress;
    use flate2::FlushCompress;
    use flate2::Status;

    // Reserve space for header (we'll write block size later)
    let header_start = output.len();

    // Write gzip header with FEXTRA flag
    output.extend_from_slice(&[
        0x1f, 0x8b, // Magic
        0x08, // Compression method (deflate)
        0x04, // Flags: FEXTRA
        0, 0, 0, 0,    // MTIME (zero)
        0x00, // XFL (no extra flags)
        0xff, // OS (unknown)
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
        panic!("Unexpected compression status: {:?}", status);
    }

    let compressed_len = compress.total_out() as usize;
    output.truncate(deflate_start + compressed_len);

    // Compute CRC32 of uncompressed data
    let crc32 = crc32fast::hash(block);

    // Write gzip trailer: CRC32 + ISIZE (uncompressed size mod 2^32)
    output.extend_from_slice(&crc32.to_le_bytes());
    output.extend_from_slice(&(block.len() as u32).to_le_bytes());

    // Now write the total block size (including header and trailer)
    let total_block_size = output.len() - header_start;

    // Block size is stored as (size - 1) to match BGZF convention
    let block_size_minus_1 = if total_block_size <= 65536 {
        (total_block_size - 1) as u16
    } else {
        0 // Overflow marker - decompressor will fall back to sequential
    };
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
