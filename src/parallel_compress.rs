//! High-performance parallel gzip compression
//!
//! This module implements parallel compression using memory-mapped I/O and
//! our custom zero-overhead scheduler (no rayon).
//!
//! For large files, we use block-based parallel compression where each block
//! is a complete gzip member that can be concatenated.
//!
//! Key optimizations:
//! - Memory-mapped files for zero-copy access (no read_to_end latency)
//! - Custom scheduler with streaming output (no bulk collection)
//! - Thread-local buffer reuse to minimize allocations
//! - 128KB fixed blocks (matches pigz default)
//! - libdeflate for L1-L6 (30-50% faster than zlib-ng)
//! - BGZF-style block size markers in FEXTRA for fast parallel decompression

use crate::scheduler::compress_parallel_independent;
use flate2::write::GzEncoder;
use flate2::Compression;
use memmap2::Mmap;
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

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

/// Default block size for parallel compression
/// BGZF format stores block size as u16, so max is 65535 bytes
/// We use 64KB to stay within this limit while maximizing parallelism
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

/// Parallel gzip compression using custom scheduler
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

        // Calculate optimal block size
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
        self.compress_parallel(&input_data, block_size, &mut writer)?;

        Ok(bytes_read)
    }

    /// Calculate optimal block size based on file size and thread count
    fn calculate_block_size(&self, file_size: usize) -> usize {
        get_optimal_block_size(self.compression_level, file_size, self.num_threads)
    }

    /// Compress a file using memory-mapped I/O for zero-copy access
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
        self.compress_parallel(&mmap, block_size, &mut writer)?;

        Ok(file_len as u64)
    }

    /// Parallel compression using custom scheduler with streaming output
    fn compress_parallel<W: Write>(
        &self,
        data: &[u8],
        block_size: usize,
        writer: &mut W,
    ) -> io::Result<()> {
        let compression_level = self.compression_level;

        // Use custom scheduler - no rayon overhead, streaming output
        compress_parallel_independent(
            data,
            block_size,
            self.num_threads,
            writer,
            |block, output| {
                compress_block_bgzf_libdeflate(output, block, compression_level);
            },
        )
    }
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
fn compress_block_bgzf_libdeflate(output: &mut Vec<u8>, block: &[u8], compression_level: u32) {
    use libdeflater::{CompressionLvl, Compressor};

    output.clear();

    // Reserve space for header (we'll write block size later)
    let header_start = output.len();

    // Write gzip header with FEXTRA flag
    output.extend_from_slice(&[
        0x1f, 0x8b, // Magic
        0x08, // Compression method (deflate)
        0x04, // Flags: FEXTRA
        0, 0, 0, 0, // MTIME (zero)
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
    let max_compressed_size = LIBDEFLATE_COMPRESSOR.with(|cache| {
        let mut cache = cache.borrow_mut();

        let compressor = match cache.as_mut() {
            Some((cached_level, comp)) if *cached_level == level => comp,
            _ => {
                let lvl = CompressionLvl::new(level).unwrap_or(CompressionLvl::default());
                *cache = Some((level, Compressor::new(lvl)));
                &mut cache.as_mut().unwrap().1
            }
        };

        compressor.deflate_compress_bound(block.len())
    });

    // Resize output buffer
    let deflate_start = output.len();
    output.resize(deflate_start + max_compressed_size, 0);

    // Do the actual compression
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
