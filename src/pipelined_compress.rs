//! Pipelined compression with dictionary sharing for maximum compression
//!
//! At high compression levels (L7-L9), users expect maximum compression ratio.
//! This module implements pigz-style pipelined compression where each block
//! uses the previous block's data as a dictionary.
//!
//! Key insight from pigz: blocks can be compressed in PARALLEL because
//! block N only needs block N-1's INPUT data as dictionary, not its output.
//! Since all input is pre-read (mmap), we can pipeline efficiently.
//!
//! Trade-off:
//! - Better compression (matches pigz output size)
//! - Sequential decompression only (like pigz)
//!
//! This is used when compression_level >= 7 and threads > 1.

use crate::parallel_compress::adjust_compression_level;
use crate::scheduler::compress_parallel;
use flate2::write::GzEncoder;
use flate2::{Compress, Compression, FlushCompress, Status};
use std::cell::{RefCell, UnsafeCell};
use std::io::{self, Read, Write};
use std::path::Path;
use std::mem::MaybeUninit;

// Thread-local Compress object to avoid reinitializing zlib state (~300KB) per block
// Note: We store Option<(level, Compress)> to cache by level
thread_local! {
    static PIPELINED_COMPRESS: RefCell<Option<(u32, Compress)>> = RefCell::new(None);
}

/// Default block size for pipelined compression - matches pigz (128KB)
const DEFAULT_BLOCK_SIZE: usize = 128 * 1024;

/// Dictionary size (DEFLATE maximum is 32KB)
const DICT_SIZE: usize = 32 * 1024;

#[inline]
fn pipelined_block_size(input_len: usize, num_threads: usize, level: u32) -> usize {
    if level >= 9 && num_threads > 1 && input_len > 0 {
        let blocks = num_threads.saturating_mul(2).max(1);
        let target = input_len.div_ceil(blocks);
        return target.clamp(256 * 1024, 4 * 1024 * 1024);
    }

    DEFAULT_BLOCK_SIZE
}

/// Pipelined gzip compression with dictionary sharing
///
/// This produces a single gzip member with dictionary sharing between
/// internal blocks, achieving compression ratios comparable to pigz.
///
/// The output is gzip-compatible but requires sequential decompression.
pub struct PipelinedGzEncoder {
    compression_level: u32,
    num_threads: usize,
}

impl PipelinedGzEncoder {
    pub fn new(compression_level: u32, num_threads: usize) -> Self {
        Self {
            compression_level,
            num_threads,
        }
    }

    /// Compress data with dictionary sharing
    pub fn compress<R: Read, W: Write>(&self, mut reader: R, writer: W) -> io::Result<u64> {
        // Read all input data
        let mut input_data = Vec::new();
        let bytes_read = reader.read_to_end(&mut input_data)? as u64;

        if input_data.is_empty() {
            // Write empty gzip file
            let encoder = GzEncoder::new(writer, Compression::new(self.compression_level));
            encoder.finish()?;
            return Ok(0);
        }

        if self.num_threads > 1 {
            self.compress_parallel_pipeline(&input_data, writer)?;
        } else {
            self.compress_sequential(&input_data, writer)?;
        }
        Ok(bytes_read)
    }

    /// Compress file using memory-mapped I/O with dictionary sharing
    pub fn compress_file<P: AsRef<Path>, W: Write>(&self, path: P, writer: W) -> io::Result<u64> {
        use memmap2::Mmap;
        use std::fs::File;

        let file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len() as usize;

        if file_len == 0 {
            let encoder = GzEncoder::new(writer, Compression::new(self.compression_level));
            encoder.finish()?;
            return Ok(0);
        }

        // Memory-map the file for zero-copy access
        let mmap = unsafe { Mmap::map(&file)? };

        if self.num_threads > 1 {
            self.compress_parallel_pipeline(&mmap, writer)?;
        } else {
            self.compress_sequential(&mmap, writer)?;
        }
        Ok(file_len as u64)
    }

    /// Parallel pipelined compression using custom scheduler
    ///
    /// Each block is compressed with dictionary = previous block's input data.
    /// Blocks are compressed in parallel since we have all input data upfront.
    /// Output is streamed in order as blocks complete.
    ///
    /// Uses our custom scheduler instead of rayon for:
    /// - Zero work-stealing overhead (uniform block sizes)
    /// - Streaming output (no bulk collection)
    /// - Pre-allocated buffers (no allocation in hot path)
    fn compress_parallel_pipeline<W: Write>(&self, data: &[u8], mut writer: W) -> io::Result<()> {
        let level = adjust_compression_level(self.compression_level);
        let block_size = pipelined_block_size(data.len(), self.num_threads, level);
        let num_blocks = data.len().div_ceil(block_size);

        // Write gzip header
        let header = [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        writer.write_all(&header)?;

        // Use a CRC combiner to compute final CRC
        // Each block's CRC is computed in parallel, then combined
        struct CrcSlot(UnsafeCell<MaybeUninit<crc32fast::Hasher>>);
        // Safety: each slot is written by exactly one worker before all threads join.
        unsafe impl Sync for CrcSlot {}
        let crc_parts: Vec<CrcSlot> = (0..num_blocks)
            .map(|_| CrcSlot(UnsafeCell::new(MaybeUninit::uninit())))
            .collect();

        // Compress all blocks using custom scheduler
        compress_parallel(
            data,
            block_size,
            self.num_threads,
            &mut writer,
            |block_idx, block, dict, is_last, output| {
                // Compress this block with dictionary
                compress_block_with_dict(block, dict, level, block_size, is_last, output);
                
                // Compute CRC for this block
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(block);
                unsafe {
                    *crc_parts[block_idx].0.get() = MaybeUninit::new(hasher);
                }
            },
        )?;

        // Combine CRCs in order
        let mut combined_hasher = crc32fast::Hasher::new();
        for part in &crc_parts {
            let hasher = unsafe { (*part.0.get()).assume_init_read() };
            combined_hasher.combine(&hasher);
        }
        let combined_crc = combined_hasher.finalize();

        // Write gzip trailer
        let isize = (data.len() as u32).to_le_bytes();
        writer.write_all(&combined_crc.to_le_bytes())?;
        writer.write_all(&isize)?;

        Ok(())
    }

    /// Sequential compression (single-threaded, for -p1)
    fn compress_sequential<W: Write>(&self, data: &[u8], mut writer: W) -> io::Result<()> {
        use crc32fast::Hasher;

        let level = adjust_compression_level(self.compression_level);

        // Write gzip header
        let header = [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        writer.write_all(&header)?;

        let mut compress = Compress::new(Compression::new(level), false);
        let block_size = pipelined_block_size(data.len(), 1, level);
        let mut output_buf = vec![0u8; block_size * 2];
        let mut crc_hasher = Hasher::new();

        let blocks: Vec<&[u8]> = data.chunks(block_size).collect();

        for (i, block) in blocks.iter().enumerate() {
            crc_hasher.update(block);

            // Set dictionary from previous block
            if i > 0 {
                let prev = blocks[i - 1];
                let dict = if prev.len() > DICT_SIZE {
                    &prev[prev.len() - DICT_SIZE..]
                } else {
                    prev
                };
                let _ = compress.set_dictionary(dict);
            }

            let flush = if i == blocks.len() - 1 {
                FlushCompress::Finish
            } else {
                FlushCompress::Sync
            };

            let mut block_data = *block;
            loop {
                let before_out = compress.total_out();
                let status = compress.compress(block_data, &mut output_buf, flush)?;
                let produced = (compress.total_out() - before_out) as usize;

                if produced > 0 {
                    writer.write_all(&output_buf[..produced])?;
                }

                let before_in = compress.total_in();
                let _ = compress.compress(&[], &mut [], FlushCompress::None);
                let consumed = (compress.total_in() - before_in) as usize;
                if consumed > 0 && consumed <= block_data.len() {
                    block_data = &block_data[consumed..];
                }

                match status {
                    Status::Ok if block_data.is_empty() && flush != FlushCompress::Finish => break,
                    Status::BufError if produced == 0 => break,
                    Status::StreamEnd => break,
                    _ => {}
                }
            }
        }

        // Write trailer
        let crc = crc_hasher.finalize();
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&(data.len() as u32).to_le_bytes())?;

        Ok(())
    }
}

/// Compress a single block with optional dictionary
///
/// Uses thread-local Compress object to avoid per-block allocation.
fn compress_block_with_dict(
    block: &[u8],
    dict: Option<&[u8]>,
    level: u32,
    block_size: usize,
    is_last: bool,
    output: &mut Vec<u8>,
) {
    PIPELINED_COMPRESS.with(|comp_cell| {
        let mut comp_opt = comp_cell.borrow_mut();

        // Ensure buffer is large enough (block_size + 10% + 1KB for headers)
        let initial_capacity = block_size + (block_size / 10) + 1024;
        if output.capacity() < initial_capacity {
            output.reserve(initial_capacity - output.capacity());
            output.resize(initial_capacity, 0);
        } else if output.is_empty() {
            // Initialize once to avoid repeated zero-fill on reuse.
            output.resize(initial_capacity, 0);
        } else {
            // Safe because the buffer has been initialized in previous use.
            unsafe {
                output.set_len(initial_capacity);
            }
        }

        // Get or create Compress at the right level
        let compress = match comp_opt.as_mut() {
            Some((cached_level, comp)) if *cached_level == level => {
                comp.reset();
                comp
            }
            _ => {
                *comp_opt = Some((level, Compress::new(Compression::new(level), false)));
                &mut comp_opt.as_mut().unwrap().1
            }
        };

        // Set dictionary if provided (last 32KB of previous block's input)
        if let Some(d) = dict {
            // Only use last DICT_SIZE bytes
            let dict_slice = if d.len() > DICT_SIZE {
                &d[d.len() - DICT_SIZE..]
            } else {
                d
            };
            let _ = compress.set_dictionary(dict_slice);
        }

        let flush = if is_last {
            FlushCompress::Finish
        } else {
            FlushCompress::Sync
        };

        let mut total_out = 0;
        let mut input = block;

        loop {
            let before_in = compress.total_in();
            let before_out = compress.total_out();

            let status = compress
                .compress(input, &mut output[total_out..], flush)
                .expect("compression failed");

            let consumed = (compress.total_in() - before_in) as usize;
            let produced = (compress.total_out() - before_out) as usize;

            total_out += produced;
            input = &input[consumed..];

            match status {
                Status::Ok if input.is_empty() && flush != FlushCompress::Finish => break,
                Status::BufError => {
                    // Need more output space (rare case for incompressible data)
                    let new_len = output.len() * 2;
                    output.resize(new_len, 0);
                }
                Status::StreamEnd => break,
                _ => {}
            }
        }

        output.truncate(total_out);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Read;

    #[test]
    fn test_pipelined_compress() {
        let data = b"Hello, world! ".repeat(10000);
        let mut output = Vec::new();

        let encoder = PipelinedGzEncoder::new(9, 4);
        encoder
            .compress(std::io::Cursor::new(&data), &mut output)
            .unwrap();

        // Verify we can decompress
        let mut decoder = GzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_pipelined_vs_parallel_size() {
        use crate::parallel_compress::ParallelGzEncoder;

        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(5000);

        // Pipelined (with dictionary)
        let mut pipelined_output = Vec::new();
        let pipelined = PipelinedGzEncoder::new(9, 4);
        pipelined
            .compress(std::io::Cursor::new(&data), &mut pipelined_output)
            .unwrap();

        // Parallel (independent blocks)
        let mut parallel_output = Vec::new();
        let parallel = ParallelGzEncoder::new(9, 4);
        parallel
            .compress(std::io::Cursor::new(&data), &mut parallel_output)
            .unwrap();

        // Pipelined should be smaller (dictionary sharing)
        println!(
            "Pipelined: {} bytes, Parallel: {} bytes",
            pipelined_output.len(),
            parallel_output.len()
        );
        assert!(
            pipelined_output.len() <= parallel_output.len(),
            "Pipelined should produce smaller output"
        );
    }
}
