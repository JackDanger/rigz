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
use flate2::write::GzEncoder;
use flate2::{Compress, Compression, FlushCompress, Status};
use rayon::prelude::*;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::OnceLock;

/// Block size for pipelined compression
/// 128KB is the default for pigz. Larger blocks reduce sync overhead
/// but don't significantly improve compression.
const BLOCK_SIZE: usize = 128 * 1024;

/// Dictionary size (DEFLATE maximum is 32KB)
const DICT_SIZE: usize = 32 * 1024;

/// Global thread pool for pipelined compression
static PIPELINE_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn get_pipeline_pool(num_threads: usize) -> &'static rayon::ThreadPool {
    PIPELINE_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create pipeline thread pool")
    })
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

    /// Parallel pipelined compression (pigz-style)
    ///
    /// Each block is compressed with dictionary = previous block's input data.
    /// Blocks are compressed in parallel since we have all input data upfront.
    /// Output is collected and written in order.
    fn compress_parallel_pipeline<W: Write>(&self, data: &[u8], mut writer: W) -> io::Result<()> {
        use crc32fast::Hasher;

        let level = adjust_compression_level(self.compression_level);

        // Write gzip header
        let header = [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        writer.write_all(&header)?;

        // Split into blocks
        let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();
        let n_blocks = blocks.len();

        // Create jobs: each job knows its block index and can access previous block for dict
        let pool = get_pipeline_pool(self.num_threads);

        // Compress all blocks in parallel
        // Each block gets the previous block's last 32KB as dictionary
        let compressed_blocks: Vec<Vec<u8>> = pool.install(|| {
            (0..n_blocks)
                .into_par_iter()
                .map(|i| {
                    let block = blocks[i];
                    let dict = if i > 0 {
                        let prev = blocks[i - 1];
                        if prev.len() > DICT_SIZE {
                            Some(&prev[prev.len() - DICT_SIZE..])
                        } else {
                            Some(prev)
                        }
                    } else {
                        None
                    };
                    let is_last = i == n_blocks - 1;
                    compress_block_with_dict(block, dict, level, is_last)
                })
                .collect()
        });

        // Write all compressed blocks in order
        for block in &compressed_blocks {
            writer.write_all(block)?;
        }

        // Compute CRC and write trailer
        let mut crc_hasher = Hasher::new();
        crc_hasher.update(data);
        let crc = crc_hasher.finalize();
        let isize = (data.len() as u32).to_le_bytes();
        writer.write_all(&crc.to_le_bytes())?;
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
        let mut output_buf = vec![0u8; BLOCK_SIZE * 2];
        let mut crc_hasher = Hasher::new();

        let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();

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
fn compress_block_with_dict(
    block: &[u8],
    dict: Option<&[u8]>,
    level: u32,
    is_last: bool,
) -> Vec<u8> {
    let mut compress = Compress::new(Compression::new(level), false);

    // Set dictionary if provided
    if let Some(d) = dict {
        let _ = compress.set_dictionary(d);
    }

    let flush = if is_last {
        FlushCompress::Finish
    } else {
        FlushCompress::Sync
    };

    // Compress the block
    let mut output = vec![0u8; block.len() * 2 + 1024];
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
                // Need more output space
                output.resize(output.len() * 2, 0);
            }
            Status::StreamEnd => break,
            _ => {}
        }
    }

    output.truncate(total_out);
    output
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
