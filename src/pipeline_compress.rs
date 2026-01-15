//! Pipeline-based parallel compression
//!
//! Implements a true producer-consumer pipeline that overlaps I/O with compression,
//! similar to how pigz works. This eliminates the latency of reading all data
//! before starting compression.

use crossbeam_channel::{bounded, Sender, Receiver};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{self, Read, Write};
use std::thread;

/// Block size for pipeline compression (128KB like pigz)
const BLOCK_SIZE: usize = 128 * 1024;

/// Number of blocks to buffer in the pipeline
const PIPELINE_DEPTH: usize = 64;

/// A block of data with its sequence number for ordered output
struct Block {
    seq: usize,
    data: Vec<u8>,
}

/// A compressed block ready for output
struct CompressedBlock {
    seq: usize,
    data: Vec<u8>,
}

/// Pipeline parallel compression using crossbeam channels
pub struct PipelineCompressor {
    compression_level: u32,
    num_threads: usize,
}

impl PipelineCompressor {
    pub fn new(compression_level: u32, num_threads: usize) -> Self {
        Self {
            compression_level,
            num_threads,
        }
    }

    /// Compress using a multi-stage pipeline
    pub fn compress<R: Read + Send + 'static, W: Write>(&self, reader: R, mut writer: W) -> io::Result<u64> {
        let num_threads = self.num_threads;
        let compression_level = self.compression_level;

        // Channel from reader to compression workers
        let (block_tx, block_rx): (Sender<Block>, Receiver<Block>) = bounded(PIPELINE_DEPTH);
        
        // Channel from compression workers to writer
        let (compressed_tx, compressed_rx): (Sender<CompressedBlock>, Receiver<CompressedBlock>) = bounded(PIPELINE_DEPTH);

        // Spawn reader thread
        let reader_handle = thread::spawn(move || {
            read_blocks(reader, block_tx)
        });

        // Spawn compression workers
        let mut worker_handles = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let rx = block_rx.clone();
            let tx = compressed_tx.clone();
            let handle = thread::spawn(move || {
                compress_blocks(rx, tx, compression_level)
            });
            worker_handles.push(handle);
        }
        
        // Drop our copies of the channels so they close when threads finish
        drop(block_rx);
        drop(compressed_tx);

        // Collect and order compressed blocks in the main thread
        let mut pending: std::collections::BTreeMap<usize, Vec<u8>> = std::collections::BTreeMap::new();
        let mut next_seq = 0;
        let mut bytes_written = 0u64;

        for compressed in compressed_rx {
            pending.insert(compressed.seq, compressed.data);
            
            // Write any blocks that are ready in order
            while let Some(data) = pending.remove(&next_seq) {
                writer.write_all(&data)?;
                bytes_written += data.len() as u64;
                next_seq += 1;
            }
        }

        // Wait for reader thread
        let bytes_read = reader_handle.join().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "Reader thread panicked")
        })??;

        // Wait for worker threads
        for handle in worker_handles {
            handle.join().map_err(|_| {
                io::Error::new(io::ErrorKind::Other, "Worker thread panicked")
            })?;
        }

        Ok(bytes_read)
    }
}

/// Reader thread: reads blocks from input and sends to workers
fn read_blocks<R: Read>(mut reader: R, tx: Sender<Block>) -> io::Result<u64> {
    let mut seq = 0;
    let mut total_read = 0u64;

    loop {
        let mut buffer = vec![0u8; BLOCK_SIZE];
        let mut bytes_in_block = 0;
        
        // Fill the block
        while bytes_in_block < BLOCK_SIZE {
            match reader.read(&mut buffer[bytes_in_block..]) {
                Ok(0) => break, // EOF
                Ok(n) => bytes_in_block += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }

        if bytes_in_block == 0 {
            break; // EOF
        }

        buffer.truncate(bytes_in_block);
        total_read += bytes_in_block as u64;

        let block = Block { seq, data: buffer };
        seq += 1;

        // Send block to workers (blocks if pipeline is full)
        if tx.send(block).is_err() {
            break; // Receiver closed, likely due to error in worker
        }
    }

    Ok(total_read)
}

/// Compression worker: receives blocks, compresses them, sends to writer
fn compress_blocks(rx: Receiver<Block>, tx: Sender<CompressedBlock>, level: u32) {
    for block in rx {
        let mut compressed = Vec::with_capacity(block.data.len() + 256);
        
        // Compress as a complete gzip member
        let mut encoder = GzEncoder::new(&mut compressed, Compression::new(level));
        if encoder.write_all(&block.data).is_ok() {
            if encoder.finish().is_ok() {
                let _ = tx.send(CompressedBlock {
                    seq: block.seq,
                    data: compressed,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_pipeline_compress() {
        let data = b"Hello, world! ".repeat(10000); // ~140KB
        let compressor = PipelineCompressor::new(6, 4);

        let mut output = Vec::new();
        compressor.compress(Cursor::new(data.clone()), &mut output).unwrap();

        // Verify output is valid gzip
        let mut decoder = flate2::read::MultiGzDecoder::new(&output[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
    }
}
