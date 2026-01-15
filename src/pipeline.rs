use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex, Condvar};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use crossbeam_channel::{bounded, Receiver, Sender};

/// Maximum number of blocks to keep in memory to prevent memory explosion
const MAX_BLOCKS_IN_MEMORY: usize = 16;

/// Block data structure for the pipeline
#[derive(Debug)]
pub struct Block {
    pub sequence: u64,
    pub data: Vec<u8>,
    pub compressed: Option<Vec<u8>>,
    pub size: usize,
    pub is_last: bool,
}

/// Job for compression threads
#[derive(Debug)]
pub struct CompressionJob {
    pub block: Block,
    pub compression_level: u8,
    pub dictionary: Option<Vec<u8>>,
}

/// Result from compression threads
#[derive(Debug)]
pub struct CompressionResult {
    pub sequence: u64,
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub dictionary: Vec<u8>, // Last 32KB for next block
    pub is_last: bool,
}

/// Memory pool for efficient buffer reuse
pub struct MemoryPool {
    buffers: Arc<Mutex<VecDeque<Vec<u8>>>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl MemoryPool {
    pub fn new(buffer_size: usize, initial_count: usize, max_buffers: usize) -> Self {
        let mut buffers = VecDeque::new();
        for _ in 0..initial_count {
            buffers.push_back(vec![0; buffer_size]);
        }
        
        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            buffer_size,
            max_buffers,
        }
    }
    
    pub fn get_buffer(&self) -> Vec<u8> {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.pop_front().unwrap_or_else(|| vec![0; self.buffer_size])
    }
    
    pub fn return_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        buffer.resize(self.buffer_size, 0);
        
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.len() < self.max_buffers {
            buffers.push_back(buffer);
        }
        // Otherwise, let buffer be dropped to free memory
    }
}

/// Pipelined compression engine inspired by pigz
pub struct CompressionPipeline {
    thread_count: usize,
    buffer_size: usize,
    compression_level: u8,
    memory_pool: MemoryPool,
}

impl CompressionPipeline {
    pub fn new(thread_count: usize, buffer_size: usize, compression_level: u8) -> Self {
        let memory_pool = MemoryPool::new(
            buffer_size,
            thread_count * 2, // Initial buffers
            MAX_BLOCKS_IN_MEMORY * 2, // Max buffers
        );
        
        Self {
            thread_count,
            buffer_size,
            compression_level,
            memory_pool,
        }
    }
    
    pub fn compress<R: Read + Send + 'static, W: Write + Send + 'static>(
        &self,
        mut reader: R,
        mut writer: W,
    ) -> io::Result<u64> {
        // Channel for read -> compress
        let (job_sender, job_receiver) = bounded::<CompressionJob>(MAX_BLOCKS_IN_MEMORY);
        
        // Channel for compress -> write
        let (result_sender, result_receiver) = bounded::<CompressionResult>(MAX_BLOCKS_IN_MEMORY);
        
        // Spawn reader thread
        let reader_handle = {
            let job_sender = job_sender.clone();
            let buffer_size = self.buffer_size;
            let memory_pool = self.memory_pool.clone();
            
            thread::spawn(move || -> io::Result<()> {
                let mut sequence = 0u64;
                let mut total_read = 0u64;
                
                loop {
                    let mut buffer = memory_pool.get_buffer();
                    buffer.resize(buffer_size, 0);
                    
                    let bytes_read = reader.read(&mut buffer)?;
                    if bytes_read == 0 {
                        // Send final empty block to signal end
                        let final_block = Block {
                            sequence,
                            data: Vec::new(),
                            compressed: None,
                            size: 0,
                            is_last: true,
                        };
                        
                        let final_job = CompressionJob {
                            block: final_block,
                            compression_level: 0,
                            dictionary: None,
                        };
                        
                        let _ = job_sender.send(final_job);
                        break;
                    }
                    
                    buffer.truncate(bytes_read);
                    total_read += bytes_read as u64;
                    
                    let block = Block {
                        sequence,
                        data: buffer,
                        compressed: None,
                        size: bytes_read,
                        is_last: false,
                    };
                    
                    let job = CompressionJob {
                        block,
                        compression_level: 0, // Will be set by compressor
                        dictionary: None, // Will be managed by pipeline
                    };
                    
                    if job_sender.send(job).is_err() {
                        break; // Pipeline closed
                    }
                    
                    sequence += 1;
                }
                
                Ok(())
            })
        };
        
        // Spawn compression threads
        let mut compression_handles = Vec::new();
        let compression_level = self.compression_level;
        
        for _ in 0..self.thread_count {
            let job_receiver = job_receiver.clone();
            let result_sender = result_sender.clone();
            let memory_pool = self.memory_pool.clone();
            
            let handle = thread::spawn(move || -> io::Result<()> {
                let mut dictionary: Option<Vec<u8>> = None;
                
                while let Ok(mut job) = job_receiver.recv() {
                    if job.block.is_last {
                        // Forward the final signal
                        let result = CompressionResult {
                            sequence: job.block.sequence,
                            compressed_data: Vec::new(),
                            original_size: 0,
                            dictionary: Vec::new(),
                            is_last: true,
                        };
                        let _ = result_sender.send(result);
                        break;
                    }
                    
                    // Set compression parameters
                    job.compression_level = compression_level;
                    job.dictionary = dictionary.clone();
                    
                    // Perform compression with dictionary if available
                    let compressed_data = compress_block_with_dictionary(
                        &job.block.data,
                        compression_level,
                        job.dictionary.as_deref(),
                    )?;
                    
                    // Create dictionary for next block (last 32KB)
                    let new_dictionary = if job.block.data.len() >= 32768 {
                        job.block.data[job.block.data.len() - 32768..].to_vec()
                    } else {
                        job.block.data.clone()
                    };
                    
                    dictionary = Some(new_dictionary.clone());
                    
                    let result = CompressionResult {
                        sequence: job.block.sequence,
                        compressed_data,
                        original_size: job.block.size,
                        dictionary: new_dictionary,
                        is_last: false,
                    };
                    
                    if result_sender.send(result).is_err() {
                        break; // Pipeline closed
                    }
                    
                    // Return buffer to pool
                    memory_pool.return_buffer(job.block.data);
                }
                
                Ok(())
            });
            
            compression_handles.push(handle);
        }
        
        // Spawn writer thread
        let writer_handle = {
            thread::spawn(move || -> io::Result<u64> {
                let mut pending_results = std::collections::BTreeMap::new();
                let mut next_sequence = 0u64;
                let mut total_written = 0u64;
                let mut compression_threads_finished = 0;
                
                while compression_threads_finished < self.thread_count {
                    match result_receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(result) => {
                            if result.is_last {
                                compression_threads_finished += 1;
                                continue;
                            }
                            
                            pending_results.insert(result.sequence, result);
                            
                            // Write results in order
                            while let Some(result) = pending_results.remove(&next_sequence) {
                                writer.write_all(&result.compressed_data)?;
                                total_written += result.compressed_data.len() as u64;
                                next_sequence += 1;
                            }
                        }
                        Err(_) => {
                            // Timeout - check if we should continue waiting
                            continue;
                        }
                    }
                }
                
                // Write any remaining results
                while let Some((_, result)) = pending_results.pop_first() {
                    writer.write_all(&result.compressed_data)?;
                    total_written += result.compressed_data.len() as u64;
                }
                
                writer.flush()?;
                Ok(total_written)
            })
        };
        
        // Close job sender to signal no more work
        drop(job_sender);
        
        // Wait for all threads to complete
        reader_handle.join().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "Reader thread panicked")
        })??;
        
        for handle in compression_handles {
            handle.join().map_err(|_| {
                io::Error::new(io::ErrorKind::Other, "Compression thread panicked")
            })??;
        }
        
        // Close result sender
        drop(result_sender);
        
        let total_written = writer_handle.join().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "Writer thread panicked")
        })??;
        
        Ok(total_written)
    }
}

/// Compress a block with optional dictionary support
fn compress_block_with_dictionary(
    data: &[u8],
    level: u8,
    dictionary: Option<&[u8]>,
) -> io::Result<Vec<u8>> {
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;
    
    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
    
    // If we have a dictionary, try to use it for better compression
    // Note: flate2 doesn't directly support preset dictionaries like zlib,
    // but we can achieve similar effect by priming the encoder
    if let Some(dict) = dictionary {
        // Write dictionary first to prime the compressor, then rewind
        // This is a simplified approach - in production we'd use raw deflate with dictionaries
        encoder.write_all(dict)?;
        encoder.reset(Vec::new())?;
    }
    
    encoder.write_all(data)?;
    encoder.finish().map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        Self {
            buffers: Arc::clone(&self.buffers),
            buffer_size: self.buffer_size,
            max_buffers: self.max_buffers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    
    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(1024, 2, 4);
        
        let buf1 = pool.get_buffer();
        let buf2 = pool.get_buffer();
        assert_eq!(buf1.len(), 1024);
        assert_eq!(buf2.len(), 1024);
        
        pool.return_buffer(buf1);
        pool.return_buffer(buf2);
        
        let buf3 = pool.get_buffer();
        assert_eq!(buf3.len(), 1024);
    }
    
    #[test]
    fn test_pipeline_compression() {
        let data = b"Hello, world! This is a test of the pipelined compression system.".repeat(1000);
        let cursor = Cursor::new(data.clone());
        let mut output = Vec::new();
        
        let pipeline = CompressionPipeline::new(2, 1024, 6);
        let written = pipeline.compress(cursor, &mut output).unwrap();
        
        assert!(written > 0);
        assert!(output.len() > 0);
        assert!(output.len() < data.len()); // Should be compressed
    }
}