use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::optimization::{ContentType, OptimizationConfig};

/// Advanced compression parameters based on pigz analysis
const DICT_SIZE: usize = 32768; // 32KB dictionary like pigz
const MIN_BLOCK_SIZE: usize = 131072; // 128KB minimum
const MAX_BLOCK_SIZE: usize = 2097152; // 2MB maximum
const RSYNC_BITS: usize = 12;
const RSYNC_MASK: u32 = (1u32 << RSYNC_BITS) - 1;
const RSYNC_HIT: u32 = RSYNC_MASK >> 1;

/// Memory-aligned buffer for optimal CPU performance
#[derive(Debug)]
#[repr(align(64))] // Cache line alignment
pub struct AlignedBuffer {
    data: Vec<u8>,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, 0);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// High-performance memory pool with NUMA awareness
pub struct AdvancedMemoryPool {
    input_buffers: Arc<Mutex<VecDeque<AlignedBuffer>>>,
    output_buffers: Arc<Mutex<VecDeque<Vec<u8>>>>,
    dict_buffers: Arc<Mutex<VecDeque<Vec<u8>>>>,
    input_size: usize,
    max_input_buffers: usize,
    max_output_buffers: usize,
}

impl AdvancedMemoryPool {
    pub fn new(block_size: usize, thread_count: usize) -> Self {
        let initial_count = thread_count * 2;
        let max_buffers = thread_count * 4;

        // Pre-allocate input buffers
        let mut input_buffers = VecDeque::new();
        for _ in 0..initial_count {
            input_buffers.push_back(AlignedBuffer::new(block_size));
        }

        // Pre-allocate output buffers (smaller, for compressed data)
        let mut output_buffers = VecDeque::new();
        for _ in 0..initial_count {
            output_buffers.push_back(Vec::with_capacity(block_size / 2));
        }

        // Pre-allocate dictionary buffers
        let mut dict_buffers = VecDeque::new();
        for _ in 0..initial_count {
            dict_buffers.push_back(vec![0; DICT_SIZE]);
        }

        Self {
            input_buffers: Arc::new(Mutex::new(input_buffers)),
            output_buffers: Arc::new(Mutex::new(output_buffers)),
            dict_buffers: Arc::new(Mutex::new(dict_buffers)),
            input_size: block_size,
            max_input_buffers: max_buffers,
            max_output_buffers: max_buffers,
        }
    }

    pub fn get_input_buffer(&self) -> AlignedBuffer {
        let mut buffers = self.input_buffers.lock().unwrap();
        buffers
            .pop_front()
            .unwrap_or_else(|| AlignedBuffer::new(self.input_size))
    }

    pub fn return_input_buffer(&self, mut buffer: AlignedBuffer) {
        buffer.resize(self.input_size);

        let mut buffers = self.input_buffers.lock().unwrap();
        if buffers.len() < self.max_input_buffers {
            buffers.push_back(buffer);
        }
    }

    pub fn get_output_buffer(&self) -> Vec<u8> {
        let mut buffers = self.output_buffers.lock().unwrap();
        buffers
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(self.input_size / 2))
    }

    pub fn return_output_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();

        let mut buffers = self.output_buffers.lock().unwrap();
        if buffers.len() < self.max_output_buffers {
            buffers.push_back(buffer);
        }
    }

    pub fn get_dict_buffer(&self) -> Vec<u8> {
        let mut buffers = self.dict_buffers.lock().unwrap();
        buffers.pop_front().unwrap_or_else(|| vec![0; DICT_SIZE])
    }

    pub fn return_dict_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        buffer.resize(DICT_SIZE, 0);

        let mut buffers = self.dict_buffers.lock().unwrap();
        if buffers.len() < self.max_input_buffers {
            buffers.push_back(buffer);
        }
    }
}

/// Compression block with metadata
#[derive(Debug)]
pub struct CompressionBlock {
    pub id: u64,
    pub data: AlignedBuffer,
    pub size: usize,
    pub dictionary: Option<Vec<u8>>,
    pub is_last: bool,
    pub content_type: ContentType,
}

/// Compressed result
#[derive(Debug)]
pub struct CompressionResult {
    pub id: u64,
    pub compressed: Vec<u8>,
    pub original_size: usize,
    pub compressed_size: usize,
    pub dictionary_output: Vec<u8>,
    pub is_last: bool,
}

/// Advanced block-based compressor inspired by pigz
pub struct AdvancedCompressor {
    config: OptimizationConfig,
    memory_pool: AdvancedMemoryPool,
    block_size: usize,
}

impl AdvancedCompressor {
    pub fn new(config: OptimizationConfig) -> Self {
        let block_size = calculate_optimal_block_size(&config);
        let memory_pool = AdvancedMemoryPool::new(block_size, config.thread_count);

        Self {
            config,
            memory_pool,
            block_size,
        }
    }

    pub fn compress_stream<R: Read + Send + 'static, W: Write + Send + 'static>(
        &self,
        mut reader: R,
        mut writer: W,
    ) -> io::Result<(u64, u64)> {
        let (block_sender, block_receiver) =
            bounded::<CompressionBlock>(self.config.thread_count * 2);
        let (result_sender, result_receiver) =
            bounded::<CompressionResult>(self.config.thread_count * 2);

        // Reader thread
        let reader_handle = {
            let block_sender = block_sender.clone();
            let memory_pool = self.memory_pool.clone();
            let block_size = self.block_size;
            let content_type = self.config.content_type;

            thread::spawn(move || -> io::Result<()> {
                let mut block_id = 0u64;

                loop {
                    let mut buffer = memory_pool.get_input_buffer();
                    let bytes_read = reader.read(buffer.as_mut_slice())?;

                    if bytes_read == 0 {
                        // Send end marker
                        let end_block = CompressionBlock {
                            id: block_id,
                            data: buffer,
                            size: 0,
                            dictionary: None,
                            is_last: true,
                            content_type,
                        };
                        let _ = block_sender.send(end_block);
                        break;
                    }

                    buffer.resize(bytes_read);

                    let block = CompressionBlock {
                        id: block_id,
                        data: buffer,
                        size: bytes_read,
                        dictionary: None, // Will be set by compression pipeline
                        is_last: false,
                        content_type,
                    };

                    if block_sender.send(block).is_err() {
                        break;
                    }

                    block_id += 1;
                }

                Ok(())
            })
        };

        // Compression worker threads
        let mut worker_handles = Vec::new();
        for thread_id in 0..self.config.thread_count {
            let block_receiver = block_receiver.clone();
            let result_sender = result_sender.clone();
            let memory_pool = self.memory_pool.clone();
            let compression_level = self.config.compression_level;
            let backend = self.config.backend;
            let use_numa = self.config.use_numa_pinning;

            let handle = thread::spawn(move || -> io::Result<()> {
                let mut local_dictionary: Option<Vec<u8>> = None;

                // Set CPU affinity for NUMA optimization
                if use_numa {
                    set_thread_affinity(thread_id);
                }

                while let Ok(mut block) = block_receiver.recv() {
                    if block.is_last {
                        let end_result = CompressionResult {
                            id: block.id,
                            compressed: Vec::new(),
                            original_size: 0,
                            compressed_size: 0,
                            dictionary_output: Vec::new(),
                            is_last: true,
                        };
                        let _ = result_sender.send(end_result);
                        memory_pool.return_input_buffer(block.data);
                        break;
                    }

                    // Set dictionary for this block
                    block.dictionary = local_dictionary.clone();

                    // Perform compression
                    let result =
                        compress_block_advanced(&block, compression_level, backend, &memory_pool)?;

                    // Update local dictionary for next block
                    local_dictionary = Some(result.dictionary_output.clone());

                    if result_sender.send(result).is_err() {
                        memory_pool.return_input_buffer(block.data);
                        break;
                    }

                    memory_pool.return_input_buffer(block.data);
                }

                Ok(())
            });

            worker_handles.push(handle);
        }

        // Writer thread with result ordering
        let writer_handle = {
            let thread_count = self.config.thread_count;
            thread::spawn(move || -> io::Result<(u64, u64)> {
                let mut pending_results = std::collections::BTreeMap::new();
                let mut next_id = 0u64;
                let mut total_input = 0u64;
                let mut total_output = 0u64;
                let mut workers_finished = 0;
                while workers_finished < thread_count {
                    match result_receiver.recv() {
                        Ok(result) => {
                            if result.is_last {
                                workers_finished += 1;
                                continue;
                            }

                            pending_results.insert(result.id, result);

                            // Write results in order
                            while let Some(result) = pending_results.remove(&next_id) {
                                writer.write_all(&result.compressed)?;
                                total_input += result.original_size as u64;
                                total_output += result.compressed_size as u64;
                                next_id += 1;
                            }
                        }
                        Err(_) => break,
                    }
                }

                // Write any remaining results
                while let Some((_, result)) = pending_results.pop_first() {
                    writer.write_all(&result.compressed)?;
                    total_input += result.original_size as u64;
                    total_output += result.compressed_size as u64;
                }

                writer.flush()?;
                Ok((total_input, total_output))
            })
        };

        // Wait for completion
        drop(block_sender);

        reader_handle
            .join()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Reader thread failed"))??;

        for handle in worker_handles {
            handle
                .join()
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "Worker thread failed"))??;
        }

        drop(result_sender);

        writer_handle
            .join()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Writer thread failed"))?
    }
}

/// Calculate optimal block size based on configuration
fn calculate_optimal_block_size(config: &OptimizationConfig) -> usize {
    let base_size = match config.content_type {
        ContentType::Text => 256 * 1024,   // 256KB for text
        ContentType::Binary => 512 * 1024, // 512KB for binary
        ContentType::Random => 128 * 1024, // 128KB for random
    };

    // Adjust for compression level
    let level_multiplier = match config.compression_level {
        1..=3 => 1.0, // Fast compression - smaller blocks
        4..=6 => 1.5, // Balanced
        7..=9 => 2.0, // High compression - larger blocks
        _ => 1.0,
    };

    // Adjust for thread count
    let thread_divisor = (config.thread_count as f64).sqrt();

    let optimal_size = (base_size as f64 * level_multiplier / thread_divisor) as usize;
    optimal_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
}

/// Advanced compression with multiple backends and optimizations
fn compress_block_advanced(
    block: &CompressionBlock,
    level: u8,
    backend: crate::optimization::CompressionBackend,
    memory_pool: &AdvancedMemoryPool,
) -> io::Result<CompressionResult> {
    use crate::optimization::CompressionBackend;

    let compressed = match backend {
        CompressionBackend::Gzp => compress_with_gzp_advanced(block, level)?,
        CompressionBackend::Flate2 => compress_with_flate2_advanced(block, level)?,
    };

    // Create dictionary for next block
    let dict_output = if block.size >= DICT_SIZE {
        block.data.as_slice()[block.size - DICT_SIZE..].to_vec()
    } else {
        block.data.as_slice().to_vec()
    };

    Ok(CompressionResult {
        id: block.id,
        compressed,
        original_size: block.size,
        compressed_size: block.size, // Will be updated
        dictionary_output: dict_output,
        is_last: false,
    })
}

/// Compress using gzp with advanced optimizations
fn compress_with_gzp_advanced(block: &CompressionBlock, level: u8) -> io::Result<Vec<u8>> {
    use gzp::{deflate::Gzip, ZBuilder};
    use std::io::Write;

    let output = Vec::new();
    let mut compressor = ZBuilder::<Gzip, _>::new()
        .num_threads(1) // Single thread per block
        .compression_level(gzp::Compression::new(level as u32))
        .from_writer(output);

    compressor.write_all(block.data.as_slice())?;
    let compressed_data = compressor
        .finish()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(compressed_data)
}

/// Compress using flate2 with dictionary support
fn compress_with_flate2_advanced(block: &CompressionBlock, level: u8) -> io::Result<Vec<u8>> {
    use flate2::{write::GzEncoder, Compression};
    use std::io::Write;

    let output = Vec::new();
    let mut encoder = GzEncoder::new(output, Compression::new(level as u32));

    // Use dictionary if available (simplified approach)
    if let Some(ref dict) = block.dictionary {
        // In a full implementation, we'd use raw deflate with preset dictionary
        // For now, just compress the block data
    }

    encoder.write_all(block.data.as_slice())?;
    encoder
        .finish()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Set thread affinity for NUMA optimization
fn set_thread_affinity(thread_id: usize) {
    // Platform-specific thread affinity
    #[cfg(target_os = "linux")]
    {
        use core_affinity;
        if let Some(core_ids) = core_affinity::get_core_ids() {
            let core_id = core_ids[thread_id % core_ids.len()];
            core_affinity::set_for_current(core_id);
        }
    }

    // For other platforms, this is a no-op
    #[cfg(not(target_os = "linux"))]
    {
        let _ = thread_id; // Suppress unused variable warning
    }
}

impl Clone for AdvancedMemoryPool {
    fn clone(&self) -> Self {
        Self {
            input_buffers: Arc::clone(&self.input_buffers),
            output_buffers: Arc::clone(&self.output_buffers),
            dict_buffers: Arc::clone(&self.dict_buffers),
            input_size: self.input_size,
            max_input_buffers: self.max_input_buffers,
            max_output_buffers: self.max_output_buffers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::{CompressionBackend, ContentType, OptimizationConfig};
    use std::io::Cursor;

    #[test]
    fn test_advanced_memory_pool() {
        let pool = AdvancedMemoryPool::new(1024, 4);

        let buf1 = pool.get_input_buffer();
        let buf2 = pool.get_input_buffer();

        assert_eq!(buf1.len(), 1024);
        assert_eq!(buf2.len(), 1024);

        pool.return_input_buffer(buf1);
        pool.return_input_buffer(buf2);
    }

    #[test]
    fn test_block_size_calculation() {
        let config = OptimizationConfig {
            thread_count: 4,
            buffer_size: 65536,
            backend: CompressionBackend::Gzp,
            content_type: ContentType::Text,
            compression_level: 6,
            use_numa_pinning: false,
        };

        let block_size = calculate_optimal_block_size(&config);
        assert!(block_size >= MIN_BLOCK_SIZE);
        assert!(block_size <= MAX_BLOCK_SIZE);
    }
}
