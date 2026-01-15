use crossbeam_channel::bounded;
use std::io::{self, Read, Write};
use std::thread;

use crate::optimization::{CompressionBackend, ContentType, OptimizationConfig};

/// Core optimization system inspired by pigz architecture
pub struct CoreOptimizer {
    config: OptimizationConfig,
}

/// Job for compression workers
#[derive(Debug)]
struct CompressionJob {
    id: u64,
    data: Vec<u8>,
    dictionary: Option<Vec<u8>>,
}

/// Result from compression
#[derive(Debug)]
struct CompressionResult {
    id: u64,
    compressed: Vec<u8>,
    dictionary_out: Vec<u8>,
}

impl CoreOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// High-performance compression using pigz-inspired patterns
    pub fn compress<R: Read, W: Write>(&self, mut reader: R, mut writer: W) -> io::Result<u64> {
        // For small files or single thread, use direct compression
        if self.config.thread_count == 1 {
            return self.compress_direct(reader, writer);
        }

        // Use multi-threaded compression for larger files
        self.compress_parallel(reader, writer)
    }

    /// Direct single-threaded compression (optimized)
    fn compress_direct<R: Read, W: Write + Send>(
        &self,
        mut reader: R,
        writer: W,
    ) -> io::Result<u64> {
        match self.config.backend {
            CompressionBackend::Gzp => {
                use gzp::{deflate::Gzip, ZBuilder};

                let mut compressor = ZBuilder::<Gzip, _>::new()
                    .num_threads(1)
                    .compression_level(gzp::Compression::new(self.config.compression_level as u32))
                    .buffer_size(self.config.buffer_size)
                    .from_writer(writer);

                let bytes_written = io::copy(&mut reader, &mut compressor)?;
                compressor
                    .finish()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                Ok(bytes_written)
            }
            CompressionBackend::Flate2 => {
                use flate2::{write::GzEncoder, Compression};

                let mut encoder = GzEncoder::new(
                    writer,
                    Compression::new(self.config.compression_level as u32),
                );
                let bytes_written = io::copy(&mut reader, &mut encoder)?;
                encoder.finish()?;
                Ok(bytes_written)
            }
        }
    }

    /// Parallel compression using work-stealing pattern
    fn compress_parallel<R: Read, W: Write>(
        &self,
        mut reader: R,
        mut writer: W,
    ) -> io::Result<u64> {
        let block_size = self.calculate_block_size();
        let (job_sender, job_receiver) = bounded(self.config.thread_count * 2);
        let (result_sender, result_receiver) = bounded(self.config.thread_count * 2);

        // Spawn compression workers
        let mut worker_handles = Vec::new();
        for worker_id in 0..self.config.thread_count {
            let job_receiver = job_receiver.clone();
            let result_sender = result_sender.clone();
            let compression_level = self.config.compression_level;
            let backend = self.config.backend;

            // Set CPU affinity for better cache locality
            let handle = thread::spawn(move || -> io::Result<()> {
                set_worker_affinity(worker_id);

                let mut local_dictionary: Option<Vec<u8>> = None;

                while let Ok(job) = job_receiver.recv() {
                    // Apply dictionary from previous block
                    let mut job: CompressionJob = job;
                    job.dictionary = local_dictionary.clone();

                    // Compress the block
                    let compressed = compress_block_optimized(&job, backend, compression_level)?;

                    // Update dictionary for next block (last 32KB)
                    local_dictionary = if job.data.len() >= 32768 {
                        Some(job.data[job.data.len() - 32768..].to_vec())
                    } else {
                        Some(job.data.clone())
                    };

                    let result = CompressionResult {
                        id: job.id,
                        compressed,
                        dictionary_out: local_dictionary.clone().unwrap_or_default(),
                    };

                    if result_sender.send(result).is_err() {
                        break;
                    }
                }

                Ok(())
            });

            worker_handles.push(handle);
        }

        // Reader thread
        let reader_handle = {
            let job_sender = job_sender.clone();
            thread::spawn(move || -> io::Result<()> {
                let mut job_id = 0u64;
                let mut buffer = vec![0u8; block_size];

                loop {
                    let bytes_read = reader.read(&mut buffer)?;
                    if bytes_read == 0 {
                        break;
                    }

                    let job = CompressionJob {
                        id: job_id,
                        data: buffer[..bytes_read].to_vec(),
                        dictionary: None, // Will be set by worker
                    };

                    if job_sender.send(job).is_err() {
                        break;
                    }

                    job_id += 1;
                }

                Ok(())
            })
        };

        // Writer thread - ensures results are written in order
        let writer_handle = {
            thread::spawn(move || -> io::Result<u64> {
                let mut pending_results = std::collections::BTreeMap::new();
                let mut next_id = 0u64;
                let mut total_written = 0u64;
                let mut workers_finished = 0;

                while workers_finished < self.config.thread_count {
                    match result_receiver.recv() {
                        Ok(result) => {
                            pending_results.insert(result.id, result);

                            // Write results in order
                            while let Some(result) = pending_results.remove(&next_id) {
                                writer.write_all(&result.compressed)?;
                                total_written += result.compressed.len() as u64;
                                next_id += 1;
                            }
                        }
                        Err(_) => {
                            workers_finished += 1;
                        }
                    }
                }

                // Write any remaining results
                for (_, result) in pending_results {
                    writer.write_all(&result.compressed)?;
                    total_written += result.compressed.len() as u64;
                }

                writer.flush()?;
                Ok(total_written)
            })
        };

        // Wait for reader to finish
        drop(job_sender);
        reader_handle
            .join()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Reader thread failed"))??;

        // Signal workers to finish
        for handle in worker_handles {
            handle
                .join()
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "Worker thread failed"))??;
        }

        // Get final result from writer
        drop(result_sender);
        writer_handle
            .join()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Writer thread failed"))?
    }

    /// Calculate optimal block size based on content and system characteristics
    fn calculate_block_size(&self) -> usize {
        let base_size = match self.config.content_type {
            ContentType::Text => 256 * 1024,   // 256KB for text
            ContentType::Binary => 512 * 1024, // 512KB for binary
            ContentType::Random => 128 * 1024, // 128KB for random data
        };

        // Adjust for compression level
        let level_factor = match self.config.compression_level {
            1..=3 => 0.8, // Smaller blocks for fast compression
            4..=6 => 1.0, // Standard blocks
            7..=9 => 1.5, // Larger blocks for high compression
            _ => 1.0,
        };

        // Adjust for thread count to balance workload
        let thread_factor = if self.config.thread_count >= 8 {
            0.8 // Smaller blocks for many threads
        } else {
            1.0
        };

        let optimal_size = (base_size as f64 * level_factor * thread_factor) as usize;
        optimal_size.clamp(64 * 1024, 2 * 1024 * 1024) // 64KB to 2MB
    }
}

/// Compress a single block with optimizations
fn compress_block_optimized(
    job: &CompressionJob,
    backend: CompressionBackend,
    level: u8,
) -> io::Result<Vec<u8>> {
    match backend {
        CompressionBackend::Gzp => {
            use gzp::{deflate::Gzip, ZBuilder};
            use std::io::Write;

            let output = Vec::new();
            let mut compressor = ZBuilder::<Gzip, _>::new()
                .num_threads(1) // Single thread per block
                .compression_level(gzp::Compression::new(level as u32))
                .from_writer(output);

            compressor.write_all(&job.data)?;
            let result = compressor
                .finish()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            Ok(result)
        }
        CompressionBackend::Flate2 => {
            use flate2::{write::GzEncoder, Compression};
            use std::io::Write;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
            encoder.write_all(&job.data)?;
            encoder
                .finish()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        }
    }
}

/// Set CPU affinity for worker threads to improve cache locality
fn set_worker_affinity(worker_id: usize) {
    // Only set affinity on Linux for now
    #[cfg(target_os = "linux")]
    {
        use core_affinity;
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if !core_ids.is_empty() {
                let core_id = core_ids[worker_id % core_ids.len()];
                let _ = core_affinity::set_for_current(core_id);
            }
        }
    }

    // For other platforms, this is a no-op
    #[cfg(not(target_os = "linux"))]
    {
        let _ = worker_id; // Suppress unused variable warning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::{CompressionBackend, ContentType};
    use std::io::Cursor;

    #[test]
    fn test_direct_compression() {
        let config = OptimizationConfig {
            thread_count: 1,
            buffer_size: 65536,
            backend: CompressionBackend::Flate2,
            content_type: ContentType::Text,
            use_numa_pinning: false,
            compression_level: 6,
        };

        let optimizer = CoreOptimizer::new(config);
        let input = b"Hello, world! This is a test.".repeat(1000);
        let cursor = Cursor::new(input.clone());
        let mut output = Vec::new();

        let result = optimizer.compress(cursor, &mut output);
        assert!(result.is_ok());
        assert!(output.len() > 0);
        assert!(output.len() < input.len()); // Should be compressed
    }

    #[test]
    fn test_block_size_calculation() {
        let config = OptimizationConfig {
            thread_count: 4,
            buffer_size: 65536,
            backend: CompressionBackend::Gzp,
            content_type: ContentType::Binary,
            use_numa_pinning: false,
            compression_level: 6,
        };

        let optimizer = CoreOptimizer::new(config);
        let block_size = optimizer.calculate_block_size();

        assert!(block_size >= 64 * 1024);
        assert!(block_size <= 2 * 1024 * 1024);
    }
}
