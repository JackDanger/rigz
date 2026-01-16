//! Simple but effective compression optimizations
//!
//! Uses system zlib via flate2 for identical output to gzip at ALL compression levels.
//! Implements custom parallel compression using rayon instead of gzp.
//!
//! Key optimizations:
//! - Memory-mapped I/O for zero-copy file access (eliminates read latency)
//! - Global thread pool to avoid per-call initialization
//! - System zlib for gzip-compatible output at all compression levels

use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::optimization::{CompressionBackend, OptimizationConfig};
use crate::parallel_compress::ParallelGzEncoder;

/// Simple but effective optimizations that address pigz performance gaps
pub struct SimpleOptimizer {
    config: OptimizationConfig,
}

impl SimpleOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Optimized compression with improved threading and buffer management
    pub fn compress<R: Read, W: Write>(&self, reader: R, writer: W) -> io::Result<u64> {
        match self.config.backend {
            CompressionBackend::Parallel => self.compress_parallel(reader, writer),
            CompressionBackend::SingleThreaded => self.compress_single_threaded(reader, writer),
        }
    }

    /// Parallel compression using our custom implementation with system zlib
    fn compress_parallel<R: Read, W: Write>(&self, reader: R, writer: W) -> io::Result<u64> {
        let optimal_threads = self.calculate_optimal_threads();

        // Use the actual requested compression level - system zlib works correctly
        let compression_level = self.config.compression_level as u32;

        let encoder = ParallelGzEncoder::new(compression_level, optimal_threads);
        encoder.compress(reader, writer)
    }

    /// File-based parallel compression using memory-mapped I/O
    /// This eliminates the latency of reading the file into memory
    pub fn compress_file<P: AsRef<Path>, W: Write>(&self, path: P, writer: W) -> io::Result<u64> {
        let optimal_threads = self.calculate_optimal_threads();
        let compression_level = self.config.compression_level as u32;

        // For single-threaded, fall back to regular compression
        if optimal_threads == 1 {
            let file = std::fs::File::open(path)?;
            return self.compress_single_threaded(file, writer);
        }

        // Use mmap-based parallel compression
        let encoder = ParallelGzEncoder::new(compression_level, optimal_threads);
        encoder.compress_file(path, writer)
    }

    /// Single-threaded compression using flate2 with system zlib
    fn compress_single_threaded<R: Read, W: Write>(
        &self,
        mut reader: R,
        writer: W,
    ) -> io::Result<u64> {
        // Use the actual compression level - system zlib produces correct output
        let compression = Compression::new(self.config.compression_level as u32);

        let mut encoder = GzEncoder::new(writer, compression);
        let bytes_written = io::copy(&mut reader, &mut encoder)?;
        encoder.finish()?;

        Ok(bytes_written)
    }

    /// Calculate optimal thread count based on pigz analysis
    fn calculate_optimal_threads(&self) -> usize {
        let base_threads = self.config.thread_count;

        // Use requested threads - the "2-thread problem" was specific to gzp
        // which we no longer use. Our rayon-based parallel compression works
        // correctly with any thread count.
        base_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::{CompressionBackend, ContentType};
    use std::io::Cursor;

    #[test]
    fn test_thread_count_passthrough() {
        let config = OptimizationConfig {
            thread_count: 4,
            buffer_size: 65536,
            backend: CompressionBackend::Parallel,
            content_type: ContentType::Binary,
            use_numa_pinning: false,
            compression_level: 6,
        };

        let optimizer = SimpleOptimizer::new(config);
        // Thread count should pass through (no longer reduced)
        assert_eq!(optimizer.calculate_optimal_threads(), 4);
    }

    #[test]
    fn test_compression() {
        let config = OptimizationConfig {
            thread_count: 4,
            buffer_size: 65536,
            backend: CompressionBackend::Parallel,
            content_type: ContentType::Text,
            use_numa_pinning: false,
            compression_level: 6,
        };

        let optimizer = SimpleOptimizer::new(config);
        let input = b"Hello, world! This is a test of the simple optimizer.".repeat(1000);
        let cursor = Cursor::new(input.clone());
        let mut output = Vec::new();

        let result = optimizer.compress(cursor, &mut output);
        assert!(result.is_ok());
        assert!(!output.is_empty());
        assert!(output.len() < input.len());
    }

    #[test]
    fn test_single_threaded_compression() {
        let config = OptimizationConfig {
            thread_count: 1,
            buffer_size: 65536,
            backend: CompressionBackend::SingleThreaded,
            content_type: ContentType::Text,
            use_numa_pinning: false,
            compression_level: 6,
        };

        let optimizer = SimpleOptimizer::new(config);
        let input = b"Hello, world! This is a test of single-threaded compression.".repeat(1000);
        let cursor = Cursor::new(input.clone());
        let mut output = Vec::new();

        let result = optimizer.compress(cursor, &mut output);
        assert!(result.is_ok());
        assert!(!output.is_empty());
        assert!(output.len() < input.len());
    }
}
