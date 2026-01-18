//! Simple but effective compression optimizations
//!
//! Uses system zlib via flate2 for identical output to gzip at ALL compression levels.
//! Implements custom parallel compression using rayon instead of gzp.
//!
//! Key optimizations:
//! - Memory-mapped I/O for zero-copy file access (eliminates read latency)
//! - Global thread pool to avoid per-call initialization
//! - System zlib for gzip-compatible output at all compression levels
//! - Cache-aware block sizing based on detected L2 cache

use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::optimization::{CompressionBackend, CpuFeatures, OptimizationConfig};
use crate::parallel_compress::ParallelGzEncoder;
use crate::pipelined_compress::PipelinedGzEncoder;

/// Simple but effective optimizations that address pigz performance gaps
pub struct SimpleOptimizer {
    config: OptimizationConfig,
}

impl SimpleOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Optimized compression with improved threading and buffer management
    pub fn compress<R: Read, W: Write + Send>(&self, reader: R, writer: W) -> io::Result<u64> {
        // L10-L12 always use parallel path (libdeflate) since zlib doesn't support these levels
        if self.config.compression_level >= 10 {
            return self.compress_parallel(reader, writer);
        }

        match self.config.backend {
            CompressionBackend::Parallel => self.compress_parallel(reader, writer),
            CompressionBackend::SingleThreaded => self.compress_single_threaded(reader, writer),
        }
    }

    /// Parallel compression using our custom implementation
    ///
    /// Strategy per compression level:
    /// - L1-L6: libdeflate independent blocks (fast + parallel decompress)
    /// - L7-L9: pipelined zlib-ng with dictionary (gzip-compatible ratio)
    /// - L10-L12: libdeflate high levels (ultra compression, near-zopfli)
    ///
    /// L10-L12 use libdeflate's exhaustive search which achieves near-zopfli
    /// compression ratios (~5% smaller than gzip -9) while being 10-20x faster.
    fn compress_parallel<R: Read, W: Write + Send>(&self, reader: R, writer: W) -> io::Result<u64> {
        let optimal_threads = self.calculate_optimal_threads();
        let compression_level = self.config.compression_level as u32;

        // L10-L12: Ultra compression using libdeflate high levels
        // These use exhaustive search for near-zopfli compression ratios
        if self.config.compression_level >= 10 {
            let encoder = ParallelGzEncoder::new(compression_level, optimal_threads);
            return encoder.compress(reader, writer);
        }

        // L6-L9: Use pipelined compression with dictionary sharing like pigz
        // This ensures we match or beat pigz's compression ratio at all levels
        if self.config.compression_level >= 6 && optimal_threads > 1 {
            let encoder = PipelinedGzEncoder::new(compression_level, optimal_threads);
            return encoder.compress(reader, writer);
        }

        // L1-L5: Use independent blocks for parallel decompression (fast)
        let encoder = ParallelGzEncoder::new(compression_level, optimal_threads);
        encoder.compress(reader, writer)
    }

    /// File-based parallel compression using memory-mapped I/O
    /// This eliminates the latency of reading the file into memory
    ///
    /// Strategy per compression level:
    /// - L10-L12: libdeflate ultra compression (near-zopfli ratio)
    /// - L7-L9: pipelined zlib-ng for gzip-compatible ratio
    /// - L1-L6: independent blocks for parallel decompression
    pub fn compress_file<P: AsRef<Path>, W: Write + Send>(
        &self,
        path: P,
        writer: W,
    ) -> io::Result<u64> {
        let optimal_threads = self.calculate_optimal_threads();
        let compression_level = self.config.compression_level as u32;

        // L10-L12: Ultra compression using libdeflate high levels
        if self.config.compression_level >= 10 {
            let encoder = ParallelGzEncoder::new(compression_level, optimal_threads);
            return encoder.compress_file(path, writer);
        }

        // For single-threaded L6-L9, use pipelined compression for max ratio
        if optimal_threads == 1 {
            if self.config.compression_level >= 6 {
                let encoder = PipelinedGzEncoder::new(compression_level, 1);
                return encoder.compress_file(path, writer);
            }
            let file = std::fs::File::open(&path)?;
            return self.compress_single_threaded(file, writer);
        }

        // L6-L9: Use pipelined compression with dictionary sharing
        if self.config.compression_level >= 6 {
            let encoder = PipelinedGzEncoder::new(compression_level, optimal_threads);
            return encoder.compress_file(path, writer);
        }

        // L1-L5: Use mmap-based parallel compression with independent blocks
        let encoder = ParallelGzEncoder::new(compression_level, optimal_threads);
        encoder.compress_file(path, writer)
    }

    /// Single-threaded compression using flate2 with system zlib
    /// Note: L10-L12 are handled by compress_parallel even for single-threaded
    fn compress_single_threaded<R: Read, W: Write>(
        &self,
        mut reader: R,
        writer: W,
    ) -> io::Result<u64> {
        // zlib-ng level 1 uses a different strategy that produces worse output.
        // Map level 1 → 2 for better compression ratio with similar speed.
        let adjusted_level = if self.config.compression_level == 1 {
            2
        } else {
            self.config.compression_level.min(9) // Cap at 9 for zlib
        };
        let compression = Compression::new(adjusted_level as u32);

        let mut encoder = GzEncoder::new(writer, compression);
        let bytes_written = io::copy(&mut reader, &mut encoder)?;
        encoder.finish()?;

        Ok(bytes_written)
    }

    /// Calculate optimal thread count based on request
    fn calculate_optimal_threads(&self) -> usize {
        // Use the requested thread count directly.
        // On GHA and similar VMs, physical_cores may report fewer cores
        // than available vCPUs (e.g., 2 physical on 4 vCPU), which hurts
        // performance. pigz uses all available threads and so should we.
        self.config.thread_count
    }

    /// Get CPU feature summary for debugging/verbosity
    #[allow(dead_code)]
    pub fn cpu_features_summary() -> String {
        let cpu = CpuFeatures::get();
        let mut features = Vec::new();

        if cpu.has_avx512 {
            features.push("AVX-512");
        } else if cpu.has_avx2 {
            features.push("AVX2");
        }
        if cpu.has_neon {
            features.push("NEON");
        }
        if cpu.has_crc32 {
            features.push("CRC32");
        }

        format!(
            "CPU: {} cores, L2={}KB, features=[{}]",
            cpu.physical_cores,
            cpu.l2_cache_size / 1024,
            features.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::{CompressionBackend, ContentType};
    use std::io::Cursor;

    #[test]
    fn test_thread_count_respects_request() {
        let config = OptimizationConfig {
            thread_count: 4,
            buffer_size: 65536,
            backend: CompressionBackend::Parallel,
            content_type: ContentType::Binary,
            use_numa_pinning: false,
            compression_level: 6,
        };

        let optimizer = SimpleOptimizer::new(config);
        // Thread count should match the requested count exactly
        // (no longer capped at physical cores - VM environments report
        // fewer physical cores than available vCPUs)
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
