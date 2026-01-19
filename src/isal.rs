//! High-Performance Decompression
//!
//! This module provides the fastest decompression using libdeflate,
//! a highly optimized, statically-linked deflate library.
//!
//! Design decision: We use libdeflate instead of ISA-L because:
//! - libdeflate is pure Rust/C with simple FFI (automatic via libdeflater crate)
//! - ISA-L's complex struct layout (200KB+ with nested Huffman tables) is hard for FFI
//! - ISA-L uses pure C fallback on ARM/Apple Silicon anyway (no SIMD advantage)
//! - libdeflate is only ~10-15% slower than ISA-L SIMD on x86_64
//!
//! All dependencies are statically linked - no dynamic library requirements.

use std::io;

/// Check if high-performance decompression is available
/// Always true since libdeflate is always available
#[allow(dead_code)]
pub fn is_available() -> bool {
    true
}

/// High-performance inflater using libdeflate
///
/// This is the recommended decompression backend for gzippy.
/// It's highly optimized and statically linked.
pub struct IsalInflater {
    inner: libdeflater::Decompressor,
}

impl IsalInflater {
    /// Create a new inflater
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            inner: libdeflater::Decompressor::new(),
        })
    }

    /// Reset the inflater for a new stream
    pub fn reset(&mut self) -> io::Result<()> {
        self.inner = libdeflater::Decompressor::new();
        Ok(())
    }

    /// Set dictionary - not supported by libdeflate for gzip
    /// Returns error since this is only available with ISA-L
    #[allow(dead_code)]
    pub fn set_dict(&mut self, _dict: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Dictionary not supported (ISA-L not available)",
        ))
    }

    /// Decompress gzip data into output buffer
    #[allow(dead_code)]
    pub fn decompress(&mut self, input: &[u8], output: &mut [u8]) -> io::Result<usize> {
        match self.inner.gzip_decompress(input, output) {
            Ok(n) => Ok(n),
            Err(e) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Decompression failed: {:?}", e),
            )),
        }
    }

    /// Decompress a complete gzip stream, auto-growing output buffer
    pub fn decompress_all(&mut self, input: &[u8], initial_size: usize) -> io::Result<Vec<u8>> {
        let mut output = vec![0u8; initial_size];

        loop {
            match self.inner.gzip_decompress(input, &mut output) {
                Ok(n) => {
                    output.truncate(n);
                    return Ok(output);
                }
                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                    output.resize(output.len() * 2, 0);
                }
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Decompression failed: {:?}", e),
                    ));
                }
            }
        }
    }
}

impl Default for IsalInflater {
    fn default() -> Self {
        Self::new().expect("Failed to create inflater")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_decompress() {
        // Create some test data
        let original = b"Hello, World! This is a test of gzip decompression.";

        // Compress with flate2
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress with our inflater
        let mut inflater = IsalInflater::new().unwrap();
        let decompressed = inflater.decompress_all(&compressed, 1024).unwrap();

        assert_eq!(&decompressed, original);
    }

    #[test]
    fn test_large_decompress() {
        // Create 1MB of test data
        let original: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

        // Compress
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress
        let mut inflater = IsalInflater::new().unwrap();
        let decompressed = inflater
            .decompress_all(&compressed, compressed.len() * 2)
            .unwrap();

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_reset() {
        let original = b"Test data for reset";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut inflater = IsalInflater::new().unwrap();

        // Decompress twice with reset
        let result1 = inflater.decompress_all(&compressed, 1024).unwrap();
        inflater.reset().unwrap();
        let result2 = inflater.decompress_all(&compressed, 1024).unwrap();

        assert_eq!(result1, result2);
        assert_eq!(&result1[..], original);
    }
}
