//! Golden Tests: Byte-exact verification of decompression output
//!
//! These tests ensure that any optimization changes don't introduce subtle
//! decode errors. They compare output against known-good reference data.

#![allow(dead_code)]
#![allow(unused_imports)]

use std::io::Write;

/// Generate golden reference data from libdeflate (the trusted implementation)
fn generate_golden_data(deflate_data: &[u8], expected_size: usize) -> Vec<u8> {
    let mut output = vec![0u8; expected_size];
    let size = libdeflater::Decompressor::new()
        .deflate_decompress(deflate_data, &mut output)
        .expect("libdeflate failed");
    output.truncate(size);
    output
}

/// Hash first N bytes for quick comparison
fn hash_bytes(data: &[u8], n: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    data[..n.min(data.len())].hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;

    /// Golden test: Simple literals
    #[test]
    fn golden_simple_literals() {
        let original = b"Hello, World! This is a test of simple literal data.";

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Get libdeflate reference
        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .expect("libdeflate failed");

        // Get our turbo output
        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size =
            crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).expect("turbo failed");

        // Byte-exact comparison
        assert_eq!(turbo_size, libdeflate_size, "Size mismatch");
        assert_slices_eq!(
            &turbo_out[..turbo_size],
            &libdeflate_out[..libdeflate_size],
            "Content mismatch"
        );
        assert_slices_eq!(
            &turbo_out[..turbo_size],
            original.as_slice(),
            "Original mismatch"
        );

        eprintln!("[GOLDEN] simple_literals: ✓ {} bytes verified", turbo_size);
    }

    /// Golden test: RLE pattern (distance=1)
    #[test]
    fn golden_rle_pattern() {
        // Create data with lots of RLE opportunities
        let original: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Compare
        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size = crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).unwrap();

        assert_eq!(turbo_size, libdeflate_size);
        assert_slices_eq!(&turbo_out[..turbo_size], &libdeflate_out[..libdeflate_size]);

        eprintln!("[GOLDEN] rle_pattern: ✓ {} bytes verified", turbo_size);
    }

    /// Golden test: Short distance matches (d=2-7)
    #[test]
    fn golden_short_distance() {
        // Pattern that creates short distance matches
        let original = b"abcabcabcabcabcabcabcabc".repeat(1000);

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size = crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).unwrap();

        assert_eq!(turbo_size, libdeflate_size);
        assert_slices_eq!(&turbo_out[..turbo_size], &libdeflate_out[..libdeflate_size]);

        eprintln!("[GOLDEN] short_distance: ✓ {} bytes verified", turbo_size);
    }

    /// Golden test: Long distance matches (d > 1024)
    #[test]
    fn golden_long_distance() {
        // Create pattern with long distance references
        let mut original = Vec::with_capacity(100_000);
        let pattern = b"This is a unique pattern that will be repeated later.";
        original.extend_from_slice(pattern);
        // Add filler
        for i in 0..50_000 {
            original.push((i % 256) as u8);
        }
        // Repeat the pattern (long distance match)
        original.extend_from_slice(pattern);

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size = crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).unwrap();

        assert_eq!(turbo_size, libdeflate_size);
        assert_slices_eq!(&turbo_out[..turbo_size], &libdeflate_out[..libdeflate_size]);

        eprintln!("[GOLDEN] long_distance: ✓ {} bytes verified", turbo_size);
    }

    /// Golden test: Silesia first 100KB (the real-world benchmark)
    #[test]
    fn golden_silesia_100kb() {
        let gzip_data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[GOLDEN] Skipping silesia test - no benchmark file");
                return;
            }
        };

        // Extract deflate data
        let deflate_start = 10
            + if (gzip_data[3] & 0x08) != 0 {
                gzip_data[10..].iter().position(|&b| b == 0).unwrap_or(0) + 1
            } else {
                0
            };
        let deflate_end = gzip_data.len() - 8;
        let deflate_data = &gzip_data[deflate_start..deflate_end];

        // Get expected size from ISIZE
        let isize_bytes = &gzip_data[gzip_data.len() - 4..];
        let isize = u32::from_le_bytes([
            isize_bytes[0],
            isize_bytes[1],
            isize_bytes[2],
            isize_bytes[3],
        ]) as usize;

        // Compare first 100KB
        let test_size = 100_000.min(isize);

        let mut libdeflate_out = vec![0u8; isize + 1000];
        let _ = libdeflater::Decompressor::new()
            .deflate_decompress(deflate_data, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; isize + 1000];
        let turbo_size = crate::bgzf::inflate_into_pub(deflate_data, &mut turbo_out).unwrap();

        // Find first mismatch
        let first_mismatch = turbo_out[..test_size]
            .iter()
            .zip(libdeflate_out[..test_size].iter())
            .enumerate()
            .find(|(_, (a, b))| a != b);

        if let Some((pos, (got, exp))) = first_mismatch {
            panic!(
                "[GOLDEN] silesia mismatch at byte {}: got {} expected {}",
                pos, got, exp
            );
        }

        eprintln!(
            "[GOLDEN] silesia_100kb: ✓ {} bytes verified (total output: {} bytes)",
            test_size, turbo_size
        );
    }

    /// Golden test: Multi-block deflate
    #[test]
    fn golden_multi_block() {
        // Create data large enough to force multiple deflate blocks
        let original = b"Multi block test data. ".repeat(50_000); // ~1.1MB

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size = crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).unwrap();

        assert_eq!(turbo_size, libdeflate_size);
        assert_slices_eq!(&turbo_out[..turbo_size], &libdeflate_out[..libdeflate_size]);

        eprintln!("[GOLDEN] multi_block: ✓ {} bytes verified", turbo_size);
    }

    /// Golden test: Binary data (all byte values)
    #[test]
    fn golden_binary() {
        // Create data with all possible byte values
        let mut original = Vec::with_capacity(256 * 100);
        for _ in 0..100 {
            for b in 0u8..=255 {
                original.push(b);
            }
        }

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size = crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).unwrap();

        assert_eq!(turbo_size, libdeflate_size);
        assert_slices_eq!(&turbo_out[..turbo_size], &libdeflate_out[..libdeflate_size]);

        eprintln!("[GOLDEN] binary: ✓ {} bytes verified", turbo_size);
    }

    /// Golden test: Maximum length match (258 bytes)
    #[test]
    fn golden_max_length_match() {
        // Create data that produces maximum length matches
        let original = b"X".repeat(10000); // All same character = max length RLE

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut libdeflate_out = vec![0u8; original.len() + 100];
        let libdeflate_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut libdeflate_out)
            .unwrap();

        let mut turbo_out = vec![0u8; original.len() + 100];
        let turbo_size = crate::bgzf::inflate_into_pub(&compressed, &mut turbo_out).unwrap();

        assert_eq!(turbo_size, libdeflate_size);
        assert_slices_eq!(&turbo_out[..turbo_size], &libdeflate_out[..libdeflate_size]);

        eprintln!("[GOLDEN] max_length: ✓ {} bytes verified", turbo_size);
    }

    /// Summary test: Run all golden tests
    #[test]
    fn golden_summary() {
        eprintln!("\n=== GOLDEN TEST SUITE ===");
        eprintln!("All golden tests verify byte-exact match with libdeflate");
        eprintln!("Any optimization that breaks these tests introduces bugs");
        eprintln!("=========================\n");
    }
}
