//! Benchmark libdeflate vs flate2/zlib-ng compression
//!
//! Run with: cargo run --release --example bench_compression

use flate2::write::GzEncoder;
use flate2::Compression;
use libdeflater::{CompressionLvl, Compressor};
use std::io::Write;
use std::time::Instant;

fn main() {
    // Use real test data file if available, otherwise generate
    let data: Vec<u8> = if std::path::Path::new("test_data/text-1MB.txt").exists() {
        // Read and repeat to get 10MB
        let seed = std::fs::read("test_data/text-1MB.txt").unwrap();
        let mut data = Vec::with_capacity(10 * 1024 * 1024);
        while data.len() < 10 * 1024 * 1024 {
            let remaining = 10 * 1024 * 1024 - data.len();
            data.extend_from_slice(&seed[..remaining.min(seed.len())]);
        }
        data
    } else {
        // Generate pseudo-random data
        (0..10 * 1024 * 1024)
            .map(|i| ((i as u64 * 1103515245 + 12345) % 256) as u8)
            .collect()
    };

    let size_mb = data.len() as f64 / (1024.0 * 1024.0);
    println!("Test data: {:.1}MB (base64-like)", size_mb);
    println!();
    println!(
        "{:>6} | {:>12} {:>8} | {:>12} {:>8} | {:>7}",
        "Level", "flate2", "size", "libdeflate", "size", "speedup"
    );
    println!("{}", "-".repeat(70));

    for level in [1u32, 6, 9, 12] {
        let flate2_level = level.min(9);

        // Benchmark flate2/zlib-ng (5 runs, take median)
        let mut flate2_times = Vec::new();
        let mut flate2_size = 0;
        for _ in 0..5 {
            let start = Instant::now();
            let mut encoder = GzEncoder::new(Vec::new(), Compression::new(flate2_level));
            encoder.write_all(&data).unwrap();
            let result = encoder.finish().unwrap();
            flate2_times.push(start.elapsed());
            flate2_size = result.len();
        }
        flate2_times.sort();
        let flate2_time = flate2_times[2]; // median

        // Benchmark libdeflate (5 runs, take median)
        let mut libdeflate_times = Vec::new();
        let mut libdeflate_size = 0;
        for _ in 0..5 {
            let start = Instant::now();
            let lvl = CompressionLvl::new(level as i32).unwrap_or_default();
            let mut compressor = Compressor::new(lvl);
            let max_size = compressor.gzip_compress_bound(data.len());
            let mut result = vec![0u8; max_size];
            let actual_size = compressor.gzip_compress(&data, &mut result).unwrap();
            result.truncate(actual_size);
            libdeflate_times.push(start.elapsed());
            libdeflate_size = result.len();
        }
        libdeflate_times.sort();
        let libdeflate_time = libdeflate_times[2]; // median

        let speedup = flate2_time.as_secs_f64() / libdeflate_time.as_secs_f64();

        println!(
            "{:>6} | {:>9.3}s {:>6.1}MB | {:>9.3}s {:>6.1}MB | {:>6.2}x",
            level,
            flate2_time.as_secs_f64(),
            flate2_size as f64 / (1024.0 * 1024.0),
            libdeflate_time.as_secs_f64(),
            libdeflate_size as f64 / (1024.0 * 1024.0),
            speedup
        );
    }

    println!();
    println!("Note: libdeflate supports levels 1-12, flate2 supports 1-9");
    println!("libdeflate level 12 = maximum compression (slower but smaller)");
}
