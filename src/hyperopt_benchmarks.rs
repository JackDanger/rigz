//! Benchmarks for hyperoptimized multi-path decompression
//!
//! Compares performance across different archive types:
//! - silesia: mixed content (our baseline)
//! - software: source code patterns
//! - logs: highly repetitive

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::time::Instant;

    /// Run benchmark on a dataset
    fn bench_dataset(name: &str, data: &[u8], runs: usize) {
        eprintln!(
            "\n=== Benchmarking {} ({:.1} MB) ===",
            name,
            data.len() as f64 / 1_000_000.0
        );

        // Benchmark libdeflate path
        let mut times_libdeflate = Vec::new();
        for _ in 0..runs {
            let mut output = Vec::new();
            let start = Instant::now();
            crate::hyperopt_dispatcher::decompress_libdeflate(data, &mut output).unwrap();
            times_libdeflate.push(start.elapsed().as_secs_f64());
        }
        let avg_libdeflate = times_libdeflate.iter().sum::<f64>() / runs as f64;
        let throughput_libdeflate = data.len() as f64 / avg_libdeflate / 1_000_000.0;

        // Benchmark consume_first path
        let mut times_consume = Vec::new();
        for _ in 0..runs {
            let mut output = Vec::new();
            let start = Instant::now();
            crate::hyperopt_dispatcher::decompress_consume_first(data, &mut output).unwrap();
            times_consume.push(start.elapsed().as_secs_f64());
        }
        let avg_consume = times_consume.iter().sum::<f64>() / runs as f64;
        let throughput_consume = data.len() as f64 / avg_consume / 1_000_000.0;

        // Benchmark hyperopt dispatcher (auto-routing)
        let mut times_hyperopt = Vec::new();
        for _ in 0..runs {
            let mut output = Vec::new();
            let start = Instant::now();
            crate::hyperopt_dispatcher::decompress_hyperopt(data, &mut output, 4).unwrap();
            times_hyperopt.push(start.elapsed().as_secs_f64());
        }
        let avg_hyperopt = times_hyperopt.iter().sum::<f64>() / runs as f64;
        let throughput_hyperopt = data.len() as f64 / avg_hyperopt / 1_000_000.0;

        eprintln!("\nResults ({} runs):", runs);
        eprintln!("  libdeflate:    {:.1} MB/s", throughput_libdeflate);
        eprintln!("  consume_first: {:.1} MB/s", throughput_consume);
        eprintln!("  hyperopt:      {:.1} MB/s", throughput_hyperopt);

        // Determine best path
        let best = throughput_libdeflate.max(throughput_consume);
        let hyperopt_ratio = throughput_hyperopt / best;

        eprintln!("\nHyperopt effectiveness: {:.1}%", hyperopt_ratio * 100.0);
        if hyperopt_ratio > 0.95 {
            eprintln!("✓ GOOD: Hyperopt selected near-optimal path");
        } else {
            eprintln!(
                "✗ POOR: Hyperopt selected suboptimal path ({:.1}% of best)",
                hyperopt_ratio * 100.0
            );
        }
    }

    #[test]
    fn bench_hyperopt_silesia() {
        let path = "benchmark_data/silesia-gzip.tar.gz";
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping: {} not found", path);
                return;
            }
        };

        let runs = std::env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        bench_dataset("SILESIA", &data, runs);
    }

    #[test]
    fn bench_hyperopt_software() {
        let path = "benchmark_data/software.archive.gz";
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping: {} not found (run prepare_datasets first)", path);
                return;
            }
        };

        let runs = std::env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        bench_dataset("SOFTWARE", &data, runs);
    }

    #[test]
    fn bench_hyperopt_logs() {
        let path = "benchmark_data/logs.txt.gz";
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping: {} not found (run prepare_datasets first)", path);
                return;
            }
        };

        let runs = std::env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        bench_dataset("LOGS", &data, runs);
    }

    #[test]
    fn bench_hyperopt_all() {
        // Prepare datasets
        if let Err(e) = crate::benchmark_datasets::prepare_datasets() {
            eprintln!("Failed to prepare datasets: {}", e);
            return;
        }

        let runs = std::env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        eprintln!("\n");
        eprintln!("╔════════════════════════════════════════════════════════════╗");
        eprintln!("║        HYPEROPT MULTI-PATH DECOMPRESSION BENCHMARK        ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝");

        for (name, _, gz_path) in [
            (
                "SILESIA",
                "benchmark_data/silesia.tar",
                "benchmark_data/silesia-gzip.tar.gz",
            ),
            (
                "SOFTWARE",
                "benchmark_data/software.archive",
                "benchmark_data/software.archive.gz",
            ),
            (
                "LOGS",
                "benchmark_data/logs.txt",
                "benchmark_data/logs.txt.gz",
            ),
        ] {
            if let Ok(data) = fs::read(gz_path) {
                bench_dataset(name, &data, runs);
            } else {
                eprintln!("\nSkipping {}: file not found", name);
            }
        }

        eprintln!("\n");
        eprintln!("════════════════════════════════════════════════════════════");
        eprintln!("Benchmark complete. Set BENCH_RUNS=N for more iterations.");
        eprintln!("════════════════════════════════════════════════════════════");
    }

    /// Test that hyperopt produces correct output
    #[test]
    fn test_hyperopt_correctness() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let test_cases = vec![
            ("repetitive", vec![b'A'; 10_000]),
            (
                "source_code",
                b"fn main() {\n    println!(\"test\");\n}\n".repeat(500),
            ),
            ("random", (0..10_000).map(|i| (i * 7 % 256) as u8).collect()),
        ];

        for (name, original) in test_cases {
            // Compress with flate2
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&original).unwrap();
            let compressed = encoder.finish().unwrap();

            // Decompress with hyperopt
            let mut output = Vec::new();
            crate::hyperopt_dispatcher::decompress_hyperopt(&compressed, &mut output, 4).unwrap();

            assert_eq!(
                output, original,
                "Hyperopt produced incorrect output for {}",
                name
            );
        }
    }
}
