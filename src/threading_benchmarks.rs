//! Threading-aware decompression benchmarks
//!
//! Compares single-threaded vs multi-threaded performance across different archive types

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Instant;

    /// Benchmark single-threaded decompression
    fn bench_single_threaded(name: &str, data: &[u8], runs: usize) -> f64 {
        let mut times = Vec::new();

        for _ in 0..runs {
            let mut output = Vec::new();
            let start = Instant::now();
            crate::threading_dispatcher::decompress_with_threading(data, &mut output, 1).unwrap();
            times.push(start.elapsed().as_secs_f64());
        }

        let avg = times.iter().sum::<f64>() / runs as f64;
        let throughput = data.len() as f64 / avg / 1_000_000.0;

        eprintln!(
            "[{:>12}] Single-thread: {:>6.1} MB/s ({:.3}s avg)",
            name, throughput, avg
        );

        throughput
    }

    /// Benchmark multi-threaded decompression
    fn bench_multi_threaded(name: &str, data: &[u8], runs: usize, threads: usize) -> f64 {
        let mut times = Vec::new();

        for _ in 0..runs {
            let mut output = Vec::new();
            let start = Instant::now();
            crate::threading_dispatcher::decompress_with_threading(data, &mut output, threads)
                .unwrap();
            times.push(start.elapsed().as_secs_f64());
        }

        let avg = times.iter().sum::<f64>() / runs as f64;
        let throughput = data.len() as f64 / avg / 1_000_000.0;

        eprintln!(
            "[{:>12}] Multi-thread ({}): {:>6.1} MB/s ({:.3}s avg)",
            name, threads, throughput, avg
        );

        throughput
    }

    #[test]
    fn bench_threading_silesia() {
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

        eprintln!("\n=== SILESIA Threading Benchmark ({} runs) ===", runs);
        eprintln!("File size: {:.1} MB\n", data.len() as f64 / 1_000_000.0);

        let single = bench_single_threaded("SILESIA", &data, runs);

        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let multi = bench_multi_threaded("SILESIA", &data, runs, threads);

        eprintln!("\nSpeedup: {:.2}x\n", multi / single);
    }

    #[test]
    fn bench_threading_all() {
        // Prepare datasets
        if let Err(e) = crate::benchmark_datasets::prepare_datasets() {
            eprintln!("Failed to prepare datasets: {}", e);
            return;
        }

        let runs = std::env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        eprintln!("\n");
        eprintln!("╔════════════════════════════════════════════════════════════╗");
        eprintln!("║      THREADING-AWARE DECOMPRESSION BENCHMARK             ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝");
        eprintln!("\nRuns: {}, Threads: {}\n", runs, threads);

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
                eprintln!(
                    "\n=== {} ({:.1} MB) ===\n",
                    name,
                    data.len() as f64 / 1_000_000.0
                );

                let single = bench_single_threaded(name, &data, runs);
                let multi = bench_multi_threaded(name, &data, runs, threads);

                let speedup = multi / single;
                let efficiency = speedup / threads as f64 * 100.0;

                eprintln!("\nSpeedup: {:.2}x", speedup);
                eprintln!("Parallel efficiency: {:.1}%", efficiency);

                if speedup > 1.5 {
                    eprintln!("✓ GOOD: Multi-threading provides significant speedup");
                } else {
                    eprintln!("⚠ WARNING: Limited multi-threading benefit");
                }
            } else {
                eprintln!("\nSkipping {}: file not found", name);
            }
        }

        eprintln!("\n");
        eprintln!("════════════════════════════════════════════════════════════");
        eprintln!("Threading benchmark complete.");
        eprintln!("════════════════════════════════════════════════════════════\n");
    }

    /// Compare against external tools (pigz, rapidgzip if available)
    #[test]
    fn bench_compare_external() {
        use std::process::Command;

        let path = "benchmark_data/silesia-gzip.tar.gz";
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping external comparison: {} not found", path);
                return;
            }
        };

        let runs = std::env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        eprintln!("\n");
        eprintln!("╔════════════════════════════════════════════════════════════╗");
        eprintln!("║        EXTERNAL TOOL COMPARISON (SILESIA)                ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝");
        eprintln!(
            "\nFile: {:.1} MB, Runs: {}\n",
            data.len() as f64 / 1_000_000.0,
            runs
        );

        // Benchmark gzippy
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let gzippy_single = bench_single_threaded("gzippy", &data, runs);
        let gzippy_multi = bench_multi_threaded("gzippy", &data, runs, threads);

        // Try to benchmark pigz if available
        if Command::new("pigz").arg("--version").output().is_ok() {
            eprintln!("\nBenchmarking pigz...");

            let mut times = Vec::new();
            for _ in 0..runs {
                let start = Instant::now();
                let output = Command::new("pigz").arg("-dc").arg(path).output();

                if let Ok(out) = output {
                    if out.status.success() {
                        times.push(start.elapsed().as_secs_f64());
                    }
                }
            }

            if !times.is_empty() {
                let avg = times.iter().sum::<f64>() / times.len() as f64;
                let throughput = data.len() as f64 / avg / 1_000_000.0;
                eprintln!(
                    "[{:>12}] pigz -dc: {:>6.1} MB/s ({:.3}s avg)",
                    "pigz", throughput, avg
                );
                eprintln!("\ngzippy vs pigz:");
                eprintln!("  Single-thread: {:.1}x", gzippy_single / throughput);
                eprintln!("  Multi-thread:  {:.1}x", gzippy_multi / throughput);
            }
        } else {
            eprintln!("\npigz not found - skipping comparison");
        }

        // Try to benchmark rapidgzip if available
        if Command::new("rapidgzip").arg("--version").output().is_ok() {
            eprintln!("\nBenchmarking rapidgzip...");

            let mut times = Vec::new();
            for _ in 0..runs {
                let start = Instant::now();
                let output = Command::new("rapidgzip")
                    .arg("-d")
                    .arg("-c")
                    .arg(path)
                    .output();

                if let Ok(out) = output {
                    if out.status.success() {
                        times.push(start.elapsed().as_secs_f64());
                    }
                }
            }

            if !times.is_empty() {
                let avg = times.iter().sum::<f64>() / times.len() as f64;
                let throughput = data.len() as f64 / avg / 1_000_000.0;
                eprintln!(
                    "[{:>12}] rapidgzip: {:>6.1} MB/s ({:.3}s avg)",
                    "rapidgzip", throughput, avg
                );
                eprintln!("\ngzippy vs rapidgzip:");
                eprintln!("  Multi-thread: {:.1}x", gzippy_multi / throughput);
            }
        } else {
            eprintln!("\nrapidgzip not found - skipping comparison");
        }

        eprintln!("\n");
    }
}
