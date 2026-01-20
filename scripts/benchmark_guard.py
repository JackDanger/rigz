#!/usr/bin/env python3
"""
Performance guard benchmark script for CI.

Usage:
    ./scripts/benchmark_guard.py --type bgzf --gzippy ./target/release/gzippy --pigz ./pigz/pigz
    ./scripts/benchmark_guard.py --type rapidgzip-multi --gzippy ./target/release/gzippy --rapidgzip ./rapidgzip/build/rapidgzip
    ./scripts/benchmark_guard.py --type rapidgzip-single --gzippy ./target/release/gzippy --rapidgzip ./rapidgzip/build/rapidgzip
    ./scripts/benchmark_guard.py --type igzip --gzippy ./target/release/gzippy --igzip ./isa-l/build/bin/igzip

Outputs results to stdout and $GITHUB_STEP_SUMMARY if available.
"""

import argparse
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Adaptive benchmarking parameters
MIN_TRIALS = 10
MAX_TRIALS = 30
TARGET_CV = 0.05  # 5% coefficient of variation

# Performance thresholds
THRESHOLDS = {
    "bgzf": 1.0,           # Must be faster than pigz
    "multi-member": 1.0,   # Must be faster than pigz
    "single-member": 1.0,  # Must be faster than gzip
    "rapidgzip-multi": 0.99,   # Within 1% of rapidgzip
    "rapidgzip-single": 0.99,  # Within 1% of rapidgzip
    "igzip": 0.90,         # At least 90% of igzip speed
}


def generate_test_data(path: Path, size_lines: int = 200000) -> None:
    """Generate reproducible compressible test data."""
    random.seed(42)
    with open(path, "w") as f:
        for _ in range(size_lines):
            f.write("".join(random.choices("abcdefghij", k=100)) + "\n")


def run_timed(cmd: list[str], stdin_file: Path | None = None) -> float:
    """Run a command and return elapsed time in seconds."""
    stdin = open(stdin_file, "rb") if stdin_file else None
    try:
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        end = time.perf_counter()
        return end - start
    finally:
        if stdin:
            stdin.close()


def adaptive_benchmark(cmd: list[str], stdin_file: Path | None = None) -> tuple[float, int]:
    """
    Run adaptive benchmarking until CV < TARGET_CV or MAX_TRIALS reached.
    Returns (mean_time, num_trials).
    """
    # Warmup
    try:
        run_timed(cmd, stdin_file)
    except subprocess.CalledProcessError:
        return (0.0, 0)  # Tool failed

    times = []
    for trial in range(1, MAX_TRIALS + 1):
        t = run_timed(cmd, stdin_file)
        times.append(t)

        if trial >= MIN_TRIALS and len(times) > 1:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            cv = stdev / mean if mean > 0 else 1.0
            if cv < TARGET_CV:
                break

    return (statistics.mean(times), len(times))


def write_summary(lines: list[str]) -> None:
    """Write to GitHub step summary if available, else print."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            for line in lines:
                f.write(line + "\n")
    for line in lines:
        print(line)


def benchmark_vs_tool(
    test_type: str,
    gzippy_bin: Path,
    other_bin: Path,
    other_name: str,
    compressed_file: Path,
    original_size: int,
    gzippy_extra_args: list[str] | None = None,
    other_extra_args: list[str] | None = None,
) -> bool:
    """
    Benchmark gzippy vs another tool on decompression.
    Returns True if threshold is met.
    """
    gzippy_args = gzippy_extra_args or []
    other_args = other_extra_args or []

    # Benchmark gzippy
    gzippy_cmd = [str(gzippy_bin), "-d"] + gzippy_args
    gzippy_time, gzippy_trials = adaptive_benchmark(gzippy_cmd, compressed_file)
    
    if gzippy_time == 0:
        write_summary([f"❌ gzippy failed to run"])
        return False
    
    gzippy_speed = original_size / gzippy_time / 1_000_000

    # Benchmark other tool
    other_cmd = [str(other_bin), "-d"] + other_args
    other_time, other_trials = adaptive_benchmark(other_cmd, compressed_file)
    
    if other_time == 0:
        write_summary([
            f"⚠️ {other_name} not available or failed, skipping comparison",
            f"gzippy speed: {gzippy_speed:.0f} MB/s ({gzippy_trials} trials)",
        ])
        return True  # Don't fail if comparison tool isn't available

    other_speed = original_size / other_time / 1_000_000
    ratio = gzippy_speed / other_speed
    threshold = THRESHOLDS.get(test_type, 1.0)

    summary = [
        f"## {test_type}: gzippy vs {other_name}",
        "",
        "| Tool | Speed (MB/s) | Trials |",
        "|------|-------------|--------|",
        f"| gzippy | {gzippy_speed:.0f} | {gzippy_trials} |",
        f"| {other_name} | {other_speed:.0f} | {other_trials} |",
        "",
        f"**Ratio**: gzippy is {ratio:.2f}x of {other_name}",
        "",
    ]

    passed = ratio >= threshold
    if passed:
        summary.append(f"✅ **PASS**: gzippy meets threshold ({ratio:.2f}x >= {threshold}x)")
    else:
        summary.append(f"❌ **FAIL**: gzippy below threshold ({ratio:.2f}x < {threshold}x)")

    write_summary(summary)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Performance guard benchmarks")
    parser.add_argument("--type", required=True, 
                       choices=["bgzf", "multi-member", "single-member", 
                               "rapidgzip-multi", "rapidgzip-single", "igzip"],
                       help="Type of benchmark to run")
    parser.add_argument("--gzippy", required=True, type=Path, help="Path to gzippy binary")
    parser.add_argument("--pigz", type=Path, help="Path to pigz binary")
    parser.add_argument("--unpigz", type=Path, help="Path to unpigz binary")
    parser.add_argument("--rapidgzip", type=Path, help="Path to rapidgzip binary")
    parser.add_argument("--igzip", type=Path, help="Path to igzip binary")
    parser.add_argument("--cores", type=int, default=os.cpu_count(), help="Number of cores")
    args = parser.parse_args()

    # Validate gzippy exists
    if not args.gzippy.exists():
        print(f"Error: gzippy binary not found at {args.gzippy}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.txt"
        
        # Generate test data
        print("Generating test data...")
        generate_test_data(test_file)
        original_size = test_file.stat().st_size

        if args.type == "bgzf":
            # Compress with gzippy (creates BGZF)
            compressed = tmpdir / "test.gz"
            subprocess.run([str(args.gzippy), "-1", "-c", str(test_file)],
                          stdout=open(compressed, "wb"), check=True)
            
            unpigz = args.unpigz or (args.pigz.parent / "unpigz" if args.pigz else None)
            if not unpigz or not unpigz.exists():
                print("Error: unpigz binary required for BGZF test")
                sys.exit(1)
            
            passed = benchmark_vs_tool(
                "bgzf", args.gzippy, unpigz, "pigz",
                compressed, original_size,
                other_extra_args=["-c"],
            )

        elif args.type == "multi-member":
            # Compress with pigz (creates multi-member)
            if not args.pigz or not args.pigz.exists():
                print("Error: pigz binary required for multi-member test")
                sys.exit(1)
            
            compressed = tmpdir / "test-pigz.gz"
            subprocess.run([str(args.pigz), f"-p{args.cores}", "-c", str(test_file)],
                          stdout=open(compressed, "wb"), check=True)
            
            unpigz = args.unpigz or (args.pigz.parent / "unpigz")
            passed = benchmark_vs_tool(
                "multi-member", args.gzippy, unpigz, "pigz",
                compressed, original_size,
                other_extra_args=["-c"],
            )

        elif args.type == "single-member":
            # Compress with gzip (single-member)
            compressed = tmpdir / "test-gzip.gz"
            subprocess.run(["gzip", "-c", str(test_file)],
                          stdout=open(compressed, "wb"), check=True)
            
            passed = benchmark_vs_tool(
                "single-member", args.gzippy, Path("/bin/gzip"), "gzip",
                compressed, original_size,
                other_extra_args=["-d", "-c"],
            )

        elif args.type == "rapidgzip-multi":
            if not args.rapidgzip or not args.rapidgzip.exists():
                print("⚠️ rapidgzip binary not available, skipping")
                sys.exit(0)
            
            if not args.pigz or not args.pigz.exists():
                print("Error: pigz binary required to create multi-member file")
                sys.exit(1)
            
            # Compress with pigz (creates multi-member)
            compressed = tmpdir / "test-pigz.gz"
            subprocess.run([str(args.pigz), f"-p{args.cores}", "-c", str(test_file)],
                          stdout=open(compressed, "wb"), check=True)
            
            passed = benchmark_vs_tool(
                "rapidgzip-multi", args.gzippy, args.rapidgzip, "rapidgzip",
                compressed, original_size,
                other_extra_args=["-d", f"-P{args.cores}"],
            )

        elif args.type == "rapidgzip-single":
            if not args.rapidgzip or not args.rapidgzip.exists():
                print("⚠️ rapidgzip binary not available, skipping")
                sys.exit(0)
            
            # Compress with gzip (single-member)
            compressed = tmpdir / "test-gzip.gz"
            subprocess.run(["gzip", "-c", str(test_file)],
                          stdout=open(compressed, "wb"), check=True)
            
            passed = benchmark_vs_tool(
                "rapidgzip-single", args.gzippy, args.rapidgzip, "rapidgzip",
                compressed, original_size,
                other_extra_args=["-d", f"-P{args.cores}"],
            )

        elif args.type == "igzip":
            if not args.igzip or not args.igzip.exists():
                print("⚠️ igzip binary not available, skipping")
                sys.exit(0)
            
            # Compress with gzip
            compressed = tmpdir / "test.gz"
            subprocess.run(["gzip", "-c", str(test_file)],
                          stdout=open(compressed, "wb"), check=True)
            
            passed = benchmark_vs_tool(
                "igzip", args.gzippy, args.igzip, "igzip",
                compressed, original_size,
                gzippy_extra_args=["-p1"],  # Single-threaded for fair comparison
            )

        else:
            print(f"Unknown test type: {args.type}")
            sys.exit(1)

        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
