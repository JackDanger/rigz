#!/usr/bin/env python3
"""
Rigz Performance Suite

Benchmarks compression AND decompression against gzip and pigz.
Tests that rigz beats or matches both tools.

Usage:
    python3 scripts/perf.py                    # Quick test (1MB, 10MB)
    python3 scripts/perf.py --full             # Full suite (1MB, 10MB, 100MB)
    python3 scripts/perf.py --levels 1,6,9     # Specific levels
    python3 scripts/perf.py --threads 1,4,8    # Specific thread counts
    python3 scripts/perf.py --sizes 10,100     # Specific sizes in MB
"""

import argparse
import os
import shutil
import subprocess
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

# Tool paths - prefer local builds, fall back to system
import shutil

def find_gzip():
    if os.path.isfile("./gzip/gzip") and os.access("./gzip/gzip", os.X_OK):
        return "./gzip/gzip"
    return shutil.which("gzip") or "gzip"

GZIP = find_gzip()
PIGZ = "./pigz/pigz"
RIGZ = "./target/release/rigz"

# Defaults
DEFAULT_LEVELS = [1, 6, 9]
DEFAULT_THREADS = [1, 4]
DEFAULT_SIZES_QUICK = [1, 10]      # MB
DEFAULT_SIZES_FULL = [1, 10, 100]  # MB

# Statistical config - more runs for shorter tests
RUNS_BY_SIZE = {1: 10, 10: 5, 100: 3}

# Acceptable overhead (rigz can be up to 5% slower and still "pass")
MAX_OVERHEAD_PCT = 5.0


@dataclass
class BenchResult:
    tool: str
    operation: str  # "compress" or "decompress"
    level: int
    threads: int
    size_mb: int
    time_median: float
    time_stdev: float
    output_size: int = 0


def format_time(seconds: float) -> str:
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 1:
        return f"{seconds:.3f}s"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.1f}s"


def format_size(bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"


def generate_test_file(path: str, size_mb: int) -> None:
    """Generate a compressible test file."""
    size_bytes = size_mb * 1024 * 1024
    # Use base64 of random data - compressible but not trivially so
    cmd = f"head -c {size_bytes} /dev/urandom | base64 > {path}"
    subprocess.run(cmd, shell=True, check=True, stderr=subprocess.DEVNULL)


def run_timed(cmd: List[str], stdin_file: str = None, stdout_file: str = None) -> Tuple[bool, float]:
    """Run a command and return (success, elapsed_time)."""
    start = time.perf_counter()
    
    stdin = open(stdin_file, 'rb') if stdin_file else None
    stdout = open(stdout_file, 'wb') if stdout_file else subprocess.DEVNULL
    
    try:
        result = subprocess.run(cmd, stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL)
        elapsed = time.perf_counter() - start
        return result.returncode == 0, elapsed
    finally:
        if stdin:
            stdin.close()
        if stdout_file:
            stdout.close()


def benchmark_compress(tool: str, level: int, threads: int, 
                       input_file: str, output_file: str, runs: int) -> Tuple[float, float, int]:
    """Benchmark compression. Returns (median_time, stdev, output_size)."""
    bin_path = {"gzip": GZIP, "pigz": PIGZ, "rigz": RIGZ}[tool]
    
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "rigz"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - start)
    
    output_size = os.path.getsize(output_file)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    
    return median, stdev, output_size


def benchmark_decompress(tool: str, input_file: str, output_file: str, runs: int) -> Tuple[float, float]:
    """Benchmark decompression. Returns (median_time, stdev)."""
    bin_path = {"gzip": GZIP, "pigz": PIGZ, "rigz": RIGZ}[tool]
    
    cmd = [bin_path, "-d", "-c", input_file]
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - start)
    
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    
    return median, stdev


def check_tools() -> bool:
    """Verify all tools exist."""
    missing = [t for t in [GZIP, PIGZ, RIGZ] if not os.path.isfile(t)]
    if missing:
        print("Missing tools:")
        for t in missing:
            print(f"  {t}")
        print("\nRun 'make deps build' first.")
        return False
    return True


def run_benchmark(levels: List[int], threads: List[int], sizes: List[int]) -> Dict:
    """Run the full benchmark suite."""
    
    results = {
        "compress": [],
        "decompress": [],
        "wins": 0,
        "losses": 0,
        "details": []
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for size_mb in sizes:
            # Generate test file
            test_file = tmpdir / f"test_{size_mb}mb.txt"
            print(f"Generating {size_mb}MB test file...")
            generate_test_file(str(test_file), size_mb)
            
            runs = RUNS_BY_SIZE.get(size_mb, 3)
            
            for level in levels:
                for thread_count in threads:
                    print()
                    print(f"{'='*60}")
                    print(f"  {size_mb}MB, Level {level}, {thread_count} thread(s) ({runs} runs)")
                    print(f"{'='*60}")
                    
                    # === COMPRESSION ===
                    print()
                    print("Compression:")
                    
                    comp_results = {}
                    comp_files = {}
                    
                    for tool in ["gzip", "pigz", "rigz"]:
                        out_file = tmpdir / f"test.{tool}.l{level}.t{thread_count}.gz"
                        median, stdev, out_size = benchmark_compress(
                            tool, level, thread_count, str(test_file), str(out_file), runs
                        )
                        comp_results[tool] = (median, stdev, out_size)
                        comp_files[tool] = out_file
                        
                        print(f"  {tool:5}: {format_time(median):>8} ±{format_time(stdev):>6}  {format_size(out_size):>10}")
                    
                    # Check rigz compression performance
                    rigz_time = comp_results["rigz"][0]
                    baseline_tool = "gzip" if thread_count == 1 else "pigz"
                    baseline_time = comp_results[baseline_tool][0]
                    
                    diff_pct = (rigz_time / baseline_time - 1) * 100
                    if diff_pct <= MAX_OVERHEAD_PCT:
                        status = "✓ WIN" if diff_pct < 0 else "✓ OK"
                        results["wins"] += 1
                    else:
                        status = "✗ SLOW"
                        results["losses"] += 1
                    
                    print(f"\n  rigz vs {baseline_tool}: {diff_pct:+.1f}% {status}")
                    results["details"].append(f"Compress {size_mb}MB L{level} T{thread_count}: {diff_pct:+.1f}%")
                    
                    # === DECOMPRESSION ===
                    print()
                    print("Decompression (using rigz-compressed file):")
                    
                    rigz_compressed = comp_files["rigz"]
                    decomp_results = {}
                    
                    for tool in ["gzip", "pigz", "rigz"]:
                        out_file = tmpdir / f"test.decomp.{tool}.tar"
                        median, stdev = benchmark_decompress(tool, str(rigz_compressed), str(out_file), runs)
                        decomp_results[tool] = (median, stdev)
                        print(f"  {tool:5}: {format_time(median):>8} ±{format_time(stdev):>6}")
                        out_file.unlink()
                    
                    # Check rigz decompression performance
                    rigz_time = decomp_results["rigz"][0]
                    # For decompression, compare against gzip (pigz decompression isn't parallelized much)
                    baseline_time = decomp_results["gzip"][0]
                    pigz_time = decomp_results["pigz"][0]
                    
                    # Use the faster of gzip/pigz as baseline
                    best_baseline = min(baseline_time, pigz_time)
                    best_name = "gzip" if baseline_time <= pigz_time else "pigz"
                    
                    diff_pct = (rigz_time / best_baseline - 1) * 100
                    if diff_pct <= MAX_OVERHEAD_PCT:
                        status = "✓ WIN" if diff_pct < 0 else "✓ OK"
                        results["wins"] += 1
                    else:
                        status = "✗ SLOW"
                        results["losses"] += 1
                    
                    print(f"\n  rigz vs {best_name}: {diff_pct:+.1f}% {status}")
                    results["details"].append(f"Decompress {size_mb}MB L{level} T{thread_count}: {diff_pct:+.1f}%")
                    
                    # Clean up compressed files
                    for f in comp_files.values():
                        if f.exists():
                            f.unlink()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rigz Performance Suite - benchmark compression and decompression"
    )
    parser.add_argument("--full", action="store_true", 
                       help="Run full suite including 100MB files")
    parser.add_argument("--levels", type=str, default=None,
                       help="Comma-separated compression levels (default: 1,6,9)")
    parser.add_argument("--threads", type=str, default=None,
                       help="Comma-separated thread counts (default: 1,4)")
    parser.add_argument("--sizes", type=str, default=None,
                       help="Comma-separated sizes in MB (default: 1,10 or 1,10,100 with --full)")
    
    args = parser.parse_args()
    
    # Parse arguments
    levels = [int(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS
    threads = [int(x) for x in args.threads.split(",")] if args.threads else DEFAULT_THREADS
    
    if args.sizes:
        sizes = [int(x) for x in args.sizes.split(",")]
    else:
        sizes = DEFAULT_SIZES_FULL if args.full else DEFAULT_SIZES_QUICK
    
    print("=" * 60)
    print("  Rigz Performance Suite")
    print("=" * 60)
    print()
    print(f"Levels:  {levels}")
    print(f"Threads: {threads}")
    print(f"Sizes:   {sizes} MB")
    print(f"Target:  rigz within {MAX_OVERHEAD_PCT}% of gzip/pigz")
    print()
    
    if not check_tools():
        return 1
    
    results = run_benchmark(levels, threads, sizes)
    
    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    
    total = results["wins"] + results["losses"]
    print(f"Results: {results['wins']}/{total} passed, {results['losses']} failed")
    print()
    
    if results["losses"] > 0:
        print("Failures:")
        for detail in results["details"]:
            if "SLOW" in detail or float(detail.split(":")[1].strip().rstrip("%")) > MAX_OVERHEAD_PCT:
                print(f"  ✗ {detail}")
        print()
        return 1
    else:
        print("✓ rigz beats or matches gzip/pigz in all tests!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
