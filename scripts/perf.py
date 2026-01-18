#!/usr/bin/env python3
"""
Gzippy Performance Suite

Benchmarks compression AND decompression against gzip and pigz.
Tests that gzippy beats or matches both tools.

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
IGZIP = "./isa-l/build/igzip"
ZOPFLI = "./zopfli/zopfli"
GZIPPY = "./target/release/gzippy"

# Defaults
DEFAULT_LEVELS = [1, 6, 9]
DEFAULT_THREADS = [1, 4]
DEFAULT_SIZES_QUICK = [1, 10]      # MB
DEFAULT_SIZES_FULL = [1, 10, 100]  # MB

# Statistical config - more runs for shorter tests to reduce noise
RUNS_BY_SIZE = {1: 30, 10: 15, 100: 7}

# Acceptable overhead (gzippy can be up to 5% slower and still "pass")
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
    bin_path = {"gzip": GZIP, "pigz": PIGZ, "igzip": IGZIP, "zopfli": ZOPFLI, "gzippy": GZIPPY}[tool]
    
    # For L10-L12 benchmarks, compare other tools at their max level (9)
    effective_level = level if tool == "gzippy" else min(level, 9)
    
    # Handle tool-specific command line syntax
    if tool == "igzip":
        # Map gzip levels 1-9 to igzip levels 0-3
        igzip_level = min(3, max(0, (effective_level - 1) // 3))
        cmd = [bin_path, f"-{igzip_level}"]
        if threads > 1:
            cmd.append(f"-T{threads}")
        cmd.extend(["-c", input_file])
    elif tool == "zopfli":
        # zopfli uses iterations, not levels. Use 5 iterations for speed.
        cmd = [bin_path, "--i5", "-c", input_file]
    elif tool == "gzippy" and level >= 10:
        # Ultra compression levels need --level flag
        cmd = [bin_path, "--level", str(level), f"-p{threads}", "-c", input_file]
    else:
        cmd = [bin_path, f"-{effective_level}"]
        if tool in ("pigz", "gzippy"):
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
    bin_path = {"gzip": GZIP, "pigz": PIGZ, "igzip": IGZIP, "gzippy": GZIPPY}[tool]
    
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
    missing = [t for t in [GZIP, PIGZ, IGZIP, ZOPFLI, GZIPPY] if not os.path.isfile(t)]
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
                    
                    # zopfli only at L9 (very slow, only for max compression)
                    tools = ["gzip", "pigz", "igzip", "gzippy"]
                    if level >= 9:
                        tools.append("zopfli")
                    
                    for tool in tools:
                        out_file = tmpdir / f"test.{tool}.l{level}.t{thread_count}.gz"
                        median, stdev, out_size = benchmark_compress(
                            tool, level, thread_count, str(test_file), str(out_file), runs
                        )
                        comp_results[tool] = (median, stdev, out_size)
                        comp_files[tool] = out_file
                        
                        print(f"  {tool:7}: {format_time(median):>8} ±{format_time(stdev):>6}  {format_size(out_size):>10}")
                    
                    # Check gzippy compression performance against all competitors
                    gzippy_time = comp_results["gzippy"][0]
                    
                    # Compare against gzip (single-thread) or pigz (multi-thread)
                    # NOTE: igzip is benchmarked for info but not used as comparison
                    # because it produces ~15% larger files (speed-only optimization)
                    if thread_count == 1:
                        competitors = {"gzip": comp_results["gzip"][0]}
                    else:
                        competitors = {"pigz": comp_results["pigz"][0]}
                    
                    fastest_name = min(competitors, key=competitors.get)
                    fastest_time = competitors[fastest_name]
                    
                    diff_pct = (gzippy_time / fastest_time - 1) * 100
                    if diff_pct <= MAX_OVERHEAD_PCT:
                        status = "✓ WIN" if diff_pct < 0 else "✓ OK"
                        results["wins"] += 1
                    else:
                        status = "✗ SLOW"
                        results["losses"] += 1
                    
                    print(f"\n  gzippy vs {fastest_name}: {diff_pct:+.1f}% {status}")
                    results["details"].append(f"Compress {size_mb}MB L{level} T{thread_count}: {diff_pct:+.1f}%")
                    
                    # === DECOMPRESSION ===
                    print()
                    print("Decompression (using gzippy-compressed file):")
                    
                    gzippy_compressed = comp_files["gzippy"]
                    decomp_results = {}
                    
                    for tool in ["gzip", "pigz", "igzip", "gzippy"]:
                        out_file = tmpdir / f"test.decomp.{tool}.tar"
                        median, stdev = benchmark_decompress(tool, str(gzippy_compressed), str(out_file), runs)
                        decomp_results[tool] = (median, stdev)
                        print(f"  {tool:6}: {format_time(median):>8} ±{format_time(stdev):>6}")
                        out_file.unlink()
                    
                    # Check gzippy decompression performance
                    gzippy_time = decomp_results["gzippy"][0]
                    
                    # Compare against all competitors
                    competitors = {
                        "gzip": decomp_results["gzip"][0],
                        "pigz": decomp_results["pigz"][0],
                        "igzip": decomp_results["igzip"][0],
                    }
                    
                    fastest_name = min(competitors, key=competitors.get)
                    fastest_time = competitors[fastest_name]
                    
                    diff_pct = (gzippy_time / fastest_time - 1) * 100
                    if diff_pct <= MAX_OVERHEAD_PCT:
                        status = "✓ WIN" if diff_pct < 0 else "✓ OK"
                        results["wins"] += 1
                    else:
                        status = "✗ SLOW"
                        results["losses"] += 1
                    
                    print(f"\n  gzippy vs {fastest_name}: {diff_pct:+.1f}% {status}")
                    results["details"].append(f"Decompress {size_mb}MB L{level} T{thread_count}: {diff_pct:+.1f}%")
                    
                    # Clean up compressed files
                    for f in comp_files.values():
                        if f.exists():
                            f.unlink()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Gzippy Performance Suite - benchmark compression and decompression"
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

    if args.full:
        levels = [x + 1 for x in range(9)]
        threads = [1, 2, 3, 7, os.cpu_count()]
        sizes = DEFAULT_SIZES_FULL
    else:
        levels = [int(x) for x in args.levels.split(",")] if args.levels else DEFAULT_LEVELS
        threads = [int(x) for x in args.threads.split(",")] if args.threads else DEFAULT_THREADS
        
        if args.sizes:
            sizes = [int(x) for x in args.sizes.split(",")]
        else:
            sizes = DEFAULT_SIZES_QUICK
    
    print("=" * 60)
    print("  Gzippy Performance Suite")
    print("=" * 60)
    print()
    print(f"Levels:  {levels}")
    print(f"Threads: {threads}")
    print(f"Sizes:   {sizes} MB")
    print(f"Target:  gzippy within {MAX_OVERHEAD_PCT}% of gzip/pigz")
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
        print("✓ gzippy beats or matches gzip/pigz in all tests!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
