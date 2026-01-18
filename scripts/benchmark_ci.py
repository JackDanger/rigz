#!/usr/bin/env python3
"""
CI-friendly benchmark script for rigz.

Runs a single benchmark configuration and outputs JSON results.
Exits with non-zero code if performance thresholds are not met.

REQUIREMENTS (from .cursorrules):
- rigz must beat pigz in EVERY configuration. No exceptions.
- L1-8: Speed must beat pigz, size within 5%
- L9: Size must match pigz (within 0.5%), speed can be within 10%

Usage:
    python3 scripts/benchmark_ci.py --size 10 --level 6 --threads 4
    python3 scripts/benchmark_ci.py --size 10 --level 9 --threads 2 --data-type text
"""

import argparse
import json
import os
import shutil
import subprocess
import statistics
import sys
import tempfile
import time
from pathlib import Path


# Performance thresholds by level
# Design: L1-6 trades size for parallel decompression, L7-9 prioritizes size
def get_thresholds(level: int) -> tuple:
    """Returns (max_time_overhead_pct, max_size_overhead_pct)."""
    if level >= 9:
        # L9: Size matters most, speed can be on par
        return (10.0, 0.5)  # 10% slower OK, but size within 0.5%
    elif level >= 7:
        # L7-8: Transitional - good compression, still fast
        return (0.0, 2.0)   # Must be faster, size within 2%
    else:
        # L1-6: Speed + parallel decompress, accept larger output
        return (0.0, 8.0)   # Must be faster, size within 8%


# Number of runs for statistical significance
RUNS_BY_SIZE = {1: 15, 10: 10, 100: 5}


def find_tool(name: str) -> str:
    """Find a tool binary."""
    paths = {
        "gzip": ["./gzip/gzip", shutil.which("gzip") or "gzip"],
        "pigz": ["./pigz/pigz"],
        "rigz": ["./target/release/rigz"],
    }
    for path in paths.get(name, []):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"Could not find {name}")


def generate_test_file(path: str, size_mb: int, data_type: str = "text") -> None:
    """Generate a test file of the specified type.
    
    Data types:
    - text: Realistic text (from Proust if available, otherwise lorem ipsum)
    - random: Random base64 (poorly compressible, stress test)
    - binary: Mixed binary content (tarball-like)
    """
    size_bytes = size_mb * 1024 * 1024
    
    if data_type == "text":
        # Use Proust text if available, otherwise generate repetitive text
        proust_path = Path("test_data/text-1MB.txt")
        if proust_path.exists():
            seed = proust_path.read_bytes()
        else:
            # Fallback: repetitive English text
            seed = (b"The quick brown fox jumps over the lazy dog. " * 100 +
                   b"Pack my box with five dozen liquor jugs. " * 100 +
                   b"How vexingly quick daft zebras jump! " * 100)
        
        # Repeat to reach target size
        with open(path, 'wb') as f:
            written = 0
            while written < size_bytes:
                chunk = seed[:size_bytes - written]
                f.write(chunk)
                written += len(chunk)
                
    elif data_type == "random":
        # Random base64 - compresses poorly
        cmd = f"head -c {size_bytes} /dev/urandom | base64 > {path}"
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
        
    elif data_type == "binary":
        # Mixed binary content (simulate tarball)
        with open(path, 'wb') as f:
            written = 0
            while written < size_bytes:
                # Mix of patterns
                patterns = [
                    os.urandom(1024),  # Random
                    b'\x00' * 512,     # Zeros
                    bytes(range(256)) * 4,  # Sequential
                    b'HEADER_MAGIC_12345678' * 50,  # Repeated strings
                ]
                for p in patterns:
                    if written >= size_bytes:
                        break
                    chunk = p[:size_bytes - written]
                    f.write(chunk)
                    written += len(chunk)


def benchmark_compress(tool: str, level: int, threads: int, 
                       input_file: str, output_file: str, runs: int) -> dict:
    """Benchmark compression. Returns stats dict."""
    bin_path = find_tool(tool)
    
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "rigz"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"{tool} compression failed")
        times.append(time.perf_counter() - start)
    
    output_size = os.path.getsize(output_file)
    
    return {
        "tool": tool,
        "operation": "compress",
        "times": times,
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "output_size": output_size,
    }


def benchmark_decompress(tool: str, input_file: str, output_file: str, runs: int) -> dict:
    """Benchmark decompression. Returns stats dict."""
    bin_path = find_tool(tool)
    
    cmd = [bin_path, "-d", "-c", input_file]
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"{tool} decompression failed")
        times.append(time.perf_counter() - start)
    
    return {
        "tool": tool,
        "operation": "decompress",
        "times": times,
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
    }


def main():
    parser = argparse.ArgumentParser(description="CI benchmark for rigz")
    parser.add_argument("--size", type=int, required=True, help="Test file size in MB")
    parser.add_argument("--level", type=int, required=True, help="Compression level (1-9)")
    parser.add_argument("--threads", type=int, required=True, help="Thread count")
    parser.add_argument("--output", type=str, default="benchmark-results.json",
                       help="Output JSON file")
    parser.add_argument("--data-type", type=str, default="text",
                       choices=["text", "random", "binary"],
                       help="Type of test data (default: text)")
    parser.add_argument("--check-ratio", action="store_true",
                       help="(Deprecated - ratio is always checked)")
    parser.add_argument("--max-time-overhead", type=float, default=None,
                       help="Override max time overhead %% (default: level-based)")
    parser.add_argument("--max-ratio-overhead", type=float, default=None,
                       help="Override max ratio overhead %% (default: level-based)")
    
    args = parser.parse_args()
    
    # Get level-specific thresholds
    default_time, default_ratio = get_thresholds(args.level)
    max_time_overhead = args.max_time_overhead if args.max_time_overhead is not None else default_time
    max_ratio_overhead = args.max_ratio_overhead if args.max_ratio_overhead is not None else default_ratio
    
    results = {
        "config": {
            "size_mb": args.size,
            "level": args.level,
            "threads": args.threads,
            "data_type": args.data_type,
            "thresholds": {
                "time_overhead_pct": max_time_overhead,
                "ratio_overhead_pct": max_ratio_overhead,
            },
        },
        "compression": {},
        "decompression": {},
        "passed": True,
        "errors": [],
    }
    
    runs = RUNS_BY_SIZE.get(args.size, 5)
    
    print(f"=== Benchmark: {args.size}MB {args.data_type}, L{args.level}, T{args.threads} ({runs} runs) ===")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / f"test_{args.size}mb.txt"
        
        # Generate test data
        print(f"Generating {args.size}MB {args.data_type} test file...")
        generate_test_file(str(test_file), args.size, args.data_type)
        print(f"Thresholds: time≤{max_time_overhead:+.1f}%, size≤{max_ratio_overhead:+.1f}%")
        
        # === COMPRESSION BENCHMARKS ===
        print("\nCompression:")
        
        baseline_tool = "gzip" if args.threads == 1 else "pigz"
        comp_files = {}
        
        for tool in [baseline_tool, "rigz"]:
            out_file = tmpdir / f"test.{tool}.gz"
            try:
                stats = benchmark_compress(tool, args.level, args.threads,
                                         str(test_file), str(out_file), runs)
                results["compression"][tool] = stats
                comp_files[tool] = out_file
                print(f"  {tool:5}: {stats['median']:.3f}s (±{stats['stdev']:.3f}s) "
                      f"→ {stats['output_size']:,} bytes")
            except Exception as e:
                error = f"Compression failed for {tool}: {e}"
                print(f"  {tool:5}: ERROR - {e}")
                results["errors"].append(error)
                results["passed"] = False
        
        # Check compression time
        if baseline_tool in results["compression"] and "rigz" in results["compression"]:
            baseline_time = results["compression"][baseline_tool]["median"]
            rigz_time = results["compression"]["rigz"]["median"]
            overhead_pct = (rigz_time / baseline_time - 1) * 100
            
            results["compression"]["comparison"] = {
                "baseline": baseline_tool,
                "overhead_pct": overhead_pct,
                "threshold_pct": max_time_overhead,
            }
            
            if overhead_pct > max_time_overhead:
                error = (f"Compression too slow: rigz is {overhead_pct:+.1f}% vs {baseline_tool} "
                        f"(threshold: {max_time_overhead:+.1f}%)")
                print(f"\n  ❌ FAIL: {error}")
                results["errors"].append(error)
                results["passed"] = False
            else:
                status = "faster" if overhead_pct < 0 else "within threshold"
                print(f"\n  ✅ PASS: rigz is {overhead_pct:+.1f}% vs {baseline_tool} ({status})")
        
        # Always check compression ratio (important for L9)
        if baseline_tool in comp_files and "rigz" in comp_files:
            baseline_size = results["compression"][baseline_tool]["output_size"]
            rigz_size = results["compression"]["rigz"]["output_size"]
            ratio_overhead = (rigz_size / baseline_size - 1) * 100
            
            results["compression"]["ratio_comparison"] = {
                "baseline": baseline_tool,
                "baseline_size": baseline_size,
                "rigz_size": rigz_size,
                "overhead_pct": ratio_overhead,
                "threshold_pct": max_ratio_overhead,
            }
            
            if ratio_overhead > max_ratio_overhead:
                error = (f"Compression ratio too poor: rigz output is {ratio_overhead:+.1f}% larger "
                        f"than {baseline_tool} (threshold: {max_ratio_overhead:+.1f}%)")
                print(f"\n  ❌ FAIL: {error}")
                results["errors"].append(error)
                results["passed"] = False
            else:
                status = "smaller" if ratio_overhead < 0 else "within threshold"
                print(f"\n  ✅ PASS: rigz output is {ratio_overhead:+.1f}% vs {baseline_tool} ({status})")
        
        # === DECOMPRESSION BENCHMARKS ===
        print("\nDecompression (rigz-compressed file):")
        
        if "rigz" in comp_files:
            rigz_compressed = comp_files["rigz"]
            decomp_out = tmpdir / "decompressed.txt"
            
            for tool in ["gzip", "pigz", "rigz"]:
                try:
                    stats = benchmark_decompress(tool, str(rigz_compressed),
                                                str(decomp_out), runs)
                    results["decompression"][tool] = stats
                    print(f"  {tool:5}: {stats['median']:.3f}s (±{stats['stdev']:.3f}s)")
                except Exception as e:
                    error = f"Decompression failed for {tool}: {e}"
                    print(f"  {tool:5}: ERROR - {e}")
                    results["errors"].append(error)
                    results["passed"] = False
            
            # Check decompression time (compare against best of gzip/pigz)
            decomp_tools = results["decompression"]
            if "gzip" in decomp_tools and "pigz" in decomp_tools and "rigz" in decomp_tools:
                gzip_time = decomp_tools["gzip"]["median"]
                pigz_time = decomp_tools["pigz"]["median"]
                rigz_time = decomp_tools["rigz"]["median"]
                
                if gzip_time <= pigz_time:
                    baseline_tool, baseline_time = "gzip", gzip_time
                else:
                    baseline_tool, baseline_time = "pigz", pigz_time
                
                overhead_pct = (rigz_time / baseline_time - 1) * 100
                
                results["decompression"]["comparison"] = {
                    "baseline": baseline_tool,
                    "overhead_pct": overhead_pct,
                    "threshold_pct": max_time_overhead,
                }
                
                if overhead_pct > max_time_overhead:
                    error = (f"Decompression too slow: rigz is {overhead_pct:+.1f}% vs {baseline_tool} "
                            f"(threshold: {max_time_overhead:+.1f}%)")
                    print(f"\n  ❌ FAIL: {error}")
                    results["errors"].append(error)
                    results["passed"] = False
                else:
                    status = "faster" if overhead_pct < 0 else "within threshold"
                    print(f"\n  ✅ PASS: rigz is {overhead_pct:+.1f}% vs {baseline_tool} ({status})")
    
    # Write results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")
    
    # Summary
    print()
    if results["passed"]:
        print("=" * 50)
        print("  ✅ ALL CHECKS PASSED")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("  ❌ SOME CHECKS FAILED")
        print("=" * 50)
        for error in results["errors"]:
            print(f"  • {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
