#!/usr/bin/env python3
"""
Unified benchmark runner for compression and decompression.

This script benchmarks both compression and decompression in a single run,
outputting results to a JSON file. It's designed to be called by the unified
benchmarks.yml workflow.

Usage:
    python3 scripts/run_benchmarks.py \
        --binaries ./bin \
        --data-file ./data/test-data.bin \
        --compressed-dir ./data/compressed \
        --level 6 \
        --threads 4 \
        --size 10 \
        --data-type text \
        --output results/benchmark.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Benchmark configuration
MIN_TRIALS = 5
MAX_TRIALS = 30
TARGET_CV = 0.05  # 5% coefficient of variation


def find_binary(binaries_dir: Path, name: str) -> str | None:
    """Find a binary in the binaries directory."""
    candidates = [
        binaries_dir / name,
        binaries_dir / f"{name}-cli",
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def run_timed(cmd: list, input_file: str | None = None, output_file: str | None = None) -> float | None:
    """Run a command and return execution time in seconds, or None on failure."""
    try:
        start = time.perf_counter()
        if input_file and output_file:
            with open(input_file, 'rb') as fin, open(output_file, 'wb') as fout:
                result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
        elif output_file:
            with open(output_file, 'wb') as fout:
                result = subprocess.run(cmd, stdout=fout, stderr=subprocess.DEVNULL)
        else:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        elapsed = time.perf_counter() - start
        return elapsed if result.returncode == 0 else None
    except Exception as e:
        print(f"  Error running {cmd[0]}: {e}", file=sys.stderr)
        return None


def adaptive_benchmark(cmd: list, input_file: str | None, output_file: str) -> dict:
    """
    Run adaptive benchmarking until results converge.
    
    Returns dict with: times, median, stdev, cv, trials, converged
    """
    times = []
    converged = False
    
    # Warmup run
    run_timed(cmd, input_file, output_file)
    
    for trial in range(MAX_TRIALS):
        elapsed = run_timed(cmd, input_file, output_file)
        if elapsed is None:
            return {"error": "command failed", "times": times}
        
        times.append(elapsed)
        
        if len(times) >= MIN_TRIALS:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            cv = stdev / mean if mean > 0 else 1.0
            
            if cv < TARGET_CV:
                converged = True
                break
    
    if not times:
        return {"error": "no successful runs", "times": []}
    
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    cv = stdev / mean if mean > 0 else 0
    
    return {
        "times": times,
        "median": statistics.median(times),
        "mean": mean,
        "stdev": stdev,
        "cv": cv,
        "trials": len(times),
        "converged": converged,
    }


def benchmark_compression(
    tool: str,
    bin_path: str,
    input_file: str,
    output_file: str,
    level: int,
    threads: int,
) -> dict:
    """Benchmark compression for a single tool."""
    
    # Build command based on tool
    if tool == "gzippy":
        if level >= 10:
            cmd = [bin_path, "--level", str(level), f"-p{threads}", "-c", input_file]
        else:
            cmd = [bin_path, f"-{level}", f"-p{threads}", "-c", input_file]
    elif tool == "pigz":
        cmd = [bin_path, f"-{level}", f"-p{threads}", "-c", input_file]
    elif tool == "gzip":
        cmd = [bin_path, f"-{level}", "-c", input_file]
    elif tool == "igzip":
        # igzip only supports levels 0-3
        igzip_level = min(3, level)
        cmd = [bin_path, f"-{igzip_level}", "-c", input_file]
    elif tool == "zopfli":
        # zopfli: only run once (very slow)
        cmd = [bin_path, "--i5", "-c", input_file]
    else:
        return {"error": f"unknown tool: {tool}"}
    
    # For zopfli, just run once
    if tool == "zopfli":
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            return {"error": "zopfli failed"}
        
        output_size = os.path.getsize(output_file)
        input_size = os.path.getsize(input_file)
        
        return {
            "tool": tool,
            "operation": "compress",
            "times": [elapsed],
            "median": elapsed,
            "mean": elapsed,
            "stdev": 0,
            "cv": 0,
            "trials": 1,
            "converged": True,
            "output_size": output_size,
            "input_size": input_size,
            "ratio": output_size / input_size,
            "speed_mbps": input_size / elapsed / 1_000_000,
        }
    
    # Normal adaptive benchmark
    # Note: cmd writes to stdout, so we use a different approach
    def run_compress():
        with open(output_file, 'wb') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    
    times = []
    converged = False
    
    # Warmup
    run_compress()
    
    for trial in range(MAX_TRIALS):
        start = time.perf_counter()
        if not run_compress():
            return {"error": f"{tool} compression failed"}
        times.append(time.perf_counter() - start)
        
        if len(times) >= MIN_TRIALS:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            cv = stdev / mean if mean > 0 else 1.0
            if cv < TARGET_CV:
                converged = True
                break
    
    output_size = os.path.getsize(output_file)
    input_size = os.path.getsize(input_file)
    median = statistics.median(times)
    
    return {
        "tool": tool,
        "operation": "compress",
        "times": times,
        "median": median,
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "cv": statistics.stdev(times) / statistics.mean(times) if len(times) > 1 else 0,
        "trials": len(times),
        "converged": converged,
        "output_size": output_size,
        "input_size": input_size,
        "ratio": output_size / input_size,
        "speed_mbps": input_size / median / 1_000_000,
    }


def benchmark_decompression(
    tool: str,
    bin_path: str,
    compressed_file: str,
    output_file: str,
    original_file: str,
    threads: int,
) -> dict:
    """Benchmark decompression for a single tool."""
    
    original_size = os.path.getsize(original_file)
    
    # Build command based on tool
    if tool == "gzippy":
        cmd = [bin_path, "-d", f"-p{threads}"]
    elif tool == "pigz":
        # pigz decompression uses unpigz or pigz -d
        cmd = [bin_path, "-d", f"-p{threads}"]
    elif tool == "unpigz":
        cmd = [bin_path, f"-p{threads}"]
    elif tool == "gzip":
        cmd = [bin_path, "-d"]
    elif tool == "igzip":
        cmd = [bin_path, "-d"]
    elif tool == "rapidgzip":
        cmd = [bin_path, "-d", "-P", str(threads)]
    else:
        return {"error": f"unknown tool: {tool}"}
    
    # Adaptive benchmark with stdin/stdout
    times = []
    converged = False
    
    def run_decompress():
        with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
            result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    
    # Warmup
    if not run_decompress():
        return {"error": f"{tool} decompression failed on warmup"}
    
    # Verify correctness
    import filecmp
    if not filecmp.cmp(original_file, output_file, shallow=False):
        return {"error": f"{tool} decompression produced incorrect output", "status": "fail"}
    
    for trial in range(MAX_TRIALS):
        start = time.perf_counter()
        if not run_decompress():
            return {"error": f"{tool} decompression failed"}
        times.append(time.perf_counter() - start)
        
        if len(times) >= MIN_TRIALS:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            cv = stdev / mean if mean > 0 else 1.0
            if cv < TARGET_CV:
                converged = True
                break
    
    median = statistics.median(times)
    
    return {
        "tool": tool,
        "operation": "decompress",
        "times": times,
        "median": median,
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "cv": statistics.stdev(times) / statistics.mean(times) if len(times) > 1 else 0,
        "trials": len(times),
        "converged": converged,
        "speed_mbps": original_size / median / 1_000_000,
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark runner")
    parser.add_argument("--binaries", type=str, required=True,
                       help="Directory containing tool binaries")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to uncompressed test data")
    parser.add_argument("--compressed-dir", type=str, required=True,
                       help="Directory containing pre-compressed variants")
    parser.add_argument("--level", type=int, required=True,
                       help="Compression level to test")
    parser.add_argument("--threads", type=int, required=True,
                       help="Number of threads to use")
    parser.add_argument("--size", type=int, required=True,
                       help="Size of test data in MB")
    parser.add_argument("--data-type", type=str, required=True,
                       choices=["text", "tarball"],
                       help="Type of test data")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    binaries_dir = Path(args.binaries)
    compressed_dir = Path(args.compressed_dir)
    data_file = args.data_file
    
    # Find available tools
    tools = {
        "gzippy": find_binary(binaries_dir, "gzippy"),
        "pigz": find_binary(binaries_dir, "pigz"),
        "unpigz": find_binary(binaries_dir, "unpigz"),
        "igzip": find_binary(binaries_dir, "igzip"),
        "rapidgzip": find_binary(binaries_dir, "rapidgzip"),
        "zopfli": find_binary(binaries_dir, "zopfli"),
        "gzip": "/usr/bin/gzip",  # System gzip
    }
    
    print(f"=== Benchmark: {args.size}MB {args.data_type}, L{args.level}, T{args.threads} ===")
    print(f"Available tools: {[k for k, v in tools.items() if v]}")
    print()
    
    results = {
        "level": args.level,
        "threads": args.threads,
        "size_mb": args.size,
        "data_type": args.data_type,
        "compression": [],
        "decompression": [],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # ============================
        # COMPRESSION BENCHMARKS
        # ============================
        print("--- Compression ---")
        
        # Tools to benchmark for compression
        # igzip only for L1-L3, zopfli only for L9
        comp_tools = ["gzippy", "pigz", "gzip"]
        if args.level <= 3 and tools["igzip"]:
            comp_tools.append("igzip")
        if args.level >= 9 and tools["zopfli"]:
            comp_tools.append("zopfli")
        
        for tool in comp_tools:
            bin_path = tools.get(tool)
            if not bin_path or not os.path.exists(bin_path):
                continue
            
            out_file = str(tmpdir / f"{tool}.gz")
            result = benchmark_compression(
                tool, bin_path, data_file, out_file,
                args.level, args.threads
            )
            
            if "error" not in result:
                print(f"  {tool}: {result['speed_mbps']:.1f} MB/s, "
                      f"{result['output_size']/1_000_000:.2f} MB, "
                      f"{result['trials']} trials")
            else:
                print(f"  {tool}: {result['error']}")
            
            result["level"] = args.level
            result["threads"] = args.threads
            results["compression"].append(result)
        
        # ============================
        # DECOMPRESSION BENCHMARKS
        # ============================
        print("\n--- Decompression ---")
        
        # Find compressed files to decompress
        compressed_files = list(compressed_dir.glob("*.gz"))
        
        # Decompression tools (use unpigz for pigz decompression)
        decomp_tools = []
        if tools["gzippy"]:
            decomp_tools.append(("gzippy", tools["gzippy"]))
        if tools["unpigz"]:
            decomp_tools.append(("unpigz", tools["unpigz"]))
        elif tools["pigz"]:
            decomp_tools.append(("pigz", tools["pigz"]))
        if tools["igzip"]:
            decomp_tools.append(("igzip", tools["igzip"]))
        if tools["rapidgzip"]:
            decomp_tools.append(("rapidgzip", tools["rapidgzip"]))
        decomp_tools.append(("gzip", tools["gzip"]))
        
        for gz_file in compressed_files:
            # Extract source info from filename (e.g., "gzippy-L6.gz")
            name = gz_file.stem
            parts = name.split("-")
            source = parts[0]
            source_level = int(parts[1].replace("L", "")) if len(parts) > 1 else 0
            
            print(f"\n  Source: {name}")
            
            for tool_name, bin_path in decomp_tools:
                # Skip multi-threaded for single-threaded tools
                if tool_name in ("gzip", "igzip") and args.threads > 1:
                    continue
                
                out_file = str(tmpdir / f"out-{tool_name}.bin")
                result = benchmark_decompression(
                    tool_name, bin_path, str(gz_file), out_file,
                    data_file, args.threads
                )
                
                if "error" not in result:
                    print(f"    {tool_name}: {result['speed_mbps']:.1f} MB/s, "
                          f"{result['trials']} trials")
                else:
                    print(f"    {tool_name}: {result.get('error', 'failed')}")
                
                result["source"] = source
                result["source_level"] = source_level
                result["threads"] = args.threads
                results["decompression"].append(result)
    
    # ============================
    # PASS/FAIL DETERMINATION
    # ============================
    gzippy_comp = next((r for r in results["compression"] if r["tool"] == "gzippy"), None)
    pigz_comp = next((r for r in results["compression"] if r["tool"] == "pigz"), None)
    
    passed = True
    reasons = []
    
    if gzippy_comp and pigz_comp and "error" not in gzippy_comp and "error" not in pigz_comp:
        # Speed comparison
        speed_ratio = gzippy_comp["speed_mbps"] / pigz_comp["speed_mbps"]
        if speed_ratio < 0.95:  # Allow 5% slower
            passed = False
            reasons.append(f"gzippy {speed_ratio:.2f}x pigz speed")
        
        # Size comparison (only at L9)
        if args.level >= 9:
            size_ratio = gzippy_comp["output_size"] / pigz_comp["output_size"]
            if size_ratio > 1.005:  # Must be within 0.5%
                passed = False
                reasons.append(f"gzippy {size_ratio:.3f}x pigz size")
    
    results["passed"] = passed
    results["reasons"] = reasons
    
    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
