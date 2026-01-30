#!/usr/bin/env python3
"""
Decompression benchmark runner.

Benchmarks decompression performance for gzippy vs pigz, gzip, igzip, rapidgzip.

Usage:
    python3 scripts/benchmark_decompression.py \
        --binaries ./bin \
        --compressed-file ./archive/compressed.gz \
        --original-file ./archive/original.bin \
        --threads 4 \
        --archive-type silesia-dynamic \
        --output results/decompression.json
"""

import argparse
import filecmp
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
    compressed_size = os.path.getsize(compressed_file)

    # Build command based on tool
    if tool == "gzippy":
        cmd = [bin_path, "-d", f"-p{threads}"]
    elif tool == "pigz":
        cmd = [bin_path, "-d", f"-p{threads}"]
    elif tool == "unpigz":
        cmd = [bin_path, f"-p{threads}"]
    elif tool == "gzip":
        cmd = [bin_path, "-d"]
    elif tool == "igzip":
        cmd = [bin_path, "-d"]
    elif tool == "rapidgzip":
        cmd = [bin_path, "-d", "-P", str(threads)]
    elif tool == "libdeflate":
        cmd = [bin_path, "-d"]  # libdeflate-gzip is single-threaded
    else:
        return {"error": f"unknown tool: {tool}"}

    def run_decompress():
        with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
            result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
        return result.returncode == 0

    # Warmup
    if not run_decompress():
        return {"error": f"{tool} decompression failed on warmup"}

    # Verify correctness
    if not filecmp.cmp(original_file, output_file, shallow=False):
        return {"error": f"{tool} decompression produced incorrect output", "status": "fail"}

    # Adaptive benchmark
    times = []
    converged = False

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
        "original_size": original_size,
        "compressed_size": compressed_size,
        "speed_mbps": original_size / median / 1_000_000,
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description="Decompression benchmark runner")
    parser.add_argument("--binaries", type=str, required=True,
                       help="Directory containing tool binaries")
    parser.add_argument("--compressed-file", type=str, required=True,
                       help="Path to compressed file")
    parser.add_argument("--original-file", type=str, required=True,
                       help="Path to original uncompressed file")
    parser.add_argument("--threads", type=int, required=True,
                       help="Number of threads to use")
    parser.add_argument("--archive-type", type=str, required=True,
                       help="Type of archive being decompressed")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")

    args = parser.parse_args()

    binaries_dir = Path(args.binaries)
    original_size = os.path.getsize(args.original_file)
    compressed_size = os.path.getsize(args.compressed_file)

    # Find available tools
    tools = {
        "gzippy": find_binary(binaries_dir, "gzippy"),
        "pigz": find_binary(binaries_dir, "pigz"),
        "unpigz": find_binary(binaries_dir, "unpigz"),
        "igzip": find_binary(binaries_dir, "igzip"),
        "rapidgzip": find_binary(binaries_dir, "rapidgzip"),
        "libdeflate": find_binary(binaries_dir, "libdeflate-gzip"),
        "gzip": "/usr/bin/gzip",
    }

    print(f"=== Decompression Benchmark ===")
    print(f"Archive type: {args.archive_type}")
    print(f"Original size: {original_size / 1_000_000:.1f} MB")
    print(f"Compressed size: {compressed_size / 1_000_000:.1f} MB")
    print(f"Threads: {args.threads}")
    print(f"Available tools: {[k for k, v in tools.items() if v]}")
    print()

    results = {
        "archive_type": args.archive_type,
        "threads": args.threads,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "results": [],
    }

    # Decompression tools to benchmark
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
    if tools["libdeflate"]:
        decomp_tools.append(("libdeflate", tools["libdeflate"]))
    decomp_tools.append(("gzip", tools["gzip"]))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for tool_name, bin_path in decomp_tools:
            # Skip multi-threaded benchmark for single-threaded tools
            if tool_name in ("gzip", "igzip", "libdeflate") and args.threads > 1:
                continue

            out_file = str(tmpdir / f"out-{tool_name}.bin")
            result = benchmark_decompression(
                tool_name, bin_path, args.compressed_file, out_file,
                args.original_file, args.threads
            )

            if "error" not in result:
                print(f"  {tool_name}: {result['speed_mbps']:.1f} MB/s, "
                      f"{result['trials']} trials")
            else:
                print(f"  {tool_name}: {result.get('error', 'failed')}")

            result["archive_type"] = args.archive_type
            result["threads"] = args.threads
            results["results"].append(result)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
