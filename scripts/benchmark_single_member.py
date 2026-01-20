#!/usr/bin/env python3
"""
Benchmark single-member gzip decompression.

This specifically tests the 4-phase hyper-parallel pipeline:
  Phase 1: Window Boot (sequential first chunk for 32KB window)
  Phase 2: Speculative Parallel Decode (markers for unresolved back-refs)
  Phase 3: Window Propagation + SIMD Marker Replacement
  Phase 4: Write Output

Single-member files are created by standard gzip (not pigz/gzippy which create
multi-member files). They're the hardest case for parallel decompression because
there's no natural parallelization boundary.

Usage:
    python3 scripts/benchmark_single_member.py \
        --binaries ./bin \
        --compressed-file /tmp/giant.tar.gz \
        --original-file /tmp/giant.tar \
        --threads 8 \
        --output results/single-member.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


# Benchmark configuration
MIN_TRIALS = 3       # Single-member is slow, fewer trials
MAX_TRIALS = 10
TARGET_CV = 0.05


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


def benchmark_decompress(
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
    else:
        return {"error": f"unknown tool: {tool}"}
    
    def run_decompress():
        with open(compressed_file, 'rb') as fin, open(output_file, 'wb') as fout:
            result = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    
    # Warmup
    print(f"  {tool}: warming up...", end="", flush=True)
    if not run_decompress():
        print(" FAILED")
        return {"error": f"{tool} decompression failed on warmup", "tool": tool}
    
    # Verify correctness
    import filecmp
    if not filecmp.cmp(original_file, output_file, shallow=False):
        print(" INCORRECT OUTPUT")
        return {"error": f"{tool} decompression produced incorrect output", "tool": tool, "status": "fail"}
    
    # Benchmark
    times = []
    converged = False
    
    for trial in range(MAX_TRIALS):
        start = time.perf_counter()
        if not run_decompress():
            return {"error": f"{tool} decompression failed on trial {trial}", "tool": tool}
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if len(times) >= MIN_TRIALS:
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            cv = stdev / mean if mean > 0 else 1.0
            if cv < TARGET_CV:
                converged = True
                break
    
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0
    cv = stdev / mean if mean > 0 else 0
    speed = original_size / median / 1_000_000
    
    print(f" {speed:.1f} MB/s ({len(times)} trials, CV={cv:.2%})")
    
    return {
        "tool": tool,
        "operation": "decompress",
        "threads": threads,
        "times": times,
        "median": median,
        "mean": mean,
        "stdev": stdev,
        "cv": cv,
        "trials": len(times),
        "converged": converged,
        "speed_mbps": speed,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": compressed_size / original_size,
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description="Single-member decompression benchmark")
    parser.add_argument("--binaries", type=str, required=True,
                       help="Directory containing tool binaries")
    parser.add_argument("--compressed-file", type=str, required=True,
                       help="Path to single-member gzip file")
    parser.add_argument("--original-file", type=str, required=True,
                       help="Path to original uncompressed file")
    parser.add_argument("--threads", type=int, required=True,
                       help="Number of threads to use")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    binaries_dir = Path(args.binaries)
    
    # Find available tools
    tools = {
        "gzippy": find_binary(binaries_dir, "gzippy"),
        "unpigz": find_binary(binaries_dir, "unpigz"),
        "pigz": find_binary(binaries_dir, "pigz"),
        "igzip": find_binary(binaries_dir, "igzip"),
        "rapidgzip": find_binary(binaries_dir, "rapidgzip"),
        "gzip": "/usr/bin/gzip",
    }
    
    original_size = os.path.getsize(args.original_file)
    compressed_size = os.path.getsize(args.compressed_file)
    
    print("=" * 70)
    print("SINGLE-MEMBER DECOMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"Original:   {original_size / 1_000_000:.1f} MB")
    print(f"Compressed: {compressed_size / 1_000_000:.1f} MB ({compressed_size * 100 / original_size:.1f}%)")
    print(f"Threads:    {args.threads}")
    print()
    print("This tests the 4-phase hyper-parallel pipeline:")
    print("  Phase 1: Window Boot (sequential)")
    print("  Phase 2: Speculative Parallel Decode")
    print("  Phase 3: Window Propagation + SIMD Marker Replacement")
    print("  Phase 4: Write Output")
    print()
    print("Tools to test:", [k for k, v in tools.items() if v])
    print("-" * 70)
    
    results = {
        "benchmark": "single-member",
        "threads": args.threads,
        "original_size_mb": original_size / 1_000_000,
        "compressed_size_mb": compressed_size / 1_000_000,
        "results": [],
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_file = tmp.name
    
    try:
        # Benchmark each tool
        # Order matters: gzippy first, then competitors
        tool_order = ["gzippy", "rapidgzip", "unpigz", "igzip", "gzip"]
        
        for tool_name in tool_order:
            bin_path = tools.get(tool_name)
            if not bin_path:
                continue
            
            # Skip multi-threaded for single-threaded tools
            if tool_name in ("gzip", "igzip") and args.threads > 1:
                continue
            
            # Use unpigz instead of pigz for decompression
            if tool_name == "pigz" and tools.get("unpigz"):
                continue
            
            result = benchmark_decompress(
                tool_name, bin_path,
                args.compressed_file, output_file,
                args.original_file, args.threads
            )
            results["results"].append(result)
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)
    
    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Tool':<12} {'Speed (MB/s)':<15} {'Trials':<8} {'Status'}")
    print("-" * 70)
    
    gzippy_speed = None
    for r in results["results"]:
        if "error" in r:
            print(f"{r['tool']:<12} {'FAILED':<15} {'-':<8} ❌")
        else:
            status = "✅" if r["status"] == "pass" else "❌"
            print(f"{r['tool']:<12} {r['speed_mbps']:<15.1f} {r['trials']:<8} {status}")
            if r["tool"] == "gzippy":
                gzippy_speed = r["speed_mbps"]
    
    # Comparisons
    if gzippy_speed:
        print()
        print("GZIPPY vs COMPETITORS:")
        for r in results["results"]:
            if r["tool"] == "gzippy" or "error" in r:
                continue
            ratio = gzippy_speed / r["speed_mbps"]
            icon = "🏆" if ratio >= 1.0 else "📉"
            print(f"  {icon} vs {r['tool']}: {ratio:.2f}x")
    
    # Pass/fail determination
    passed = True
    reasons = []
    
    gzippy = next((r for r in results["results"] if r["tool"] == "gzippy" and "error" not in r), None)
    rapidgzip = next((r for r in results["results"] if r["tool"] == "rapidgzip" and "error" not in r), None)
    unpigz = next((r for r in results["results"] if r["tool"] == "unpigz" and "error" not in r), None)
    
    if gzippy and rapidgzip:
        ratio = gzippy["speed_mbps"] / rapidgzip["speed_mbps"]
        if ratio < 0.99:  # Must be within 1% of rapidgzip
            passed = False
            reasons.append(f"gzippy {ratio:.2f}x rapidgzip (need ≥0.99)")
    
    if gzippy and unpigz:
        ratio = gzippy["speed_mbps"] / unpigz["speed_mbps"]
        if ratio < 1.0:  # Must beat pigz
            passed = False
            reasons.append(f"gzippy {ratio:.2f}x unpigz (need ≥1.0)")
    
    results["passed"] = passed
    results["reasons"] = reasons
    
    print()
    print("=" * 70)
    print(f"{'✅ PASSED' if passed else '❌ FAILED'}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")
    print("=" * 70)
    
    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults written to {args.output}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
