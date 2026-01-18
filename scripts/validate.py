#!/usr/bin/env python3
"""
Cross-tool validation matrix for gzippy.

Tests that gzippy produces gzip-compatible output by:
1. Creating a tarball from the repo
2. Compressing at multiple levels with multiple thread counts
3. Decompressing with gzip, pigz, and gzippy
4. Verifying all outputs are byte-identical

Uses adaptive trial count (3-10) with statistical convergence detection.
Outputs JSON for charting when --json flag is used.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import statistics
from pathlib import Path

# Tool paths - prefer local builds, fall back to system
import shutil

# Adaptive trial configuration
MIN_TRIALS = 3
MAX_TRIALS = 17
# Coefficient of variation threshold for "stable" results (5%)
CV_THRESHOLD = 0.05


def find_gzip():
    if os.path.isfile("./gzip/gzip") and os.access("./gzip/gzip", os.X_OK):
        return "./gzip/gzip"
    return shutil.which("gzip") or "gzip"


GZIP = find_gzip()
PIGZ = "./pigz/pigz"
GZIPPY = "./target/release/gzippy"
UNGZIPPY = "./target/release/ungzippy"

# Test matrix
LEVELS = [1, 6, 9]
THREADS = [1, 4]
TOOLS = ["gzip", "pigz", "gzippy"]


def run(cmd, capture=False):
    """Run a command, optionally capturing output."""
    if capture:
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0, result.stdout
    else:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0, None


def get_tool_path(tool):
    """Get the binary path for a tool."""
    return {"gzip": GZIP, "pigz": PIGZ, "gzippy": GZIPPY}[tool]


def robust_stats(times):
    """
    Calculate robust statistics, handling outliers from task scheduling.
    
    Uses trimmed statistics: removes highest value if it's an outlier
    (defined as >2x the median). This handles the common case where
    OS task scheduling causes one slow run.
    """
    if len(times) < 2:
        return times[0] if times else 0, 0, times
    
    median = statistics.median(times)
    
    # Identify and remove high outliers (scheduling interference)
    # Keep low outliers as they represent achievable performance
    trimmed = [t for t in times if t <= median * 2]
    
    # If we removed too many, use original
    if len(trimmed) < len(times) // 2:
        trimmed = times
    
    if len(trimmed) >= 2:
        return statistics.median(trimmed), statistics.stdev(trimmed), trimmed
    return trimmed[0], 0, trimmed


def is_stable(times):
    """
    Check if timing results are statistically stable.
    Uses coefficient of variation (CV = stdev/mean).
    """
    if len(times) < MIN_TRIALS:
        return False
    
    median, stdev, _ = robust_stats(times)
    if median == 0:
        return True
    
    cv = stdev / median
    return cv <= CV_THRESHOLD


def run_adaptive(func, *args, **kwargs):
    """
    Run a function adaptively until results are stable or max trials reached.
    Returns (success, median_time, all_times).
    """
    times = []
    
    for trial in range(MAX_TRIALS):
        success, elapsed = func(*args, **kwargs)
        if not success:
            return False, 0, times
        times.append(elapsed)
        
        # Check for stability after minimum trials
        if trial >= MIN_TRIALS - 1 and is_stable(times):
            break
    
    median, _, _ = robust_stats(times)
    return True, median, times


def compress_once(tool, level, threads, input_file, output_file):
    """Compress a file once. Returns (success, elapsed_time)."""
    bin_path = get_tool_path(tool)
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "gzippy"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed


def compress(tool, level, threads, input_file, output_file):
    """Compress with adaptive trials. Returns (success, median_time, times)."""
    return run_adaptive(compress_once, tool, level, threads, input_file, output_file)


def decompress_once(tool, input_file, output_file):
    """Decompress a file once. Returns (success, elapsed_time)."""
    bin_path = get_tool_path(tool)
    cmd = [bin_path, "-d", "-c", input_file]
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed


def decompress(tool, input_file, output_file):
    """Decompress with adaptive trials. Returns (success, median_time, times)."""
    return run_adaptive(decompress_once, tool, input_file, output_file)


def files_identical(file1, file2):
    """Check if two files are byte-identical."""
    result = subprocess.run(["diff", "-q", file1, file2], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def format_size(path):
    """Format file size nicely."""
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def format_time(seconds):
    """Format time with appropriate precision."""
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.0f}s"


def format_stats(times):
    """Format timing statistics."""
    if not times:
        return ""
    median, stdev, trimmed = robust_stats(times)
    min_t = min(times)
    max_t = max(times)
    n = len(times)
    
    if len(times) >= 2:
        return f"med={format_time(median)} (±{format_time(stdev)}, {format_time(min_t)}-{format_time(max_t)}, n={n})"
    return f"{format_time(median)} (n={n})"


def check_tools():
    """Verify all tools exist and are executable."""
    tools = [GZIP, PIGZ, GZIPPY, UNGZIPPY]
    missing = [t for t in tools if not os.path.isfile(t)]
    if missing:
        print("Missing tools:")
        for t in missing:
            print(f"  {t}")
        print("\nRun 'make deps build' first.")
        return False
    return True


def create_tarball(output_path):
    """Create a tarball from repo contents."""
    cmd = [
        "tar", "cf", output_path,
        "--exclude=.git",
        "--exclude=test_data",
        "--exclude=test_results",
        "--exclude=*.gz",
        "--exclude=*.o",
        "target/", "src/", "scripts/", "Makefile", "Cargo.toml", "README.md"
    ]
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def run_validation(output_json=False):
    """Run the full validation suite. Returns (results_dict, passed, failed)."""
    results = {
        "compression": [],
        "decompression": [],
        "test_size_bytes": 0,
    }
    
    if not output_json:
        print("=" * 70)
        print("  Cross-Tool Validation Matrix")
        print(f"  (adaptive {MIN_TRIALS}-{MAX_TRIALS} trials, CV threshold {CV_THRESHOLD*100:.0f}%)")
        print("=" * 70)
        print()
    
    # Check tools exist
    if not check_tools():
        return results, 0, 1
    
    passed = 0
    failed = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tarball = tmpdir / "test.tar"
        
        # Create tarball
        if not output_json:
            print("Creating test tarball from repo...")
        if not create_tarball(str(tarball)):
            if not output_json:
                print("  ✗ Failed to create tarball")
            return results, 0, 1
        
        test_size = os.path.getsize(tarball)
        results["test_size_bytes"] = test_size
        
        if not output_json:
            print(f"  Created test.tar ({format_size(tarball)})")
            print()
        
        # Test each level and thread count
        for level in LEVELS:
            for threads in THREADS:
                if not output_json:
                    print(f"Level {level}, {threads} thread(s):")
                    print("-" * 70)
                
                # Compress with each tool
                compressed = {}
                for tool in TOOLS:
                    out = tmpdir / f"test.{tool}.l{level}.t{threads}.gz"
                    success, median_time, times = compress(tool, level, threads, str(tarball), str(out))
                    
                    comp_result = {
                        "tool": tool,
                        "level": level,
                        "threads": threads,
                        "success": success,
                        "median_seconds": median_time,
                        "times": times,
                        "output_size_bytes": os.path.getsize(out) if success else 0,
                        "input_size_bytes": test_size,
                    }
                    results["compression"].append(comp_result)
                    
                    if success:
                        compressed[tool] = out
                        if not output_json:
                            size_str = format_size(out)
                            stats_str = format_stats(times)
                            print(f"  {tool:5}: {size_str:>10}  {stats_str}")
                    else:
                        if not output_json:
                            print(f"  {tool:5}: ✗ compression failed")
                        failed += 1
                
                if not output_json:
                    print()
                
                # Decompression matrix
                for comp_tool, comp_file in compressed.items():
                    for decomp_tool in TOOLS:
                        out = tmpdir / f"test.{comp_tool}.{decomp_tool}.tar"
                        
                        success, median_time, times = decompress(decomp_tool, str(comp_file), str(out))
                        
                        correct = success and files_identical(str(tarball), str(out))
                        
                        decomp_result = {
                            "compressor": comp_tool,
                            "decompressor": decomp_tool,
                            "level": level,
                            "threads": threads,
                            "success": success,
                            "correct": correct,
                            "median_seconds": median_time,
                            "times": times,
                        }
                        results["decompression"].append(decomp_result)
                        
                        if not success:
                            if not output_json:
                                print(f"  ✗ {comp_tool} → {decomp_tool}: decompression failed")
                            failed += 1
                        elif not correct:
                            if not output_json:
                                stats_str = format_stats(times)
                                print(f"  ✗ {comp_tool:5} → {decomp_tool:5}  {stats_str}  MISMATCH")
                            failed += 1
                        else:
                            if not output_json:
                                stats_str = format_stats(times)
                                print(f"  ✓ {comp_tool:5} → {decomp_tool:5}  {stats_str}")
                            passed += 1
                        
                        if out.exists():
                            out.unlink()
                
                if not output_json:
                    print()
        
        # Test ungzippy symlink
        if not output_json:
            print("Testing ungzippy symlink...")
        gzippy_file = tmpdir / "test.gzippy.gz"
        success, _, _ = compress("gzippy", 6, 4, str(tarball), str(gzippy_file))
        if success:
            ungzippy_out = tmpdir / "test.ungzippy.tar"
            success, median_time, times = run_adaptive(
                lambda: decompress_once_ungzippy(str(gzippy_file), str(ungzippy_out))
            )
            
            correct = success and files_identical(str(tarball), str(ungzippy_out))
            if correct:
                if not output_json:
                    stats_str = format_stats(times)
                    print(f"  ✓ ungzippy             {stats_str}")
                passed += 1
            else:
                if not output_json:
                    stats_str = format_stats(times)
                    print(f"  ✗ ungzippy             {stats_str}  FAILED")
                failed += 1
    
    if not output_json:
        print()
        print("=" * 70)
        total = passed + failed
        print(f"  Results: {passed}/{total} passed, {failed} failed")
        print("=" * 70)
    
    return results, passed, failed


def decompress_once_ungzippy(input_file, output_file):
    """Decompress using ungzippy."""
    cmd = [UNGZIPPY, "-c", input_file]
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Cross-tool validation for gzippy")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", "-o", help="Output file for JSON results")
    args = parser.parse_args()
    
    results, passed, failed = run_validation(output_json=args.json)
    
    if args.json or args.output:
        results["summary"] = {"passed": passed, "failed": failed}
        json_str = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"Results written to {args.output}")
        else:
            print(json_str)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
