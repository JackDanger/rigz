#!/usr/bin/env python3
"""
Comprehensive CI validation for rigz.

Tests rigz against gzip and pigz across:
- Multiple data types (text, random, tarball)
- Multiple compression levels (1, 6, 9)
- Multiple thread counts (1, max)

Validates correctness via cross-tool decompression matrix.
Reports performance with statistical significance.

Usage:
    python3 scripts/validate_ci.py
    python3 scripts/validate_ci.py --size 10 --trials 5
    python3 scripts/validate_ci.py --output results.json
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Configuration
DEFAULT_TRIALS = 5
DEFAULT_SIZE_MB = 10
SEED_FILE = "test_data/text-1MB.txt"


def find_tool(name: str) -> str:
    """Find a tool binary."""
    paths = {
        "gzip": ["./gzip/gzip", shutil.which("gzip") or "gzip"],
        "pigz": ["./pigz/pigz"],
        "rigz": ["./target/release/rigz"],
        "unrigz": ["./target/release/unrigz"],
    }
    for path in paths.get(name, []):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(f"Could not find {name}")


def format_time(seconds: float) -> str:
    """Format time for display."""
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.1f}s"


def format_size(bytes_val: int) -> str:
    """Format size for display."""
    mb = bytes_val / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.1f}MB"
    kb = bytes_val / 1024
    return f"{kb:.1f}KB"


def format_ratio(original: int, compressed: int) -> str:
    """Format compression ratio."""
    if original == 0:
        return "—"
    ratio = compressed / original * 100
    return f"{ratio:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Test Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_file(output_path: Path, size_mb: int) -> bool:
    """Generate text by repeating the Gutenberg seed."""
    seed = Path(SEED_FILE)
    if not seed.exists():
        # Fall back to generating pseudo-text
        print(f"    (seed not found, generating Lorem Ipsum)")
        lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100
        target = size_mb * 1024 * 1024
        with open(output_path, 'w') as f:
            while f.tell() < target:
                f.write(lorem)
        return True
    
    seed_data = seed.read_bytes()
    target = size_mb * 1024 * 1024
    with open(output_path, 'wb') as f:
        while f.tell() < target:
            remaining = target - f.tell()
            f.write(seed_data[:remaining])
    return True


def generate_random_file(output_path: Path, size_mb: int) -> bool:
    """Generate random data from /dev/urandom."""
    result = subprocess.run(
        ["dd", "if=/dev/urandom", f"of={output_path}", "bs=1M", f"count={size_mb}"],
        capture_output=True
    )
    return result.returncode == 0


def generate_tarball(output_path: Path, target_mb: int) -> bool:
    """Generate tarball from entire repo including .git (realistic mixed content)."""
    # Include everything: source, .git objects, binaries - realistic workload
    tar_cmd = [
        "tar", "cf", str(output_path),
        "--exclude=target",      # Build artifacts (huge)
        "--exclude=test_data",   # Generated test files
        "--exclude=test_results",
        "."
    ]
    result = subprocess.run(tar_cmd, capture_output=True)
    if result.returncode != 0:
        return False
    
    # Pad by repeating if too small
    actual = output_path.stat().st_size
    target = target_mb * 1024 * 1024
    if actual < target * 0.8:
        base = output_path.read_bytes()
        with open(output_path, 'wb') as f:
            while f.tell() < target:
                remaining = target - f.tell()
                f.write(base[:remaining])
    
    return True


def create_test_files(tmpdir: Path, size_mb: int) -> Dict[str, Path]:
    """Create all test files and return paths."""
    files = {}
    
    print(f"  Generating {size_mb}MB test files...")
    
    # Text (highly compressible)
    text_path = tmpdir / "text.txt"
    if generate_text_file(text_path, size_mb):
        files["text"] = text_path
        print(f"    text:    {format_size(text_path.stat().st_size)}")
    
    # Random (poorly compressible)
    random_path = tmpdir / "random.dat"
    if generate_random_file(random_path, size_mb):
        files["random"] = random_path
        print(f"    random:  {format_size(random_path.stat().st_size)}")
    
    # Tarball (mixed content)
    tar_path = tmpdir / "repo.tar"
    if generate_tarball(tar_path, size_mb):
        files["tarball"] = tar_path
        print(f"    tarball: {format_size(tar_path.stat().st_size)}")
    
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Compression / Decompression
# ─────────────────────────────────────────────────────────────────────────────

def compress_once(tool: str, level: int, threads: int, input_file: str, output_file: str) -> Tuple[bool, float]:
    """Compress once, return (success, elapsed)."""
    bin_path = find_tool(tool)
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "rigz"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed


def decompress_once(tool: str, input_file: str, output_file: str) -> Tuple[bool, float]:
    """Decompress once, return (success, elapsed)."""
    bin_path = find_tool(tool)
    cmd = [bin_path, "-d", "-c", input_file]
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed


def run_trials(func, trials: int) -> Dict:
    """Run a function multiple times and collect stats."""
    times = []
    success = True
    
    for _ in range(trials):
        ok, elapsed = func()
        if not ok:
            return {"success": False}
        times.append(elapsed)
    
    return {
        "success": True,
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) >= 2 else 0,
        "times": times,
    }


def files_identical(file1: str, file2: str) -> bool:
    """Check if two files are byte-identical."""
    result = subprocess.run(["diff", "-q", file1, file2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    test_files: Dict[str, Path],
    levels: List[int],
    thread_counts: List[int],
    tools: List[str],
    trials: int,
    tmpdir: Path
) -> Dict:
    """Run full validation matrix."""
    
    results = {
        "config": {
            "levels": levels,
            "threads": thread_counts,
            "trials": trials,
            "data_types": list(test_files.keys()),
        },
        "compression_stats": [],
        "tests": [],
        "passed": 0,
        "failed": 0,
        "errors": [],
    }
    
    for data_type, test_file in test_files.items():
        original_size = test_file.stat().st_size
        
        print(f"\n{'='*60}")
        print(f"  Data: {data_type} ({format_size(original_size)})")
        print(f"{'='*60}")
        
        for level in levels:
            for threads in thread_counts:
                print(f"\n  Level {level}, {threads} thread(s):")
                print(f"  {'-'*50}")
                
                # Compress with each tool
                compressed = {}
                for tool in tools:
                    out = tmpdir / f"{data_type}.{tool}.l{level}.t{threads}.gz"
                    
                    stats = run_trials(
                        lambda t=tool, o=out: compress_once(t, level, threads, str(test_file), str(o)),
                        trials
                    )
                    
                    if stats["success"]:
                        compressed[tool] = out
                        comp_size = out.stat().st_size
                        throughput = (original_size / (1024*1024)) / stats["median"]
                        ratio = format_ratio(original_size, comp_size)
                        
                        print(f"    {tool:5} {format_size(comp_size):>8} ({ratio:>5})  "
                              f"{format_time(stats['median']):>7}  {throughput:5.0f} MB/s")
                        
                        results["compression_stats"].append({
                            "data_type": data_type,
                            "tool": tool,
                            "level": level,
                            "threads": threads,
                            "input_size": original_size,
                            "output_size": comp_size,
                            "median_time": stats["median"],
                            "min_time": stats["min"],
                            "max_time": stats["max"],
                            "stdev": stats["stdev"],
                            "throughput_mbps": throughput,
                        })
                    else:
                        print(f"    {tool:5} FAILED")
                        results["errors"].append(f"{tool} compression failed ({data_type} L{level} T{threads})")
                        results["failed"] += 1
                
                # Cross-tool decompression
                print()
                for comp_tool, comp_file in compressed.items():
                    for decomp_tool in tools:
                        out = tmpdir / f"{data_type}.{comp_tool}.{decomp_tool}.bin"
                        
                        stats = run_trials(
                            lambda dt=decomp_tool, cf=comp_file, o=out: decompress_once(dt, str(cf), str(o)),
                            trials
                        )
                        
                        test_id = f"{comp_tool}→{decomp_tool}"
                        test_result = {
                            "data_type": data_type,
                            "compress_tool": comp_tool,
                            "decompress_tool": decomp_tool,
                            "level": level,
                            "threads": threads,
                        }
                        
                        if not stats["success"]:
                            print(f"    ❌ {test_id}: decompression failed")
                            test_result["passed"] = False
                            test_result["error"] = "decompression failed"
                            results["failed"] += 1
                            results["errors"].append(f"{test_id} ({data_type} L{level} T{threads}): decompression failed")
                        elif not files_identical(str(test_file), str(out)):
                            print(f"    ❌ {test_id}: output mismatch")
                            test_result["passed"] = False
                            test_result["error"] = "output mismatch"
                            results["failed"] += 1
                            results["errors"].append(f"{test_id} ({data_type} L{level} T{threads}): output mismatch")
                        else:
                            print(f"    ✅ {test_id}: OK ({format_time(stats['median'])})")
                            test_result["passed"] = True
                            test_result["median_time"] = stats["median"]
                            results["passed"] += 1
                        
                        results["tests"].append(test_result)
                        
                        if out.exists():
                            out.unlink()
    
    return results


def test_unrigz(tmpdir: Path, test_file: Path, trials: int) -> Dict:
    """Test unrigz symlink."""
    try:
        unrigz = find_tool("unrigz")
    except FileNotFoundError:
        return {"passed": False, "error": "unrigz not found"}
    
    # Compress with rigz
    compressed = tmpdir / "unrigz_test.gz"
    ok, _ = compress_once("rigz", 6, 1, str(test_file), str(compressed))
    if not ok:
        return {"passed": False, "error": "compression failed"}
    
    # Decompress with unrigz
    output = tmpdir / "unrigz_test.bin"
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        with open(output, "wb") as f:
            result = subprocess.run([unrigz, "-c", str(compressed)], stdout=f, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - start)
        
        if result.returncode != 0:
            return {"passed": False, "error": "decompression failed"}
    
    if not files_identical(str(test_file), str(output)):
        return {"passed": False, "error": "output mismatch"}
    
    return {
        "passed": True,
        "median_time": statistics.median(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive rigz validation")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE_MB,
                       help=f"Test file size in MB (default: {DEFAULT_SIZE_MB})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                       help=f"Trials per test (default: {DEFAULT_TRIALS})")
    parser.add_argument("--levels", type=str, default="1,6,9",
                       help="Compression levels (default: 1,6,9)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    levels = [int(x) for x in args.levels.split(",")]
    max_threads = os.cpu_count() or 2
    thread_counts = [1, max_threads] if max_threads > 1 else [1]
    tools = ["gzip", "pigz", "rigz"]
    
    print("=" * 60)
    print("  rigz Comprehensive Validation Suite")
    print("=" * 60)
    print(f"  Size:    {args.size}MB per data type")
    print(f"  Trials:  {args.trials} per test")
    print(f"  Levels:  {levels}")
    print(f"  Threads: {thread_counts}")
    print(f"  Tools:   {tools}")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Generate test data
        test_files = create_test_files(tmpdir, args.size)
        if not test_files:
            print("ERROR: Failed to create test files")
            return 1
        
        # Run validation
        results = run_validation(
            test_files, levels, thread_counts, tools, args.trials, tmpdir
        )
        
        # Add metadata
        results["config"]["size_mb"] = args.size
        
        # Test unrigz
        print(f"\n{'='*60}")
        print("  Testing unrigz symlink...")
        print(f"{'='*60}")
        
        # Use the text file for unrigz test
        if "text" in test_files:
            unrigz_result = test_unrigz(tmpdir, test_files["text"], args.trials)
            if unrigz_result["passed"]:
                print(f"  ✅ unrigz: OK ({format_time(unrigz_result.get('median_time', 0))})")
                results["passed"] += 1
            else:
                print(f"  ❌ unrigz: {unrigz_result.get('error', 'unknown error')}")
                results["failed"] += 1
                results["errors"].append(f"unrigz: {unrigz_result.get('error')}")
            
            results["tests"].append({
                "tool": "unrigz",
                **unrigz_result
            })
    
    # Summary
    total = results["passed"] + results["failed"]
    print()
    print("=" * 60)
    print(f"  Results: {results['passed']}/{total} passed, {results['failed']} failed")
    print(f"  ({args.trials} trials × {len(test_files)} data types)")
    print("=" * 60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")
    
    if results["failed"] > 0:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  • {error}")
        return 1
    
    print("\n✅ All validation tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
