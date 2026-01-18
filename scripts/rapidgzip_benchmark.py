#!/usr/bin/env python3
"""
Local benchmark comparing gzippy decompression against rapidgzip.

This script downloads the Silesia corpus and runs compression/decompression
benchmarks comparing gzippy, pigz, igzip, and rapidgzip.

Usage:
    python3 scripts/rapidgzip_benchmark.py

Prerequisites:
    - gzippy built: cargo build --release
    - pigz built: make -C pigz
    - igzip built (optional): mkdir -p isa-l/build && cd isa-l/build && cmake .. && make igzip
    - rapidgzip installed: pip install rapidgzip

The Silesia corpus will be downloaded automatically if not present.
"""

import os
import sys
import subprocess
import time
import statistics
import tempfile
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Tool paths (relative to repo root)
GZIPPY = "./target/release/gzippy"
PIGZ = "./pigz/pigz"
IGZIP = "./isa-l/build/igzip"
RAPIDGZIP = "./rapidgzip/librapidarchive/build/src/tools/rapidgzip"
ZOPFLI = "./zopfli/zopfli"
GZIP = "gzip"

# Silesia corpus URL
SILESIA_URL = "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"

RUNS = 5


def find_repo_root() -> Path:
    """Find the repository root."""
    path = Path(__file__).resolve().parent.parent
    if (path / "Cargo.toml").exists():
        return path
    raise RuntimeError("Could not find repository root")


def check_tool(path: str, name: str) -> bool:
    """Check if a tool exists and is executable."""
    if path.startswith("./"):
        full_path = find_repo_root() / path[2:]
    else:
        # System command
        try:
            subprocess.run([path, "--version"], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False
    return full_path.exists() and os.access(full_path, os.X_OK)


def check_rapidgzip() -> bool:
    """Check if rapidgzip CLI is available."""
    return check_tool(RAPIDGZIP, "rapidgzip")


def download_silesia(work_dir: Path) -> Path:
    """Download and extract the Silesia corpus."""
    silesia_zip = work_dir / "silesia.zip"
    silesia_dir = work_dir / "silesia"
    silesia_tar = work_dir / "silesia.tar"
    
    if silesia_tar.exists():
        print(f"Using existing {silesia_tar}")
        return silesia_tar
    
    if not silesia_zip.exists():
        print(f"Downloading Silesia corpus from {SILESIA_URL}...")
        urllib.request.urlretrieve(SILESIA_URL, silesia_zip)
        print(f"Downloaded to {silesia_zip}")
    
    if not silesia_dir.exists():
        print(f"Extracting {silesia_zip}...")
        with zipfile.ZipFile(silesia_zip, 'r') as zf:
            zf.extractall(work_dir)
        print(f"Extracted to {silesia_dir}")
    
    # Create a tarball
    print(f"Creating {silesia_tar}...")
    with tarfile.open(silesia_tar, 'w') as tf:
        tf.add(silesia_dir, arcname="silesia")
    print(f"Created {silesia_tar}: {silesia_tar.stat().st_size / 1024 / 1024:.1f} MB")
    
    return silesia_tar


def benchmark_compress(tool: str, level: int, threads: int, 
                       input_file: Path, output_file: Path, 
                       runs: int = RUNS) -> Tuple[float, int]:
    """Benchmark compression. Returns (median_time, output_size)."""
    repo_root = find_repo_root()
    
    if tool == "gzippy":
        bin_path = str(repo_root / "target/release/gzippy")
        cmd = [bin_path, f"-{level}", f"-p{threads}", "-c", str(input_file)]
    elif tool == "pigz":
        bin_path = str(repo_root / "pigz/pigz")
        cmd = [bin_path, f"-{level}", f"-p", str(threads), "-c", str(input_file)]
    elif tool == "igzip":
        bin_path = str(repo_root / "isa-l/build/igzip")
        igzip_level = min(3, max(0, (level - 1) // 3))
        cmd = [bin_path, f"-{igzip_level}", f"-T", str(threads), "-c", str(input_file)]
    elif tool == "gzip":
        cmd = ["gzip", f"-{level}", "-c", str(input_file)]
    else:
        raise ValueError(f"Unknown tool: {tool}")
    
    times = []
    for _ in range(runs):
        output_file.unlink(missing_ok=True)
        
        start = time.perf_counter()
        with open(output_file, 'wb') as f:
            subprocess.run(cmd, stdout=f, check=True, stderr=subprocess.DEVNULL)
        end = time.perf_counter()
        
        times.append(end - start)
    
    median_time = statistics.median(times)
    output_size = output_file.stat().st_size
    return median_time, output_size


def benchmark_decompress(tool: str, threads: int, input_file: Path, 
                         runs: int = RUNS) -> float:
    """Benchmark decompression. Returns median time."""
    repo_root = find_repo_root()
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        
        if tool == "gzippy":
            bin_path = str(repo_root / "target/release/gzippy")
            subprocess.run([bin_path, "-d", f"-p{threads}", "-c", str(input_file)],
                          stdout=subprocess.DEVNULL, check=True, stderr=subprocess.DEVNULL)
        elif tool == "pigz":
            bin_path = str(repo_root / "pigz/pigz")
            subprocess.run([bin_path, "-d", "-p", str(threads), "-c", str(input_file)],
                          stdout=subprocess.DEVNULL, check=True, stderr=subprocess.DEVNULL)
        elif tool == "igzip":
            bin_path = str(repo_root / "isa-l/build/igzip")
            subprocess.run([bin_path, "-d", "-T", str(threads), "-c", str(input_file)],
                          stdout=subprocess.DEVNULL, check=True, stderr=subprocess.DEVNULL)
        elif tool == "rapidgzip":
            # rapidgzip CLI: -d decompress, -P parallelism, -c stdout
            bin_path = str(repo_root / "rapidgzip/librapidarchive/build/src/tools/rapidgzip")
            subprocess.run([bin_path, "-d", f"-P{threads}", "-c", str(input_file)],
                          stdout=subprocess.DEVNULL, check=True, stderr=subprocess.DEVNULL)
        elif tool == "gzip":
            subprocess.run(["gzip", "-d", "-c", str(input_file)],
                          stdout=subprocess.DEVNULL, check=True, stderr=subprocess.DEVNULL)
        else:
            raise ValueError(f"Unknown tool: {tool}")
        
        end = time.perf_counter()
        times.append(end - start)
    
    return statistics.median(times)


def run_benchmarks():
    """Run all benchmarks."""
    repo_root = find_repo_root()
    os.chdir(repo_root)
    
    # Check tools
    print("=== Checking Tools ===")
    available_compress = []
    available_decompress = []
    
    if check_tool(GZIPPY, "gzippy"):
        available_compress.append("gzippy")
        available_decompress.append("gzippy")
        print("✓ gzippy")
    else:
        print("✗ gzippy (run: cargo build --release)")
    
    if check_tool(PIGZ, "pigz"):
        available_compress.append("pigz")
        available_decompress.append("pigz")
        print("✓ pigz")
    else:
        print("✗ pigz (run: make -C pigz)")
    
    if check_tool(IGZIP, "igzip"):
        available_compress.append("igzip")
        available_decompress.append("igzip")
        print("✓ igzip")
    else:
        print("✗ igzip (optional)")
    
    if check_tool(GZIP, "gzip"):
        available_compress.append("gzip")
        available_decompress.append("gzip")
        print("✓ gzip (system)")
    
    if check_rapidgzip():
        available_decompress.append("rapidgzip")
        # Get version from CLI
        result = subprocess.run([str(find_repo_root() / RAPIDGZIP[2:]), "--version"],
                               capture_output=True, text=True)
        version = result.stdout.strip().split()[-1] if result.returncode == 0 else "unknown"
        print(f"✓ rapidgzip ({version})")
    else:
        print("✗ rapidgzip (run: make deps)")
    
    if len(available_compress) < 2:
        print("\nNeed at least 2 tools to run comparison benchmark")
        sys.exit(1)
    
    # Setup
    work_dir = repo_root / "benchmark_data"
    work_dir.mkdir(exist_ok=True)
    
    print("\n=== Downloading Test Data ===")
    silesia_tar = download_silesia(work_dir)
    input_size = silesia_tar.stat().st_size
    print(f"Input size: {input_size / 1024 / 1024:.1f} MB")
    
    cores = os.cpu_count() or 4
    print(f"\n=== System Info ===")
    print(f"CPU cores: {cores}")
    
    # Compression benchmarks
    print("\n" + "=" * 60)
    print("=== Compression Benchmarks ===")
    print("=" * 60)
    
    for level in [1, 6, 9]:
        print(f"\n### Level {level}")
        print(f"{'Tool':<10} {'Threads':>8} {'Time (s)':>10} {'Size':>12} {'Ratio':>8} {'MB/s':>10}")
        print("-" * 60)
        
        for threads in [1, cores // 2, cores]:
            for tool in available_compress:
                # gzip is single-threaded only
                if tool == "gzip" and threads != 1:
                    continue
                
                output_file = work_dir / f"test-{tool}-l{level}-t{threads}.gz"
                try:
                    time_taken, size = benchmark_compress(tool, level, threads, 
                                                          silesia_tar, output_file, 
                                                          runs=RUNS)
                    ratio = input_size / size
                    bandwidth = input_size / time_taken / 1024 / 1024
                    
                    print(f"{tool:<10} {threads:>8} {time_taken:>10.3f} "
                          f"{size / 1024 / 1024:>10.1f} MB {ratio:>7.2f}x {bandwidth:>9.1f}")
                except Exception as e:
                    print(f"{tool:<10} {threads:>8} ERROR: {e}")
    
    # Decompression benchmarks
    print("\n" + "=" * 60)
    print("=== Decompression Benchmarks ===")
    print("=" * 60)
    
    # Create test files compressed by different tools
    test_files = {}
    for compressor in ["gzippy", "pigz", "gzip"]:
        if compressor in available_compress:
            output_file = work_dir / f"silesia-{compressor}.tar.gz"
            if not output_file.exists():
                print(f"Creating {output_file.name}...")
                benchmark_compress(compressor, 9, cores, silesia_tar, output_file, runs=1)
            test_files[compressor] = output_file
    
    for compressed_by, input_file in test_files.items():
        print(f"\n### Decompressing file compressed by {compressed_by}")
        compressed_size = input_file.stat().st_size
        print(f"Compressed size: {compressed_size / 1024 / 1024:.1f} MB")
        print(f"{'Tool':<12} {'Threads':>8} {'Time (s)':>10} {'MB/s':>10} {'Speedup':>10}")
        print("-" * 55)
        
        # Baseline: single-threaded gzip
        gzip_time = benchmark_decompress("gzip", 1, input_file, runs=RUNS)
        gzip_bandwidth = input_size / gzip_time / 1024 / 1024
        print(f"{'gzip':<12} {1:>8} {gzip_time:>10.3f} {gzip_bandwidth:>9.1f} {1.0:>9.1f}x")
        
        for threads in [1, cores // 2, cores]:
            for tool in available_decompress:
                if tool == "gzip":
                    continue  # Already shown
                
                try:
                    time_taken = benchmark_decompress(tool, threads, input_file, runs=RUNS)
                    bandwidth = input_size / time_taken / 1024 / 1024
                    speedup = gzip_time / time_taken
                    
                    print(f"{tool:<12} {threads:>8} {time_taken:>10.3f} "
                          f"{bandwidth:>9.1f} {speedup:>9.1f}x")
                except Exception as e:
                    print(f"{tool:<12} {threads:>8} ERROR: {e}")
    
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    run_benchmarks()
