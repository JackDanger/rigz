#!/usr/bin/env python3
"""
Cross-tool validation matrix for rigz.

Tests that rigz produces gzip-compatible output by:
1. Creating a tarball from the repo
2. Compressing at multiple levels with multiple thread counts
3. Decompressing with gzip, pigz, and rigz
4. Verifying all outputs are byte-identical
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Tool paths - prefer local builds, fall back to system
import shutil

def find_gzip():
    if os.path.isfile("./gzip/gzip") and os.access("./gzip/gzip", os.X_OK):
        return "./gzip/gzip"
    return shutil.which("gzip") or "gzip"

GZIP = find_gzip()
PIGZ = "./pigz/pigz"
RIGZ = "./target/release/rigz"
UNRIGZ = "./target/release/unrigz"

# Test matrix
LEVELS = [1, 6, 9]
THREADS = [1, 4]
TOOLS = ["gzip", "pigz", "rigz"]

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
    return {"gzip": GZIP, "pigz": PIGZ, "rigz": RIGZ}[tool]

def compress(tool, level, threads, input_file, output_file):
    """Compress a file with the given tool. Returns (success, elapsed_time)."""
    bin_path = get_tool_path(tool)
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "rigz"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed

def decompress(tool, input_file, output_file):
    """Decompress a file with the given tool. Returns (success, elapsed_time)."""
    bin_path = get_tool_path(tool)
    cmd = [bin_path, "-d", "-c", input_file]
    
    start = time.perf_counter()
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    
    return result.returncode == 0, elapsed

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

def check_tools():
    """Verify all tools exist and are executable."""
    tools = [GZIP, PIGZ, RIGZ, UNRIGZ]
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

def main():
    print("=" * 60)
    print("  Cross-Tool Validation Matrix")
    print("=" * 60)
    print()
    
    # Check tools exist
    if not check_tools():
        return 1
    
    passed = 0
    failed = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tarball = tmpdir / "test.tar"
        
        # Create tarball
        print("Creating test tarball from repo...")
        if not create_tarball(str(tarball)):
            print("  ✗ Failed to create tarball")
            return 1
        print(f"  Created test.tar ({format_size(tarball)})")
        print()
        
        # Test each level and thread count
        for level in LEVELS:
            for threads in THREADS:
                print(f"Level {level}, {threads} thread(s):")
                print("-" * 60)
                
                # Compress with each tool
                compressed = {}
                comp_times = {}
                for tool in TOOLS:
                    out = tmpdir / f"test.{tool}.l{level}.t{threads}.gz"
                    success, elapsed = compress(tool, level, threads, str(tarball), str(out))
                    if success:
                        compressed[tool] = out
                        comp_times[tool] = elapsed
                        size_str = format_size(out)
                        time_str = format_time(elapsed)
                        print(f"  {tool:5}: {size_str:>10}  {time_str:>8}")
                    else:
                        print(f"  {tool:5}: ✗ compression failed")
                        failed += 1
                
                print()
                
                # Decompression matrix
                for comp_tool, comp_file in compressed.items():
                    for decomp_tool in TOOLS:
                        out = tmpdir / f"test.{comp_tool}.{decomp_tool}.tar"
                        
                        success, elapsed = decompress(decomp_tool, str(comp_file), str(out))
                        if not success:
                            print(f"  ✗ {comp_tool} → {decomp_tool}: decompression failed")
                            failed += 1
                            continue
                        
                        time_str = format_time(elapsed)
                        if files_identical(str(tarball), str(out)):
                            print(f"  ✓ {comp_tool:5} → {decomp_tool:5}  {time_str:>8}")
                            passed += 1
                        else:
                            print(f"  ✗ {comp_tool:5} → {decomp_tool:5}  {time_str:>8}  MISMATCH")
                            failed += 1
                        
                        out.unlink()  # Clean up
                
                print()
        
        # Test unrigz symlink
        print("Testing unrigz symlink...")
        rigz_file = tmpdir / "test.rigz.gz"
        success, _ = compress("rigz", 6, 4, str(tarball), str(rigz_file))
        if success:
            unrigz_out = tmpdir / "test.unrigz.tar"
            start = time.perf_counter()
            cmd = [UNRIGZ, "-c", str(rigz_file)]
            with open(unrigz_out, "wb") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
            elapsed = time.perf_counter() - start
            
            time_str = format_time(elapsed)
            if result.returncode == 0 and files_identical(str(tarball), str(unrigz_out)):
                print(f"  ✓ unrigz             {time_str:>8}")
                passed += 1
            else:
                print(f"  ✗ unrigz             {time_str:>8}  FAILED")
                failed += 1
    
    # Summary
    print()
    print("=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
