#!/usr/bin/env python3
"""
CI-friendly validation script for rigz.

Tests that rigz produces gzip-compatible output by running a cross-tool
decompression matrix. Outputs JSON results and exits with non-zero code
on any failure.

Usage:
    python3 scripts/validate_ci.py
    python3 scripts/validate_ci.py --level 6 --threads 4
    python3 scripts/validate_ci.py --output results.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def compress(tool: str, level: int, threads: int, input_file: str, output_file: str) -> bool:
    """Compress a file. Returns success."""
    bin_path = find_tool(tool)
    cmd = [bin_path, f"-{level}"]
    if tool in ("pigz", "rigz"):
        cmd.append(f"-p{threads}")
    cmd.extend(["-c", input_file])
    
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def decompress(tool: str, input_file: str, output_file: str) -> bool:
    """Decompress a file. Returns success."""
    bin_path = find_tool(tool)
    cmd = [bin_path, "-d", "-c", input_file]
    
    with open(output_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def files_identical(file1: str, file2: str) -> bool:
    """Check if two files are byte-identical."""
    result = subprocess.run(["diff", "-q", file1, file2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def create_test_data(output_path: str, size_mb: int = 10) -> bool:
    """Create test data file."""
    size_bytes = size_mb * 1024 * 1024
    cmd = f"head -c {size_bytes} /dev/urandom | base64 > {output_path}"
    result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="CI validation for rigz")
    parser.add_argument("--level", type=int, default=None,
                       help="Specific compression level (default: test 1, 6, 9)")
    parser.add_argument("--threads", type=int, default=None,
                       help="Specific thread count (default: test 1, 4)")
    parser.add_argument("--size", type=int, default=10,
                       help="Test file size in MB (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    levels = [args.level] if args.level else [1, 6, 9]
    thread_counts = [args.threads] if args.threads else [1, 4]
    tools = ["gzip", "pigz", "rigz"]
    
    results = {
        "config": {
            "levels": levels,
            "threads": thread_counts,
            "size_mb": args.size,
        },
        "tests": [],
        "passed": 0,
        "failed": 0,
        "errors": [],
    }
    
    print(f"=== Cross-Tool Validation Matrix ===")
    print(f"Levels: {levels}, Threads: {thread_counts}, Size: {args.size}MB")
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.bin"
        
        print(f"Creating {args.size}MB test file...")
        if not create_test_data(str(test_file), args.size):
            print("ERROR: Failed to create test data")
            results["errors"].append("Failed to create test data")
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            return 1
        
        for level in levels:
            for threads in thread_counts:
                print(f"\n--- Level {level}, {threads} thread(s) ---")
                
                # Compress with each tool
                compressed = {}
                for tool in tools:
                    out = tmpdir / f"test.{tool}.l{level}.t{threads}.gz"
                    if compress(tool, level, threads, str(test_file), str(out)):
                        compressed[tool] = out
                        size = os.path.getsize(out)
                        print(f"  {tool:5} compressed: {size:,} bytes")
                    else:
                        error = f"{tool} compression failed (L{level} T{threads})"
                        print(f"  {tool:5} compressed: FAILED")
                        results["errors"].append(error)
                        results["failed"] += 1
                        results["tests"].append({
                            "compress_tool": tool,
                            "level": level,
                            "threads": threads,
                            "passed": False,
                            "error": "compression failed",
                        })
                
                # Cross-tool decompression matrix
                print()
                for comp_tool, comp_file in compressed.items():
                    for decomp_tool in tools:
                        out = tmpdir / f"test.{comp_tool}.{decomp_tool}.bin"
                        test_id = f"{comp_tool}→{decomp_tool} (L{level} T{threads})"
                        
                        test_result = {
                            "compress_tool": comp_tool,
                            "decompress_tool": decomp_tool,
                            "level": level,
                            "threads": threads,
                        }
                        
                        if not decompress(decomp_tool, str(comp_file), str(out)):
                            print(f"  ❌ {comp_tool:5} → {decomp_tool:5}: decompression failed")
                            test_result["passed"] = False
                            test_result["error"] = "decompression failed"
                            results["errors"].append(f"{test_id}: decompression failed")
                            results["failed"] += 1
                        elif not files_identical(str(test_file), str(out)):
                            print(f"  ❌ {comp_tool:5} → {decomp_tool:5}: output mismatch")
                            test_result["passed"] = False
                            test_result["error"] = "output mismatch"
                            results["errors"].append(f"{test_id}: output mismatch")
                            results["failed"] += 1
                        else:
                            print(f"  ✅ {comp_tool:5} → {decomp_tool:5}: OK")
                            test_result["passed"] = True
                            results["passed"] += 1
                        
                        results["tests"].append(test_result)
                        
                        if out.exists():
                            out.unlink()
        
        # Test unrigz symlink
        print("\n--- Testing unrigz symlink ---")
        try:
            unrigz_path = find_tool("unrigz")
            rigz_compressed = tmpdir / "unrigz_test.gz"
            unrigz_out = tmpdir / "unrigz_test.bin"
            
            if compress("rigz", 6, 4, str(test_file), str(rigz_compressed)):
                cmd = [unrigz_path, "-c", str(rigz_compressed)]
                with open(unrigz_out, "wb") as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
                
                if result.returncode == 0 and files_identical(str(test_file), str(unrigz_out)):
                    print("  ✅ unrigz: OK")
                    results["passed"] += 1
                    results["tests"].append({"tool": "unrigz", "passed": True})
                else:
                    print("  ❌ unrigz: FAILED")
                    results["failed"] += 1
                    results["errors"].append("unrigz decompression failed or mismatch")
                    results["tests"].append({"tool": "unrigz", "passed": False})
            else:
                raise RuntimeError("rigz compression failed for unrigz test")
        except Exception as e:
            print(f"  ❌ unrigz: ERROR - {e}")
            results["failed"] += 1
            results["errors"].append(f"unrigz test error: {e}")
            results["tests"].append({"tool": "unrigz", "passed": False, "error": str(e)})
    
    # Summary
    total = results["passed"] + results["failed"]
    print()
    print("=" * 50)
    print(f"  Results: {results['passed']}/{total} passed, {results['failed']} failed")
    print("=" * 50)
    
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
