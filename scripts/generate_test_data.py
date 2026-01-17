#!/usr/bin/env python3
"""
Generate test data for rigz validation.

Creates test files of various types and sizes:
- text: Highly compressible (Project Gutenberg repeated)
- random: Poorly compressible (urandom)
- tarball: Mixed content (repo archive)

Usage:
    python3 scripts/generate_test_data.py --output-dir test_data
    python3 scripts/generate_test_data.py --size 10  # 10MB files
    python3 scripts/generate_test_data.py --size 50 --types text,random
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SEED_FILE = "test_data/text-1MB.txt"  # 1MB of Proust, checked into git


def generate_text_file(output_path: Path, size_mb: int, seed_file: Path) -> bool:
    """Generate text file by repeating the seed file."""
    if not seed_file.exists():
        print(f"  ERROR: Seed file not found: {seed_file}")
        return False
    
    seed_size = seed_file.stat().st_size
    target_size = size_mb * 1024 * 1024
    repeats = (target_size // seed_size) + 1
    
    print(f"  Generating {size_mb}MB text file ({repeats} copies of seed)...")
    
    with open(output_path, 'wb') as out:
        seed_data = seed_file.read_bytes()
        bytes_written = 0
        for _ in range(repeats):
            if bytes_written >= target_size:
                break
            chunk = seed_data[:target_size - bytes_written]
            out.write(chunk)
            bytes_written += len(chunk)
    
    # Truncate to exact size
    with open(output_path, 'r+b') as f:
        f.truncate(target_size)
    
    actual_size = output_path.stat().st_size
    print(f"  Created: {output_path} ({actual_size / 1024 / 1024:.1f}MB)")
    return True


def generate_random_file(output_path: Path, size_mb: int) -> bool:
    """Generate random data file from /dev/urandom."""
    target_size = size_mb * 1024 * 1024
    
    print(f"  Generating {size_mb}MB random file...")
    
    result = subprocess.run(
        ["dd", "if=/dev/urandom", f"of={output_path}", f"bs=1M", f"count={size_mb}"],
        capture_output=True
    )
    
    if result.returncode != 0:
        print(f"  ERROR: dd failed: {result.stderr.decode()}")
        return False
    
    actual_size = output_path.stat().st_size
    print(f"  Created: {output_path} ({actual_size / 1024 / 1024:.1f}MB)")
    return True


def generate_tarball(output_path: Path, size_mb: int) -> bool:
    """Generate tarball from entire repo including .git (realistic mixed content)."""
    print(f"  Generating ~{size_mb}MB tarball from repo...")
    
    # Include .git for realistic binary/text mix
    tar_cmd = [
        "tar", "cf", str(output_path),
        "--exclude=target",       # Build artifacts
        "--exclude=test_data",    # Generated test files
        "--exclude=test_results",
        "."
    ]
    
    result = subprocess.run(tar_cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: tar failed: {result.stderr.decode()}")
        return False
    
    actual_size = output_path.stat().st_size
    actual_mb = actual_size / 1024 / 1024
    
    # Pad by repeating if too small
    if actual_mb < size_mb * 0.8:
        print(f"  Tarball is {actual_mb:.1f}MB, padding to reach ~{size_mb}MB...")
        base_tar = output_path.read_bytes()
        target = size_mb * 1024 * 1024
        with open(output_path, 'wb') as f:
            while f.tell() < target:
                remaining = target - f.tell()
                f.write(base_tar[:remaining])
    
    actual_size = output_path.stat().st_size
    print(f"  Created: {output_path} ({actual_size / 1024 / 1024:.1f}MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate test data for rigz")
    parser.add_argument("--output-dir", type=str, default="test_data",
                       help="Output directory for test files")
    parser.add_argument("--size", type=int, default=10,
                       help="Size in MB for each test file (default: 10)")
    parser.add_argument("--types", type=str, default="text,random,tarball",
                       help="Comma-separated list of types: text,random,tarball")
    parser.add_argument("--seed", type=str, default=SEED_FILE,
                       help=f"Path to seed text file (default: {SEED_FILE})")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_file = Path(args.seed)
    types = [t.strip() for t in args.types.split(",")]
    size_mb = args.size
    
    print(f"Generating {size_mb}MB test files in {output_dir}/")
    print(f"Types: {', '.join(types)}")
    print()
    
    success = True
    generated = []
    
    for data_type in types:
        if data_type == "text":
            path = output_dir / f"text-{size_mb}MB.txt"
            if generate_text_file(path, size_mb, seed_file):
                generated.append(("text", path))
            else:
                success = False
                
        elif data_type == "random":
            path = output_dir / f"random-{size_mb}MB.dat"
            if generate_random_file(path, size_mb):
                generated.append(("random", path))
            else:
                success = False
                
        elif data_type == "tarball":
            path = output_dir / f"repo-{size_mb}MB.tar"
            if generate_tarball(path, size_mb):
                generated.append(("tarball", path))
            else:
                success = False
        else:
            print(f"  WARNING: Unknown type '{data_type}', skipping")
    
    print()
    print("Summary:")
    for dtype, path in generated:
        size = path.stat().st_size / 1024 / 1024
        print(f"  {dtype:10} {path.name:30} {size:6.1f}MB")
    
    if not success:
        print("\nSome files failed to generate!")
        return 1
    
    print(f"\n✓ Generated {len(generated)} test files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
