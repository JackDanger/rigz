#!/bin/bash
# Profile-Guided Optimization (PGO) Build Script
#
# PGO improves performance by 5-15% by optimizing branch prediction and inlining
# based on actual runtime behavior.
#
# Usage:
#   ./scripts/pgo-build.sh
#
# Requirements:
#   - Rust nightly (for -C profile-generate/-C profile-use)
#   - Test data in benchmark_data/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== PGO Build for gzippy ==="
echo ""

# Check for nightly
if ! rustup run nightly rustc --version &>/dev/null; then
    echo "Error: Rust nightly required for PGO"
    echo "Install with: rustup install nightly"
    exit 1
fi

# Create directories
PGO_DIR="$PROJECT_DIR/target/pgo-data"
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"

echo "Step 1: Build instrumented binary..."
RUSTFLAGS="-Cprofile-generate=$PGO_DIR" \
    cargo +nightly build --release

echo ""
echo "Step 2: Generate profile data by running benchmarks..."

# Create test data if needed
mkdir -p benchmark_data
if [ ! -f benchmark_data/1mb.txt ]; then
    echo "Generating test data..."
    dd if=/dev/urandom bs=1M count=1 2>/dev/null | base64 > benchmark_data/1mb.txt
    dd if=/dev/urandom bs=1M count=10 2>/dev/null | base64 > benchmark_data/10mb.txt
    # Also create compressible data
    yes "The quick brown fox jumps over the lazy dog. " | head -c 1000000 > benchmark_data/1mb-text.txt
    yes "The quick brown fox jumps over the lazy dog. " | head -c 10000000 > benchmark_data/10mb-text.txt
fi

# Compress files with various levels
echo "Running compression workloads..."
for level in 1 6 9; do
    ./target/release/gzippy -$level -k -f benchmark_data/1mb.txt -o /tmp/pgo-test.gz 2>/dev/null || true
    ./target/release/gzippy -$level -k -f benchmark_data/10mb.txt -o /tmp/pgo-test.gz 2>/dev/null || true
    ./target/release/gzippy -$level -k -f benchmark_data/1mb-text.txt -o /tmp/pgo-test.gz 2>/dev/null || true
done

# Decompress files
echo "Running decompression workloads..."
gzip -k -f benchmark_data/1mb.txt 2>/dev/null || true
gzip -k -f benchmark_data/10mb.txt 2>/dev/null || true
./target/release/gzippy -d benchmark_data/1mb.txt.gz -o /tmp/pgo-out.txt -f 2>/dev/null || true
./target/release/gzippy -d benchmark_data/10mb.txt.gz -o /tmp/pgo-out.txt -f 2>/dev/null || true

# Merge profile data
echo ""
echo "Step 3: Merge profile data..."
if command -v llvm-profdata &>/dev/null; then
    llvm-profdata merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw 2>/dev/null || true
elif command -v llvm-profdata-18 &>/dev/null; then
    llvm-profdata-18 merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw 2>/dev/null || true
else
    echo "Warning: llvm-profdata not found, using raw profiles"
fi

echo ""
echo "Step 4: Build optimized binary with profile data..."
if [ -f "$PGO_DIR/merged.profdata" ]; then
    RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" \
        cargo +nightly build --release
else
    # Use raw profile directory
    RUSTFLAGS="-Cprofile-use=$PGO_DIR" \
        cargo +nightly build --release
fi

echo ""
echo "=== PGO Build Complete ==="
echo "Optimized binary: $PROJECT_DIR/target/release/gzippy"
echo ""
echo "Expected improvement: 5-15% faster on typical workloads"

# Cleanup
rm -f /tmp/pgo-test.gz /tmp/pgo-out.txt
