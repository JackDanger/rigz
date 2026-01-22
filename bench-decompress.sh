#!/bin/bash
set -e

# Decompression benchmark: gzippy (Rust) vs libdeflater crate (C libdeflate)
# Tests 3 archive types: silesia (mixed), software (source code), logs (repetitive)
#
# Usage:
#   ./bench-decompress.sh           Run speed benchmark
#   ./bench-decompress.sh --analyze Run detailed analysis (block types, cache stats)

export RUSTFLAGS="-C target-cpu=native"

if [[ "$1" == "--analyze" ]]; then
    cargo test --release bench_analyze -- --nocapture
else
    cargo test --release bench_decompress -- --nocapture
fi
