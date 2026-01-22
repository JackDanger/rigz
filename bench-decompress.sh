#!/bin/bash
set -e

# Decompression benchmark: gzippy (Rust) vs libdeflater crate (C libdeflate)
# Tests 3 archive types: silesia (mixed), software (source code), logs (repetitive)
#
# Usage:
#   ./bench-decompress.sh                  Run speed benchmark (10 iterations)
#   ./bench-decompress.sh --runs 50        Run with 50 iterations for stable results
#   ./bench-decompress.sh --analyze        Run detailed analysis (block types, cache stats)

BENCHMARK_DIR="benchmark_data"
SILESIA_URL="https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
SILESIA_ZIP="$BENCHMARK_DIR/silesia.zip"
SILESIA_TAR="$BENCHMARK_DIR/silesia.tar"
SILESIA_GZ="$BENCHMARK_DIR/silesia-gzip.tar.gz"

# Create benchmark data directory if needed
mkdir -p "$BENCHMARK_DIR"

# Download and prepare silesia corpus if not present
if [[ ! -f "$SILESIA_GZ" ]]; then
    echo "Silesia benchmark corpus not found. Preparing..."

    # Download if zip doesn't exist
    if [[ ! -f "$SILESIA_ZIP" ]]; then
        echo "Downloading silesia corpus from $SILESIA_URL..."
        curl -L -o "$SILESIA_ZIP" "$SILESIA_URL"
    fi

    # Extract zip to get individual files
    if [[ -f "$SILESIA_ZIP" ]]; then
        echo "Extracting silesia.zip..."
        SILESIA_EXTRACT="$BENCHMARK_DIR/silesia_extract"
        mkdir -p "$SILESIA_EXTRACT"
        unzip -o "$SILESIA_ZIP" -d "$SILESIA_EXTRACT"

        # Create tar archive from extracted files
        echo "Creating silesia.tar..."
        tar -cf "$SILESIA_TAR" -C "$SILESIA_EXTRACT" .

        # Compress with gzip (best compression to match typical benchmark setup)
        echo "Compressing to silesia-gzip.tar.gz..."
        gzip -9 -c "$SILESIA_TAR" > "$SILESIA_GZ"

        # Cleanup
        rm -rf "$SILESIA_EXTRACT"
        echo "Silesia corpus ready: $SILESIA_GZ"
    fi
fi

export RUSTFLAGS="-C target-cpu=native"

# Parse arguments
ANALYZE=false
PROFILE=false
RUNS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --analyze)
            ANALYZE=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./bench-decompress.sh [--runs N] [--analyze] [--profile]"
            exit 1
            ;;
    esac
done

# Export BENCH_RUNS if specified
if [[ -n "$RUNS" ]]; then
    export BENCH_RUNS="$RUNS"
fi

if [[ "$PROFILE" == "true" ]]; then
    cargo test --release --features profile bench_profile -- --nocapture
elif [[ "$ANALYZE" == "true" ]]; then
    cargo test --release bench_analyze -- --nocapture
else
    cargo test --release bench_decompress -- --nocapture
fi
