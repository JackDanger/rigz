#!/bin/bash
# Rigz Full Performance Suite
# Compares speed AND compression ratio against gzip/pigz

set -euo pipefail

RIGZ_BIN="./target/release/rigz"
PIGZ_BIN="./pigz/pigz"
GZIP_BIN="./gzip/gzip"
TEST_DATA_DIR="test_data"

# Configuration - Full matrix
LEVELS="1 2 3 4 5 6 7 8 9"
THREADS="1 2 4 8"
SIZES="1MB:1048576 10MB:10485760 100MB:104857600"
TYPES="text random"

# Statistical configuration
RUNS_SHORT=20   # For <0.1s tests (1MB)
RUNS_MEDIUM=10  # For 0.1-1s tests (10MB)
RUNS_LONG=5     # For >1s tests (100MB)

echo "============================================"
echo "  Rigz Full Performance Suite"
echo "============================================"
echo ""
echo "Using:"
echo "  gzip: $GZIP_BIN"
echo "  pigz: $PIGZ_BIN"
echo "  rigz: $RIGZ_BIN"
echo ""
echo "Configuration:"
echo "  Levels: $LEVELS"
echo "  Threads: $THREADS"
echo "  Sizes: $SIZES"
echo "  Types: $TYPES"
echo "  Runs: $RUNS_SHORT (short), $RUNS_MEDIUM (medium), $RUNS_LONG (long)"
echo ""
echo "Metrics:"
echo "  Speed: compression time (lower is better)"
echo "  Size:  compressed output size vs gzip (lower is better)"
echo ""

# Generate test data
echo "Generating test data..."
mkdir -p "$TEST_DATA_DIR"
for size_spec in $SIZES; do
    size_name="${size_spec%%:*}"
    size_bytes="${size_spec##*:}"
    
    for type in $TYPES; do
        ext="txt"
        [[ "$type" == "random" ]] && ext="dat"
        file="$TEST_DATA_DIR/${type}-${size_name}.${ext}"
        
        if [[ ! -f "$file" ]]; then
            echo "  Generating $file..."
            if [[ "$type" == "text" ]]; then
                head -c "$size_bytes" /dev/urandom | base64 > "$file" 2>/dev/null
            else
                head -c "$size_bytes" /dev/urandom > "$file" 2>/dev/null
            fi
        fi
    done
done
echo ""

# Benchmark function
benchmark_test() {
    local size_name="$1"
    local type="$2"
    local level="$3"
    
    local ext="txt"
    [[ "$type" == "random" ]] && ext="dat"
    local file="$TEST_DATA_DIR/${type}-${size_name}.${ext}"
    local input_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    
    # Determine number of runs based on expected timing
    local runs=$RUNS_MEDIUM
    case "$size_name" in
        1MB) runs=$RUNS_SHORT ;;
        10MB) runs=$RUNS_MEDIUM ;;
        100MB) runs=$RUNS_LONG ;;
    esac
    
    echo "=== ${size_name} ${type} Level ${level} (${runs} runs) ==="
    
    python3 << EOF
import subprocess
import time
import statistics
import tempfile
import os

GZIP_BIN = "$GZIP_BIN"
PIGZ_BIN = "$PIGZ_BIN"
RIGZ_BIN = "$RIGZ_BIN"

def benchmark(cmd, runs):
    times = []
    for _ in range(runs):
        start = time.time()
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        times.append(time.time() - start)
    return min(times), statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0

def get_compressed_size(cmd, input_file):
    """Run compression and return output size"""
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        tmp = f.name
    try:
        with open(tmp, 'wb') as out:
            subprocess.run(cmd + ["-c", input_file], stdout=out, stderr=subprocess.DEVNULL)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)

file = "$file"
level = "$level"
runs = $runs
input_size = $input_size

# Get gzip baseline (time and size)
gzip_min, gzip_med, gzip_std = benchmark([GZIP_BIN, "-" + level, "-c", file], runs)
gzip_size = get_compressed_size([GZIP_BIN, "-" + level], file)
gzip_ratio = gzip_size / input_size * 100

print(f"  gzip:       {gzip_med:.3f}s (±{gzip_std:.3f})  size: {gzip_size:,} bytes ({gzip_ratio:.1f}%)")

# rigz and pigz for each thread count
for threads in [1, 2, 4, 8]:
    rigz_min, rigz_med, rigz_std = benchmark(
        [RIGZ_BIN, "-" + level, f"-p{threads}", "-c", file], runs)
    rigz_size = get_compressed_size([RIGZ_BIN, "-" + level, f"-p{threads}"], file)
    
    # Size comparison vs gzip
    size_diff = (rigz_size / gzip_size - 1) * 100
    size_status = "=" if abs(size_diff) < 0.5 else ("+" if size_diff > 0 else "-")
    
    if threads == 1:
        time_diff = (rigz_med / gzip_med - 1) * 100
        baseline_name = "gzip"
        baseline_med = gzip_med
    else:
        pigz_min, pigz_med, pigz_std = benchmark(
            [PIGZ_BIN, "-" + level, f"-p{threads}", "-c", file], runs)
        pigz_size = get_compressed_size([PIGZ_BIN, "-" + level, f"-p{threads}"], file)
        time_diff = (rigz_med / pigz_med - 1) * 100
        baseline_name = "pigz"
        baseline_med = pigz_med
    
    # Determine status
    cv = (rigz_std / rigz_med * 100) if rigz_med > 0 else 0
    time_status = "✓" if time_diff <= 5 else "✗ SLOW"
    
    # Size penalty warning
    size_warn = ""
    if size_diff > 1:
        size_warn = f" ⚠️ +{size_diff:.1f}% larger"
    elif size_diff < -1:
        size_warn = f" 📦 {abs(size_diff):.1f}% smaller"
    
    noise_note = " (within noise)" if abs(time_diff) < cv * 2 else ""
    
    if threads == 1:
        print(f"  rigz -p{threads} :  {rigz_med:.3f}s  vs {baseline_name}: {time_diff:+.1f}% {time_status}  size: {rigz_size:,} ({size_diff:+.1f}% vs gzip){size_warn}{noise_note}")
    else:
        print(f"  rigz -p{threads} :  {rigz_med:.3f}s  vs {baseline_name}: {time_diff:+.1f}% {time_status}  size: {rigz_size:,} ({size_diff:+.1f}% vs gzip){size_warn}  (pigz: {pigz_med:.3f}s, {pigz_size:,}){noise_note}")
EOF
    echo ""
}

echo "Running benchmarks..."
echo ""

for size_spec in $SIZES; do
    size_name="${size_spec%%:*}"
    for type in $TYPES; do
        for level in $LEVELS; do
            benchmark_test "$size_name" "$type" "$level"
        done
    done
done

# Summary
python3 << 'SUMMARY'
print("")
print("============================================")
print("  LEGEND")
print("============================================")
print("")
print("Speed:")
print("  ✓      = within 5% of baseline (acceptable)")
print("  ✗ SLOW = more than 5% slower (needs work)")
print("  (within noise) = difference not statistically significant")
print("")
print("Size (vs gzip single-threaded):")
print("  =      = same size (within 0.5%)")
print("  +X%    = larger output (parallel overhead)")
print("  -X%    = smaller output (rare)")
print("  ⚠️      = significant size penalty (>1%)")
print("  📦     = better compression than gzip")
print("")
print("Note: Parallel compression typically produces 0-2% larger")
print("      output due to independent block compression.")
print("============================================")
SUMMARY
