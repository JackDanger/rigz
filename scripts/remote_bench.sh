#!/bin/bash
# Remote benchmark script for gzippy on x86_64 Linux
#
# Usage:
#   ./scripts/remote_bench.sh                    # Run benchmark on current HEAD
#   ./scripts/remote_bench.sh <sha>              # Checkout specific commit and benchmark
#   ./scripts/remote_bench.sh <sha> --quick      # Quick benchmark (fewer runs)
#   ./scripts/remote_bench.sh --cmd "command"   # Run arbitrary command

set -e

REMOTE="root@10.30.0.199"
JUMP="-J neurotic"
REPO_DIR="/root/gzippy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SHA=""
QUICK=""
CUSTOM_CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK="1"
            shift
            ;;
        --cmd)
            CUSTOM_CMD="$2"
            shift 2
            ;;
        *)
            SHA="$1"
            shift
            ;;
    esac
done

# Function to run remote command
remote() {
    ssh $JUMP $REMOTE "cd $REPO_DIR && $1" 2>&1 | grep -v "setlocale\|manpath"
}

# If custom command, just run it
if [[ -n "$CUSTOM_CMD" ]]; then
    echo -e "${BLUE}Running custom command:${NC} $CUSTOM_CMD"
    remote "$CUSTOM_CMD"
    exit 0
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  gzippy Remote Benchmark (x86_64 Linux)${NC}"
echo -e "${BLUE}========================================${NC}"

# Show machine info
echo -e "\n${YELLOW}Machine:${NC}"
remote "uname -m && nproc && cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2"

# Fetch and checkout if SHA specified
if [[ -n "$SHA" ]]; then
    echo -e "\n${YELLOW}Checking out: ${SHA}${NC}"
    remote "git fetch origin && git checkout $SHA && git reset --hard $SHA"
fi

# Show current commit
echo -e "\n${YELLOW}Current commit:${NC}"
remote "git log --oneline -1"

# Build
echo -e "\n${YELLOW}Building release...${NC}"
remote "cargo build --release 2>&1 | tail -3"

# Verify binary works
echo -e "\n${YELLOW}Verifying binary:${NC}"
remote "./target/release/gzippy --version"

# Generate test data if needed
echo -e "\n${YELLOW}Preparing test data...${NC}"
remote "make test-data 2>&1 | tail -5"

# Run benchmarks
if [[ -n "$QUICK" ]]; then
    RUNS=5
    echo -e "\n${YELLOW}Running QUICK benchmark (${RUNS} runs)...${NC}"
else
    RUNS=10
    echo -e "\n${YELLOW}Running full benchmark (${RUNS} runs)...${NC}"
fi

# Function to run compression benchmark
benchmark() {
    local level=$1
    local threads=$2
    local file=$3
    
    echo -e "\n${GREEN}=== L${level} T${threads} (${file}) ===${NC}"
    
    remote "python3 -c \"
import subprocess, time, statistics, os

def bench(cmd, runs=$RUNS):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        times.append(time.perf_counter() - start)
    times.sort()
    trimmed = times[1:-1] if len(times) > 4 else times
    return statistics.median(trimmed), statistics.stdev(trimmed) if len(trimmed) > 1 else 0

# Compression benchmark
pigz_med, pigz_std = bench('./pigz/pigz -${level} -p${threads} -c test_data/${file} > /tmp/pigz.gz')
gzippy_med, gzippy_std = bench('./target/release/gzippy -${level} -p${threads} -c test_data/${file} > /tmp/gzippy.gz')

overhead = (gzippy_med / pigz_med - 1) * 100
status = '✅' if overhead <= 5.0 else '❌'

print(f'Compression:')
print(f'  pigz: {pigz_med:.3f}s (±{pigz_std:.3f}s)')
print(f'  gzippy: {gzippy_med:.3f}s (±{gzippy_std:.3f}s)')
print(f'  {status} overhead: {overhead:+.1f}%')

# Size comparison
pigz_size = os.path.getsize('/tmp/pigz.gz')
gzippy_size = os.path.getsize('/tmp/gzippy.gz')
size_diff = (gzippy_size / pigz_size - 1) * 100
print(f'Size: pigz={pigz_size:,} gzippy={gzippy_size:,} ({size_diff:+.2f}%)')

# Decompression benchmark
gzip_med, _ = bench('gzip -dc /tmp/gzippy.gz > /dev/null')
pigz_dec, _ = bench('./pigz/pigz -dc /tmp/gzippy.gz > /dev/null')
gzippy_dec, _ = bench('./target/release/gzippy -d /tmp/gzippy.gz -c > /dev/null')

print(f'Decompression (gzippy file):')
print(f'  gzip: {gzip_med:.3f}s, pigz: {pigz_dec:.3f}s, gzippy: {gzippy_dec:.3f}s')
speedup = (pigz_dec / gzippy_dec - 1) * 100 if gzippy_dec > 0 else 0
print(f'  gzippy vs pigz: {speedup:+.1f}%')
\""
}

# Run key benchmarks
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Compression Benchmarks${NC}"
echo -e "${BLUE}========================================${NC}"

# L9 with various thread counts (the problematic configs)
benchmark 9 2 "text-100MB.txt"
benchmark 9 4 "text-100MB.txt"
benchmark 9 16 "text-100MB.txt"

# L1 and L6 for comparison
benchmark 1 16 "text-100MB.txt"
benchmark 6 16 "text-100MB.txt"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}  Benchmark complete!${NC}"
echo -e "${BLUE}========================================${NC}"
