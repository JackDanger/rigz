#!/bin/bash
# Rigz Quick Benchmark - Fast but statistically sound
# Runs in <30 seconds with enough iterations for reliable results

set -euo pipefail

RIGZ_BIN="./target/release/rigz"
PIGZ_BIN="../pigz"
GZIP_BIN=$(which gzip)
TEST_DATA_DIR="test_data"

echo "============================================"
echo "  RIGZ Quick Benchmark"
echo "============================================"
echo ""

# Ensure test data exists
mkdir -p "$TEST_DATA_DIR"
[[ ! -f "$TEST_DATA_DIR/text-1MB.txt" ]] && head -c 1048576 /dev/urandom | base64 > "$TEST_DATA_DIR/text-1MB.txt" 2>/dev/null
[[ ! -f "$TEST_DATA_DIR/text-10MB.txt" ]] && head -c 10485760 /dev/urandom | base64 > "$TEST_DATA_DIR/text-10MB.txt" 2>/dev/null

# Run statistical benchmark
python3 << 'EOF'
import subprocess
import time
import statistics

def benchmark(cmd, runs):
    times = []
    for _ in range(runs):
        start = time.time()
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        times.append(time.time() - start)
    return min(times), statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0

def compare(name, baseline_cmd, rigz_cmd, runs=10):
    b_min, b_med, b_std = benchmark(baseline_cmd, runs)
    r_min, r_med, r_std = benchmark(rigz_cmd, runs)
    
    diff = (r_med / b_med - 1) * 100
    status = "✓" if diff <= 5 else "✗ GAP"
    
    return diff, status, b_med, r_med

wins = 0
losses = 0
gaps = []

print("=== 1MB Text (20 runs for statistical significance) ===")

diff, status, gzip_t, rigz_t = compare(
    "1MB T1", ["gzip", "-6", "-c", "test_data/text-1MB.txt"],
    ["./target/release/rigz", "-6", "-p1", "-c", "test_data/text-1MB.txt"], runs=20)
print(f"  rigz -p1:  {rigz_t:.3f}s  vs gzip:  {diff:+.1f}% {status}")
if diff <= 5: wins += 1
else: losses += 1; gaps.append(f"1MB L6 T1: {diff:+.1f}%")

diff, status, pigz_t, rigz_t = compare(
    "1MB T4", ["../pigz", "-6", "-p4", "-c", "test_data/text-1MB.txt"],
    ["./target/release/rigz", "-6", "-p4", "-c", "test_data/text-1MB.txt"], runs=20)
print(f"  rigz -p4:  {rigz_t:.3f}s  vs pigz: {diff:+.1f}% {status}")
if diff <= 5: wins += 1
else: losses += 1; gaps.append(f"1MB L6 T4: {diff:+.1f}%")

print("")
print("=== 10MB Text (10 runs) ===")

diff, status, gzip_t, rigz_t = compare(
    "10MB T1", ["gzip", "-6", "-c", "test_data/text-10MB.txt"],
    ["./target/release/rigz", "-6", "-p1", "-c", "test_data/text-10MB.txt"], runs=10)
print(f"  rigz -p1:  {rigz_t:.3f}s  vs gzip:  {diff:+.1f}% {status}")
if diff <= 5: wins += 1
else: losses += 1; gaps.append(f"10MB L6 T1: {diff:+.1f}%")

diff, status, pigz_t, rigz_t = compare(
    "10MB T4", ["../pigz", "-6", "-p4", "-c", "test_data/text-10MB.txt"],
    ["./target/release/rigz", "-6", "-p4", "-c", "test_data/text-10MB.txt"], runs=10)
print(f"  rigz -p4:  {rigz_t:.3f}s  vs pigz: {diff:+.1f}% {status}")
if diff <= 5: wins += 1
else: losses += 1; gaps.append(f"10MB L6 T4: {diff:+.1f}%")

# Validation
print("")
print("=== Validation ===")
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
    tmp = f.name

subprocess.run(["./target/release/rigz", "-6", "-p4", "-c", "test_data/text-10MB.txt"], 
               stdout=open(tmp, 'wb'), stderr=subprocess.DEVNULL)
result = subprocess.run(["gunzip", "-t", tmp], capture_output=True)
os.unlink(tmp)

if result.returncode == 0:
    print("✓ Output validates with gunzip")
else:
    print("✗ Validation FAILED")
    losses += 1

print("")
print("============================================")
print(f"  SUMMARY: {wins} wins, {losses} losses")
print("============================================")

if losses == 0:
    print("✓ All tests pass - rigz matches or beats targets!")
else:
    print("")
    print("Gaps to address:")
    for gap in gaps:
        print(f"  ✗ {gap}")
EOF
