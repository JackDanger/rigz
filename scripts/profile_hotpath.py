#!/usr/bin/env python3
"""
Profile the hot path to identify where the 6% gap comes from.
"""

import subprocess
import os
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent

def run_perf_profile():
    """Run perf record on the decode loop."""
    
    # Build in release mode with debug info
    print("Building with debug info...")
    subprocess.run([
        'cargo', 'build', '--release'
    ], cwd=WORKSPACE, check=True)
    
    # Check if we have sample data
    silesia = WORKSPACE / 'benchmark_data' / 'silesia-gzip.tar.gz'
    if not silesia.exists():
        print(f"No test data at {silesia}")
        return
    
    # On macOS, use Instruments or sample
    import platform
    if platform.system() == 'Darwin':
        print("\n=== Running sample profiler on macOS ===")
        
        # Run with sample
        result = subprocess.run([
            'sample', 
            str(WORKSPACE / 'target' / 'release' / 'gzippy'),
            '-wait', '-mayDie'
        ], capture_output=True, text=True, timeout=30)
        
        # Actually, let's use the built-in benchmark with more detail
        print("\n=== Running detailed benchmark ===")
        
        result = subprocess.run([
            'cargo', 'test', '--release', 'bench_cf_silesia', '--', '--nocapture'
        ], cwd=WORKSPACE, capture_output=True, text=True, timeout=120)
        
        print(result.stdout)
        print(result.stderr)

def analyze_assembly():
    """Analyze the generated assembly for the hot path."""
    
    print("\n=== Analyzing Generated Assembly ===")
    
    # Generate assembly
    subprocess.run([
        'cargo', 'rustc', '--release', '--lib', '--',
        '--emit=asm', '-C', 'llvm-args=-x86-asm-syntax=intel'
    ], cwd=WORKSPACE, capture_output=True)
    
    # Find function in assembly
    import glob
    asm_files = glob.glob(str(WORKSPACE / 'target' / 'release' / 'deps' / '*.s'))
    
    if not asm_files:
        print("No assembly files found")
        return
    
    # Look for decode_huffman in the largest asm file
    asm_file = max(asm_files, key=os.path.getsize)
    print(f"Analyzing {asm_file}")
    
    with open(asm_file) as f:
        content = f.read()
    
    # Find decode_huffman function
    import re
    
    # Count key instruction patterns
    patterns = {
        'loads': len(re.findall(r'\bldr[bhwq]?\b', content, re.I)),
        'stores': len(re.findall(r'\bstr[bhwq]?\b', content, re.I)),
        'shifts': len(re.findall(r'\b(lsl|lsr|asr|ror|ubfx|bfxil)\b', content, re.I)),
        'branches': len(re.findall(r'\b(b\.[a-z]+|cbz|cbnz|tbz|tbnz)\b', content, re.I)),
        'compares': len(re.findall(r'\b(cmp|ccmp|tst)\b', content, re.I)),
        'alu': len(re.findall(r'\b(add|sub|and|orr|eor|mov)\b', content, re.I)),
    }
    
    print("\nInstruction counts in generated ASM:")
    for name, count in patterns.items():
        print(f"  {name}: {count}")

def main():
    print("=" * 60)
    print("HOT PATH PROFILING")
    print("=" * 60)
    
    print("""
The 6% gap between Rust and libdeflate C could come from:

1. REFILL: Extra instructions in branchless refill pattern
2. CONSUME: Extra masking in bitsleft -= (entry & 0xff)
3. EXTRA BITS: Sub-optimal extract_bits implementation
4. MATCH COPY: Not using LDP/STP for SIMD copies
5. REGISTER ALLOCATION: LLVM not using optimal registers

Let's analyze each...
""")
    
    # Check extract_bits implementation
    print("\n=== Checking extract_bits implementation ===")
    
    grep_result = subprocess.run([
        'grep', '-n', 'fn extract_bits', 
        str(WORKSPACE / 'src' / 'consume_first_decode.rs')
    ], capture_output=True, text=True)
    print(grep_result.stdout)
    
    # Run benchmark
    run_perf_profile()
    
    # Analyze assembly
    analyze_assembly()

if __name__ == '__main__':
    main()
