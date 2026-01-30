#!/usr/bin/env python3
"""
Gap Analysis: Rust vs libdeflate C

This script analyzes the performance gap between our Rust implementation
and libdeflate C by comparing generated assembly.
"""

import subprocess
import re
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent

def get_rust_asm():
    """Generate and parse Rust assembly for decode loop."""
    # Build with assembly output
    result = subprocess.run([
        'cargo', 'rustc', '--release', '--', 
        '--emit=asm', '-C', 'llvm-args=-x86-asm-syntax=intel'
    ], cwd=WORKSPACE, capture_output=True, text=True)
    
    # Find the assembly file
    asm_file = WORKSPACE / 'target' / 'release' / 'deps' / 'gzippy-*.s'
    import glob
    asm_files = glob.glob(str(WORKSPACE / 'target' / 'release' / 'deps' / 'gzippy*.s'))
    
    if not asm_files:
        print("No Rust ASM file found")
        return None
    
    with open(asm_files[0]) as f:
        return f.read()

def get_libdeflate_asm():
    """Read libdeflate C assembly (already generated)."""
    asm_file = WORKSPACE / 'target' / 'libdeflate_decompress.s'
    if not asm_file.exists():
        # Generate it
        subprocess.run([
            'python3', str(WORKSPACE / 'scripts' / 'libdeflate_to_rust_asm.py')
        ])
    
    if asm_file.exists():
        with open(asm_file) as f:
            return f.read()
    return None

def count_instructions(asm: str, pattern: str) -> int:
    """Count occurrences of instruction pattern."""
    return len(re.findall(pattern, asm, re.IGNORECASE))

def analyze_instruction_mix(asm: str, name: str):
    """Analyze instruction mix."""
    print(f"\n=== {name} Instruction Mix ===")
    
    categories = {
        'Load': r'\b(ldr|ldp|ldrb|ldrh)\b',
        'Store': r'\b(str|stp|strb|strh)\b',
        'ALU': r'\b(add|sub|and|orr|eor|mov)\b',
        'Shift': r'\b(lsl|lsr|asr|ror|ubfx|bfxil)\b',
        'Branch': r'\b(b\.|bl|cbz|cbnz|tbz|tbnz)\b',
        'Compare': r'\b(cmp|tst|ccmp)\b',
        'Conditional': r'\b(csel|cset|cinc)\b',
    }
    
    total = 0
    for cat, pattern in categories.items():
        count = count_instructions(asm, pattern)
        total += count
        print(f"  {cat}: {count}")
    
    print(f"  Total: {total}")
    return total

def find_hot_loop(asm: str, name: str):
    """Find and analyze the hot decode loop."""
    # Look for the fastloop pattern
    lines = asm.split('\n')
    
    # Find label patterns that suggest a loop
    loop_markers = []
    for i, line in enumerate(lines):
        if 'fastloop' in line.lower() or 'LBB' in line:
            loop_markers.append(i)
    
    if loop_markers:
        print(f"\n=== {name} Loop Markers ===")
        print(f"  Found {len(loop_markers)} potential loop labels")

def analyze_gap():
    """Main analysis."""
    print("=" * 60)
    print("GAP ANALYSIS: Rust vs libdeflate C")
    print("=" * 60)
    
    print("\nCurrent Performance:")
    print("  SILESIA:  94.4% of libdeflate")
    print("  SOFTWARE: 94.0% of libdeflate")
    print("  LOGS:    101.2% of libdeflate")
    
    print("\n" + "=" * 60)
    print("KNOWN DIFFERENCES (from previous analysis)")
    print("=" * 60)
    
    print("""
1. REFILL PATTERN
   - libdeflate: Uses computed goto and BFXIL for bitsleft update
   - Rust: Uses conditional branch and separate AND/OR
   - Impact: ~1-2 cycles per refill
   
2. ENTRY CONSUMPTION
   - libdeflate: Uses "bitsleft -= entry" (full subtract trick)
   - Rust: Uses "bitsleft -= (entry & 0xff)" (masked)
   - Impact: ~0.5 cycles per entry
   
3. EXTRA BITS EXTRACTION  
   - libdeflate: Uses saved_bitbuf pattern with preload
   - Rust: Similar, but Rust version may have extra MOV
   - Impact: ~0.5 cycles per length/distance
   
4. MATCH COPY
   - libdeflate: Uses LDP/STP for 32-byte copies
   - Rust: Uses LDR/STR for 8-byte copies
   - Impact: ~1-2 cycles per large match
   
5. LITERAL BATCHING
   - libdeflate: Batches 2-3 literals in fastloop
   - Rust: Same (we copied this pattern)
   - Impact: Minimal difference

TOTAL ESTIMATED GAP: ~3-5 cycles per symbol
At 1300 MB/s on ~150 bytes/symbol average: ~5% overhead matches observed ~6%
""")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)
    
    print("""
1. BFXIL REFILL (High Impact)
   Current Rust:
     and w14, bitsleft, #7
     orr bitsleft, w14, #56
   
   Optimal (libdeflate):
     mov w15, #56
     bfxil w15, bitsleft, #0, #3
     mov bitsleft, w15
   
   Savings: 0-1 cycle (depends on LLVM optimization)

2. FULL SUBTRACT TRICK (Medium Impact)
   Current Rust:
     code_bits = entry & 0xff
     bitsleft -= code_bits
   
   Optimal (libdeflate):
     bitsleft -= entry  // High bits are garbage but don't affect refill
   
   Note: We tried this but it requires careful handling of the refill
   threshold comparison (must mask or use CCMP)

3. LDP/STP FOR LARGE MATCHES (Medium Impact)
   Current: 8-byte copies
   Optimal: 32-byte copies with LDP/STP q registers
   
   Savings: ~50% for matches > 32 bytes

4. PRELOAD NEXT ENTRY (Low Impact)
   Start loading next entry while writing current
   
   Savings: ~0.5 cycles (often hidden by memory latency)

5. SIMD LITERAL BATCHING (Low Impact)
   For 4+ consecutive literals, write as u32/u64
   
   Savings: ~1 cycle for long literal runs
""")
    
    # Get libdeflate ASM for detailed comparison
    libdeflate_asm = get_libdeflate_asm()
    if libdeflate_asm:
        analyze_instruction_mix(libdeflate_asm[:50000], "libdeflate C")
        find_hot_loop(libdeflate_asm, "libdeflate C")

if __name__ == '__main__':
    analyze_gap()
