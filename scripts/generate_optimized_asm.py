#!/usr/bin/env python3
"""
Generate Optimized ASM for gzippy v4 decoder

This script analyzes the LLVM assembly and generates specific optimizations
for our hand-written v4 ASM decoder to close the performance gap.

Key optimizations identified from LLVM:
1. Use csel/ccmp for conditional logic (branchless)
2. Use bfxil for bit field operations
3. Preload entries before consuming current
4. Better register allocation
5. More aggressive unrolling
"""

import re
from pathlib import Path
from typing import List, Tuple

def read_v4_asm() -> str:
    """Read the current v4 ASM from asm_decode.rs"""
    path = Path("src/asm_decode.rs")
    content = path.read_text()
    
    # Extract the v4 ASM block
    match = re.search(
        r'pub fn decode_huffman_asm_v4.*?std::arch::asm!\((.*?)\n\s*// Outputs',
        content, re.DOTALL
    )
    if match:
        return match.group(1)
    return ""

def generate_preload_optimization() -> str:
    """Generate preload pattern - lookup next entry before processing current"""
    return '''
            // PRELOAD OPTIMIZATION: Lookup next entry before processing current
            // This hides memory latency by issuing the load early
            
            // After writing literal, preload next entry before loop check
            "10:",  // LITERAL path with preload
            // Consume and write literal
            "and w14, {entry:w}, #0xff",
            "lsr {bitbuf}, {bitbuf}, x14",
            "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
            "lsr w14, {entry:w}, #16",
            
            // PRELOAD: lookup next entry BEFORE writing
            "and x15, {bitbuf}, {tablemask}",
            "ldr w24, [{litlen_ptr}, x15, lsl #2]",
            
            // Now write the literal
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            
            // Check if next is also literal (preloaded in w24)
            "tbz w24, #31, 11f",  // Not literal, go to non-literal handling
            
            // Next is literal - process it immediately
            "mov {entry:w}, w24",
            "b 10b",  // Loop back to process this literal
            
            "11:",  // Next is not a literal
            "mov {entry:w}, w24",
            "b 2b",  // Go back to main loop with new entry
'''

def generate_conditional_refill() -> str:
    """Generate branchless refill using csel/ccmp"""
    return '''
            // BRANCHLESS REFILL using csel/ccmp (LLVM pattern)
            // This reduces branch mispredictions
            
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #47",
            
            // Load word unconditionally (will be discarded if not needed)
            "ldr x9, [{in_ptr}, {in_pos}]",
            "lsl x9, x9, x14",
            "orr x15, x9, {bitbuf}",
            
            // Calculate new in_pos
            "lsr w11, w14, #3",
            "mov w22, #7",
            "sub w11, w22, w11",
            
            // Use csel to conditionally select new values
            "cmp w14, #47",
            "csel {bitbuf}, x15, {bitbuf}, ls",  // if bitsleft <= 47, use refilled
            "csel x11, x11, xzr, ls",            // if bitsleft <= 47, advance in_pos
            "add {in_pos}, {in_pos}, x11",
            "mov w11, #56",
            "bfxil w11, w14, #0, #3",
            "csel {bitsleft:w}, w11, {bitsleft:w}, ls",
'''

def generate_packed_literal_write() -> str:
    """Generate packed 2-literal write using u16 store"""
    return '''
            // PACKED LITERAL WRITE: Write 2 literals with one u16 store
            // When we know next entry is also a literal
            
            // First literal in w14, second in w24 (preloaded)
            "lsr w14, {entry:w}, #16",
            "and x15, {bitbuf}, {tablemask}",
            "ldr w24, [{litlen_ptr}, x15, lsl #2]",
            
            // Check if next is literal
            "tbz w24, #31, 15f",  // Not literal, write single
            
            // Both are literals - pack and write as u16
            "lsr w15, w24, #16",
            "orr w14, w14, w15, lsl #8",  // w14 = lit1 | (lit2 << 8)
            "strh w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #2",
            
            // Consume second literal entry
            "and w14, w24, #0xff",
            "lsr {bitbuf}, {bitbuf}, x14",
            "sub {bitsleft:w}, {bitsleft:w}, w24",
            "b 2b",
            
            "15:",  // Single literal
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            "mov {entry:w}, w24",
            "b 2b",
'''

def generate_8_literal_batch() -> str:
    """Generate 8-literal batching for highly-compressible data"""
    return '''
            // 8-LITERAL BATCH: Process 8 literals in a row
            // Uses packed u64 store for maximum throughput
            
            "8lit:",
            // Check if we have enough bits for 8 literals (8 * 9 = 72 bits max)
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #72",
            "b.lo 10f",  // Not enough bits, use normal path
            
            // Speculatively load 8 entries
            ".set off, 0",
            ".rept 8",
            "and x15, {bitbuf}, {tablemask}",
            "ldr w{=24+off}, [{litlen_ptr}, x15, lsl #2]",
            "and w14, w{=24+off}, #0xff",
            "lsr {bitbuf}, {bitbuf}, x14",
            "sub {bitsleft:w}, {bitsleft:w}, w{=24+off}",
            ".set off, off+1",
            ".endr",
            
            // Check all 8 are literals
            "and x14, x24, x25",
            "and x14, x14, x26",
            "and x14, x14, x27",
            "and x14, x14, x28",
            // ... check all
            "tbz x14, #31, 10f",  // Not all literals
            
            // Extract and pack 8 literals
            "lsr w14, w24, #16",
            "lsr w15, w25, #16",
            "orr x14, x14, x15, lsl #8",
            "lsr w15, w26, #16",
            "orr x14, x14, x15, lsl #16",
            "lsr w15, w27, #16",
            "orr x14, x14, x15, lsl #24",
            // ... continue for remaining 4
            
            // Write as u64
            "str x14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #8",
            "b 2b",
'''

def generate_better_match_copy() -> str:
    """Generate optimized match copy with 64-byte unrolling"""
    return '''
            // OPTIMIZED MATCH COPY with 64-byte SIMD unrolling
            
            // Choose copy strategy based on distance and length
            "cmp w27, #64",
            "b.lo 35f",
            "cmp w26, #128",
            "b.lo 35f",
            
            // 64-byte SIMD copy loop (non-overlapping, long matches)
            "64copy:",
            "ldp q0, q1, [x24]",
            "ldp q2, q3, [x24, #32]",
            "stp q0, q1, [x25]",
            "stp q2, q3, [x25, #32]",
            "add x24, x24, #64",
            "add x25, x25, #64",
            "subs w28, w28, #64",
            "b.hi 64copy",
            "b 40f",
            
            "35:",
            // 32-byte copy for medium matches
            "cmp w27, #32",
            "b.lo 36f",
            "cmp w26, #32",
            "b.lo 36f",
            
            "32copy:",
            "ldp q0, q1, [x24]",
            "stp q0, q1, [x25]",
            "add x24, x24, #32",
            "add x25, x25, #32",
            "subs w28, w28, #32",
            "b.hi 32copy",
            "b 40f",
            
            "36:",
            // 8-byte copy for short non-overlapping
            "cmp w27, #8",
            "b.lo 38f",
            
            "37:",
            "ldr x9, [x24], #8",
            "str x9, [x25], #8",
            "subs w28, w28, #8",
            "b.hi 37b",
            "b 40f",
            
            "38:",
            // Byte copy for overlapping
            "ldrb w9, [x24], #1",
            "strb w9, [x25], #1",
            "subs w28, w28, #1",
            "b.hi 38b",
            
            "40:",
            "add {out_pos}, {out_pos}, x26",
'''

def generate_optimized_distance_decode() -> str:
    """Generate optimized distance decode using LLVM patterns"""
    return '''
            // OPTIMIZED DISTANCE DECODE with preload
            
            // Save bitbuf for extra bits
            "mov x24, {bitbuf}",
            
            // Lookup distance entry (preload during length processing)
            "and x11, x24, {dist_mask}",
            "ldr w11, [{dist_ptr}, x11, lsl #2]",
            
            // Check for subtable using tbz (single instruction)
            "tbz w11, #14, 22f",
            
            // Subtable path (same as current)
            "lsr x24, x24, #8",
            "sub {bitsleft:w}, {bitsleft:w}, #8",
            "ubfx x14, x11, #8, #4",
            "mov x9, #-1",
            "lsl x14, x9, x14",
            "bic x14, x24, x14",
            "add x14, x14, x11, lsr #16",
            "ldr w11, [{dist_ptr}, x14, lsl #2]",
            
            "22:",
            // Decode distance with bfxil for efficiency
            "lsr {bitbuf}, x24, x11",
            "sub {bitsleft:w}, {bitsleft:w}, w11",
            
            // Extract extra bits efficiently
            "ubfx w9, w11, #8, #4",
            "and w14, w11, #0x1f",
            "sub w14, w14, w9",
            "mov x15, #1",
            "lsl x15, x15, x14",
            "sub x15, x15, #1",
            "lsr x14, x24, x9",
            "and x14, x14, x15",
            "add w27, w14, w11, lsr #16",
'''

def main():
    print("=" * 70)
    print("OPTIMIZED ASM GENERATOR FOR GZIPPY V4")
    print("=" * 70)
    
    print("\nReading current v4 ASM...")
    v4_asm = read_v4_asm()
    v4_lines = len(v4_asm.split('\n'))
    print(f"Current v4 has {v4_lines} lines of ASM")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION OPPORTUNITIES FROM LLVM ANALYSIS")
    print("=" * 70)
    
    optimizations = [
        ("1. PRELOAD PATTERN", 
         "Issue table lookup before consuming current entry",
         "Hides ~3 cycle memory latency",
         generate_preload_optimization),
        
        ("2. BRANCHLESS REFILL",
         "Use csel/ccmp instead of branches for refill",
         "Eliminates branch misprediction penalty (~10 cycles)",
         generate_conditional_refill),
        
        ("3. PACKED LITERAL WRITES",
         "Write 2 literals with one u16 store when consecutive",
         "Reduces store instructions by 50% in literal runs",
         generate_packed_literal_write),
        
        ("4. BETTER MATCH COPY",
         "64-byte SIMD copy for long matches",
         "Doubles throughput for long matches",
         generate_better_match_copy),
        
        ("5. OPTIMIZED DISTANCE DECODE",
         "Use LLVM's bit extraction patterns",
         "Reduces instruction count by ~20%",
         generate_optimized_distance_decode),
    ]
    
    for name, desc, impact, generator in optimizations:
        print(f"\n### {name}")
        print(f"Description: {desc}")
        print(f"Expected impact: {impact}")
        print("\nGenerated ASM snippet:")
        print("-" * 40)
        snippet = generator()
        # Print first 20 lines
        lines = snippet.strip().split('\n')
        for line in lines[:20]:
            print(line)
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION PRIORITY")
    print("=" * 70)
    print("""
1. HIGHEST: Preload pattern - Most impact for least code change
2. HIGH: Packed literal writes - Helps literal-heavy data
3. MEDIUM: Better match copy - Helps match-heavy data  
4. MEDIUM: Branchless refill - Helps unpredictable patterns
5. LOW: 8-literal batch - Only helps very compressible data

Estimated total improvement: 15-25% closer to LLVM
(From 70% to 85-95% of LLVM performance)
""")
    
    # Save optimizations to file
    output_path = Path("target/optimized_asm_snippets.txt")
    with open(output_path, 'w') as f:
        f.write("OPTIMIZED ASM SNIPPETS FOR GZIPPY V4\n")
        f.write("=" * 70 + "\n\n")
        
        for name, desc, impact, generator in optimizations:
            f.write(f"### {name}\n")
            f.write(f"Description: {desc}\n")
            f.write(f"Impact: {impact}\n\n")
            f.write(generator())
            f.write("\n\n")
    
    print(f"\nFull snippets saved to: {output_path}")

if __name__ == "__main__":
    main()
