#!/usr/bin/env python3
"""
ASM Patch Generator

Extracts specific patterns from LLVM's assembly and generates
ready-to-use code patches for our v4 decoder.

Usage:
    python3 scripts/generate_asm_patches.py
"""

import re
from pathlib import Path

def extract_llvm_patterns():
    """Extract key patterns from LLVM's assembly"""
    raw_file = Path("target/llvm_raw_decode.s")
    if not raw_file.exists():
        print("Run llvm_to_inline_asm.py first")
        return {}
    
    content = raw_file.read_text()
    lines = content.split('\n')
    
    patterns = {}
    
    # Find refill pattern
    for i, line in enumerate(lines):
        if 'cmp' in line.lower() and '#47' in line:
            patterns['refill_fast'] = extract_block(lines, i, 15)
            break
    
    # Find literal decode with preload
    for i, line in enumerate(lines):
        if 'tbnz' in line.lower() and '#31' in line and 'LBB' in line:
            # This is the literal check - get the code after it
            patterns['literal_with_preload'] = extract_block(lines, i, 20)
            break
    
    # Find match copy pattern
    for i, line in enumerate(lines):
        if 'ldp' in line.lower() and 'q0' in line.lower():
            patterns['simd_match_copy'] = extract_block(lines, i-2, 15)
            break
    
    # Find distance decode
    for i, line in enumerate(lines):
        if 'dist' in line.lower() and 'ldr' in line.lower():
            patterns['distance_decode'] = extract_block(lines, i-2, 12)
            break
    
    # Find bounds check pattern
    for i, line in enumerate(lines):
        if 'ccmp' in line.lower():
            patterns['bounds_check'] = extract_block(lines, i-3, 8)
            break
    
    return patterns

def extract_block(lines, start, count):
    """Extract a block of lines"""
    block = []
    for i in range(start, min(start + count, len(lines))):
        line = lines[i].strip()
        if line and not line.startswith('.') and not line.startswith('//'):
            block.append(line)
    return block

def convert_to_inline_asm(block, description):
    """Convert a block to Rust inline ASM format"""
    result = [f"        // {description}"]
    for line in block:
        # Clean up the line
        line = re.sub(r'\s+', ' ', line).strip()
        if line.endswith(':'):
            result.append(f'        "{line}",')
        else:
            result.append(f'        "{line}",')
    return result

def generate_optimized_refill():
    """Generate optimized refill pattern using BFXIL"""
    return '''
        // OPTIMIZED REFILL - using BFXIL like LLVM
        // Check if refill needed
        "and w14, {bitsleft:w}, #0xff",
        "cmp w14, #47",
        "b.hi 3f",                    // Skip if enough bits
        
        // Do refill
        "ldr x19, [{in_ptr}, {in_pos}]",
        "lsl x19, x19, {bitsleft}",   // Shift new data
        "orr x23, x19, {bitbuf}",     // Merge with existing
        
        // Update in_pos: add (7 - (bitsleft >> 3)) & 7 = 7 - floor(bitsleft/8)
        "sub w14, w16, w14, lsr #3",  // w16 contains 7
        "add {in_pos}, {in_pos}, x14",
        
        // Update bitsleft to 56 | (bitsleft & 7)
        "mov w14, #56",
        "bfxil w14, {bitsleft:w}, #0, #3",  // Insert low 3 bits
        "mov {bitsleft}, x14",
        
        // Save old bitbuf for extra bits extraction
        "mov {bitbuf}, x23",          // Now bitbuf = merged buffer
        "3:",
'''

def generate_literal_batch():
    """Generate batched literal decode with preload"""
    return '''
        // LITERAL BATCH - decode 2 literals with preload
        // Entry already consumed, literal value in entry>>16
        
        "10:",  // Literal path
        // Consume entry bits
        "and x14, {entry}, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
        
        // PRELOAD next entry while we process current
        "and x19, {bitbuf}, #0x7ff",
        "ldr w24, [{litlen_ptr}, x19, lsl #2]",
        
        // Extract and write literal
        "lsr w14, {entry:w}, #16",
        "strb w14, [{out_ptr}, {out_pos}]",
        "add {out_pos}, {out_pos}, #1",
        
        // Check if preloaded is also literal
        "tbz w24, #31, 11f",          // Not literal? Go process entry
        
        // LITERAL 2 - process preloaded
        "and x14, x24, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, w24",
        "lsr w14, w24, #16",
        "strb w14, [{out_ptr}, {out_pos}]",
        "add {out_pos}, {out_pos}, #1",
        
        // Preload again
        "and x19, {bitbuf}, #0x7ff",
        "ldr {entry:w}, [{litlen_ptr}, x19, lsl #2]",
        "b 2b",                       // Back to loop check
        
        "11:",  // Preloaded wasn't literal
        "mov {entry}, x24",
        "b 2b",
'''

def generate_ccmp_bounds_check():
    """Generate chained bounds check using CCMP"""
    return '''
        // CCMP BOUNDS CHECK - chain two conditions
        // Check: in_pos < in_end && out_pos + 320 < out_len
        "add x14, {out_pos}, #320",
        "cmp {in_pos}, {in_end}",
        "ccmp x14, {out_len}, #2, lo",  // Only check if in_pos < in_end
        "b.hi 99f",                     // Exit if either fails
'''

def generate_csel_refill():
    """Generate branchless refill using CSEL"""
    return '''
        // BRANCHLESS REFILL using CSEL
        // Note: May be slower due to unconditional load
        "and w14, {bitsleft:w}, #0xff",
        "cmp w14, #47",
        
        // Unconditionally load (may be wasted)
        "ldr x19, [{in_ptr}, {in_pos}]",
        "lsl x23, x19, {bitsleft}",
        "orr x23, x23, {bitbuf}",
        
        // Calculate new in_pos
        "sub w24, w16, w14, lsr #3",
        "add x25, {in_pos}, x24",
        
        // Calculate new bitsleft
        "mov w26, #56",
        "bfxil w26, {bitsleft:w}, #0, #3",
        
        // Conditionally select old or new values
        "csel {bitbuf}, x23, {bitbuf}, ls",
        "csel {in_pos}, x25, {in_pos}, ls",
        "csel {bitsleft:w}, w26, {bitsleft:w}, ls",
'''

def generate_distance_with_bfxil():
    """Generate distance decode using bit manipulation"""
    return '''
        // DISTANCE DECODE with BFXIL
        // Entry format: [extra_bits:6][extra_count:5][base_offset:16]
        "and w14, {entry:w}, #0x3f",     // Extract extra count (bits 0-5)
        "lsl x14, x17, x14",             // x17 = -1, create mask
        "bic x14, x23, x14",             // Extract extra bits from saved_bitbuf
        "lsr w19, {entry:w}, #8",        // Get shift amount
        "and w19, w19, #0x1f",
        "lsr x14, x14, x19",             // Shift extra bits
        "add w14, w14, {entry:w}, lsr #16",  // Add base offset
'''

def generate_packed_literal_write():
    """Generate packed 2-byte literal write"""
    return '''
        // PACKED 2-BYTE LITERAL WRITE
        // When we have 2 consecutive literals, write as u16
        "lsr w14, {entry:w}, #16",       // First literal
        "lsr w19, w24, #16",             // Second literal
        "orr w14, w14, w19, lsl #8",     // Pack into u16
        "strh w14, [{out_ptr}, {out_pos}]",  // Write 2 bytes
        "add {out_pos}, {out_pos}, #2",
'''

def main():
    print("=" * 70)
    print("ASM Patch Generator")
    print("=" * 70)
    
    patterns = extract_llvm_patterns()
    
    print(f"\nExtracted {len(patterns)} patterns from LLVM")
    for name, block in patterns.items():
        print(f"  - {name}: {len(block)} lines")
    
    print("\n" + "=" * 70)
    print("READY-TO-USE ASM PATCHES")
    print("=" * 70)
    
    patches = [
        ("1. OPTIMIZED REFILL (with BFXIL)", generate_optimized_refill()),
        ("2. LITERAL BATCH (2 literals with preload)", generate_literal_batch()),
        ("3. CCMP BOUNDS CHECK (chained conditions)", generate_ccmp_bounds_check()),
        ("4. BRANCHLESS REFILL (CSEL version)", generate_csel_refill()),
        ("5. DISTANCE WITH BFXIL", generate_distance_with_bfxil()),
        ("6. PACKED LITERAL WRITE (2 bytes)", generate_packed_literal_write()),
    ]
    
    for title, code in patches:
        print(f"\n### {title}")
        print("```rust")
        print(code)
        print("```")
    
    # Save to file
    output = Path("target/asm_patches.rs")
    with open(output, 'w') as f:
        f.write("// ASM Patches - ready to integrate into v4 decoder\n")
        f.write("// Generated by scripts/generate_asm_patches.py\n\n")
        for title, code in patches:
            f.write(f"// {title}\n")
            f.write(f"/*{code}*/\n\n")
    
    print(f"\n\nPatches saved to: {output}")
    
    print("\n" + "=" * 70)
    print("LLVM PATTERNS (extracted directly)")
    print("=" * 70)
    
    for name, block in patterns.items():
        print(f"\n### {name}")
        for line in block[:10]:
            print(f"    {line}")
        if len(block) > 10:
            print(f"    ... ({len(block) - 10} more lines)")

if __name__ == "__main__":
    main()
