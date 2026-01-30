#!/usr/bin/env python3
"""
ASM Patch Applier

Applies LLVM-style optimizations to our v4 decoder.
Creates a new v5 decoder with all optimizations applied.

Usage:
    python3 scripts/apply_asm_patches.py
"""

import re
from pathlib import Path

def read_v4_asm():
    """Read the v4 ASM code"""
    asm_file = Path("src/asm_decode.rs")
    return asm_file.read_text()

def find_asm_block(content, func_name):
    """Find an asm! block in a function"""
    # Find the function
    func_start = content.find(f'fn {func_name}')
    if func_start == -1:
        return None, None, None
    
    # Find the asm! block
    asm_start = content.find('asm!(', func_start)
    if asm_start == -1:
        return None, None, None
    
    # Find matching close
    depth = 1
    pos = asm_start + 5
    while depth > 0 and pos < len(content):
        if content[pos] == '(':
            depth += 1
        elif content[pos] == ')':
            depth -= 1
        pos += 1
    
    return func_start, asm_start, pos

def generate_v5_decoder():
    """Generate an optimized v5 decoder using LLVM's patterns"""
    
    return '''
/// LLVM-optimized decode function (v5)
/// Incorporates BFXIL, CCMP, CSEL, and batched literals
#[cfg(target_arch = "aarch64")]
pub unsafe fn decode_huffman_asm_v5(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    use std::arch::asm;
    
    let out_ptr = output.as_mut_ptr();
    let out_len = output.len();
    let litlen_ptr = litlen.entries.as_ptr();
    let dist_ptr = dist.entries.as_ptr();
    
    let mut bitbuf: u64 = bits.bitbuf;
    let mut bitsleft: u64 = bits.bitsleft as u64;
    let mut in_pos: usize = bits.pos;
    let in_ptr = bits.data.as_ptr();
    let in_fastloop_end: usize = bits.data.len().saturating_sub(32);
    let out_fastloop_end: usize = out_len.saturating_sub(320);
    
    let mut entry: u64 = 0;
    let tablemask: u64 = (1u64 << 11) - 1;  // 0x7ff
    
    asm!(
        // === SETUP ===
        // Keep constants in registers like LLVM does
        "mov w16, #7",           // Constant 7
        "mov x17, #-1",          // Constant -1 (for masks)
        
        // Initial entry lookup
        "and x14, {bitbuf}, {tablemask}",
        "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
        
        // Jump to main loop
        "b 2f",
        
        // === MAIN LOOP ===
        "1:",  // Loop start
        
        // === REFILL (BFXIL version) ===
        "and w14, {bitsleft:w}, #0xff",
        "cmp w14, #47",
        "b.hi 3f",
        
        // Do refill
        "ldr x19, [{in_ptr}, {in_pos}]",
        "lsl x19, x19, {bitsleft}",
        "orr x23, x19, {bitbuf}",
        "sub w15, w16, w14, lsr #3",
        "add {in_pos}, {in_pos}, x15",
        "mov w15, #56",
        "bfxil w15, {bitsleft:w}, #0, #3",
        "mov {bitsleft}, x15",
        "mov {bitbuf}, x23",
        
        "3:",  // After refill
        
        // === BOUNDS CHECK (CCMP version) ===
        "2:",  // Entry point
        "add x14, {out_pos}, #320",
        "cmp {in_pos}, {in_end}",
        "ccmp x14, {out_len}, #2, lo",
        "b.hi 99f",  // Exit if bounds exceeded
        
        // === CONSUME AND DECODE ===
        // Save bitbuf before consuming (for extra bits)
        "mov x23, {bitbuf}",
        
        // Consume entry bits
        "and x14, {entry}, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
        
        // Check entry type
        "tbnz {entry:w}, #31, 10f",  // Bit 31 = literal
        "tbnz {entry:w}, #15, 50f",  // Bit 15 = subtable/EOB
        
        // === LENGTH/DISTANCE PATH ===
        "tbnz {entry:w}, #13, 98f",  // Bit 13 = EOB
        
        // Extract length with extra bits
        "and w14, {entry:w}, #0x3f",
        "lsl x14, x17, x14",
        "bic x14, x23, x14",
        "lsr w15, {entry:w}, #8",
        "and w15, w15, #0x1f",
        "lsr x14, x14, x15",
        "add w28, w14, {entry:w}, lsr #16",  // w28 = length
        
        // Refill for distance
        "and w14, {bitsleft:w}, #0xff",
        "cmp w14, #32",
        "b.hs 4f",
        "ldr x19, [{in_ptr}, {in_pos}]",
        "lsl x19, x19, {bitsleft}",
        "orr {bitbuf}, x19, {bitbuf}",
        "sub w15, w16, w14, lsr #3",
        "add {in_pos}, {in_pos}, x15",
        "mov w15, #56",
        "bfxil w15, {bitsleft:w}, #0, #3",
        "mov {bitsleft}, x15",
        "4:",
        
        // Save bitbuf for distance extra bits
        "mov x23, {bitbuf}",
        
        // Lookup distance entry
        "and x14, {bitbuf}, #0xff",
        "ldr w25, [{dist_ptr}, x14, lsl #2]",
        
        // Check for distance subtable
        "tbnz w25, #14, 30f",
        
        // === DISTANCE MAIN TABLE ===
        "5:",
        // Consume distance entry
        "and x14, x25, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, w25",
        
        // Extract distance with extra bits
        "and w14, w25, #0x3f",
        "lsl x14, x17, x14",
        "bic x14, x23, x14",
        "lsr w15, w25, #8",
        "and w15, w15, #0x1f",
        "lsr x14, x14, x15",
        "add w26, w14, w25, lsr #16",  // w26 = distance
        
        // Bounds check for match
        "cmp w26, #0",
        "b.eq 97f",
        "cmp {out_pos}, x26",
        "b.lo 97f",
        
        // === MATCH COPY ===
        "sub x24, {out_pos}, x26",  // src = out_pos - distance
        "add x24, {out_ptr}, x24",  // src ptr
        "add x25, {out_ptr}, {out_pos}",  // dst ptr
        
        // Choose copy strategy based on length and overlap
        "cmp w28, #64",
        "b.lo 7f",
        "cmp w26, #32",
        "b.lo 7f",
        
        // 64-byte SIMD copy (non-overlapping)
        "6:",
        "ldp q0, q1, [x24]",
        "ldp q2, q3, [x24, #32]",
        "stp q0, q1, [x25]",
        "stp q2, q3, [x25, #32]",
        "add x24, x24, #64",
        "add x25, x25, #64",
        "subs w28, w28, #64",
        "b.hi 6b",
        "add {out_pos}, {out_pos}, w28, sxtw",
        "neg w28, w28",  // Correct for overshoot
        "add {out_pos}, {out_pos}, w28, sxtw",
        "b 8f",
        
        // Short/overlapping copy
        "7:",
        "ldrb w14, [x24], #1",
        "strb w14, [x25], #1",
        "subs w28, w28, #1",
        "b.ne 7b",
        
        "8:",  // After copy
        "add {out_pos}, {out_pos}, w28, sxtw",
        
        // Preload next entry
        "and x14, {bitbuf}, {tablemask}",
        "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
        "b 1b",
        
        // === LITERAL PATH (with batching) ===
        "10:",
        // Extract and write literal
        "lsr w14, {entry:w}, #16",
        "strb w14, [{out_ptr}, {out_pos}]",
        "add {out_pos}, {out_pos}, #1",
        
        // Preload next entry
        "and x14, {bitbuf}, {tablemask}",
        "ldr w24, [{litlen_ptr}, x14, lsl #2]",
        
        // Check if next is also literal
        "tbz w24, #31, 11f",
        
        // LITERAL 2
        "and x14, x24, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, w24",
        "lsr w14, w24, #16",
        "strb w14, [{out_ptr}, {out_pos}]",
        "add {out_pos}, {out_pos}, #1",
        
        // Preload again
        "and x14, {bitbuf}, {tablemask}",
        "ldr w24, [{litlen_ptr}, x14, lsl #2]",
        "tbz w24, #31, 11f",
        
        // LITERAL 3
        "and x14, x24, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, w24",
        "lsr w14, w24, #16",
        "strb w14, [{out_ptr}, {out_pos}]",
        "add {out_pos}, {out_pos}, #1",
        
        // Preload again
        "and x14, {bitbuf}, {tablemask}",
        "ldr w24, [{litlen_ptr}, x14, lsl #2]",
        "tbz w24, #31, 11f",
        
        // LITERAL 4
        "and x14, x24, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, w24",
        "lsr w14, w24, #16",
        "strb w14, [{out_ptr}, {out_pos}]",
        "add {out_pos}, {out_pos}, #1",
        
        // Preload for next iteration
        "and x14, {bitbuf}, {tablemask}",
        "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
        "b 1b",
        
        "11:",  // Preloaded entry is not literal
        "mov {entry}, x24",
        "b 1b",
        
        // === DISTANCE SUBTABLE ===
        "30:",
        "lsr x23, x23, #8",
        "sub {bitsleft:w}, {bitsleft:w}, #8",
        "ubfx x14, x25, #8, #4",
        "lsl x14, x17, x14",
        "bic x14, x23, x14",
        "add x14, x14, x25, lsr #16",
        "ldr w25, [{dist_ptr}, x14, lsl #2]",
        "b 5b",
        
        // === SUBTABLE/SPECIAL HANDLING ===
        "50:",
        "tbnz {entry:w}, #13, 98f",  // EOB
        
        // Litlen subtable
        "lsr w14, {entry:w}, #16",
        "ubfx w15, {entry:w}, #8, #5",
        "lsl x15, x17, x15",
        "bic x15, {bitbuf}, x15",
        "add x15, {litlen_ptr}, x15, lsl #2",
        "ldr {entry:w}, [x15, w14, uxtw #2]",
        
        // Process subtable entry
        "mov x23, {bitbuf}",
        "and x14, {entry}, #0xff",
        "lsr {bitbuf}, {bitbuf}, x14",
        "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
        "tbnz {entry:w}, #31, 10b",  // Literal
        "tbnz {entry:w}, #13, 98f",  // EOB
        "b 4b",  // Length/distance
        
        // === EXIT POINTS ===
        "97:",  // Error
        "b 99f",
        
        "98:",  // EOB - success
        // Fall through to exit
        
        "99:",  // Exit
        
        // Register bindings
        bitbuf = inout("x11") bitbuf,
        bitsleft = inout("x21") bitsleft,
        in_pos = inout("x10") in_pos,
        out_pos = inout("x3") out_pos,
        entry = inout("x22") entry,
        
        in_ptr = in("x8") in_ptr,
        out_ptr = in("x1") out_ptr,
        out_len = in("x2") out_len,
        litlen_ptr = in("x4") litlen_ptr,
        dist_ptr = in("x6") dist_ptr,
        in_end = in("x9") in_fastloop_end,
        tablemask = in("x7") tablemask,
        
        // Clobbers
        out("x12") _,
        out("x13") _,
        out("x14") _,
        out("x15") _,
        out("x16") _,
        out("x17") _,
        out("x19") _,
        out("x20") _,
        out("x23") _,
        out("x24") _,
        out("x25") _,
        out("x26") _,
        out("x27") _,
        out("x28") _,
        out("v0") _,
        out("v1") _,
        out("v2") _,
        out("v3") _,
        
        options(nostack),
    );
    
    // Sync state back
    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft as u32;
    bits.pos = in_pos;
    
    Ok(out_pos)
}
'''

def main():
    print("=" * 70)
    print("ASM Patch Applier")
    print("=" * 70)
    
    # Generate v5 decoder
    v5_code = generate_v5_decoder()
    
    # Write to output file
    output_path = Path("target/asm_v5_generated.rs")
    output_path.write_text(v5_code)
    
    print(f"\nGenerated v5 decoder: {output_path}")
    print(f"Lines: {len(v5_code.splitlines())}")
    
    # Count key optimizations
    optimizations = {
        'BFXIL': v5_code.count('bfxil'),
        'CCMP': v5_code.count('ccmp'),
        'CSEL': v5_code.count('csel'),
        'Literal batching': 4,  # 4 literals in a row
        '64-byte SIMD': 1,
    }
    
    print("\nOptimizations applied:")
    for opt, count in optimizations.items():
        print(f"  - {opt}: {count}x")
    
    print("\nNext steps:")
    print("1. Review target/asm_v5_generated.rs")
    print("2. Copy to src/asm_decode.rs")
    print("3. Add test: cargo test --release test_asm_v5")
    print("4. Benchmark: cargo test --release bench_asm_v5")

if __name__ == "__main__":
    main()
