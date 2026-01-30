# Inline Assembly Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for implementing custom inline assembly for gzippy's Huffman decode hot path, based on lessons learned from our first attempt.

## Lessons from First Attempt

### What Went Wrong

1. **Register Allocation Conflicts (ARM64)**: Using `inout(reg)` with explicit `mov` instructions inside the asm block caused LLVM to allocate the same register for multiple purposes, resulting in garbage output values.

2. **Single-Literal Loop Was Too Slow**: The simplified ASM loop processed one literal at a time, while the Rust version batches 8 literals. The overhead of the Rust↔ASM boundary per literal destroyed performance.

3. **Complex State Management**: The full decode loop has ~15 live variables (bitbuf, bitsleft, in_pos, out_pos, entry, 8 packed literals, etc.) which exceeded the available general-purpose registers on both architectures.

4. **Exit Condition Handling**: Detecting and handling EOB, subtable lookups, and length codes required complex branching that was error-prone in assembly.

---

## Failure Modes and Mitigation Strategies

### FAILURE 1: Register Allocation Conflicts

**Problem**: Rust's inline asm macro allocates registers, but we also manually use specific registers inside the asm block, causing conflicts.

**Strategies**:

1. **Use Only Named Operands (No Explicit Register Names)**
   - Instead of `mov x0, {bb}` followed by operations on `x0`, use the operand directly: `lsr {bb}, {bb}, {shift}`
   - Let LLVM handle ALL register allocation
   
2. **Use `clobber_abi("C")` Instead of Individual Clobbers**
   - Tells LLVM that all caller-saved registers may be modified
   - Simpler and less error-prone than listing each register

3. **Separate Input/Output Operands Completely**
   - Use `in(reg)` for read-only values
   - Use `out(reg)` for write-only values  
   - Use `inout(reg)` ONLY when the same register is read then written
   - Never use explicit register names like `x0`, `rax` inside the asm

4. **Write a Test Function That Returns All Outputs**
   - Before building the full loop, verify a simple function that takes inputs and returns outputs works correctly
   - Example: `fn test_asm(a: u64, b: u64) -> (u64, u64)` that does `a+b, a*b`

### FAILURE 2: Single-Literal Loop Too Slow

**Problem**: Processing one literal at a time has too much overhead.

**Strategies**:

1. **Batch 8 Literals in Assembly**
   - Match the Rust implementation: process up to 8 literals before returning to Rust
   - Use packed writes (u64 store for 8 literals, u32 for 4, u16 for 2)
   - Only exit ASM when encountering a non-literal

2. **Minimize ASM Entry/Exit**
   - Keep the ENTIRE fastloop in assembly, including length/distance handling
   - Only exit for: (a) end of block, (b) bounds check, (c) error
   - Handle subtables within the assembly block

3. **Use Global ASM Instead of Inline ASM**
   - Write a standalone `.s` file with the full decode function
   - Link it with `global_asm!` or build.rs
   - Avoids inline asm register allocation complexity entirely

4. **Hybrid Approach: ASM for Tight Inner Loop Only**
   - Keep Rust wrapper for bounds checking and error handling
   - Use ASM only for the literal burst: decode up to N literals and return count + last entry
   - This limits the ASM complexity while still optimizing the hot path

### FAILURE 3: Too Many Live Variables

**Problem**: The decode loop needs ~15 variables but ARM64 has 31 GPRs (some reserved) and x86_64 has 16.

**Strategies**:

1. **Keep Constants in Memory**
   - TABLE_MASK can be computed once and stored on stack
   - Table pointers (litlen_ptr, dist_ptr) can be loaded from stack when needed
   - Only keep the truly hot variables (bitbuf, bitsleft, entry) in registers

2. **Use SIMD Registers for Storage**
   - ARM64: Use v0-v7 as 64-bit storage for non-arithmetic values
   - x86_64: Use xmm registers for the same purpose
   - Example: Store table pointers in SIMD, move to GPR when needed

3. **Reduce Variable Count by Restructuring**
   - Pack out_pos into the high bits of bitsleft (since bitsleft only uses 0-63)
   - Use in_pos as an offset from end instead of from start (simplifies bounds check)
   - Compute addresses on-the-fly instead of storing pointers

4. **Two-Phase Decode**
   - Phase 1: Decode symbols into a buffer (no match handling)
   - Phase 2: Process the buffer, handling matches
   - Each phase has fewer live variables

### FAILURE 4: Complex Exit Condition Handling

**Problem**: Detecting EOB, subtables, and length codes requires complex branching.

**Strategies**:

1. **Unified Entry Format**
   - Redesign the Huffman table entry to make dispatch simpler
   - Use a single check (sign bit) for literal vs everything-else
   - For non-literals, use a jump table indexed by entry type

2. **Speculative Execution**
   - Assume literals are most common (they are: ~70% of symbols in typical data)
   - Decode as if literal, then backtrack if wrong
   - This keeps the hot path branchless

3. **Return to Rust for Complex Cases**
   - ASM handles ONLY literals (the common case)
   - Returns to Rust wrapper for: length/distance, EOB, subtable
   - Simplifies ASM dramatically while still optimizing the hot path

4. **Use Computed Goto / Jump Table**
   - Extract entry type bits and use them as jump table index
   - Each type (literal, length, EOB, subtable) has its own code path
   - More branches but each path is simpler

### FAILURE 5: Incorrect Bit Operations

**Problem**: Subtle bugs in shift directions, mask widths, or byte order.

**Strategies**:

1. **Test Each Primitive Independently**
   - Write unit tests for refill, consume, lookup as separate functions
   - Verify against Rust reference implementation with random inputs
   - Use property-based testing (proptest/quickcheck)

2. **Use Consistent Conventions Document**
   - Write down: "bitbuf LSB = next bit to decode", "little-endian loads", etc.
   - Every asm instruction should have a comment explaining what it does

3. **Build Incrementally**
   - Step 1: Get single-literal decode working and tested
   - Step 2: Add batching (2, 4, 8 literals)
   - Step 3: Add length/distance handling
   - Step 4: Add full loop with bounds checking
   - Test exhaustively at each step

4. **Compare Output Byte-by-Byte**
   - After each ASM decode, compare against Rust decode of same input
   - Report exact position and values of first mismatch
   - Use small test inputs for easier debugging

---

## Implementation Plan

### Phase 1: Foundation (Estimated: 1-2 hours)

1. **Create test harness for ASM primitives**
   - Function that calls ASM, compares to Rust reference
   - Random input generation
   - Detailed mismatch reporting

2. **Implement and test `consume_bits_asm`**
   - Simplest primitive: `(bitbuf >> n, bitsleft - n)`
   - Verify on both ARM64 and x86_64
   - Use ONLY named operands, no explicit registers

3. **Implement and test `refill_asm`**
   - More complex: load, shift, OR, update in_pos
   - Verify byte order is correct
   - Test edge cases: bitsleft=0, bitsleft=56

4. **Implement and test `lookup_asm`**
   - Simple: mask bits, load from table
   - Verify table indexing is correct

### Phase 2: Literal Burst (Estimated: 2-3 hours)

1. **Implement single-literal decode in ASM**
   - lookup → check literal → extract → consume → write
   - Returns (new_bitbuf, new_bitsleft, entry_if_not_literal)
   - Exit if not literal

2. **Extend to 4-literal batch**
   - Decode up to 4 literals before returning
   - Pack into u32 for single write
   - Refill after every 2 literals

3. **Extend to 8-literal batch**
   - Match the Rust implementation exactly
   - Pack into u64 for single write
   - Handle all exit conditions (2, 3, 4, 5, 6, 7, 8 literals)

4. **Benchmark against Rust implementation**
   - Should be within 10% of Rust at this point
   - If slower, profile to find bottleneck

### Phase 3: Full Fast Loop (Estimated: 3-4 hours)

1. **Add length/distance handling inside ASM**
   - When literal check fails and it's a length code:
   - Decode length with extra bits
   - Lookup distance entry
   - Decode distance with extra bits
   - Call match copy (can be Rust function via indirect call)

2. **Add subtable handling**
   - When exceptional bit is set but not EOB:
   - Compute subtable index
   - Load from subtable
   - Continue with regular decode

3. **Add bounds checking**
   - Check in_pos and out_pos at top of loop
   - Exit when near boundaries

4. **Add EOB handling**
   - When EOB entry detected:
   - Consume EOB bits
   - Return to Rust with out_pos

### Phase 4: Optimization (Estimated: 2-3 hours)

1. **Profile and identify bottlenecks**
   - Use perf/Instruments to find hot instructions
   - Look for cache misses, branch mispredictions

2. **Architecture-specific tuning**
   - x86_64: Use BMI2 (shrx, bzhi) where beneficial
   - ARM64: Use appropriate instruction forms (LSL vs UBFX)

3. **Instruction scheduling**
   - Reorder to hide latency
   - Preload next entry while writing current

4. **Memory access optimization**
   - Prefetch table entries
   - Align output writes

---

## Architecture-Specific Notes

### x86_64

```asm
; Key instructions:
; - shrx: shift right by register (BMI2, 3 operands)
; - bzhi: zero high bits (BMI2)
; - shlx: shift left by register (BMI2)
; - movzx: zero-extend load

; Register allocation suggestion:
; rax = bitbuf (hot)
; rbx = scratch
; rcx = shift amount (required for some shifts)
; rdx = bitsleft (low 32 bits)
; rsi = in_pos
; rdi = out_pos
; r8 = in_ptr
; r9 = out_ptr
; r10 = litlen_ptr
; r11 = scratch
; r12-r15 = preserved (for caller)
```

### ARM64

```asm
; Key instructions:
; - lsr: logical shift right
; - lsl: logical shift left
; - and: bitwise AND (can use immediate)
; - ldr: load (with register offset + shift)
; - str/strb/strh: store variants
; - tbnz/tbz: test bit and branch

; Register allocation suggestion:
; x0 = bitbuf (hot)
; w1 = bitsleft (32-bit)
; x2 = in_pos
; x3 = out_pos
; x4 = in_ptr (input, preserved)
; x5 = out_ptr (input, preserved)
; x6 = litlen_ptr (input, preserved)
; x7 = in_end (input, preserved)
; x8 = out_end (input, preserved)
; x9-x15 = scratch
; w16 = entry
; x19-x28 = callee-saved (avoid)
```

---

## Success Criteria

1. **Correctness**: All existing tests pass with ASM decoder
2. **Performance**: ≥95% of libdeflate throughput on SILESIA
3. **Maintainability**: Well-commented, with test coverage for each primitive
4. **Portability**: Works on both x86_64 and ARM64

---

## Fallback Plan

If inline assembly continues to have issues:

1. **Global ASM**: Write standalone `.s` files, link via build.rs
2. **C with Inline ASM**: Write the hot loop in C with inline asm, call from Rust via FFI
3. **Pure Rust with Intrinsics**: Use `std::arch` intrinsics for specific operations

---

## References

- [Rust Inline Assembly](https://doc.rust-lang.org/reference/inline-assembly.html)
- [libdeflate decompress_template.h](../libdeflate/lib/decompress_template.h)
- [ISA-L igzip_inflate.c](../isa-l/igzip/igzip_inflate.c)
- [ARM64 Instruction Set](https://developer.arm.com/documentation/ddi0602/latest/)
- [x86_64 Instruction Set](https://www.felixcloutier.com/x86/)
