# Optimization Audit Summary

**Date:** January 2026
**Current Performance:** 976 MB/s (69.0% of libdeflate's 1414 MB/s)
**Target:** 1840+ MB/s (130%+ of libdeflate)

---

## Executive Summary

**We have most optimizations implemented, but they're either:**
1. **Not integrated into the hot path** (multi_symbol.rs, parts of vector_huffman.rs)
2. **Already integrated but not enough** (libdeflate_decode.rs has many optimizations)
3. **Tried and regressed** (DoubleLitCache -73%)

**The 31% gap to parity** (69% → 100%) is likely due to:
- Register allocation differences (Rust vs C)
- Memory access patterns
- CPU-specific optimizations in libdeflate
- Remaining micro-optimizations

**The path to 130%+ requires novel approaches:**
- Full SIMD lane parallelism (vector_huffman infrastructure exists)
- JIT code generation for repeated tables
- Hardware acceleration

---

## Detailed Findings

### ✅ Already Implemented in libdeflate_decode.rs

| Optimization | Location | Status |
|--------------|----------|--------|
| saved_bitbuf pattern | line 588 | ✅ DONE |
| Multi-literal unroll (4x) | lines 608-672 | ✅ DONE |
| Signed comparison for literal | line 598 | ✅ DONE |
| Unsafe pointer writes | lines 602-647 | ✅ DONE |
| Fastloop with margin | lines 582-584 | ✅ DONE |
| Branchless refill | bits.refill_branchless() | ✅ DONE |
| Subtable resolution | lines 592-594 | ✅ DONE |
| Two-level Huffman tables | LitLenTable/DistTable | ✅ DONE |

### ⚠️ Implemented but NOT Integrated

| Module | Purpose | Status | Reason |
|--------|---------|--------|--------|
| `multi_symbol.rs` | Decode 1-2 symbols per lookup | Built, never called | No integration |
| `double_literal.rs` | Decode 2 literals at once | Tried, reverted | **-73% regression** |
| `vector_huffman.rs` decode_fixed_multi_literal | Multi-literal for fixed | Exists | Only used in consume_first path |
| `vector_huffman.rs` decode_8_symbols (SIMD) | 8-lane parallel | Built | Never integrated |

### ❌ Missing from libdeflate

| Optimization | Expected Gain | Difficulty | Notes |
|--------------|---------------|------------|-------|
| Branch hints (`likely/unlikely`) | +2-5% | Easy | `core::intrinsics::likely` |
| `EXTRACT_VARBITS8` u8 cast | +1-2% | Easy | Cast before shift hint |
| Preload next entry during write | +3-5% | Medium | Reduce latency |
| `CAN_CONSUME_AND_THEN_PRELOAD` | +5-10% | Medium | Compile-time bit budget |

### ❌ Missing from rapidgzip

| Optimization | Expected Gain | Difficulty | Notes |
|--------------|---------------|------------|-------|
| `needToReadDistanceBits` flag | +2-5% | Easy | Avoid branch |
| Pre-computed length+extra | +5-10% | Medium | In entry format |
| `DISTANCE_OFFSET` pattern | +5-10% | Medium | Combine length+distance |
| Shared window dedup | +10-20% parallel | Hard | For parallel chunks |

---

## Key Comparisons

### rapidgzip Source Analysis

**HuffmanCodingShortBitsMultiCached.hpp:**
```cpp
struct CacheEntry {
    bool needToReadDistanceBits : 1;  // Avoid branch
    uint8_t bitsToSkip : 6;            // Total bits
    uint8_t symbolCount : 2;           // 1 or 2 symbols
    uint32_t symbols : 18;             // Packed symbols
};
```

**We have this in `multi_symbol.rs`**, but it's never called!

### libdeflate Source Analysis

**decompress_template.h:358-434 (Fastloop):**
```c
saved_bitbuf = bitbuf;              // Save BEFORE consumption
bitbuf >>= (u8)entry;               // Consume
bitsleft -= entry;                   // Subtract full entry

// Decode up to 3 literals with CAN_CONSUME_AND_THEN_PRELOAD check
if (CAN_CONSUME_AND_THEN_PRELOAD(2*LITLEN_TABLEBITS + LENGTH_MAXBITS,
                                  OFFSET_TABLEBITS)) {
    lit = entry >> 16;
    entry = d->u.litlen_decode_table[bitbuf & litlen_tablemask];
    *out_next++ = lit;
    // ... (2 more literals)
}
```

**We have this in `libdeflate_decode.rs:608-672`**, but maybe not as optimized as libdeflate's version.

---

## Performance Regression History

| Attempt | Expected | Actual | Root Cause |
|---------|----------|--------|------------|
| DoubleLitCache per block | +15% | **-73%** | Build cost (4x) exceeds decode gain |
| `#[cold]` on errors | +5% | **-4%** | Function call overhead |
| Table-free fixed Huffman | +20% | **-325%** | Bit reversal kills perf |
| Unconditional refill | +5% | **-12%** | Conditional was faster |
| consume_first feature | +40% | **-20%** | Control flow overhead |

**Lesson: Micro-optimizations often backfire. LLVM already does well.**

---

## Why 69% Instead of 100%?

Likely causes of the 31% gap:

### 1. Register Allocation (10-15%)
- C compilers can hand-tune register usage
- Rust compiler may not allocate as optimally for tight loops
- libdeflate uses explicit register hints (`register` keyword in some paths)

### 2. Memory Access Patterns (5-10%)
- Cache line prefetching
- libdeflate may have better spatial locality
- Pointer aliasing assumptions (C's `restrict`)

### 3. CPU-Specific Optimizations (5-10%)
- libdeflate uses BMI2 instructions on x86_64 (`_bzhi_u64`, `_pext_u64`)
- We check for these but may not use them as effectively
- Branch prediction tuning (libdeflate has been profiled extensively)

### 4. Missed Micro-Optimizations (3-5%)
- `unlikely()` branch hints missing
- `EXTRACT_VARBITS8` pattern (cast to u8) missing
- Possible differences in loop unrolling
- Preload timing (we preload but maybe not at optimal point)

### 5. Measurement Noise (2-3%)
- Different compilation flags
- Cache state differences
- CPU frequency scaling

---

## Path to 100% (Parity)

### High Priority (Est: +10-20%)
1. **Add `unlikely()` hints** on EOB and subtable checks
2. **Add `EXTRACT_VARBITS8`** - cast to u8 before shift for extra bits
3. **Optimize preload timing** - ensure next entry loaded DURING write, not after
4. **Profile with perf** - find actual hotspots, not assumed ones

### Medium Priority (Est: +5-10%)
5. **Try `#[target_feature]` attributes** - force SIMD codegen
6. **Experiment with `#[inline(always)]` vs `#[inline(never)]`** - sometimes less inlining is faster
7. **Check assembly output** - compare to libdeflate's with `cargo asm`

### Low Priority (Est: +2-5%)
8. **Benchmark different refill strategies** - maybe conditional IS better for some CPUs
9. **Try different table sizes** - 10-bit vs 11-bit vs 12-bit
10. **Experiment with `#[repr(C)]` on entry structs** - might affect layout

---

## Path to 130%+ (Target)

**Reaching 130%+ requires going BEYOND what libdeflate does.**

### Novel Approach 1: Full SIMD Lanes (+50-100%)
- `vector_huffman.rs` has infrastructure for 8-lane parallel decode
- Use AVX-512 for 16 lanes on newer CPUs
- Challenge: handling matches breaks parallelism
- Solution: hybrid approach - SIMD for literal-heavy chunks, scalar for match-heavy

### Novel Approach 2: JIT Code Generation (+30-60%)
- Detect repeated Huffman tables in file (common in BGZF, pigz output)
- Generate native machine code for that specific table
- Eliminates table lookups entirely - just jump table
- Use cranelift or LLVM for codegen
- Amortize build cost over many blocks

### Novel Approach 3: Two-Pass with Table Caching (+20-40%)
- Pass 1: Scan file, fingerprint all dynamic tables
- Pass 2: Decode with cached tables, parallel per-block
- Only build each unique table once
- Works great for files with repeated structure (logs, databases)

### Novel Approach 4: Hardware Acceleration (+100-500%)
- FPGA implementation of Huffman FSM
- GPU tensor cores for parallel table lookup (insane but possible)
- Custom RISC-V instruction (`huffdec rd, rs1, rs2`)

---

## Recommended Next Steps

### Week 1: Low-Hanging Fruit
1. Add `likely()/unlikely()` branch hints
2. Add `EXTRACT_VARBITS8` u8 cast pattern
3. Profile with `perf` to find actual hotspots
4. Benchmark after each change

### Week 2-3: Integration
5. **Try integrating `MultiSymbolLUT`** - but ONLY for fixed blocks (skip build cost)
6. Optimize preload timing in fastloop
7. Compare assembly output to libdeflate

### Month 2: Novel Approaches
8. Implement JIT table caching for repeated blocks
9. Full SIMD lane integration (vector_huffman)
10. Two-pass with table fingerprinting

---

## Conclusion

**We're at 69% with most traditional optimizations already in place.**

The remaining 31% to parity is likely:
- Compiler/register allocation differences (hard to fix)
- Micro-optimizations (unlikely/EXTRACT_VARBITS8) (easy to add)
- Timing/measurement differences (unavoidable)

**Realistic target: 85-90% of libdeflate with remaining easy optimizations.**

**To reach 130%+, we MUST pursue novel approaches:**
- SIMD lane parallelism (infrastructure exists in vector_huffman.rs)
- JIT code generation
- Two-pass table caching

**OR accept that single-threaded parity is enough, and focus on parallel performance, where we already exceed libdeflate (148% with 8 threads).**

---

## Files to Investigate Next

1. **src/libdeflate_decode.rs:568-800** - Current best decoder, add branch hints
2. **src/vector_huffman.rs:615-656** - Multi-literal lookahead, integrate into main path
3. **src/multi_symbol.rs:51-293** - Multi-symbol table, integrate for fixed blocks
4. **rapidgzip/huffman/HuffmanCodingShortBitsMultiCached.hpp** - Study their CacheEntry pattern
5. **libdeflate/lib/decompress_template.h:350-500** - Study their fastloop carefully
