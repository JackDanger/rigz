# Gap Analysis: gzippy vs State-of-the-Art Submodules

**Generated:** Jan 2026  
**Purpose:** Identify optimization opportunities by comparing our implementation to libdeflate, ISA-L, and rapidgzip.

---

## Executive Summary

| Category | Our Implementation | State-of-Art | Gap | Priority |
|----------|-------------------|--------------|-----|----------|
| **Huffman Decode Loop** | 91% of libdeflate | libdeflate | 9% | HIGH |
| **Multi-Symbol Tables** | Implemented, NOT integrated | ISA-L: 1-3 symbols/lookup | **LARGE** | HIGH |
| **Runtime CPU Detection** | BMI2 done, AVX2 missing | Both have full detection | 5% | MEDIUM |
| **Parallel Single-Member** | marker_turbo working | rapidgzip | ~same | LOW |
| **Match Copying** | SIMD for dist>=32 | ISA-L: full SIMD | 2% | LOW |

---

## Detailed Comparison

### 1. Huffman Decode Table Format

#### libdeflate (decompress_template.h)
```c
// 32-bit entry format:
// Bit 31:     HUFFDEC_LITERAL (1 = literal, high bit for fast test)
// Bit 23-16:  literal value or length base
// Bit 15:     HUFFDEC_EXCEPTIONAL (subtable or EOB)
// Bit 14:     HUFFDEC_SUBTABLE_POINTER
// Bit 13:     HUFFDEC_END_OF_BLOCK
// Bit 11-8:   remaining codeword length
// Bit 4-0:    total bits to consume (codeword + extra)
```

**Key trick:** `bitsleft -= entry` subtracts the FULL entry, not masked bits. High bits are garbage but refill handles it.

#### Our Implementation (libdeflate_entry.rs)
```rust
// LitLenEntry: 32-bit, same format as libdeflate
// ✅ HUFFDEC_LITERAL in bit 31
// ✅ bitsleft -= entry full subtract
// ✅ saved_bitbuf pattern for extra bits
```

**Gap:** None - we match libdeflate exactly.

---

### 2. Multi-Symbol Decode (ISA-L's Key Innovation)

#### ISA-L (igzip_inflate.c)
```c
// ISA-L encodes 1, 2, or 3 symbols per lookup entry:
#define LARGE_SYM_COUNT_OFFSET 26
#define TRIPLE_SYM_FLAG  0
#define DOUBLE_SYM_FLAG  1
#define SINGLE_SYM_FLAG  2

// Entry can contain:
// - 1 symbol (any type)
// - 2 symbols (both must be literals < 256)
// - 3 symbols (all must be literals < 256)

// Dynamic mode selection based on input size:
if (state->avail_in <= SINGLE_SYM_THRESH) {
    multisym = SINGLE_SYM_FLAG;  // Small input: 1 symbol/lookup
} else if (state->avail_in <= DOUBLE_SYM_THRESH) {
    multisym = DOUBLE_SYM_FLAG;  // Medium: 2 symbols/lookup
} else {
    multisym = TRIPLE_SYM_FLAG;  // Large: 3 symbols/lookup
}
```

#### Our Implementation
```rust
// src/multi_symbol.rs - EXISTS but NOT INTEGRATED
// src/double_literal.rs - EXISTS but NOT INTEGRATED
// Both were tried and FAILED due to "build cost > decode gain"
```

**Gap:** LARGE - ISA-L's approach is different:
1. They build multi-symbol tables ONCE per Huffman table
2. The table encodes ALL possible pairs/triples statically
3. No per-lookup overhead

**Why our attempts failed:**
- We built DoubleLitCache per-block (too slow)
- ISA-L builds it during table construction (amortized)

**Recommendation:** Revisit multi-symbol with ISA-L's approach:
- Build multi-symbol entries during `LitLenTable::build()`
- Use existing table, don't build separate cache

---

### 3. Runtime CPU Feature Detection

#### libdeflate (x86/cpu_features.c)
```c
// Detects at runtime: SSE2, BMI2, AVX2, AVX512
static const struct cpu_feature x86_cpu_features[] = {
    {X86_CPU_FEATURE_SSE2, "sse2", 3, 26, 0, 0},
    {X86_CPU_FEATURE_BMI2, "bmi2", 7, 0, 8, 0},
    {X86_CPU_FEATURE_AVX2, "avx2", 7, 0, 5, 0},
    // ...
};
```

#### Our Implementation
```rust
// src/consume_first_decode.rs
fn has_bmi2() -> bool {
    // ✅ Runtime BMI2 detection with caching
    is_x86_feature_detected!("bmi2")
}

// AVX2 for match copy: MISSING runtime detection
// Currently uses compile-time #[cfg(target_feature = "avx2")]
```

**Gap:** AVX2 match copying is compile-time only

**Recommendation:** Add runtime AVX2 detection for `copy_match_fast`:
```rust
#[target_feature(enable = "avx2")]
unsafe fn copy_match_avx2(...) { ... }

fn copy_match_fast(...) {
    if has_avx2() {
        unsafe { copy_match_avx2(...) }
    } else {
        copy_match_scalar(...)
    }
}
```

---

### 4. Bit Buffer Refill Strategy

#### libdeflate
```c
// CAN_CONSUME_AND_THEN_PRELOAD - compile-time check
#define CAN_CONSUME_AND_THEN_PRELOAD(consume, preload) \
    (CONSUMABLE_NBITS >= (consume) && \
     FASTLOOP_PRELOADABLE_NBITS >= (consume) + (preload))

// Used to batch multiple literals without intermediate refills
if (CAN_CONSUME_AND_THEN_PRELOAD(2 * LITLEN_TABLEBITS + LENGTH_MAXBITS,
                                  OFFSET_TABLEBITS)) {
    // Decode 2 extra fast literals + length + offset preload
}
```

#### Our Implementation
```rust
// We do conditional refills, not compile-time batching
if (bitsleft as u8) < 48 {
    refill_branchless_fast!();
}
```

**Gap:** We use runtime checks; libdeflate uses compile-time guarantees.

**Analysis:** This is a design tradeoff. Our runtime checks work but add 1 branch per check. libdeflate's compile-time approach is slightly faster but less flexible.

**Recommendation:** LOW priority - the difference is ~1-2%.

---

### 5. Parallel Decompression

#### rapidgzip (HuffmanCodingShortBitsMultiCached.hpp)
```cpp
struct CacheEntry {
    bool needToReadDistanceBits : 1;  // If true, need distance after length
    uint8_t bitsToSkip : 6;           // Total bits consumed
    uint8_t symbolCount : 2;          // 1, 2, or 3 symbols
    uint32_t symbols : 18;            // Packed symbol values
};
```

#### Our Implementation
```rust
// marker_turbo.rs - 2129 MB/s marker-based decoder
// hyper_parallel.rs - parallel single-member using marker_turbo
// ✅ Working and integrated via hyperion.rs
```

**Gap:** None for architecture - we match rapidgzip's approach.

**Remaining issue:** hyper_parallel fallback rate could be optimized.

---

### 6. Match Copy Optimization

#### ISA-L (igzip_inflate.c)
```c
// Full SIMD for all distances with careful overlap handling
// Uses AVX2 for 32-byte copies, handles overlap via stride
```

#### libdeflate (decompress_template.h)
```c
// Unconditional 5-word copy first, then loop
store_word_unaligned(load_word_unaligned(src), dst);
src += WORDBYTES; dst += WORDBYTES;
// ... repeat 5 times
// Then loop for remainder
```

#### Our Implementation
```rust
// copy_match_fast in consume_first_decode.rs
// ✅ NEON for ARM (32-byte copies)
// ✅ Scalar fallback for x86
// ❌ AVX2 disabled (requires runtime detection)
```

**Gap:** x86 uses scalar, not AVX2.

---

## Action Items (Priority Order)

### HIGHEST Priority: INTEGRATE EXISTING MULTI-SYMBOL CODE

**CRITICAL FINDING:** We already have `src/multi_symbol.rs` with:
- `MultiSymbolLUT::build()` that creates double-literal entries
- `upgrade_to_double_literals()` that packs literal pairs during table build
- Working tests proving correctness

**BUT IT IS NOT INTEGRATED INTO THE HOT PATH!**

The integration path:
1. Replace `LitLenTable` with `MultiSymbolLUT` in `consume_first_decode.rs`
2. Add multi-symbol decode path to `decode_huffman_libdeflate_style()`
3. Benchmark on SILESIA

**Expected improvement:** 10-20% on literal-heavy data (SILESIA is 60%+ literals)

### HIGH Priority

1. **Runtime AVX2 Detection for Match Copy**
   - Add `has_avx2()` with caching
   - Add `#[target_feature(enable = "avx2")]` functions
   - ~5% improvement expected on x86_64

2. **Investigate Why multi_symbol.rs Was Never Integrated**
   - Check git history for previous integration attempts
   - May have been blocked by build cost (now amortized into table build)

### MEDIUM Priority

3. **Profile the 9% SILESIA Gap**
   - Use `perf` or Instruments to find the bottleneck
   - Compare instruction counts with libdeflate
   - Identify specific hot spots

4. **Reduce hyper_parallel Fallback Rate**
   - Debug why some chunks fail
   - Add better error recovery

### LOW Priority

5. **Compile-Time Bit Budget Checks**
   - Implement CAN_CONSUME_AND_THEN_PRELOAD pattern
   - May provide 1-2% improvement

6. **Triple-Symbol Encoding**
   - ISA-L does 3 symbols per lookup for large inputs
   - Our double-literal already exists, extend to triple

---

## Files to Study

| Submodule File | Our Equivalent | Key Insight |
|---------------|----------------|-------------|
| `libdeflate/lib/decompress_template.h:350-500` | `src/consume_first_decode.rs:649-1200` | Hot path comparison |
| `isa-l/igzip/igzip_inflate.c:400-550` | `src/multi_symbol.rs` | Multi-symbol table building |
| `rapidgzip/src/huffman/HuffmanCodingShortBitsMultiCached.hpp` | `src/double_literal.rs` | Cache entry format |
| `libdeflate/lib/x86/cpu_features.c` | `src/consume_first_decode.rs:163-210` | Runtime detection |

---

## Conclusion

We are at **91-101% of state-of-the-art** in pure Rust. The main gaps are:

1. **Multi-symbol tables not integrated** - ISA-L's approach of building during table construction could help
2. **AVX2 match copy missing on x86** - Easy win with runtime detection
3. **9% SILESIA gap unexplained** - Needs profiling

The good news: Our architecture is sound. We just need targeted optimizations.
