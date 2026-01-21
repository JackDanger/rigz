# Optimization Status Summary

**Current Performance: 976 MB/s (69.0% of libdeflate's 1414 MB/s)**
**Target: 1840+ MB/s (130%+ of libdeflate)**

---

## What We Have Implemented ✅

### From rapidgzip:
- [x] **Multi-literal lookahead** (`vector_huffman.rs`) - decodes up to 4 literals per lookup
- [x] **DoubleLitCache** (`double_literal.rs`) - pairs of literals in single lookup
- [x] **Multi-symbol table** (`multi_symbol.rs`) - 1-2 symbols per lookup (NOT integrated)
- [x] **Marker-based parallel decode** - for BGZF/multi-member files
- [x] **SIMD infrastructure** - AVX2/NEON support in vector_huffman.rs

### From libdeflate:
- [x] **Two-level Huffman tables** - 11-bit L1 cache + subtables
- [x] **Bit buffer with refill** - lazy refill pattern
- [x] **Fastloop architecture** - separate fastloop/generic paths
- [x] **Branchless bit refill** - `REFILL_BITS_IN_FASTLOOP` equivalent
- [x] **Subtables inline** - subtables follow main table in memory
- [x] **Copy with overlapping** - 8-byte chunk copies for matches
- [x] **Distance=1 memset** - special case for RLE patterns
- [x] **Static fixed tables** - cached with OnceLock

---

## What's NOT Integrated ❌

### Implemented but not in hot path:
- [ ] **MultiSymbolLUT** (`multi_symbol.rs`) - exists but never called
- [ ] **DoubleLitCache for dynamic** - tried, but **REGRESSED 73%** due to build cost per block

### Missing from libdeflate:
- [ ] **saved_bitbuf pattern** - save bitbuf BEFORE consumption for extracting extra bits
- [ ] **Multi-literal in fastloop** - decode 2-3 fast literals when bits available
- [ ] **CAN_CONSUME_AND_THEN_PRELOAD** - compile-time bit budget checking
- [ ] **Preload during write** - start next lookup before writing current literal
- [ ] **EXTRACT_VARBITS8 optimization** - cast to u8 before shift
- [ ] **unlikely() branch hints** - `core::intrinsics::likely/unlikely`
- [ ] **Entry preload** - preload next entry during current operation

### Missing from rapidgzip:
- [ ] **needToReadDistanceBits flag** - avoid branch for literal vs length check
- [ ] **Pre-computed length in entry** - length base + extra bits combined
- [ ] **DISTANCE_OFFSET pattern** - combine length+distance in single symbol space
- [ ] **Shared window deduplication** - for parallel chunks

---

## Performance Regressions Found ⚠️

| Optimization | Expected | Actual | Reason |
|--------------|----------|--------|--------|
| DoubleLitCache per block | +15% | **-73%** | Build cost exceeds decode gain |
| `#[cold]` on error paths | +5% | **-4%** | Function call overhead |
| Table-free fixed Huffman | +20% | **-325%** | Bit reversal kills performance |
| Unconditional refill | +5% | **-12%** | Conditional was better |

**Key Lesson: Simple micro-optimizations often make things WORSE. LLVM already optimizes well.**

---

## Critical Missing Pieces for 130%+ Target

### Tier 1: Proven Optimizations (Est: +10-20%)
1. **saved_bitbuf + EXTRACT_VARBITS8** - libdeflate's pattern for extra bits extraction
2. **Multi-literal fastloop** - decode 2-3 literals when bit budget allows
3. **Preload next entry** - during current literal write (reduce latency)
4. **Branch hints** - `likely()/unlikely()` on literal check

### Tier 2: Novel Approaches (Est: +20-40%)
5. **JIT decode table** - generate native code for specific Huffman table
6. **SIMD parallel lanes** - decode 8 lanes simultaneously (vector_huffman has infrastructure)
7. **Precomputed multi-symbol sequences** - 16-22 bit lookup → 2-4 symbols

### Tier 3: Radical (Est: +50-100%)
8. **Two-pass with table reuse** - scan file, identify repeated tables, cache them
9. **Hardware acceleration** - use GPU tensor cores or FPGA

---

## Next Steps (Priority Order)

### Immediate (Week 1):
1. ✅ **Audit what exists** - this document
2. **Integrate saved_bitbuf pattern** - into libdeflate_decode.rs
3. **Add multi-literal fastloop** - decode 2-3 literals when bits available
4. **Add preload during write** - reduce latency

### Short-term (Week 2-3):
5. **Integrate MultiSymbolLUT** - but ONLY for fixed blocks (avoid build cost)
6. **Add branch hints** - `likely(literal)` in fastloop
7. **Benchmark after each change** - prevent regressions

### Medium-term (Month 2):
8. **JIT decode table** - for repeated dynamic blocks
9. **Full vector SIMD** - integrate vector_huffman lanes into main path
10. **Profile-guided optimization** - use perf to find actual hotspots

---

## Comparison Matrix

| Feature | libdeflate | rapidgzip | gzippy | Gap |
|---------|-----------|-----------|---------|-----|
| Multi-literal decode | ✓ (2-3 lits) | ✓ (CacheEntry) | ✓ (4 lits) | ≈ Same |
| saved_bitbuf pattern | ✓ | ✗ | ✗ | **Missing** |
| Preload next entry | ✓ | ✗ | Partial | **Incomplete** |
| CAN_CONSUME_PRELOAD | ✓ | ✗ | ✗ | **Missing** |
| Branch hints | ✓ | ✗ | ✗ | **Missing** |
| SIMD parallelism | ✗ | ✗ | ✓ (AVX2/NEON) | **Advantage!** |
| JIT tables | ✗ | ✗ | Planned | **Novel** |
| Marker-based parallel | ✗ | ✓ | ✓ | ✓ Same |

---

## Why We're at 69% Instead of 130%

The **30% gap to parity** (69% → 100%) is due to:
1. Missing saved_bitbuf pattern (~5-10%)
2. Missing multi-literal fastloop (~8-15%)
3. Missing preload optimization (~3-5%)
4. Missing branch hints (~2-5%)
5. Register allocation differences (~5-10%)

The **additional 30% for 130% target** requires:
- Novel optimizations (JIT, SIMD lanes, hardware acceleration)
- OR accepting that single-threaded parity is enough, and focus on parallel (already at 148%!)

---

## Conclusion

**We have most of the infrastructure, but it's not integrated.**

- `multi_symbol.rs` exists but is never called
- `vector_huffman.rs` has multi-literal but only used in one path
- `double_literal.rs` was tried but regressed performance

**The path forward:**
1. Add libdeflate's saved_bitbuf pattern
2. Add 2-3 literal fastloop with bit budget check
3. Add entry preload during write
4. If those don't reach 100%, then pursue novel approaches (JIT, SIMD lanes)

**The 130%+ target likely requires going beyond traditional optimizations into JIT or SIMD territory.**
