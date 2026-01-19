# Remaining Optimizations to Surpass rapidgzip

**Date**: January 2026  
**Last Updated**: After P1 multi-member fix

---

## Current Status

| Metric | gzippy | rapidgzip | gzip | Status |
|--------|--------|-----------|------|--------|
| Single-member (202MB) | ~0.18s | ~0.07s | ~0.20s | ⚠️ 2.5x slower than rapidgzip |
| BGZF parallel | 3400+ MB/s | 3168 MB/s | N/A | ✅ **Faster** |
| Multi-member | Parallel ✅ | Parallel | Sequential | ✅ Fixed |
| Arbitrary gzip parallel | Not integrated | ✅ Core strength | N/A | ❌ **Key gap** |

**The #1 gap**: rapidgzip achieves 2-3x speedup on arbitrary single-member gzip files through speculative parallel decompression. We have the code (`marker_decode.rs`) but it's disabled due to reliability issues.

---

## Critical Path to Beat rapidgzip

### 🔴 Gap 1: Speculative Parallel Decompression (CRITICAL)

**This is the key to matching rapidgzip on arbitrary gzip files.**

**What rapidgzip does**:
1. Partition input at fixed intervals (e.g., 4MB chunks)
2. Speculatively start decoding at each chunk boundary
3. Use markers (uint16_t output) for unresolved back-references
4. Once chunk 0 finishes, propagate its 32KB window to chunk 1
5. Replace markers in chunk 1 with resolved values
6. Repeat for all chunks

**What we have**:
- `marker_decode.rs`: Complete marker-based decoder ✅
- `MarkerDecoder`: Decodes deflate with markers for unknown back-refs ✅
- `try_decode_chunk()`: Tries decoding at chunk boundaries ✅
- `decompress_parallel()`: Full parallel pipeline ✅

**What's broken**:
- Disabled in `ultra_decompress.rs` because it was causing infinite loops
- `try_decode_chunk()` success rate too low on real data
- Window propagation may have bugs

**Expected Gain**: 2-3x on single-member gzip files

**Effort**: 1-2 weeks to debug and stabilize

**Files**:
- `src/marker_decode.rs` - Core algorithm (implemented)
- `src/ultra_decompress.rs` - Integration point (disabled)

---

### 🟡 Gap 2: Block Boundary Finding

**Alternative to speculative decode**: Find actual deflate block boundaries.

**What rapidgzip also does**:
- Scan for potential block start signatures
- Validate by attempting decode
- Only start chunks at verified block boundaries

**What we have**:
- `block_finder.rs` was deleted (low success rate <5%)

**Expected Gain**: Similar to Gap 1, but more reliable

**Effort**: 1 week

---

### 🟡 Gap 3: Dynamic Block Multi-Symbol Decode

**Current**: Single symbol per lookup in dynamic Huffman blocks  
**Target**: 2-3 symbols per lookup when codes are short

**What's missing**:
```rust
// Build multi-symbol table at runtime for dynamic blocks
// when max code length <= 12 bits
fn build_dynamic_multi_sym_table(lengths: &[u8]) -> MultiSymTable {
    // Pack 2-3 symbols when total bits <= 12
}
```

**Expected Gain**: 15-25% on dynamic-heavy files

**Effort**: 3 days

**Files**: `src/ultra_fast_inflate.rs`, `src/two_level_table.rs`

---

### 🟡 Gap 4: Table Construction Optimization

**Current**: Heap-allocate Vec, zero-fill 1024+ entries  
**Target**: Stack-allocate L1, lazy L2

```rust
// Current
pub struct TwoLevelTable {
    l1: [u16; 1024],  // ✅ Stack
    l2: Vec<u16>,     // ❌ Heap alloc per table
}

// Optimal  
pub struct TwoLevelTable {
    l1: [u16; 1024],
    l2: [u16; 512],   // Stack, fixed max size
    l2_len: usize,
}
```

**Expected Gain**: 3-5% on files with many dynamic blocks

**Effort**: 1 day

---

### 🟢 Gap 5: Prefetching

**Current**: No explicit prefetching  
**Target**: Prefetch input and output during decode

```rust
// In decode loop
std::arch::x86_64::_mm_prefetch(input.add(128), _MM_HINT_T0);

// In LZ77 copy
std::arch::x86_64::_mm_prefetch(output.add(64), _MM_HINT_T0);
```

**Expected Gain**: 2-5%

**Effort**: 1 day

---

### 🟢 Gap 6: AVX-512 Support

**Current**: AVX2 (32-byte), NEON (16-byte)  
**Target**: AVX-512 (64-byte) for modern Intel/AMD

```rust
#[cfg(target_feature = "avx512f")]
unsafe fn copy_64_avx512(src: *const u8, dst: *mut u8) {
    let data = _mm512_loadu_si512(src as *const __m512i);
    _mm512_storeu_si512(dst as *mut __m512i, data);
}
```

**Expected Gain**: 5-10% on AVX-512 CPUs

**Effort**: 2 days

---

### 🟢 Gap 7: Branch Prediction Hints

**Current**: No hints  
**Target**: likely/unlikely on hot paths

```rust
if std::intrinsics::likely(symbol < 256) {
    // literal - most common
    output.push(symbol as u8);
} else if std::intrinsics::unlikely(symbol == 256) {
    // end of block - rare
    break;
}
```

**Expected Gain**: 1-3%

**Effort**: 0.5 days

---

### 🟢 Gap 8: Huge Pages

**Current**: Standard 4KB pages  
**Target**: 2MB huge pages for large files

```rust
let mmap = MmapOptions::new()
    .huge(Some(2 * 1024 * 1024))
    .populate()
    .map(&file)?;
```

**Expected Gain**: 5-10% on very large files

**Effort**: 1 day

---

### 🟢 Gap 9: Thread Affinity

**Current**: OS-scheduled threads  
**Target**: Pin threads to specific cores

```rust
core_affinity::set_for_current(CoreId { id: thread_id });
```

**Expected Gain**: 2-5% on NUMA systems

**Effort**: 0.5 days

---

### 🟢 Gap 10: Index Caching (rapidgzip feature)

**Current**: No persistent index  
**Target**: Save block index to `.gzidx` file

rapidgzip can create an index file that allows:
- Near-instant random access to any position
- Fast re-decompression of the same file

**Expected Gain**: 10-100x for repeated access

**Effort**: 2 days

---

### 🟢 Gap 11: Streaming Ultra-Fast Inflate

**Current**: stdin uses flate2 (slower)  
**Target**: Use ultra_fast_inflate for streaming

**Expected Gain**: 2x for piped input

**Effort**: 2 days

---

## Priority Matrix

| Priority | Gap | Expected Gain | Effort | Impact |
|----------|-----|---------------|--------|--------|
| 🔴 **P0** | Speculative parallel | 2-3x | 2 weeks | **CRITICAL** |
| 🟡 **P1** | Dynamic multi-symbol | 15-25% | 3 days | High |
| 🟡 **P1** | Table construction | 3-5% | 1 day | Medium |
| 🟢 **P2** | Prefetching | 2-5% | 1 day | Medium |
| 🟢 **P2** | AVX-512 | 5-10% | 2 days | Platform-specific |
| 🟢 **P2** | Branch hints | 1-3% | 0.5 days | Low |
| 🟢 **P3** | Huge pages | 5-10% | 1 day | Large files only |
| 🟢 **P3** | Thread affinity | 2-5% | 0.5 days | NUMA only |
| 🟢 **P3** | Index caching | 10-100x | 2 days | Repeated access |
| 🟢 **P3** | Streaming inflate | 2x | 2 days | Pipes only |

---

## What We've Already Completed ✅

| Optimization | Status | Location |
|--------------|--------|----------|
| Two-level Huffman tables | ✅ | `two_level_table.rs` |
| SIMD pattern expansion (dist 1-7) | ✅ | `simd_copy.rs` |
| SIMD overlapping copy (dist 8-31) | ✅ | `simd_copy.rs` |
| AVX2/NEON LZ77 copy | ✅ | `simd_copy.rs` |
| 64-bit bit buffer | ✅ | `two_level_table.rs` |
| BGZF parallel decompression | ✅ | `ultra_inflate.rs` |
| Multi-member parallel | ✅ | `ultra_decompress.rs` |
| Parallel compression L1-L9 | ✅ | `parallel_compress.rs` |
| libdeflate integration | ✅ | `isal.rs` |
| Fixed Huffman multi-symbol | ✅ | `turbo_inflate.rs` |
| Cache-aligned buffers | ✅ | `decompression.rs` |
| Thread-local decompressor | ✅ | `decompression.rs` |
| Conservative multi-member detection | ✅ | `decompression.rs` |

---

## The Path to Victory

### To match rapidgzip:
1. **Fix speculative parallel decompression** - This alone would get us to parity

### To beat rapidgzip:
1. Fix speculative parallel (2-3x)
2. Add dynamic multi-symbol (15-25%)
3. Add prefetching (2-5%)
4. Total: **3-4x improvement** possible

### Current bottleneck:
On single-member gzip files, we use sequential libdeflate (0.18s). rapidgzip uses parallel speculative decode (0.07s). The 2.5x gap is entirely due to lack of parallelism.

---

## Quick Wins (< 1 day each)

1. **Table construction**: Stack-allocate L2 table
2. **Branch hints**: Add likely/unlikely macros
3. **Thread affinity**: Pin worker threads
4. **Prefetching**: Add prefetch intrinsics

These combined could give 5-10% improvement with minimal effort.
