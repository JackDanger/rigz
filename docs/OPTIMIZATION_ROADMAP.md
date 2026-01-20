# Optimization Roadmap: Exceed All Tools

**Goal: Exceed libdeflate (1389 MB/s) and rapidgzip (720 MB/s single-threaded) on silesia**

## Current Status

| Metric | Current | libdeflate | Gap |
|--------|---------|------------|-----|
| Silesia throughput | **773 MB/s** | 1389 MB/s | 55.6% |
| Simple data | ~3000 MB/s | ~3500 MB/s | 86% |
| BGZF parallel (8 threads) | 3770 MB/s | N/A | ✓ |

## Gap Analysis: What We've Implemented ✓

| Optimization | Source | Status |
|--------------|--------|--------|
| Signed literal check `(entry as i32) < 0` | libdeflate | ✅ Done |
| Entry consumption `bitsleft -= entry` | libdeflate | ✅ Done |
| Unsafe unchecked table lookups | custom | ✅ Done |
| Unsafe pointer writes for literals | custom | ✅ Done |
| Unrolled 4-literal decode | libdeflate | ✅ Done |
| Unsafe 8-byte overlapping match copy | libdeflate | ✅ Done |
| Unaligned 8-byte refill loads | libdeflate | ✅ Done |
| JIT table caching (HashMap) | custom | ✅ Done |
| Static fixed table caching (OnceLock) | custom | ✅ Done |
| BMI2 `_bzhi_u64` intrinsic | libdeflate | ✅ Done |
| 11-bit litlen table | libdeflate | ✅ Done |
| 8-bit offset table | libdeflate | ✅ Done |
| Golden tests for correctness | custom | ✅ Done |

## Gap Analysis: What We Haven't Implemented Yet

### High Impact (>10% each)

| Optimization | Source | Expected Gain | Difficulty |
|--------------|--------|---------------|------------|
| **Double-literal cache** | rapidgzip | 15-25% | Hard |
| **Preload-before-write pattern** | libdeflate | 10-15% | Medium |
| **Inline assembly decode loop** | libdeflate | 10-20% | Very Hard |

### Medium Impact (5-10% each)

| Optimization | Source | Expected Gain | Difficulty |
|--------------|--------|---------------|------------|
| Compile-time bit budget checking | libdeflate | 5-10% | Medium |
| Distance preloading during length decode | libdeflate | 5-8% | Medium |
| PRELOAD_SLACK optimization | libdeflate | 3-5% | Easy |

### Low Impact (<5% each)

| Optimization | Source | Expected Gain | Difficulty |
|--------------|--------|---------------|------------|
| Cache line alignment for tables | general | 2-3% | Easy |
| Prefetch next block | general | 1-3% | Easy |

## Implementation Priority

### Phase 1: Double-Literal Cache (NEXT)
From rapidgzip's `HuffmanCodingDoubleLiteralCached.hpp`:
- Cache pairs of consecutive literals in a single lookup
- Key insight: most deflate streams have runs of literals
- Implementation: 16-bit cache size (64KB), stores symbol1+length in low entry, symbol2 in high entry
- Expected gain: **15-25%** → should reach ~900-950 MB/s

### Phase 2: Preload-Before-Write Pattern
From libdeflate's `decompress_template.h` lines 389-415:
```c
// libdeflate pattern:
lit = entry >> 16;                    // extract literal
entry = table[bitbuf & mask];         // PRELOAD NEXT before write
*out_next++ = lit;                    // write AFTER preload
```
- Key insight: hides memory latency by starting next lookup while writing
- Expected gain: **10-15%** → should reach ~1050-1100 MB/s

### Phase 3: Inline Assembly
From libdeflate's BMI2 path:
- Hand-tuned x86_64 assembly for the inner decode loop
- Use PEXT, BZHI instructions directly
- Expected gain: **10-20%** → should reach ~1200-1350 MB/s

### Phase 4: Combined Optimizations
Layer all optimizations together:
- Double-literal + preload + assembly
- Expected: **Exceed libdeflate** at 1400+ MB/s

## Progress Log

| Date | Version | Throughput | % of libdeflate | Key Changes |
|------|---------|------------|-----------------|-------------|
| Jan 2026 | v1 | 618 MB/s | 45% | Initial libdeflate-compatible decode |
| Jan 2026 | v2 | 773 MB/s | 55.6% | Unsafe optimizations (unchecked, pointers) |
| Jan 2026 | v3 | 778 MB/s | 55% | Double-literal cache built (not yet integrated) |

## Lessons Learned

### Preload-Before-Write Complexity
The libdeflate preload pattern (preload next entry before writing current) is complex to 
implement correctly in Rust due to control flow:
- When entry2/entry3 are non-literals, we need to fall through to length handling
- But saved_bitbuf must be preserved from the right point for extra bit extraction
- The attempt caused "Invalid distance" errors due to incorrect saved_bitbuf state

**Solution**: Keep the simpler unrolled loop that doesn't fall through, or restructure
the entire decode loop to match libdeflate's exact control flow.

### Double-Literal Cache Limitations
- 16-bit cache = 256KB, only beneficial for fixed Huffman blocks
- Dynamic blocks would need per-block cache building (expensive)
- 31% double-literal hit rate on fixed Huffman is promising but silesia is mostly dynamic

## Reference Implementations

- **libdeflate**: `libdeflate/lib/decompress_template.h`
- **rapidgzip double-literal**: `rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingDoubleLiteralCached.hpp`
- **rapidgzip multi-cached**: `rapidgzip/librapidarchive/src/rapidgzip/huffman/HuffmanCodingShortBitsMultiCached.hpp`

## Key Insights from Reference Code

### rapidgzip Double-Literal Cache
```cpp
// Cache stores pairs: [symbol1 | length_bits] and [symbol2]
m_doubleCodeCache[paddedCode * 2] = symbol | (length << LENGTH_SHIFT);
m_doubleCodeCache[paddedCode * 2 + 1] = symbol2;
```
- Uses 2 * minCodeLength + 1 as cache bit count (typically 13 bits for base64)
- Only caches literal pairs (symbol < 256), not length codes
- NONE_SYMBOL (0xFFFF) indicates single-symbol entry

### libdeflate Preload Pattern
```c
// Key: preload BEFORE write to hide memory latency
lit = entry >> 16;
entry = d->u.litlen_decode_table[bitbuf & litlen_tablemask];  // preload
saved_bitbuf = bitbuf;
bitbuf >>= (u8)entry;
bitsleft -= entry;
*out_next++ = lit;  // write AFTER preload
```
