# gzippy Optimization Roadmap

**Mission: Exceed the performance of ALL existing decompression tools.**

## Current Status (Jan 2026)

| Metric | gzippy | libdeflate | rapidgzip | Target |
|--------|--------|------------|-----------|--------|
| Single-thread silesia | ~630 MB/s | 1,200 MB/s | 720 MB/s (ISA-L) | **1,400+ MB/s** |
| BGZF parallel (8T) | 3,770 MB/s | N/A | N/A | **5,000+ MB/s** |
| Ratio vs libdeflate | 52% | 100% | 60% | **115%** |

## Competitive Intelligence

### libdeflate Key Optimizations (from source analysis)

| Optimization | Description | Our Status |
|--------------|-------------|------------|
| **64-bit bitbuffer** | Word-sized buffer, refill every ~56 bits | ✅ Implemented |
| **Branchless refill** | `bitsleft \|= MAX_BITSLEFT & ~7` single instruction | ⚠️ Partial |
| **Consume-first pattern** | Consume bits unconditionally, then branch | ❌ Blocked by table design |
| **Subtables (never BITS=0)** | Every entry is valid, enables consume-first | ❌ We use BITS=0 markers |
| **11-bit primary table** | 2KB table fits L1 cache | ⚠️ We use 12-bit (4KB) |
| **BMI2 on x86_64** | `_bzhi_u64()` for bit extraction | ❌ Not using |
| **5-word unconditional copy** | 40 bytes written unconditionally | ❌ Tested, SLOWER for us |
| **RLE special case (d=1)** | `memset()` for single-byte repeat | ✅ Implemented |
| **Word-at-a-time copy** | 8-byte chunks for d≥8 | ✅ Implemented |
| **Stride copy for d=2-7** | Overlapping word writes | ✅ Implemented (3.78x faster) |
| **bitsleft -= entry** | Subtract full u32, bits in low byte | ⚠️ Partial |
| **Entry preloading** | Load next entry before processing current | ✅ Implemented |
| **Fastloop/generic split** | Fastloop with margin, generic near edges | ✅ Implemented |
| **Inline macros** | C macros eliminate function call overhead | ❌ Rust uses functions |

### rapidgzip Key Optimizations (from source analysis)

| Optimization | Description | Our Status |
|--------------|-------------|------------|
| **10-11 bit LUT** | Optimal for L1 cache (their benchmarks) | ⚠️ We use 12-bit |
| **ISA-L integration** | Uses Intel's hand-optimized assembly | ❌ Not integrated |
| **Marker-based parallel** | `uint16_t` output, markers for back-refs | ✅ Implemented |
| **Speculative decode** | Decode without knowing window, resolve later | ✅ Implemented |
| **Multi-cached decoder** | Cache multiple symbols per lookup | ❌ Not implemented |
| **Double-literal cache** | Cache two consecutive literals | ❌ Not implemented |
| **Window memcpy optimization** | For large uncompressed blocks | ✅ Implemented |
| **Static Huffman table cache** | Avoid rebuilding fixed Huffman | ✅ Implemented |
| **Thread pool parallelism** | Lock-free work distribution | ✅ Implemented |

## Gap Analysis: Why We're at 52%

### What We've Ruled Out as Bottlenecks

| Component | Test Result | Conclusion |
|-----------|-------------|------------|
| Table size (10/11/12 bit) | <2% difference | NOT bottleneck |
| Copy function performance | All paths 4000+ MB/s | NOT bottleneck |
| Table lookup speed | 3,250 M lookups/s | NOT bottleneck |

### What IS the Bottleneck

| Issue | Impact | Fix Difficulty |
|-------|--------|----------------|
| **Table design (BITS=0 entries)** | 12% (blocks consume-first) | HIGH |
| **No subtables** | Can't use libdeflate's pattern | HIGH |
| **Function call overhead** | 5-10% (Rust vs C inline) | MEDIUM |
| **Branch prediction** | 5-10% (nested conditionals) | MEDIUM |
| **No BMI2 intrinsics** | 5% on supported CPUs | LOW |

## Roadmap to Exceed All Tools

### Phase 1: Foundation (Target: 70% of libdeflate)

#### 1.1 Subtable Implementation
**Goal**: Eliminate BITS=0 entries, enable consume-first pattern

Currently our table has `BITS=0` for codes that don't fit in 12 bits. libdeflate uses
actual subtables instead, ensuring every primary entry is valid.

**Implementation**:
```rust
// Current: BITS=0 signals "use fallback"
if entry & BITS_MASK == 0 { slow_path() }

// Target: Subtable pointer in entry
if entry & SUBTABLE_FLAG {
    let subtable_idx = (entry >> 16) as usize;
    let extra_bits = (entry >> 8) & 0x3F;
    entry = subtable[subtable_idx + (bits & ((1 << extra_bits) - 1))];
}
```

**Micro-benchmark**: `bench_subtable_vs_fallback`

#### 1.2 Consume-First Pattern
**Goal**: 12% speedup in decode loop

Once subtables are implemented, restructure decode loop:
```rust
// Before (check-first):
if is_literal(entry) { consume(entry); process_literal(); }
else { consume(entry); process_match(); }

// After (consume-first):
consume(entry);  // Unconditional
if is_literal(entry) { process_literal(); }
else { process_match(); }
```

**Micro-benchmark**: `bench_branch_pattern` (already shows 12% potential)

### Phase 2: Bit Operations (Target: 80% of libdeflate)

#### 2.1 Branchless Refill Optimization
**Goal**: Single-instruction bitsleft update

```rust
// Current: Multiple operations
self.bits = self.bits.saturating_sub(n);
self.refill_if_needed();

// Target: libdeflate pattern
self.bits |= (BITBUF_NBITS - 1) & !7;  // Single OR instruction
```

**Micro-benchmark**: `bench_refill_patterns`

#### 2.2 Entry Subtraction Optimization
**Goal**: Subtract full entry instead of masked bits

```rust
// Current:
bits.consume((entry & BITS_MASK) as u32);

// Target:
bits.consume_entry_raw(entry);  // Bits in low byte, subtract full u32
```

**Micro-benchmark**: `bench_entry_subtraction`

### Phase 3: Architecture-Specific (Target: 90% of libdeflate)

#### 3.1 BMI2 Intrinsics (x86_64)
**Goal**: 5% speedup on modern Intel/AMD

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
fn extract_bits(word: u64, count: u32) -> u64 {
    unsafe { std::arch::x86_64::_bzhi_u64(word, count) }
}
```

**Micro-benchmark**: `bench_bit_extraction` (already implemented)

#### 3.2 NEON Optimizations (aarch64)
**Goal**: 5% speedup on ARM

- Use NEON for bulk memory operations
- Optimize copy paths with vector instructions

**Micro-benchmark**: `bench_simd_copy`

### Phase 4: Advanced Optimizations (Target: 100%+ of libdeflate)

#### 4.1 Multi-Symbol Decode
**Goal**: Decode 2-3 symbols per table lookup

rapidgzip's `HuffmanCodingShortBitsMultiCached` achieves this. Entry format:
```
Entry = [Symbol1:8][Bits1:4][Symbol2:8][Bits2:4][Flags:8]
```

**Micro-benchmark**: `bench_multi_symbol_decode`

#### 4.2 JIT Table Cache
**Goal**: Avoid rebuilding identical Huffman tables

Fingerprint code lengths, cache built tables:
```rust
fn get_or_build_table(code_lengths: &[u8]) -> &PackedLUT {
    let fingerprint = hash(code_lengths);
    TABLE_CACHE.get_or_insert(fingerprint, || build_table(code_lengths))
}
```

**Micro-benchmark**: `bench_table_build_vs_cache`

#### 4.3 Speculative Decode with Work Stealing
**Goal**: Better load balancing for parallel decode

- Divide input into chunks
- Workers speculatively decode
- Work stealing for uneven chunk sizes

### Phase 5: Exceed Everything (Target: 115%+ of libdeflate)

#### 5.1 Hybrid Parallel Strategy
**Goal**: Optimal parallelism for all file types

| File Type | Strategy | Expected Speed |
|-----------|----------|----------------|
| BGZF | Parallel blocks | 5,000+ MB/s |
| Multi-member | Parallel members | 4,000+ MB/s |
| Single-member | Two-pass parallel | 2,500+ MB/s |
| Small files | Fast sequential | 1,400+ MB/s |

#### 5.2 Profile-Guided Optimization
**Goal**: Let real-world data guide optimizations

- Enable PGO in release builds
- Optimize hot paths based on actual usage

#### 5.3 Assembly Hot Paths (Optional)
**Goal**: Match or beat ISA-L on critical paths

If Rust compiler doesn't optimize adequately, hand-write assembly for:
- Inner decode loop
- Bit buffer operations
- Match copy

## Implementation Priority

| Phase | Effort | Expected Gain | Priority |
|-------|--------|---------------|----------|
| 1.1 Subtables | HIGH | +10-15% | **P0** |
| 1.2 Consume-first | MEDIUM | +12% | **P0** |
| 2.1 Branchless refill | LOW | +3-5% | P1 |
| 2.2 Entry subtraction | LOW | +2-3% | P1 |
| 3.1 BMI2 | LOW | +5% | P1 |
| 4.1 Multi-symbol | HIGH | +15-20% | P2 |
| 4.2 JIT cache | MEDIUM | +5-10% | P2 |
| 5.1 Hybrid parallel | MEDIUM | N/A (already fast) | P2 |

## Success Metrics

### Milestone 1: Match libdeflate (100%)
- Single-thread silesia: 1,200 MB/s
- All existing tests pass
- No regression in BGZF parallel performance

### Milestone 2: Exceed libdeflate (115%)
- Single-thread silesia: 1,400+ MB/s
- Beat rapidgzip+ISA-L on all benchmarks
- Become the fastest pure-Rust decompressor

### Milestone 3: Best in Class (130%+)
- Single-thread silesia: 1,600+ MB/s
- BGZF parallel: 5,000+ MB/s
- Recognized as fastest gzip decompressor

## Testing Strategy

Every optimization MUST have:
1. **Micro-benchmark** - Isolate and measure the specific improvement
2. **Correctness test** - Verify output matches reference implementation
3. **Integration test** - Ensure no regression in overall performance
4. **Cross-platform test** - Works on x86_64, aarch64, and other targets

## Reference Implementations

- **libdeflate**: `./libdeflate/lib/decompress_template.h`
- **rapidgzip**: `./rapidgzip/librapidarchive/src/rapidgzip/gzip/deflate.hpp`
- **ISA-L**: `./isa-l/igzip/`
- **zlib-ng**: Industry-standard reference

---

*Last updated: January 2026*
*Goal: Exceed ALL competition in performance*
