# Optimization Roadmap: Beating libdeflate

**Current Status**: 667 MB/s (54% of libdeflate's 1239 MB/s)
**Target**: 1400+ MB/s (115% of libdeflate)

## Key Discoveries (Jan 2026)

From profiling and analysis:

1. **Slow path is rarely hit** - Our LUT pre-computes length+distance for most matches
2. **Entry format limitation** - Current format doesn't store codeword length separately from total bits, limiting saved_bitbuf optimization
3. **Unconditional copy is risky** - Requires guaranteed buffer margin; broke tests when attempted
4. **Profile data shows**:
   - 2792 dynamic blocks, 10 stored, 0 fixed (silesia)
   - 3.13x compression ratio
   - Most time in fastloop, not slow path
5. **DecodeTrace infrastructure exists** but isn't connected to decode loop

### Optimizations Attempted

| Change | Result | Learning |
|--------|--------|----------|
| Literal loop preloading | +0.7% (53.8→54.5%) | Small win, keep it |
| Pattern-based small distance copy | **BROKE TESTS** | Can't pre-build pattern when distance<8 |
| Simplified slow path to use copy_match_into | Neutral | Slow path rarely hit |
| 5-word unconditional copy (d>=8) | **55.4%** then **BROKE TESTS** | Works in fastloop only, needs margin |
| Loop unrolling (4x 8-byte) | 50.5% | Not better than simple loop |
| Tracing overhead optimization | 52.3% | #[cold] nested function helps |

### Data-Driven Insights (from tracing)

**Distance Distribution (17.5M matches in silesia):**
- d=1 (RLE): 0.1% - Already optimized with memset
- d=2-7: 0.7% - Byte-by-byte (only 130K matches, not worth optimizing)
- d=8-39: 7.8% - Chunk copy with potential for libdeflate-style stride copy
- d>=40: **91.3%** - Uses memcpy when d>=length (which is often)

**Length Distribution:**
- len 3-8: 70.7% - Most matches are short
- len 9-32: 24.5%
- len 33+: 4.8%

**Key Insight**: 91% of matches have d>=40. For most of these (when len<=d), 
we already use memcpy. The remaining gap is in the decode loop itself, not the copy function.

### What DOESN'T Work (Tested Jan 2026)

| Attempted | Result | Why |
|-----------|--------|-----|
| `saved_bitbuf` pattern | N/A | LUT pre-computes all matches, no extra bits to read |
| Unconditional 5-word copy | **48% (slower!)** | Matches avg 10.5 bytes, writing 40 wastes bandwidth |
| Loop unrolling | 50.5% | No improvement over simple loop |
| `wrapping_sub` vs `saturating_sub` | 47.9% | Counter-intuitively worse |

### What DOES Work (Tested Jan 2026)

| Optimization | Before | After | Gain |
|--------------|--------|-------|------|
| Short distance copy (d=2-7) | 1,088 MB/s | 4,109 MB/s | **3.78x** |

**Key Insight**: Short distance copy was 5x slower than other paths due to `i % distance`
modulo. Implemented libdeflate's word-at-a-time + stride approach.

### Current Path Benchmarks (Jan 2026)

| Decode Path | Speed | Status |
|-------------|-------|--------|
| Pure Literals | 10,000+ MB/s | ✅ Fast |
| RLE (d=1) | 6,221 MB/s | ✅ Fast |
| Long Distance (d>=40) | 4,449 MB/s | ✅ Fast |
| Short Distance (d=2-7) | 4,109 MB/s | ✅ Fixed |

### Micro-benchmark Results (Jan 2026)

| Component | Speed | Finding |
|-----------|-------|---------|
| Table lookup | 3250 M/sec | NOT bottleneck (<2% between 10/11/12 bit) |
| Consume-first pattern | 12% faster | Can't use - our table has BITS=0 entries |

**Key insight**: libdeflate's table NEVER has invalid entries (BITS=0). They use
subtables for long codes. Our table has BITS=0 for subtable pointers, which means
we must check-then-consume, not consume-then-check.

### Where the Gap Actually Is

All copy paths are now optimized. The remaining ~50% gap to libdeflate is:
1. **Table entry format** - libdeflate uses subtables, never BITS=0, enabling consume-first
2. **Inline macros** - libdeflate uses C macros, we use function calls
3. **BMI2 intrinsics** - libdeflate uses bzhi/pext on x86_64
4. **Compiler differences** - C with -O3 vs Rust with LTO

### What Actually Helps

1. **Entry preloading in literal loop** - Preload next entry BEFORE processing current (small but real gain)
2. **3-literal chain decode** - Already implemented, helps a lot
3. **RLE (distance=1) memset** - Already implemented, common case
4. **Conditional refill (ensure())** - Already implemented

## Gap Analysis

### What libdeflate Does That We Don't

| Optimization | libdeflate | gzippy | Impact | Status |
|--------------|-----------|--------|--------|--------|
| `saved_bitbuf` for extra bits | ✅ | ❌ | 10-15% | Tests written |
| `bitsleft -= entry` (full u32) | ✅ | ✅ | Done | ✅ Implemented |
| Entry preloading before copy | ✅ | ⚠️ | 5-10% | Partial |
| 5-word unconditional copy | ✅ | ❌ | 5-8% | Needs margin |
| RLE (offset=1) memset | ✅ | ✅ | 3-5% | ✅ Implemented |
| JIT table rebuild avoidance | ✅ | ❌ | 5-10% | Not started |
| BMI2 intrinsics (x86_64) | ✅ | ❌ | 5-10% | Tests written |
| 11-bit litlen table | ✅ | ❌ 12-bit | 2-3% | Not started |
| Subtable for long codes | ✅ | ✅ L2 | 3-5% | ✅ Implemented |
| Preload slack (conditional refill) | ✅ | ✅ | 2-3% | ✅ Implemented |

**Remaining potential gain: 30-50%** (gets us to 85-105% of libdeflate)

---

## Phase 1: saved_bitbuf Pattern (10-15% gain)

### The Problem
When decoding length/distance pairs, libdeflate saves the bitbuf BEFORE consuming:
```c
saved_bitbuf = bitbuf;
bitbuf >>= (u8)entry;
bitsleft -= entry;

// Later, extract extra bits from saved_bitbuf:
length = entry >> 16;
length += EXTRACT_VARBITS8(saved_bitbuf, entry) >> (u8)(entry >> 8);
```

This avoids separate `bits.read(extra_bits)` calls, which require:
1. Masking to extract bits
2. Shifting the buffer
3. Updating bitsleft

### The Fix
Add `saved_bitbuf` to our entry consumption:
```rust
// Before: 
bits.consume_entry(entry);
let extra = bits.read(extra_bits);
let length = base + extra;

// After:
let saved = bits.buffer();
bits.consume_entry(entry);  // Just shifts, doesn't read extra
let length = (entry >> 16) + ((saved >> shift_from_entry) & mask);
```

### Implementation
1. Modify `TurboBits::consume_entry()` to not touch extra bits
2. Add helper `extract_extra(saved_buf, entry) -> u32`
3. Pack extra bit count and shift into entry format

---

## Phase 2: Unconditional 5-Word Copy (5-8% gain)

### The Problem
libdeflate unconditionally copies 5 machine words (40 bytes on x64):
```c
store_word_unaligned(load_word_unaligned(src), dst);
src += WORDBYTES; dst += WORDBYTES;  // Repeat 5x
while (dst < out_next) { ... }  // Only loop if length > 40
```

Our code uses bounds-checked loops even for small matches.

### The Fix
```rust
// Unconditional 5-word copy (40 bytes)
unsafe {
    let s = output.as_ptr().add(src_start);
    let d = output.as_mut_ptr().add(out_pos);
    ptr::copy_nonoverlapping(s, d, 8);
    ptr::copy_nonoverlapping(s.add(8), d.add(8), 8);
    ptr::copy_nonoverlapping(s.add(16), d.add(16), 8);
    ptr::copy_nonoverlapping(s.add(24), d.add(24), 8);
    ptr::copy_nonoverlapping(s.add(32), d.add(32), 8);
}
if length > 40 {
    // Continue with loop for remaining bytes
}
```

---

## Phase 3: JIT Table Caching (5-10% gain)

### The Problem
For each dynamic block, we rebuild Huffman tables from scratch. libdeflate:
1. Caches static code tables
2. Uses a `static_codes_loaded` flag
3. For dynamic blocks with similar structure, can skip some work

### The Fix: JIT Table Fingerprinting
```rust
struct TableCache {
    fingerprint: u64,  // Hash of code lengths
    litlen_table: PackedLUT,
    dist_table: TwoLevelTable,
}

fn decode_dynamic_block(bits: &mut TurboBits, cache: &mut TableCache) {
    let code_lengths = read_code_lengths(bits);
    let fingerprint = hash_code_lengths(&code_lengths);
    
    if fingerprint == cache.fingerprint {
        // Reuse cached tables - skip expensive build!
        return decode_with_tables(bits, &cache.litlen_table, &cache.dist_table);
    }
    
    // Build new tables and cache them
    cache.litlen_table = PackedLUT::build(&code_lengths.litlen);
    cache.dist_table = TwoLevelTable::build(&code_lengths.dist);
    cache.fingerprint = fingerprint;
}
```

**Key insight**: Many gzip files (especially from same compressor) have identical or similar Huffman tables across blocks.

---

## Phase 4: BMI2 Intrinsics for x86_64 (5-10% gain)

### Key BMI2 Instructions
| Instruction | Use Case |
|-------------|----------|
| `shrx` | Variable shift without moving to CL register |
| `bzhi` | Zero high bits (extract N low bits) |
| `pext` | Parallel bit extract |

### Current vs. BMI2
```rust
// Current (3-4 instructions)
let bits_to_consume = (entry & 0xFF) as u32;
self.buf >>= bits_to_consume;
self.bits -= bits_to_consume;

// BMI2 (1-2 instructions)
use core::arch::x86_64::{_shrx_u64, _bzhi_u64};
self.buf = _shrx_u64(self.buf, entry as u64);  // entry contains shift amount
```

### Implementation
```rust
#[cfg(target_arch = "x86_64")]
mod bmi2 {
    #[target_feature(enable = "bmi2")]
    pub unsafe fn decode_loop_bmi2(...) {
        // Use _shrx_u64, _bzhi_u64 for critical operations
    }
}

// Runtime dispatch
if is_x86_feature_detected!("bmi2") {
    unsafe { bmi2::decode_loop_bmi2(...) }
} else {
    decode_loop_generic(...)
}
```

---

## Phase 5: Optimal Table Size (2-3% gain)

### The Problem
We use 12-bit primary table (4096 entries = 16KB).
libdeflate uses 11-bit (2048 entries = 8KB) + subtables.

8KB fits better in L1 cache (32KB typical).

### The Fix
```rust
const LITLEN_TABLEBITS: u32 = 11;
const TABLE_SIZE: usize = 1 << LITLEN_TABLEBITS;  // 2048

struct PackedLUT {
    primary: [PackedEntry; TABLE_SIZE],  // 8KB
    subtables: Vec<PackedEntry>,         // Overflow for 12-15 bit codes
}

fn decode(&self, bits: u64) -> PackedEntry {
    let entry = self.primary[(bits & 0x7FF) as usize];
    if entry.needs_subtable() {
        let subtable_idx = entry.subtable_index();
        let extra_bits = (bits >> 11) & entry.subtable_mask();
        return self.subtables[subtable_idx + extra_bits as usize];
    }
    entry
}
```

---

## Phase 6: Preload Slack & Refill Optimization (2-3% gain)

### The Problem
We call `bits.ensure(56)` every iteration. libdeflate calculates exactly
how many bits are preloadable vs. consumable and only refills when needed.

### The Concept
```
CONSUMABLE_NBITS = 56 (can be consumed after refill)
PRELOADABLE_NBITS = 63 (can be looked at but not consumed)
PRELOAD_SLACK = 7 (extra lookahead bits)
```

### The Fix
Track whether refill is needed based on actual bit consumption:
```rust
// Only refill when we've consumed enough to need it
if bits.available() < MIN_FASTLOOP_BITS {
    bits.refill_branchless();
}

// Preload next entry even when bits aren't "officially" available
// (using the 7 slack bits)
let next_entry = table[(bits.peek(12)) as usize];
```

---

## Phase 7: Parallel BGZF Optimization (Already done ✅)

We already beat libdeflate on BGZF parallel decompression:
- **3770 MB/s with 8 threads**
- Linear scaling with thread count
- Zero lock contention via UnsafeCell

---

## Implementation Priority

| Phase | Optimization | Est. Gain | Effort | Priority |
|-------|--------------|-----------|--------|----------|
| 1 | saved_bitbuf | 10-15% | Medium | **HIGH** |
| 2 | 5-word copy | 5-8% | Low | **HIGH** |
| 3 | JIT table cache | 5-10% | High | Medium |
| 4 | BMI2 intrinsics | 5-10% | Medium | **HIGH** |
| 5 | 11-bit table | 2-3% | Medium | Low |
| 6 | Preload slack | 2-3% | Low | Medium |

**Recommended order**: Phase 2 → Phase 1 → Phase 4 → Phase 6 → Phase 3 → Phase 5

---

## Verification Strategy

After each phase:
1. Run `GZIPPY_TRACE=1` to get timing
2. Compare with libdeflate benchmark
3. Ensure all 163 tests pass
4. Profile to find new bottleneck

```bash
# Quick benchmark
cargo test --release benchmark_turbo_vs_all -- --nocapture

# Full trace
GZIPPY_TRACE=1 ./target/release/gzippy -d benchmark_data/silesia-gzip.tar.gz -k
```

---

## Target Milestones

| Milestone | Speed | % of libdeflate |
|-----------|-------|-----------------|
| Current | 703 MB/s | 57.5% |
| After Phase 1+2 | ~850 MB/s | 70% |
| After Phase 3+4 | ~1050 MB/s | 86% |
| After Phase 5+6 | ~1150 MB/s | 94% |
| With JIT cache | ~1300 MB/s | 106% |
| Optimal | ~1400 MB/s | 115% |

---

## Appendix: libdeflate Entry Format

```
Literals:
    Bit 31:     1 (HUFFDEC_LITERAL - test with signed comparison)
    Bit 23-16:  literal value
    Bit 15:     0
    Bit 3-0:    codeword length

Lengths:
    Bit 31:     0
    Bit 24-16:  length base value  
    Bit 15:     0
    Bit 11-8:   codeword length (for saved_bitbuf shift)
    Bit 4-0:    codeword length + extra bit count

EOB:
    Bit 31:     0
    Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
    Bit 13:     1 (HUFFDEC_END_OF_BLOCK)
```

Our PackedLUT format is similar but uses different bit positions. Consider
aligning exactly with libdeflate for easier porting of optimizations.
