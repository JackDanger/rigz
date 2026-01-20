# Optimization Roadmap: Beating libdeflate

**Current Status**: 703 MB/s (57.5% of libdeflate's 1223 MB/s)
**Target**: 1400+ MB/s (115% of libdeflate)

## Gap Analysis

### What libdeflate Does That We Don't

| Optimization | libdeflate | gzippy | Impact |
|--------------|-----------|--------|--------|
| `saved_bitbuf` for extra bits | ✅ | ❌ | 10-15% |
| `bitsleft -= entry` (full u32) | ✅ | ✅ | Done |
| Entry preloading before copy | ✅ | ⚠️ partial | 5-10% |
| 5-word unconditional copy | ✅ | ❌ 40-byte | 5-8% |
| RLE (offset=1) broadcast | ✅ | ⚠️ partial | 3-5% |
| JIT table rebuild avoidance | ✅ | ❌ | 5-10% |
| BMI2 intrinsics (x86_64) | ✅ | ❌ | 5-10% |
| 11-bit litlen table | ✅ | ❌ 12-bit | 2-3% |
| Subtable for long codes | ✅ | ❌ L2 | 3-5% |
| Preload slack calculation | ✅ | ❌ | 2-3% |

**Total potential gain: 40-70%** (gets us to 110-120% of libdeflate)

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
