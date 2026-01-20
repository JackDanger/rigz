# Exceed All: The Path to Ultimate Performance

## Current Status (January 2026)

| Metric | Current | libdeflate | Target |
|--------|---------|------------|--------|
| Single-threaded silesia | 624 MB/s | 1321 MB/s | 1400+ MB/s |
| Parallel BGZF (8 threads) | 3770 MB/s | N/A | 4500+ MB/s |
| Performance ratio | 47.2% | 100% | 106%+ |

## Problems We Encountered

### 1. Entry Format Redesign Failed

**What happened:**
- Attempted to move `total_bits` to low byte like libdeflate
- Distance bases (up to 24577) require 15 bits
- This overlapped with type flags in bits 29-31

**Root cause:**
- Made layout changes without fully planning bit allocation
- Changed format, constructors, accessors, and all usage sites at once
- When tests failed, couldn't identify which part was wrong

### 2. Insufficient Incremental Testing

**What happened:**
- Made big changes, then tried to fix tests afterward
- No golden tests to catch subtle decode errors early

**Root cause:**
- Rushed to implement the optimization
- Didn't follow TDD (Test-Driven Development) properly

## Safety Principles Going Forward

### Principle 1: Design-First with Bit Layout Documents

Before changing any entry format, create a complete bit layout document:

```
Entry Type: LiteralEntry
┌─────────────────────────────────────────────────────────┐
│ bit 31   │ bit 30   │ bits 29-24 │ bits 23-16 │ bits 15-8  │ bits 7-0   │
│ LITERAL  │ unused   │ unused     │ lit_value  │ unused     │ total_bits │
│ 1        │ 0        │ 000000     │ 0-255      │ 00000000   │ 1-15       │
└─────────────────────────────────────────────────────────┘
```

### Principle 2: Separate Types for Different Tables

Use Rust's type system to prevent mistakes:

```rust
/// Entry for literal/length tables
#[repr(transparent)]
pub struct LitLenEntry(u32);

/// Entry for distance tables (different bit layout allowed)
#[repr(transparent)]  
pub struct DistEntry(u32);
```

### Principle 3: Feature Flags for Incremental Rollout

```toml
[features]
# Each optimization can be toggled
opt_signed_literal_check = []      # Use (entry as i32) < 0
opt_packed_total_bits = []         # Store total_bits in low byte
opt_precomputed_distance = []      # Pre-compute distance bases
opt_inline_handle_length = []      # Inline length handling
```

### Principle 4: Golden Test Suite

Create deterministic test cases with saved expected output:

```rust
#[test]
fn golden_silesia_first_1000_bytes() {
    let expected = include_bytes!("golden/silesia_first_1000.bin");
    let actual = decompress_silesia_first_1000();
    assert_eq!(actual, expected);
}
```

### Principle 5: Micro-Benchmarks for Each Path

```rust
#[bench] fn bench_literal_decode_only() { ... }
#[bench] fn bench_length_decode_only() { ... }
#[bench] fn bench_distance_decode_only() { ... }
#[bench] fn bench_copy_match_d1() { ... }
#[bench] fn bench_copy_match_d8() { ... }
```

## The Complete Optimization Roadmap

### Phase 1: Foundation (Current - COMPLETE)

- [x] PrecomputedTable with saved_bitbuf pattern
- [x] Signed comparison for literal check
- [x] Optimized copy_match_into (RLE, prefetch)
- [x] is_exceptional() grouping

**Status: 47.2% of libdeflate**

### Phase 2: Entry Format Redesign (NEXT)

The key insight from libdeflate is storing `total_bits` in the low byte so that:
```c
bitbuf >>= (u8)entry;  // Just shift by low byte
bitsleft -= entry;     // Low 8 bits only matter
```

**The Correct Approach:**

1. **Create separate `LitLenEntry` and `DistEntry` types**
2. **LitLenEntry layout** (flags are fine, base values are small):
   ```
   [31:LIT][30:SUB][29:EOB][28-16:base_or_lit][15-8:codeword][7-0:total]
   ```
3. **DistEntry layout** (no LIT/EOB needed, just SUB):
   ```
   [31:SUB][30-16:base][15-8:codeword][7-0:total]
   ```

4. **Write tests FIRST** for each entry type
5. **Build new tables alongside old ones** for comparison
6. **Switch decode loop only when tests pass**

**Expected gain: 15-20%**

### Phase 3: Eliminate Branches in Decode Loop

libdeflate's key insight: use unlikely() hints and fall-through:

```rust
// Fast path (95%+ of iterations)
if entry.is_literal() {  // Single bit test
    // Decode literal, CONTINUE
}

// Rare path grouped together
if unlikely(entry.is_exceptional()) {
    // Handle subtable OR EOB
}

// Length code - fall through, no test needed
```

**Expected gain: 5-10%**

### Phase 4: BMI2 Intrinsics on x86_64

Use `_bzhi_u64` for bit extraction when available:

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
fn extract_bits(val: u64, count: u32) -> u64 {
    unsafe { core::arch::x86_64::_bzhi_u64(val, count) }
}
```

**Expected gain: 5-10%**

### Phase 5: JIT Table Caching

Fingerprint code lengths and reuse identical tables:

```rust
static TABLE_CACHE: Lazy<Mutex<HashMap<u64, Arc<CachedTable>>>> = ...;

fn get_or_build_table(code_lengths: &[u8]) -> Arc<CachedTable> {
    let hash = hash_code_lengths(code_lengths);
    // Check cache first...
}
```

**Expected gain: 2-5% on multi-block files**

### Phase 6: SIMD Huffman Decode (Advanced)

For AVX-512 systems, decode multiple symbols in parallel:

```rust
#[cfg(target_feature = "avx512f")]
fn decode_8_symbols_simd(bitbuf: __m512i, table: &[u32]) -> [u16; 8] {
    // Parallel table lookups with vpgatherdd
    // Parallel bit consumption
}
```

**Expected gain: 20-40% on AVX-512 systems**

### Phase 7: Parallel Single-Member (rapidgzip approach)

For large single-member files:
1. Find block boundaries with speculative decode
2. Propagate windows between chunks
3. Parallel re-decode with known windows

**Expected gain: 3-5x on large single-member files**

## Implementation Order

| Week | Task | Safety Measure |
|------|------|----------------|
| 1 | Create LitLenEntry type with tests | Golden tests for each entry type |
| 1 | Create DistEntry type with tests | Golden tests for each entry type |
| 2 | Build new tables alongside old | Compare outputs byte-by-byte |
| 2 | Micro-benchmark new vs old | Ensure no regression |
| 3 | Switch decode loop to new entries | Feature flag to toggle |
| 3 | Benchmark and tune | Target: 60%+ of libdeflate |
| 4 | Add BMI2 intrinsics | Runtime detection |
| 4 | Add JIT table caching | Hash collision tests |
| 5 | Final tuning and cleanup | All 233+ tests pass |

## Success Criteria

1. **Exceed libdeflate single-threaded**: 106%+ on silesia
2. **Exceed rapidgzip parallel**: 4000+ MB/s on 8 threads
3. **Zero correctness regressions**: All tests pass
4. **No dependencies**: Pure Rust, statically linked
5. **Drop-in compatible**: Same CLI as gzip/pigz

## The Mantra

> "Measure twice, cut once. Test first, optimize second. One change at a time."

We will exceed libdeflate. We will exceed rapidgzip. We will do it safely and systematically.
