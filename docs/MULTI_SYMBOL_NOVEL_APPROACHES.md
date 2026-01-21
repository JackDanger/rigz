# Novel Approaches to Make Multi-Symbol Work

**Current Problem:** Multi-symbol regressed 3% because we do TWO lookups per iteration:
1. `multi.lookup()` - checks for literals
2. `litlen_table.lookup()` - fallback for matches

**Goal:** Make multi-symbol FASTER than single-symbol, not slower.

---

## Approach #1: Adaptive Mode Switching with Bloom Filter 🌟

**Idea:** Track recent symbol patterns and only use multi_symbol when in literal-heavy regions.

```rust
struct AdaptiveDecoder {
    recent_pattern: u64,  // Rolling 64-bit: 1=literal, 0=match
    use_multi: bool,
}

impl AdaptiveDecoder {
    fn update(&mut self, is_literal: bool) {
        self.recent_pattern = (self.recent_pattern << 1) | (is_literal as u64);

        // If > 75% of last 64 symbols were literals, enable multi_symbol
        self.use_multi = self.recent_pattern.count_ones() > 48;
    }
}
```

**Cost:** 1 shift + 1 popcnt = ~2 cycles
**Benefit:** Avoid double-lookup in match-heavy regions
**Expected gain:** +5-10% over current multi_symbol

---

## Approach #2: Unified Table (Replace, Don't Augment)

**Idea:** Make MultiSymbolLUT handle ALL cases, not just literals.

```rust
enum MultiEntry {
    DoubleLiteral(u8, u8, bits),
    SingleLiteral(u8, bits),
    Match {
        length_base: u16,
        length_extra_bits: u8,
        // Pre-load distance table index
        distance_hint: u16,
    },
    EOB,
}
```

**Cost:** Larger table (3x size)
**Benefit:** No fallback, single lookup
**Expected gain:** +10-15% over baseline

---

## Approach #3: Speculative Dual Lookup

**Idea:** Do BOTH lookups in parallel, use the right one.

```rust
// Modern CPUs can issue 2 loads simultaneously
let multi_entry = multi.lookup(bits);
let regular_entry = litlen.lookup(bits);

// Fast select based on which one is valid
let use_multi = multi_entry.is_literal();
let (symbol, bits_consumed) = if use_multi {
    (multi_entry.symbol1(), multi_entry.bits())
} else {
    (regular_entry.symbol(), regular_entry.bits())
};
```

**Cost:** 2 cache loads (but parallel on modern CPUs)
**Benefit:** No branch, always have both ready
**Expected gain:** +5-8% if branch prediction was the issue

---

## Approach #4: Block-Level Analysis and Switching

**Idea:** Analyze first 1KB of each block to determine if it's literal-heavy.

```rust
fn analyze_block(input: &[u8]) -> DecoderMode {
    // Quick scan of first 1KB
    let sample = &input[..min(1024, input.len())];
    let literal_ratio = count_literals(sample) / sample.len();

    if literal_ratio > 0.7 {
        DecoderMode::MultiSymbol
    } else {
        DecoderMode::Regular
    }
}
```

**Cost:** 1KB decode upfront (amortized)
**Benefit:** Choose optimal decoder for entire block
**Expected gain:** +10-20% on mixed blocks

---

## Approach #5: SIMD Multi-Symbol (8-16 symbols at once)

**Idea:** Use vector_huffman infrastructure to decode 8 literals simultaneously.

```rust
// For literal runs, decode 8 at a time using AVX2
if detect_literal_run() {
    let symbols: [u8; 8] = simd_decode_8_literals(bits, multi_table);
    output[out_pos..out_pos+8].copy_from_slice(&symbols);
    out_pos += 8;
}
```

**Cost:** SIMD table (16KB)
**Benefit:** 8x throughput on literal runs
**Expected gain:** +50-100% on literal-heavy data

---

## Approach #6: JIT Code Generation per Block

**Idea:** Generate native code for each unique Huffman table.

```rust
// Build executable code for this specific table
let decode_fn: fn(&[u8]) -> Vec<u8> = jit_compile_decoder(&table);

// Subsequent blocks with same table use cached JIT code
if let Some(cached_fn) = jit_cache.get(table_fingerprint) {
    return cached_fn(input);
}
```

**Cost:** Cranelift compilation (~1ms per table)
**Benefit:** No table lookups at all - direct jumps
**Expected gain:** +100-200% for repeated tables

---

## Approach #7: Probabilistic Skip

**Idea:** Use hash to predict if multi_symbol will hit.

```rust
// Hash low bits to predict symbol type
let likely_literal = (bits & 0xFF).count_ones() > 4;

if likely_literal {
    let entry = multi.lookup(bits);
    if entry.is_literal() { /* fast path */ }
}
// Else skip multi entirely
```

**Cost:** 1 popcnt = ~1 cycle
**Benefit:** Skip multi_symbol when unlikely to help
**Expected gain:** +3-5% by avoiding wasted lookups

---

## Approach #8: Hybrid Cache Line Packing

**Idea:** Pack multi_symbol and regular table entries in same cache line.

```rust
#[repr(C)]
struct HybridEntry {
    multi: MultiEntry,     // 8 bytes
    regular: RegularEntry, // 8 bytes
    // Both in same 16-byte chunk = 1 cache line
}
```

**Cost:** 2x table size
**Benefit:** Both lookups hit same cache line
**Expected gain:** +2-4% from cache optimization

---

## Recommendation: Try #1 First (Adaptive Bloom Filter)

**Why:**
- Simplest to implement (~20 lines)
- Minimal overhead (2 cycles)
- Solves the core problem (double-lookup)
- Can be combined with other approaches

**Implementation:**

```rust
fn decode_huffman_adaptive(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
    multi: &MultiSymbolLUT,
) -> Result<usize> {
    let mut pattern = 0u64;
    let mut use_multi = true;

    while out_pos < output.len() {
        bits.refill();

        if use_multi && pattern.count_ones() > 48 {
            // Try multi-symbol (only in literal-heavy regions)
            let entry = multi.lookup(bits.peek());
            if entry.is_literal() {
                output[out_pos] = entry.symbol1();
                out_pos += 1;
                bits.consume(entry.bits());
                pattern = (pattern << 1) | 1; // Mark literal
                continue;
            }
        }

        // Regular decode
        let entry = litlen.lookup(bits.peek());
        let is_literal = entry.is_literal();
        pattern = (pattern << 1) | (is_literal as u64);

        // Update mode every 64 symbols
        if pattern == 0 {
            use_multi = pattern.count_ones() > 48;
        }

        // ... decode literal or match ...
    }
}
```

**Expected Result:**
- Baseline: 935 MB/s
- With adaptive: **1020-1050 MB/s** (+9-12%)
- Achieves the goal of 1000+ MB/s!

---

## If That Doesn't Work: Try #2 (Unified Table)

Make multi_symbol handle matches too, eliminating fallback entirely.

---

## Dream Scenario: Combine Multiple Approaches

1. Adaptive bloom filter (#1)
2. SIMD for literal runs (#5)
3. JIT caching for repeated tables (#6)

**Combined Expected Gain:** +150-200% 🚀
