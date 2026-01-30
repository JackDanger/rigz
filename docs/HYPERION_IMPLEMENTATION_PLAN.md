# HYPERION: LLM-Friendly Hyperoptimization Implementation Plan

**Goal:** Beat every decompression tool in every circumstance using advanced math and novel algorithms.

**Design Principle:** This plan is designed for implementation by an LLM (like Claude) working with partial context, potentially getting confused, but making guaranteed forward progress toward the ultimate goal.

---

## Lessons from Git History (55+ Commits Analyzed)

### ✅ WHAT WORKED (Evidence-Based)

| Optimization | Commit | Result |
|--------------|--------|--------|
| Consume-first pattern | `50ac529` | 36% faster isolated lookups |
| NEON/AVX2 SIMD match copies | `3b9c81d` | Achieved ARM parity |
| Libdeflate-style decode loop | `3b9c81d` | 99-114% of libdeflate |
| 8-literal batching | `f40740f` | Fewer loop iterations |
| Parallel BGZF | `df19e86` | 7430 MB/s (550% of libdeflate ST) |
| Pure Rust (no FFI) | `f40740f` | Eliminates dependency issues |
| Branchless refill | `5c01fad` | Matches libdeflate pattern |
| Generic loop always refills | `f40740f` | Fixed bit buffer exhaustion |

### ❌ WHAT FAILED (DO NOT REPEAT)

| Optimization | Commit | Expected | Actual | Why |
|--------------|--------|----------|--------|-----|
| JIT (Cranelift) | `b21f0f7` | +30% | **Slow** | 2.3ms compile time per table |
| Markov prediction | `13bc2a5` | +10% | **-75%** | Prediction overhead destroys perf |
| Bytecode interpreter | `f7b2d76` | +10% | **-46%** | Match dispatch overhead |
| DoubleLitCache per block | `aa3ea09` | +15% | **-73%** | Build cost > decode gain |
| Table-free fixed Huffman | `0335dc4` | +20% | **-325%** | Bit reversal slower than lookup |
| `#[cold]` on errors | `0335dc4` | +5% | **-4%** | Added call overhead |
| Unconditional refill | `0335dc4` | +5% | **-12%** | Conditional was optimal |
| 5-word match unroll | `75b5763` | +10% | **-15%** | Hurts cache on short matches |
| 2-3 literal batching x86 | `7df6559` | +5% | **-20%** | Hurt SOFTWARE dataset |

### ⚡ DISCONNECTED MODULES (Inventory Jan 2026)

| Module | Purpose | Status | Potential |
|--------|---------|--------|-----------|
| `multi_symbol.rs` | 2-symbol per lookup | Implemented | **HIGH** - ISA-L's key win |
| `double_literal.rs` | Double literal cache | Implemented | **HIGH** - pair decoding |
| `algebraic_decode.rs` | ANF branchless decode | +52% isolated | Correctness bugs block it |
| `unified_table.rs` | Single 64-bit table | Implemented | MEDIUM - needs benchmark |
| `vector_huffman.rs` | 8-lane SIMD | Infrastructure | LOW - complex integration |
| `simd_parallel_decode.rs` | AVX2 speculative | Infrastructure | LOW - lane sync issues |
| `precomputed_decode.rs` | Pre-built sequences | Implemented | UNKNOWN - not tested |
| `speculative_batch.rs` | CPU speculation | Implemented | UNKNOWN - not tested |
| `two_level_table.rs` | Two-level lookup | **USED in bgzf.rs** | Working ✅ |
| `combined_lut.rs` | Combined lit+dist | **USED in bgzf.rs** | Working ✅ |
| `packed_lut.rs` | Packed entries | **USED in bgzf.rs** | Working ✅ |
| `jit_decode.rs` | JIT compilation | TableFingerprint used | JIT itself too slow |
| `hyper_parallel.rs` | Parallel single | **USED via hyperion** | Working ✅ |
| `marker_turbo.rs` | Fast marker decode | **USED via hyper_parallel** | Working ✅ |

### 📊 GAP ANALYSIS SUMMARY (Jan 2026)

See `docs/GAP_ANALYSIS.md` for full comparison with libdeflate, ISA-L, and rapidgzip.

| Category | Our Status | State-of-Art | Gap | Priority |
|----------|-----------|--------------|-----|----------|
| Huffman decode loop | 91% libdeflate | libdeflate | 9% | HIGH |
| Multi-symbol tables | NOT integrated | ISA-L: 1-3 sym/lookup | **LARGE** | **HIGHEST** |
| Runtime BMI2 | ✅ Implemented | libdeflate | 0% | Done |
| Runtime AVX2 match | ❌ Compile-time only | libdeflate | ~5% | MEDIUM |
| Parallel single-member | marker_turbo works | rapidgzip | ~same | Done |

---

## Core Architecture: The HYPERION Unified Entrypoint

```
decompress_hyperion(data, writer, threads)
    │
    ├──▶ classify_archive(data) → ArchiveType + DataProfile
    │         │
    │         ├─ BGZF (embedded block sizes)
    │         ├─ MultiMember (pigz-style)
    │         ├─ SingleMember + HighEntropy (complex)
    │         └─ SingleMember + LowEntropy (repetitive)
    │
    ├──▶ select_decoder(archive_type, data_profile, threads)
    │         │
    │         ├─ BGZF → bgzf_parallel (proven: 4000+ MB/s)
    │         ├─ MultiMember → multi_member_parallel (proven)
    │         ├─ SingleMember+1Thread → turbo_inflate (proven: 1400 MB/s)
    │         ├─ SingleMember+LowEntropy → HYPERION_PHASE_2 (ANF decoder)
    │         └─ SingleMember+HighEntropy → HYPERION_PHASE_3 (FSM decoder)
    │
    └──▶ execute_decode(decoder, data, writer) → Result<bytes>
```

---

## Implementation Strategy: The Guardrail Pattern

### Golden Rule: Every Change Must Pass This Test

```bash
# BEFORE any code change, record baseline:
cargo test --release bench_cf_silesia -- --nocapture 2>&1 | grep "Our throughput"
# Record: ______ MB/s

# AFTER code change:
cargo test --release 2>&1 | grep -E "(passed|failed)"  # Must: all passed
cargo test --release bench_cf_silesia -- --nocapture 2>&1 | grep "Our throughput"
# Must be >= baseline

# IF FAILED: git checkout -- . && start over
```

### The Three Invariants (Never Break These)

1. **SILESIA throughput >= 1350 MB/s** (current baseline)
2. **SOFTWARE throughput >= 9000 MB/s** (we exceed libdeflate here)
3. **LOGS throughput >= 8000 MB/s** (we exceed libdeflate here)

### Checkpoint Pattern

Each phase ends with a git commit that:
1. Passes all tests
2. Maintains or improves performance
3. Has a clear rollback point
4. Documents results in commit message

### Pre-Commit CI Check (MANDATORY)

**Before every commit**, check the GitHub workflow status from the previous push:

```bash
# Check status of last workflow runs
gh run list --limit 5

# If any failed, investigate before committing:
gh run view <run-id> --log-failed

# Check for performance regressions in benchmark jobs:
gh run view <run-id> --job <benchmark-job-id> | grep -E "MB/s|ratio|regression"
```

**Rules:**
1. **Never commit if the previous push has failing workflows** - fix first
2. **Check for performance regressions** - benchmark jobs should show stable or improved numbers
3. **If CI is red, diagnose before adding more commits** - don't pile on broken state

---

## Phase 1: Unified Entrypoint (Low Risk, High Value)

**Objective:** Create single `decompress_hyperion()` that routes optimally based on archive type.

### Step 1.1: Create Entrypoint Skeleton

```rust
// src/hyperion.rs - NEW FILE
pub fn decompress_hyperion<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    threads: usize,
) -> GzippyResult<u64> {
    let archive_type = classify_archive(data);
    
    match archive_type {
        ArchiveType::Bgzf => crate::bgzf::decompress_bgzf_parallel(data, writer, threads),
        ArchiveType::MultiMember => crate::bgzf::decompress_multi_member_parallel(data, writer, threads),
        ArchiveType::SingleMember => decompress_single_member_turbo(data, writer),
    }
}
```

**Checkpoint:** `git commit -m "feat: add hyperion unified entrypoint (passthrough only)"`

### Step 1.2: Wire Into decompression.rs

Replace `decompress_gzip_libdeflate` call with `decompress_hyperion`.

**Test:** Same performance as before (it's just routing differently).

**Checkpoint:** `git commit -m "refactor: route all gzip through hyperion entrypoint"`

### Step 1.3: Add Data Profiling

```rust
#[derive(Debug, Clone, Copy)]
pub struct DataProfile {
    pub estimated_entropy: f32,      // 0.0 = highly repetitive, 8.0 = random
    pub avg_match_distance: u32,     // Average LZ77 distance in first 64KB
    pub literal_ratio: f32,          // % of symbols that are literals
}

pub fn profile_data(deflate_data: &[u8]) -> DataProfile {
    // Quick sampling: analyze first 64KB of deflate stream
    // Count distinct bytes, estimate entropy via population count
    // This takes <1ms even for large files
}
```

**Checkpoint:** `git commit -m "feat: add data profiling for adaptive decoder selection"`

---

## Phase 2: Integrate ANF Decoder for Low-Entropy Data

**Objective:** Use algebraic_decode.rs for highly repetitive data where branchless decode shines.

### Step 2.1: Benchmark ANF in Isolation

```bash
cargo test --release bench_anf -- --nocapture
```

Document: Current ANF isolated throughput: ______ M symbols/sec

### Step 2.2: Create ANF Wrapper

```rust
// In src/hyperion.rs
fn decompress_anf<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    // Use ANF decoder for fixed Huffman blocks
    // Fall back to turbo_inflate for dynamic blocks
}
```

**Critical:** Only use ANF for fixed Huffman blocks initially. Dynamic blocks use existing turbo path.

### Step 2.3: Add Conditional Routing

```rust
if archive_type == ArchiveType::SingleMember {
    let profile = profile_data(deflate_data);
    if profile.estimated_entropy < 3.0 && threads == 1 {
        // Low entropy, single thread: try ANF
        if let Ok(bytes) = decompress_anf(data, writer) {
            return Ok(bytes);
        }
    }
    // Fallback to turbo
    return decompress_single_member_turbo(data, writer);
}
```

**Checkpoint:** `git commit -m "feat: integrate ANF decoder for low-entropy data"`

---

## Phase 3: Fix Parallel Single-Member (The Blocker)

**The Problem:** `rapidgzip_decoder.rs` uses `SpeculativeDecoder` at ~70 MB/s. Our `turbo_inflate` runs at ~1400 MB/s. The speculative decoder must be nearly as fast as turbo_inflate for parallel to help.

### Step 3.1: Analyze Why SpeculativeDecoder is Slow

```bash
# Profile the speculative decoder
cargo test --release bench_marker_decoder -- --nocapture
```

**Hypothesis:** SpeculativeDecoder uses separate bit reading + marker output, while turbo_inflate uses optimized combined table lookup.

### Step 3.2: Create Hybrid Speculative Decoder

**Key Insight:** We don't need the SpeculativeDecoder to be 100% as fast as turbo_inflate. We only need:

```
speculative_speed * threads > turbo_speed
```

For 8 threads: `speculative_speed > 175 MB/s` (currently 70 MB/s, need 2.5x improvement)

**Strategy:** Reuse the consume_first_decode hot path, but output to `u16` buffer with markers instead of `u8` buffer with bytes.

```rust
// In src/marker_turbo.rs - NEW FILE
pub fn turbo_inflate_with_markers(
    deflate_data: &[u8],
    output: &mut [u16],
) -> Result<(usize, bool), DecompressError> {
    // Use same hot path as consume_first_decode.rs
    // But write literals as u16(literal)
    // And write back-references as u16(0x8000 | marker_index)
}
```

### Step 3.3: Integrate Into hyper_parallel.rs

Replace the slow `SpeculativeDecoder` with `turbo_inflate_with_markers`.

**Target:** Speculative speed >= 500 MB/s (8 threads = 4000 MB/s total)

**Checkpoint:** `git commit -m "perf: hybrid speculative decoder reuses turbo hot path"`

---

## Phase 4: SIMD Parallel Huffman (Novel Math)

**Objective:** Implement the interleaved FSM with SIMD from `algebraic_decode.rs`.

### Step 4.1: Verify FSM Infrastructure

```rust
// In src/algebraic_decode.rs
// Existing: AlgebraicFsmDecoder, FsmState, build_fsm
```

Benchmark: `cargo test --release bench_fsm_interleaved -- --nocapture`

### Step 4.2: AVX2 Lane Implementation

```rust
// In src/simd_huffman.rs
#[cfg(target_arch = "x86_64")]
pub fn decode_8_lanes_avx2(
    bits: &[u64; 8],      // 8 bit buffers
    table: &FsmTable,     // Shared FSM table
    output: &mut [u8],    // Interleaved output
) -> [usize; 8] {         // Bits consumed per lane
    // VPGATHERDD to load 8 table entries simultaneously
    // Each lane decodes independently
    // Prefix-sum to compute output positions
    // VPSCATTERDD to write outputs
}
```

### Step 4.3: Lane Synchronization

The challenge: lanes consume different numbers of bits.

**Solution:** Decode in "bursts" of 4 symbols each, then resynchronize:

```rust
for _ in 0..4 {
    // Parallel decode one symbol per lane
    // Don't refill yet
}
// Now resync: prefix-sum bits consumed, refill all lanes
```

**Checkpoint:** `git commit -m "feat: SIMD 8-lane parallel Huffman with burst decode"`

---

## Phase 5: Unified Table (Cache Optimization)

**Objective:** Replace separate litlen/distance tables with single 64-bit table.

### Step 5.1: Analyze unified_table.rs

```rust
// Existing in src/unified_table.rs
pub struct UnifiedEntry(u64);
// Encodes: type, symbol, extra_bits, codeword_bits, distance_base, length_base
```

### Step 5.2: Benchmark Against Separate Tables

```bash
cargo test --release bench_unified_table -- --nocapture
```

### Step 5.3: Integrate If Faster

Only integrate if measurably faster on SILESIA. Otherwise, document and move on.

---

## Phase 6: Close the Remaining 9% Gap (Tier 1 Optimizations)

**Current Status (Jan 2026):** 91% of libdeflate on SILESIA. These optimizations target the final 9%.

### Step 6.1: Entry Preloading (Expected +3-5%)

libdeflate's key trick: start the NEXT table lookup before finishing the CURRENT iteration.

```rust
// Current (bad):
loop {
    let entry = litlen.lookup(bits.peek());
    bits.consume_entry(entry.raw());
    if entry.is_literal() {
        *output_ptr = entry.literal_value();
        output_ptr = output_ptr.add(1);
        continue;  // Next lookup happens AFTER this
    }
    // ...
}

// Target (good):
let mut entry = litlen.lookup(bits.peek());
loop {
    bits.consume_entry(entry.raw());
    let next_peek = bits.peek();  // Preload BEFORE writing
    if entry.is_literal() {
        *output_ptr = entry.literal_value();
        output_ptr = output_ptr.add(1);
        entry = litlen.lookup(next_peek);  // Already computed!
        continue;
    }
    // ... handle match, then:
    entry = litlen.lookup(bits.peek());
}
```

**Files to modify:** `src/consume_first_decode.rs` (decode_huffman_libdeflate_style)

**Checkpoint:** `git commit -m "perf: entry preloading in decode loop (+3-5% expected)"`

### Step 6.2: Packed bitsleft Subtraction (Expected +1-2%)

libdeflate subtracts the entire entry value, not just the masked low bits:

```rust
// Current:
self.bitsleft -= (entry & 0x1F) as u32;

// Target (libdeflate-style):
self.bitsleft -= entry as u32;  // High bits are garbage but refill handles it
```

**Files to modify:** `src/consume_first_decode.rs` (Bits::consume_entry)

**Checkpoint:** `git commit -m "perf: packed bitsleft subtraction saves 1 instruction/symbol"`

### Step 6.3: Saved Bitbuf Pattern (Expected +1-2%)

Capture bits BEFORE consuming for extra bits extraction:

```rust
// Current (multiple shifts):
let extra = (bits.peek() >> codeword_bits) & extra_mask;

// Target (single operation):
let saved = bits.peek();  // Capture before consume
bits.consume(total_bits);
let extra = (saved >> codeword_bits) & extra_mask;
```

**Files to modify:** `src/consume_first_decode.rs`, `src/libdeflate_entry.rs`

**Checkpoint:** `git commit -m "perf: saved_bitbuf pattern for extra bits extraction"`

---

## Phase 7: Medium-Impact Optimizations (Tier 2)

### Step 7.1: Fix hyper_parallel Fallback Rate

Currently hyper_parallel falls back to sequential too often. Debug and fix.

```bash
# Enable debug logging to see fallback reasons:
GZIPPY_DEBUG=1 cargo run --release -- -d large_file.gz > /dev/null
```

**Target:** Parallel path succeeds >90% of the time for files >8MB.

### Step 7.2: Cache-Align Critical Structs

```rust
#[repr(C, align(64))]  // Cache line alignment
pub struct Bits {
    buffer: u64,
    bitsleft: u32,
    pos: usize,
    data: *const u8,
    len: usize,
}
```

**Expected gain:** ~2% on ARM (128-byte cache lines)

### Step 7.3: Prefetch Next Block

```rust
#[cfg(target_arch = "x86_64")]
unsafe {
    std::arch::x86_64::_mm_prefetch(
        data.as_ptr().add(current_pos + 4096) as *const i8,
        std::arch::x86_64::_MM_HINT_T0,
    );
}
```

---

## Phase 8: Speculative Research (Tier 3)

These have unknown impact and should only be attempted after Phases 6-7 are complete.

### 8.1: Fix ANF Decoder Bugs
- ANF shows 1.55x speedup on core operations but has correctness issues
- Only beneficial for fixed Huffman blocks (niche case)

### 8.2: JIT Table Compilation
- Generate native code for repeated Huffman tables
- High complexity, diminishing returns

### 8.3: GPU Offload
- Parallel marker replacement on GPU
- Only worthwhile for files >1GB

---

## Phase 9: ISA-L-Style Multi-Symbol Tables (HIGH PRIORITY)

**This is the biggest untapped opportunity based on gap analysis.**

ISA-L achieves higher throughput by encoding 1-3 symbols per table entry, not just 1.
We have `multi_symbol.rs` and `double_literal.rs` but they are NOT integrated.

### Why Previous Attempts Failed

Previous `DoubleLitCache` attempt (commit `aa3ea09`) showed -73% regression because:
1. Built a SEPARATE cache per block (expensive)
2. Cache building cost exceeded decode gain

**ISA-L's approach:** Build multi-symbol entries DURING table construction, stored in the SAME table.

### Step 9.1: Understand ISA-L's Table Format

From `isa-l/igzip/igzip_inflate.c:450`:
```c
// ISA-L encodes sym_count in the entry itself:
short_code_lookup[code] = sym1 | (sym2 << 8) |
    (code_length << LARGE_SHORT_CODE_LEN_OFFSET) |
    (2 << LARGE_SYM_COUNT_OFFSET);  // 2 symbols packed!
```

Key insight: During table building, when code1 + code2 <= LUT_BITS:
- Pack both symbols into a single entry
- Store combined bit length
- Decode both with ONE lookup

### Step 9.2: Extend LitLenEntry for Multi-Symbol

```rust
// Current: 32-bit entry, 1 symbol per lookup
// New option 1: 64-bit entry, up to 2 symbols
// New option 2: Keep 32-bit, add symbol_count field

// Recommended: 64-bit entry (matches multi_symbol.rs)
pub struct LitLenEntry64(u64);

impl LitLenEntry64 {
    // Bits 63-56: symbol1 (literal byte or length code 257-285)
    // Bits 55-48: symbol2 (literal byte only, or 0xFF if single)
    // Bits 47-40: total_bits (combined length of both codes)
    // Bits 39-32: count (1 or 2)
    // Bits 31-0:  flags + extra bits info
}
```

### Step 9.3: Modify Table Building

In `src/libdeflate_entry.rs`, modify `LitLenTable::build()`:

```rust
// After building single-symbol entries, add pairs:
for code1 in 0..num_literals {
    let len1 = code_lengths[code1];
    if len1 == 0 || code1 >= 256 { continue; }  // Only pair literals
    
    for code2 in 0..num_symbols {
        let len2 = code_lengths[code2];
        if len2 == 0 || len1 + len2 > TABLE_BITS { continue; }
        
        // Combine: code1 followed by code2
        let combined_code = code1_reversed | (code2_reversed << len1);
        let combined_len = len1 + len2;
        
        // Fill all positions matching this combined pattern
        let fill_count = 1 << (TABLE_BITS - combined_len);
        for i in 0..fill_count {
            let index = combined_code | (i << combined_len);
            table[index] = LitLenEntry64::double(code1 as u8, code2 as u16, combined_len);
        }
    }
}
```

### Step 9.4: Modify Decode Loop

```rust
// In decode_huffman_libdeflate_style:
let entry = lookup!();
let count = entry.symbol_count();  // 1 or 2

if count == 2 && entry.is_double_literal() {
    // Fast path: write both literals, advance by total_bits
    let lit1 = entry.symbol1();
    let lit2 = entry.symbol2();
    unsafe {
        *out_ptr.add(out_pos) = lit1;
        *out_ptr.add(out_pos + 1) = lit2;
    }
    out_pos += 2;
    bitbuf >>= entry.total_bits() as u8;
    bitsleft -= entry.total_bits();
    entry = lookup!();
    continue;
}
// ... existing single-symbol path
```

### Step 9.5: Benchmark and Tune

```bash
cargo test --release bench_cf_silesia -- --nocapture
```

Expected improvement: 10-20% on SILESIA (which is 60%+ literals).

### Critical Considerations

1. **Table size doubles** (32-bit → 64-bit entries)
   - May increase L1 cache pressure
   - May need to reduce TABLE_BITS from 11 to 10

2. **Only pair LITERALS**
   - Don't pair length symbols (they need distance lookup)
   - Don't pair EOB symbol

3. **Fallback to single-symbol**
   - If pairs don't fit in LUT, use existing path

**Checkpoint:** `git commit -m "perf: ISA-L style multi-symbol table entries"`

---

## Phase 10: Runtime AVX2 Match Copy (MEDIUM PRIORITY)

Currently AVX2 match copying is compile-time only. Add runtime detection.

### Step 10.1: Add Runtime Detection

```rust
// In src/consume_first_decode.rs
#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHE.get_or_init(|| is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}
```

### Step 10.2: Add AVX2 Match Copy Function

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn copy_match_avx2(output: &mut [u8], pos: usize, dist: usize, len: usize) {
    // Use _mm256_loadu_si256 / _mm256_storeu_si256 for 32-byte copies
}
```

### Step 10.3: Dispatch at Runtime

```rust
fn copy_match_fast(output: &mut [u8], pos: usize, dist: usize, len: usize) {
    if has_avx2() && dist >= 32 && len >= 32 {
        unsafe { copy_match_avx2(output, pos, dist, len) }
    } else {
        copy_match_scalar(output, pos, dist, len)
    }
}
```

**Expected improvement:** 3-5% on x86_64 for match-heavy content.

**Checkpoint:** `git commit -m "perf: runtime AVX2 detection for match copy"`

---

## Current Status Checkboxes (Updated Jan 2026)

### Phase 1: Unified Entrypoint ✅ COMPLETE
- [x] `decompress_hyperion` exists and routes correctly
- [x] All tests pass (316 tests)
- [x] Performance maintained (91% of libdeflate)

### Phase 2: ANF Integration ⏸️ DEFERRED
- [x] ANF benchmarked (1.55x faster core ops)
- [ ] ANF integrated - BLOCKED: has correctness bugs
- Recommendation: Defer until simpler optimizations exhausted

### Phase 3: Parallel Single-Member ✅ COMPLETE
- [x] marker_turbo.rs created (2129 MB/s!)
- [x] Integrated into hyper_parallel.rs
- [x] Wired into hyperion router for >8MB files
- [ ] Debug high fallback rate (partial success)

### Phase 4: SIMD Parallel Huffman ⏸️ DEFERRED
- [ ] Only helps literal-heavy fixed blocks (niche)
- Recommendation: Low priority

### Phase 5: Unified Table
- [ ] Not yet benchmarked
- Recommendation: Try after Phase 6

### Phase 6: Tier 1 Optimizations ✅ ALREADY IMPLEMENTED
- [x] Entry preloading (lines 771-773, 810, 865)
- [x] Packed bitsleft subtraction (line 804: `bitsleft.wrapping_sub(entry)`)
- [x] Saved bitbuf pattern (line 788: `let saved_bitbuf = bitbuf`)

These were already in consume_first_decode.rs and responsible for 91% parity!

### Phase 9: ISA-L Multi-Symbol Tables ⚠️ COMPLEX
- [x] Studied ISA-L table building (igzip_inflate.c:400-550)
- [x] Added HUFFDEC_DOUBLE_LITERAL flag and entry format
- [x] Attempted table upgrade - FAILED (correctness issues)
- **Key learning:** Table replication means indices don't map directly to Huffman codes.
  ISA-L's algorithm is more subtle - need to validate that combined index
  actually decodes to the expected pair.
- **Current status:** Infrastructure added but disabled. Existing 8-literal
  batching achieves similar effect through sequential lookups.
- [ ] Revisit with ISA-L's exact algorithm (future work)

### Phase 10: Runtime AVX2 Match Copy
- [ ] Add runtime AVX2 detection (similar to BMI2)
- [ ] Add AVX2 match copy function
- [ ] Integrate into copy_match_fast

---

## Premortem: Top Failure Modes & Remediations

| # | What Goes Wrong | Historical Evidence | One-Line Remediation |
|---|-----------------|---------------------|----------------------|
| 1 | **Optimization that benchmarks fast in isolation is slow when integrated** | consume_first was 36% faster isolated, slower in full loop | Always benchmark on SILESIA after integration, not just isolated tests |
| 2 | **Parallel path produces wrong output at chunk boundaries** | hyper_parallel output differed at 4.1MB (CHUNK_SIZE) | Run `diff <(gzippy -d) <(gzip -d)` after every parallel change |
| 3 | **New module breaks existing passing tests** | Multiple reverts in history due to test failures | Run `cargo test --release` before AND after every commit |
| 4 | **Performance regresses without noticing** | Specialized decoder was 20% slower but shipped | Commit message MUST include benchmark numbers vs baseline |
| 5 | **LLM changes too many things at once, can't isolate failure** | Several "big bang" commits that needed reversion | ONE change per commit; if >50 lines changed, split it |
| 6 | **Fast speculative decoder still 10x slower than turbo** | SpeculativeDecoder at 70 MB/s vs 1400 MB/s | Reuse consume_first_decode hot path directly, don't rewrite |
| 7 | **Advanced math module (ANF/FSM) has subtle correctness bugs** | No evidence yet but high complexity risk | Golden test: decode 10 diverse files, compare byte-for-byte |
| 8 | **SIMD code works on dev machine, fails on CI (different CPU)** | Multiple CI failures from target-cpu=native | Never use target-cpu=native; test on both ARM and x86 |
| 9 | **L1 cache pressure from larger tables kills performance** | 12-bit table was same speed as 11-bit (cache pressure) | Profile cache misses before adding larger lookup tables |
| 10 | **Marker replacement corrupts output for large back-references** | Signed comparison bug in replace_markers_avx2 (`26f54c3`) | Use u32 for all marker math; add fuzzer for edge cases |

### Quick Reference: Emergency Recovery

```bash
# If tests fail after changes:
git stash && cargo test --release && git stash pop  # Verify it was your change

# If performance dropped:
git log --oneline -5  # Find last known-good
git checkout <sha> -- src/  # Restore source, keep docs

# If completely lost:
git checkout main && git checkout -b hyperion-attempt-N
```

---

## Guardrails for LLM Implementation

### When Starting Any Task

1. **Read the relevant test first**
   ```bash
   grep -n "fn test_" src/hyperion.rs | head -20
   ```

2. **Run baseline benchmark**
   ```bash
   cargo test --release bench_cf_silesia -- --nocapture 2>&1 | tail -5
   ```

3. **Make ONE small change**

4. **Run tests immediately**
   ```bash
   cargo test --release 2>&1 | tail -3
   ```

5. **If tests fail: revert and try differently**
   ```bash
   git checkout -- .
   ```

### When Confused

If you're unsure what to do next:

1. **Check the current state:**
   ```bash
   git status
   git log --oneline -5
   ```

2. **Run all benchmarks:**
   ```bash
   cargo test --release bench_ -- --nocapture 2>&1 | grep -E "(MB/s|M ops)"
   ```

3. **Read CLAUDE.md for current status:**
   ```bash
   head -100 CLAUDE.md
   ```

4. **Pick the next unchecked item from this plan**

### Error Recovery

If you've made changes that broke things:

```bash
# See what changed
git diff

# If it's a mess, start fresh from last good commit
git stash
git checkout main
git checkout -b fix-attempt-N

# Re-read this plan and start from Phase 1
```

---

## Success Metrics

### Phase 1 Complete When: ✅ DONE
- [x] `decompress_hyperion` exists and routes correctly
- [x] All tests pass (316 tests)
- [x] Performance unchanged (91% of libdeflate)

### Phase 2 Complete When: ⏸️ DEFERRED
- [x] ANF benchmarked (1.55x core speedup)
- [ ] ANF integrated - BLOCKED by correctness bugs
- Recommendation: Fix bugs later, focus on Tier 1 optimizations

### Phase 3 Complete When: ✅ DONE
- [x] marker_turbo.rs created (2129 MB/s)
- [x] Parallel single-member infrastructure working
- [x] Wired into hyperion router
- [ ] Debug fallback rate (partial)

### Phase 4 Complete When: ⏸️ DEFERRED
- [ ] SIMD parallel Huffman - niche benefit (fixed blocks only)
- Recommendation: Low priority

### Phase 5 Complete When:
- [ ] Unified table benchmarked
- [ ] Either integrated or documented as "not faster"

### Phase 6 Complete When: ✅ ALREADY DONE
- [x] Entry preloading implemented (lines 720-722, 1199-1201)
- [x] Packed bitsleft subtraction (line 753)
- [x] SILESIA throughput: 1402 MB/s (90.9% of libdeflate)
- [x] SOFTWARE: 18369 MB/s (93.3% of libdeflate)
- [x] LOGS: 8438 MB/s (101.2% - WE BEAT LIBDEFLATE!)

### Phase 7 Complete When:
- [ ] hyper_parallel fallback rate < 10%
- [ ] Cache alignment applied
- [ ] Prefetch implemented

### Phase 8 Complete When:
- [ ] ANF bugs fixed OR documented as unfixable
- [ ] JIT evaluated OR documented as not worth it

---

## The Ultimate Goal

### Current Status (Jan 2026)

**ARM (Apple M3) - Development Platform:**
```
               ┌──────────────────────────────────────────────────────┐
               │          ARM HYPERION PERFORMANCE                    │
               ├──────────────────────────────────────────────────────┤
               │  Dataset    Our MB/s   libdeflate   Ratio   Status   │
               ├──────────────────────────────────────────────────────┤
               │  SILESIA      1402       1543       90.9%   ⚡ CLOSE │
               │  SOFTWARE    18369      19680       93.3%   ✅ PARITY│
               │  LOGS         8438       8341      101.2%   ✅ WINS! │
               └──────────────────────────────────────────────────────┘
```

**x86_64 (Linux CI) - Production Platform:**
```
               ┌──────────────────────────────────────────────────────┐
               │          x86_64 HYPERION PERFORMANCE                 │
               ├──────────────────────────────────────────────────────┤
               │  Dataset    Our MB/s   Best Tool    Ratio   Status   │
               ├──────────────────────────────────────────────────────┤
               │  SILESIA T1   205.0    igzip 272.7   75%   ⚠️ GAP   │
               │  LOGS T1      407.6    rapidgzip 411 99%   ✅ PARITY│
               │  SOFTWARE Tmax 405.6   rapidgzip 403 100%  ✅ WINS! │
               └──────────────────────────────────────────────────────┘
```

### Target After Phase 6-7
```
               ┌──────────────────────────────────────────────────────┐
               │          TARGET HYPERION PERFORMANCE                 │
               ├──────────────────────────────────────────────────────┤
               │  Dataset    Target     libdeflate   Ratio   Gap      │
               ├──────────────────────────────────────────────────────┤
               │  SILESIA      1350       1385       97.5%   +85 MB/s │
               │  SOFTWARE    19500      19113      102%     ✅ EXCEED│
               │  LOGS         8000       7857      102%     ✅ EXCEED│
               │  BGZF (8T)    4000       N/A       110%+    ✅ WINS  │
               └──────────────────────────────────────────────────────┘
```

### Ultimate Vision
```
               ┌──────────────────────────────────────────────────────┐
               │          ULTIMATE HYPERION (POST PHASE 8)            │
               ├──────────────────────────────────────────────────────┤
               │  Dataset    Threads   Target      vs Best Tool       │
               ├──────────────────────────────────────────────────────┤
               │  SILESIA       1      1450        105% libdeflate    │
               │  SILESIA       8      5000        200% rapidgzip     │
               │  SOFTWARE      1     20000        105% libdeflate    │
               │  SOFTWARE      8     40000        fastest ever       │
               │  LOGS          1      8500        108% libdeflate    │
               │  LOGS          8     17000        fastest ever       │
               └──────────────────────────────────────────────────────┘
```

**We beat everyone, everywhere, always.**

---

## Commit Message Template

```
<type>: <description>

<body explaining what changed and why>

Benchmark Results:
- SILESIA: X MB/s (Y% of baseline)
- SOFTWARE: X MB/s (Y% of baseline)
- LOGS: X MB/s (Y% of baseline)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

Types: `feat`, `perf`, `fix`, `refactor`, `test`, `docs`

---

*This plan designed for robust LLM implementation. Each phase has clear entry/exit criteria, measurable checkpoints, and recovery procedures. Forward progress is guaranteed even if the specific implementation details change.*
