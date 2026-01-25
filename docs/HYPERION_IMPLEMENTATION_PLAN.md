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

### ⚡ DISCONNECTED MODULES (Ready but Not Integrated)

| Module | Purpose | Potential | Blocker |
|--------|---------|-----------|---------|
| `algebraic_decode.rs` | ANF branchless decode | +52% isolated | Integration complexity |
| `simd_parallel_decode.rs` | AVX2 speculative | +100% parallel | Lane synchronization |
| `unified_table.rs` | Single 64-bit table | Better cache | Not tested |
| `vector_huffman.rs` | 8-lane SIMD | +100% theory | Not integrated |
| `hyper_parallel.rs` | 4-phase pipeline | Parallel single | Marker bugs |
| `rapidgzip_decoder.rs` | Parallel single | 10x slower | Uses slow SpeculativeDecoder |
| `marker_decode.rs` | Marker-based spec | Parallel single | Needs fast speculative |

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

### Phase 1 Complete When:
- [ ] `decompress_hyperion` exists and routes correctly
- [ ] All tests pass
- [ ] Performance unchanged

### Phase 2 Complete When:
- [ ] ANF decoder integrated for low-entropy data
- [ ] LOGS dataset shows improvement
- [ ] SILESIA/SOFTWARE unchanged or better

### Phase 3 Complete When:
- [ ] Parallel single-member works
- [ ] SILESIA with 8 threads exceeds rapidgzip
- [ ] No correctness regressions

### Phase 4 Complete When:
- [ ] SIMD parallel Huffman working
- [ ] Measurable improvement on at least one dataset
- [ ] Falls back gracefully on non-SIMD platforms

### Phase 5 Complete When:
- [ ] Unified table benchmarked
- [ ] Either integrated or documented as "not faster"

---

## The Ultimate Goal

```
               ┌─────────────────────────────────────────────┐
               │          HYPERION PERFORMANCE               │
               ├─────────────────────────────────────────────┤
               │  Dataset    Threads   MB/s    vs Best Tool  │
               ├─────────────────────────────────────────────┤
               │  SILESIA       1      1500     108% libdef  │
               │  SILESIA       8      5000     200% rapidgz │
               │  SOFTWARE      1     25000     120% libdef  │
               │  SOFTWARE      8     50000     200% fastest │
               │  LOGS          1     10000     125% libdef  │
               │  LOGS          8     20000     200% fastest │
               └─────────────────────────────────────────────┘
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
