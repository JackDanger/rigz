# CLAUDE.md - Hyperoptimization Guide for gzippy

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

**ACHIEVED: 91-101% of libdeflate in pure Rust!**

Current: **1402 MB/s on SILESIA (91% of libdeflate), 8438 MB/s on LOGS (101%!)**
Status: **WE BEAT LIBDEFLATE ON LOGS!** Near parity on all other datasets.

Every change must be benchmarked. Every optimization must be measured. Speed is the only metric that matters.

## ABSOLUTE RULES

1. **NO LIBDEFLATE IN HOT PATHS** - We are REPLACING libdeflate, not using it
2. **BENCHMARK EVERYTHING** - Run `./bench-decompress.sh` after EVERY change
3. **REVERT REGRESSIONS** - If performance drops, revert immediately and try something different
4. **ALL CODE IS RUST** - No FFI, no C, pure Rust only

## Current Performance Status (Jan 2026)

### ARM (Apple M3) - Primary Development Platform
```
Dataset          Our MB/s    libdeflate MB/s    Ratio
SILESIA          1402        1543               90.9%   ✓ NEAR PARITY
SOFTWARE         18369       19680              93.3%   ✓ AT PARITY
LOGS             8438        8341               101.2%  ✓ WE WIN!

Decoder: consume_first_decode.rs → decode_huffman_libdeflate_style
Status: 91-101% of libdeflate - WE BEAT LIBDEFLATE ON LOGS!
```

### x86 (Intel i7-13700T) - Secondary Platform
```
Dataset          Our MB/s    libdeflate MB/s    Ratio      Notes
SILESIA          580-620     620-680            ~90-98%    High variance
SOFTWARE         1900-2300   1900-2200          100-115%   ✓ Often exceeds
LOGS             1500-2000   1700-2300          ~90-95%    High variance

Status: Near parity, high variance due to 35W TDP thermal throttling
```

**x86 Observations:**
- BMI2 `bzhi` is used (verified in assembly)
- High measurement variance (~10-20%) due to thermal throttling
- 8-literal batching works well; 2-3 literal batching hurts SOFTWARE
- 5-word match copy unrolling hurts cache performance

### Key Optimizations That Worked

1. **Libdeflate-style decode loop** - Exact match of libdeflate's algorithm
2. **8-literal batching** - Decode up to 8 literals before refill
3. **Packed writes** - Write 2/4/8 literals with single u16/u32/u64 store
4. **saved_bitbuf pattern** - Extract extra bits from pre-shift buffer
5. **bitsleft -= entry** - Full subtract trick (not masked!)
6. **Preload pattern** - Preload next entry before writing current
7. **Branchless refill** - Exact libdeflate refill pattern
8. **AVX2 match copy** - 64-byte copies for large matches

### What HURT Performance (disabled)

- **Specialized decoder** - Slower for match-heavy content (SOFTWARE, LOGS)
- The inline extra-bits handling added overhead vs saved_bitbuf pattern

## What Has Been Tried and FAILED

| Optimization | Expected | Actual | Why |
|--------------|----------|--------|-----|
| DoubleLitCache per block | +15% | **-73%** | Build cost exceeds decode gain |
| `#[cold]` on errors | +5% | **-4%** | Function call overhead |
| Table-free fixed Huffman | +20% | **-325%** | Bit reversal overhead |
| Unconditional refill | +5% | **-12%** | Conditional was faster |
| Multi-symbol (augment style) | +15% | **-3%** | Double lookup overhead |
| Speculative batch (basic) | +28% | **-1%** | Still sequential lookups |
| Adaptive bloom filter | +10% | **-2%** | Tracking overhead |
| `bitsleft -= entry` (full) | +5% | **BROKE** | Refill shift corrupted by high bytes |
| copy_match 5-word unroll | +10% | **-15%** | Overwrites data used by later matches |
| Combined match lookup | +20% | **-10%** | Extra table lookups canceled gains |
| x86 2-3 literal batching | +5% | **-20%** | Hurts SOFTWARE; libdeflate's advice is x86-specific |
| x86 5-word loop unroll | +5% | **-10%** | Extra writes hurt cache on short matches |
| Hand-written ASM (v1-v4) | +30% | **-30%** | LLVM's scheduling/allocation beats hand-written |

**KEY LESSON: Micro-optimizations often REGRESS. LLVM already optimizes well.**

## Hand-Written ASM Status (Jan 2026)

### Four ASM decoder versions implemented:

| Version | Approach | Performance | Notes |
|---------|----------|-------------|-------|
| v1 | ASM primitives | 50% of Rust | Function call overhead |
| v2 | Inline macros | 64% of Rust | Less overhead but still slow |
| v3 | Pure ASM loop | 71% of Rust | Single asm! block |
| v4 | LLVM-parity | 70% of Rust | Correct, full-featured |

### v4 Features:
- ✅ Literal decoding with 4-literal batching
- ✅ Length decoding with extra bits
- ✅ Distance decoding (main + subtable)
- ✅ SIMD match copy (32-byte LDP/STP)
- ✅ 8-byte and byte-by-byte fallback for overlap
- ✅ Full correctness on SILESIA (212M bytes)

### Why Hand-Written ASM is Slower:
1. **LLVM's instruction scheduling** is better at hiding latency
2. **LLVM's register allocation** avoids spills
3. **LLVM preloads entries** before consuming current
4. **Inline ASM constraints** limit optimization opportunities

### Conclusion:
Hand-written ASM achieves 70% of LLVM's performance. The effort is better spent on:
- Algorithmic improvements (parallel decode)
- Better data structures (unified tables)
- Profile-guided optimization

## ISA-L Implementation Status (Jan 2026)

### Implementation Complete ✓

The ISA-L algorithm is now fully implemented in `src/isal_decode.rs`:

| Feature | Status | Notes |
|---------|--------|-------|
| Entry format (multi-symbol packing) | ✅ Working | 1, 2, or 3 symbols per entry |
| 11-bit main table | ✅ Working | Same size as libdeflate |
| Subtable support | ✅ Working | For codes > 11 bits |
| All block types | ✅ Working | Stored, fixed, dynamic Huffman |
| Multi-block files | ✅ Working | Fixed bit buffer alignment bug |
| Full SILESIA decode | ✅ Working | 212 MB decoded correctly |

### Performance Comparison

| Path | SILESIA Throughput | % of libdeflate |
|------|-------------------|-----------------|
| libdeflate (reference C) | 1464 MB/s | 100% |
| consume_first_decode.rs | 1278 MB/s | **87%** |
| isal_decode.rs | 358 MB/s | **24%** |

**ISA-L is 3.6x slower than our libdeflate-style path!**

### Why ISA-L is Slower

The current ISA-L implementation has overhead from:
1. **Extensive error checking** - Bounds checks on every iteration
2. **No packed writes** - Writing 1 byte at a time vs u32/u64 stores
3. **No preload pattern** - Sequential lookup/consume/write instead of overlapped
4. **Sequential multi-symbol** - Processing symbols in a loop vs unrolled

### What ISA-L Gets Right

Despite slower overall, ISA-L has advantages:
1. **Cleaner entry format** - sym_count field is explicit, not inferred from flags
2. **Multi-symbol packing** - 2-3 literals decoded per lookup (when applicable)
3. **Pre-expanded lengths** - C version pre-computes length+extra into table

### Future Optimization Plan

To make ISA-L competitive, port these libdeflate optimizations:
1. Remove hot-loop guards (trust table building correctness)
2. Add packed literal writes (u16/u32/u64 for consecutive literals)
3. Add preload pattern (lookup next entry while writing current)
4. Unroll multi-symbol processing (inline sym1/sym2/sym3)
5. Pre-expand length codes like real ISA-L does

## ISA-L Implementation Status (Jan 2026)

A complete ISA-L-style decoder is available in `src/isal_decode.rs`:

| Feature | Status |
|---------|--------|
| Pre-expansion | ✅ length = sym - 254 (eliminates extra bit reads) |
| Multi-symbol entries | ✅ Works but DISABLED (build overhead > decode benefit) |
| Subtable support | ✅ Up to 20-bit codes for pre-expanded symbols |
| All 325 tests | ✅ Passing |

**Performance**: 530 MB/s (38% of libdeflate C, up from 334 MB/s initial)

The ISA-L path demonstrates pre-expansion correctly but is slower than libdeflate-style
due to entry format differences and micro-optimization gaps. Multi-symbol table building
is O(n²)/O(n³) and was found to hurt performance on match-heavy data like SILESIA.

## Multi-Threaded Decompression Status (Jan 2026)

### Benchmark Results vs rapidgzip (M3, 14 threads)

| Dataset | gzippy | rapidgzip | Ratio | Status |
|---------|--------|-----------|-------|--------|
| **LOGS** | 1749 MB/s | 691 MB/s | **253%** | ✓ WE WIN by 2.5x |
| **SOFTWARE** | 2659 MB/s | 3065 MB/s | 87% | Close |
| **SILESIA** | 856 MB/s | 2464 MB/s | 35% | ✗ Need parallel |

### BREAKTHROUGH: marker_turbo.rs (Jan 2026)

**Fast Marker-Based Decoder**: 2129 MB/s (30x faster than old 70 MB/s MarkerDecoder!)

The key insight: reuse the fast Bits struct from consume_first_decode but output
to u16 buffer with markers for unresolved back-references.

**Now fully integrated:**
- `inflate_with_markers_at()` - Start at any bit position for parallel chunks
- `hyper_parallel.rs` - Uses marker_turbo for speculative parallel decode
- `hyperion.rs` - Routes large single-member files (>8MB) to hyper_parallel

**Parallel potential:**
- 8 threads × 2129 MB/s = 17032 MB/s theoretical
- vs 1400 MB/s single-thread turbo_inflate
- **12x parallel speedup potential!**

### Why rapidgzip Wins on SILESIA

rapidgzip uses **parallel single-member decompression**:
1. Find deflate block boundaries speculatively
2. Decode each block with markers for unresolved back-references
3. Propagate windows between chunks
4. Resolve markers in parallel

We have the infrastructure (`rapidgzip_decoder.rs`, `hyper_parallel.rs`, `marker_decode.rs`)
but it's **10x slower** than our sequential turbo inflate:
- `SpeculativeDecoder`: ~70 MB/s
- `inflate_into_pub`: ~1300 MB/s

### Blocker for Parallel Single-Member

The speculative decoder must match our sequential speed to be worthwhile.
Two-pass decode (sequential boundary finding + parallel re-decode) only helps
if the second pass is >2x faster than sequential.

### What Works for Parallel

- **BGZF files** (gzippy output): Full parallel via embedded block sizes
- **Multi-member gzip** (pigz output): Parallel per-member
- **Highly-compressible data** (LOGS, SOFTWARE): Our sequential is already 2.5x faster

## What MIGHT Work (Untried or Partially Tried)

### Tier 1: High Probability (Try These First)
1. **Unified Table** - Single table for ALL symbol types, no fallback (`src/unified_table.rs`)
2. **True SIMD Lanes** - Decode 8 symbols in parallel using AVX2/NEON (`src/vector_huffman.rs`)
3. **JIT Code Generation** - Generate native code for repeated Huffman tables
4. **BMI2 PEXT/PDEP** - Hardware bit extraction (`src/bmi2.rs`)

### Tier 2: Medium Probability
5. **Precomputed Decode Sequences** - 16-22 bit lookup → multiple symbols
6. **Two-Pass with Table Caching** - Scan file, fingerprint tables, cache
7. **Branch hints** - `likely()/unlikely()` macros

### Tier 3: Radical Ideas
8. **GPU Offload** - Use tensor cores for parallel table lookup
9. **FPGA Huffman FSM** - Hardware Huffman decoder
10. **Custom RISC-V Instructions** - `huffdec` opcode

## Architecture Overview

### Main Decode Paths
- `src/consume_first_decode.rs` - **PRODUCTION PATH** (~1400 MB/s, 99-114% of libdeflate)
  - `decode_huffman_libdeflate_style()` - Main hot path for all blocks
- `src/libdeflate_decode.rs` - Entry format definitions (LitLenEntry, DistEntry)
- `src/bgzf.rs:inflate_into_pub` - Entry point for decompression

### Why We Achieved Parity

Key breakthroughs:
1. **Exact libdeflate algorithm** - Copied the decompress_template.h patterns exactly
2. **bitsleft -= entry trick** - Full subtract works when high bits are garbage
3. **8-literal batching** - More aggressive than libdeflate's 2-3 literal unroll
4. **Disabled specialized decoder** - Was slower for match-heavy content
5. **AVX2 match copy** - SIMD for large non-overlapping matches

### Key Data Structures
- `src/libdeflate_entry.rs` - LitLenEntry, DistEntry formats
- `src/multi_symbol.rs` - Multi-symbol table (not integrated)
- `src/vector_huffman.rs` - SIMD infrastructure

### Benchmarks
```bash
# Main benchmark (use this for all testing)
cargo test --release bench_cf_silesia -- --nocapture

# Multi-dataset benchmark
cargo test --release bench_diversity -- --nocapture

# Parallel benchmark (BGZF)
cargo test --release bench_bgzf -- --nocapture
```

## How to Make Progress

### Step 1: Understand the Hot Path
Read `src/libdeflate_decode.rs:607-791` (decode_huffman function). This is where 90%+ of time is spent.

### Step 2: Profile
```bash
cargo build --release
perf record ./target/release/gzippy -d < silesia.tar.gz > /dev/null
perf report
```

### Step 3: Try ONE Thing
Make ONE small change, benchmark, compare. Never change multiple things at once.

### Step 4: Document Results
Update this file with what you tried and the result.

## Key Insights from libdeflate/rapidgzip Analysis

### libdeflate's Tricks (decompress_template.h)
1. **saved_bitbuf pattern** - Save bits BEFORE consuming for extra bit extraction
2. **Multi-literal in fastloop** - Decode 2-3 literals when bits available
3. **CAN_CONSUME_AND_THEN_PRELOAD** - Compile-time bit budget check
4. **Preload during write** - Start next lookup before writing current
5. **EXTRACT_VARBITS8** - Cast to u8 before shift

### rapidgzip's Tricks (HuffmanCodingShortBitsMultiCached.hpp)
1. **CacheEntry** - `needToReadDistanceBits` flag, `symbolCount`, packed symbols
2. **DISTANCE_OFFSET** - Combine length+distance in symbol space
3. **Marker-based parallel** - Find block boundaries, decode in parallel

### What We Have That They Don't
1. **Vector Huffman infrastructure** - 8-lane SIMD ready (`src/vector_huffman.rs`)
2. **JIT table cache** - Framework for caching compiled tables
3. **Unified table** - Single table for all symbol types (`src/unified_table.rs`)
4. **Speculative batch** - CPU-speculative-execution inspired decode

## Files to Study

| File | Purpose | Priority |
|------|---------|----------|
| `src/consume_first_decode.rs` | Current best decoder (90% libdeflate) | ⭐⭐⭐⭐⭐ |
| `src/isal_decode.rs` | ISA-L style decoder with pre-expansion (530 MB/s) | ⭐⭐⭐⭐ |
| `src/hyperion.rs` | Unified routing entrypoint | ⭐⭐⭐⭐⭐ |
| `src/marker_turbo.rs` | Fast parallel marker decoder (2129 MB/s!) | ⭐⭐⭐⭐⭐ |
| `src/hyper_parallel.rs` | Parallel single-member via marker_turbo | ⭐⭐⭐⭐ |
| `src/libdeflate_entry.rs` | Entry format definitions | ⭐⭐⭐⭐ |
| `src/unified_table.rs` | Novel unified approach | ⭐⭐⭐ |
| `libdeflate/lib/decompress_template.h` | libdeflate's implementation | ⭐⭐⭐⭐⭐ |
| `isa-l/igzip/igzip_inflate.c` | ISA-L reference implementation | ⭐⭐⭐⭐ |
| `rapidgzip/huffman/HuffmanCodingShortBitsMultiCached.hpp` | rapidgzip's optimization | ⭐⭐⭐⭐ |

## Test Commands

```bash
# Quick correctness test
cargo test --release test_silesia

# Main performance benchmark
cargo test --release bench_cf_silesia -- --nocapture

# All tests
cargo test --release

# Clippy (required before commit)
cargo clippy

# Format
cargo fmt
```

## Commit Guidelines

1. Every commit must pass clippy
2. Every commit must be formatted with cargo fmt
3. Performance changes must include benchmark results in commit message
4. Use conventional commits: `feat:`, `fix:`, `perf:`, `refactor:`

## Remember

- **We MATCH or EXCEED libdeflate** - 99% on SILESIA, 106-114% on SOFTWARE/LOGS
- **Pure Rust achieved parity** - No FFI, no unsafe C code needed
- **Parallel exceeds libdeflate** - 148% with 8 threads on BGZF
- **The 130% target is partially achieved** - SOFTWARE and LOGS exceed it!
- **Simpler is often faster** - Specialized decoder was SLOWER than libdeflate-style
- **Measure, measure, measure** - The specialized path regression was unexpected

---

## HYPERION: Next-Generation Implementation Plan

See `docs/HYPERION_IMPLEMENTATION_PLAN.md` for the full LLM-friendly implementation guide.

### Summary of 55+ Commit History Analysis

**What ALWAYS Works:**
1. Copy libdeflate's algorithm exactly, then optimize
2. Benchmark before AND after every change
3. Revert immediately if performance drops
4. Keep the hot path simple (nested conditionals kill performance)

**What NEVER Works:**
1. JIT compilation (compile time > decode time)
2. Statistical prediction (prediction overhead > benefit)
3. Per-block table building (build cost > decode gain)
4. Replacing table lookups with computation (L1 cache beats compute)

**The Parallel Single-Member Blocker:**
The speculative decoder runs at ~70 MB/s while turbo_inflate runs at ~1400 MB/s.
For 8-thread parallel to beat sequential:
```
speculative_speed × 8 > 1400 MB/s
speculative_speed > 175 MB/s (currently 70, need 2.5x improvement)
```

**Solution:** Create `turbo_inflate_with_markers` that reuses the consume_first_decode
hot path but outputs to u16 buffer with markers. This should achieve 500+ MB/s,
making parallel worthwhile.

### Disconnected Advanced Modules (Ready to Integrate)

| Module | Status | Next Step |
|--------|--------|-----------|
| `algebraic_decode.rs` | 1.52x faster isolated | Integrate for fixed Huffman |
| `simd_parallel_decode.rs` | Infrastructure ready | Add AVX2 gather/scatter |
| `unified_table.rs` | Exists | Benchmark vs separate tables |
| `vector_huffman.rs` | 8-lane ready | Wire into decode loop |
| `hyper_parallel.rs` | Broken | Fix marker bugs |
| `marker_decode.rs` | Works | Needs fast speculative decoder |

### The HYPERION Goal

```
Beat EVERY tool on EVERY dataset at EVERY thread count.

SILESIA:  1 thread → 1500 MB/s,  8 threads → 5000 MB/s
SOFTWARE: 1 thread → 25000 MB/s, 8 threads → 50000 MB/s
LOGS:     1 thread → 10000 MB/s, 8 threads → 20000 MB/s
```

---

## Comprehensive Plan: Surpassing ALL Decompression Tools

### Current Decoder Inventory (Jan 2026)

| Module | Throughput | Status | Best For |
|--------|-----------|--------|----------|
| `consume_first_decode.rs` | 1278 MB/s | ✅ Production | All datasets |
| `isal_decode.rs` | 358 MB/s | ✅ Working | Research/comparison |
| `marker_turbo.rs` | 2129 MB/s | ✅ Working | Parallel chunks |
| `algebraic_decode.rs` | 1.52x isolated | ⚠️ Not integrated | Fixed Huffman |
| `vector_huffman.rs` | - | ⚠️ Infrastructure | Future SIMD |
| `hyper_parallel.rs` | - | ⚠️ Needs work | Large single-member |

### Strategy for Each Scenario

#### 1. Single-Thread Sequential (Current Focus)

**Target: 1500+ MB/s on SILESIA (115% of libdeflate)**

Path: Optimize `consume_first_decode.rs`

Remaining opportunities:
- [ ] Integrate algebraic_decode for fixed Huffman blocks (1.52x faster)
- [ ] Pre-expand length codes like ISA-L (eliminate extra bit reads)
- [ ] Profile-guided branch layout
- [ ] BMI2 `pext`/`pdep` for bit extraction on x86

#### 2. Multi-Thread Parallel (BGZF/Pigz Format)

**Target: 10000+ MB/s with 8 threads**

Path: `parallel_decompress.rs` → `marker_turbo.rs`

Status: Already **2.5x faster** than rapidgzip on LOGS!

Opportunities:
- [ ] Adaptive chunk sizing based on compression ratio
- [ ] Lock-free work stealing for better load balance
- [ ] Memory-mapped I/O to reduce copies

#### 3. Multi-Thread Single-Member (The Hard Problem)

**Target: 5000+ MB/s on SILESIA with 8 threads**

Path: `hyper_parallel.rs` → speculative block-parallel decode

Blocker: Speculative decoder is 20x slower than sequential.

Solution path:
1. Create `turbo_inflate_with_markers()` - Fast decode that outputs markers
2. Parallel speculative decode from multiple block boundaries
3. Propagate window state between chunks
4. Resolve markers in parallel

#### 4. Hybrid Approach (Best of All)

**The Ultimate gzippy:**

```
Input Analysis:
├── BGZF/Pigz format? → Parallel member decode (10000+ MB/s)
├── Single member < 1MB? → Sequential turbo (1500 MB/s)
├── Single member 1-8MB? → Sequential turbo (no parallel overhead)
├── Single member > 8MB? → Speculative parallel (5000+ MB/s)
└── Adaptive selection based on compression ratio
```

### Competitive Landscape

| Tool | Best Case | Technique | Our Advantage |
|------|-----------|-----------|---------------|
| **libdeflate** | 1464 MB/s (SILESIA) | Optimal table format, hand-tuned | Pure Rust, parallel |
| **ISA-L** | ~1000 MB/s | Multi-symbol, SIMD | Better integration |
| **zlib-ng** | ~800 MB/s | SIMD memcpy | Much faster |
| **rapidgzip** | 2464 MB/s parallel | Block-parallel | Beat on LOGS |
| **pigz** | ~400 MB/s | Simple parallel | 5x faster |

### Next Steps (Priority Order)

1. **Optimize ISA-L hot loop** - Remove guards, add packed writes
2. **Integrate algebraic_decode** - For fixed Huffman (10-15% files)
3. **Create turbo_inflate_with_markers** - Enable true parallel single-member
4. **Profile-guided optimization** - Branch prediction tuning
5. **SIMD exploration** - vector_huffman.rs integration

### Success Metrics

We will have surpassed ALL tools when:
- [ ] Single-thread SILESIA > 1500 MB/s (115% of libdeflate)
- [ ] Multi-thread SILESIA > 5000 MB/s (beats rapidgzip's 2464)
- [ ] Multi-thread LOGS > 15000 MB/s (6x current)
- [ ] Multi-thread BGZF > 20000 MB/s (memory bandwidth limited)
