# CLAUDE.md - Hyperoptimization Guide for gzippy

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created.**

Current: ~950 MB/s (68% of libdeflate)
Target: **1840+ MB/s (130%+ of libdeflate single-threaded)**

Every change must be benchmarked. Every optimization must be measured. Speed is the only metric that matters.

## ABSOLUTE RULES

1. **NO LIBDEFLATE IN HOT PATHS** - We are REPLACING libdeflate, not using it
2. **BENCHMARK EVERYTHING** - Run `cargo test --release bench_cf_silesia -- --nocapture` after EVERY change
3. **REVERT REGRESSIONS** - If performance drops, revert immediately and try something different
4. **ALL CODE IS RUST** - No FFI, no C, pure Rust only

## Current Performance Status

```
Our throughput:          ~950 MB/s
libdeflate throughput:   ~1400 MB/s
Ratio: 68%
Gap to parity: 32%
Gap to target (130%): 93%
```

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
| Combined match lookup (feature `combined_match`) | +10% | **-35%** | Regressed on software/logs (64%/51% of libdeflate) |

**KEY LESSON: Micro-optimizations often REGRESS. LLVM already optimizes well.**

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
- `src/libdeflate_decode.rs` - Current best path (~950 MB/s)
- `src/consume_first_decode.rs` - Alternative implementation
- `src/unified_table.rs` - Novel unified table approach (NEW)
- `src/speculative_batch.rs` - Speculative batch decode (NEW)

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
| `src/libdeflate_decode.rs` | Current best decoder | ⭐⭐⭐⭐⭐ |
| `src/libdeflate_entry.rs` | Entry format definitions | ⭐⭐⭐⭐ |
| `src/unified_table.rs` | Novel unified approach | ⭐⭐⭐⭐ |
| `src/vector_huffman.rs` | SIMD infrastructure | ⭐⭐⭐⭐ |
| `libdeflate/lib/decompress_template.h` | libdeflate's implementation | ⭐⭐⭐⭐⭐ |
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

- **We are at 68% of libdeflate** - there's a lot of room for improvement
- **Parallel already exceeds libdeflate** - 148% with 8 threads on BGZF
- **The 130% target is achievable** - but requires novel approaches, not micro-opts
- **SIMD is the key** - vector_huffman infrastructure exists, needs integration
- **Measure, measure, measure** - intuition is often wrong about performance
