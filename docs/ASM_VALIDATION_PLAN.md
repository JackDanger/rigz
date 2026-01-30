# ASM Validation Plan: Why Isn't Our Machine Code Competitive?

## The Problem

Our hand-written ARM64 assembly (v3) achieves only ~64% of LLVM's generated code.
This should be impossible if we're doing the same work - we control the exact instructions.

## VALIDATED: Subtables Are NOT The Bottleneck

Analysis of SILESIA dynamic blocks shows:
- **Only 0.4% of table entries are subtable pointers**
- Fixed Huffman has 0% subtable entries
- The v3 ASM exits to Rust on subtable entries, but this is rare

**Conclusion: The 36% performance gap is NOT from subtable fallbacks.**

## Root Cause Hypothesis (Updated)

**One of these MUST be true:**
1. ~~We're not running our ASM~~ - Fallback paths are hit too often → **RULED OUT** (only 0.4%)
2. **We're doing MORE work** - Extra instructions, redundant operations  
3. **We're doing work BADLY** - Poor instruction scheduling, cache misses
4. **Measurement is wrong** - Thermal throttling, other processes

## Current Measurements (Jan 2026)

### Performance
- **Rust (libdeflate-style):** ~1007 MB/s
- **v3 ASM + Rust fallback:** ~765 MB/s  
- **Ratio:** 76% of Rust baseline

### Instruction Counts
- **LLVM-generated decode function:** ~1308 instructions (first 1500 lines)
- **Our v3 ASM block:** ~178 inline assembly instructions

### Key Insight
LLVM generates significantly more instructions but achieves better performance.
This suggests LLVM's instruction scheduling and register allocation is superior,
or our ASM has hidden overhead (memory stalls, branch mispredictions).

## Likely Root Causes

### 1. Instruction Scheduling
LLVM can schedule instructions to hide memory latency. Our hand-written ASM
may have dependencies that stall the pipeline.

### 2. Register Pressure  
Our v3 uses scratch registers x10-x17. LLVM might allocate better.

### 3. Match Copy Inefficiency
The match copy loop in v3 may be slower than LLVM's optimized version.
LLVM likely uses SIMD more effectively.

### 4. Hidden Overhead
The ASM block entry/exit may have overhead not visible in instruction count.
LLVM inlines aggressively and eliminates function call overhead.

## Validation Strategy

### Phase 1: Verify We're Actually Running ASM

**Goal:** Confirm what percentage of decode time is spent in our ASM vs fallback.

```bash
# Create an instrumented build that counts path executions
cargo test --release test_asm_path_coverage -- --nocapture
```

**Metrics to collect:**
- Count of ASM fast loop iterations
- Count of fallback calls (subtable, EOB)
- Bytes decoded in ASM vs fallback
- Time spent in ASM vs fallback

### Phase 2: Instruction-for-Instruction Comparison

**Goal:** Compare our ASM to LLVM's generated code for the SAME algorithm.

#### Step 2a: Extract LLVM's Assembly

```bash
# Generate assembly for the Rust decode loop
RUSTFLAGS="--emit asm" cargo build --release

# Or with better formatting:
cargo rustc --release -- --emit=asm -C "llvm-args=-x86-asm-syntax=intel"

# Find the hot function
grep -n "decode_huffman" target/release/deps/*.s
```

#### Step 2b: Create Side-by-Side Comparison

For each section of the hot loop, document:

| Operation | LLVM Instructions | Our Instructions | Delta |
|-----------|-------------------|------------------|-------|
| Refill check | `cmp w10, #48` | `cmp w10, #48` | 0 |
| Load word | `ldr x11, [x0, x1]` | `ldr x11, [ptr, pos]` | 0 |
| Shift | `lsl x11, x11, x10` | `lsl x11, x11, x10` | 0 |
| ... | ... | ... | ... |

**CRITICAL**: Count total instructions in fast loop for each.

### Phase 3: Micro-benchmark Individual Operations

**Goal:** Isolate which operation is slow.

```rust
// Benchmark just refill (1M iterations)
#[bench] fn bench_refill_only()

// Benchmark just lookup (1M iterations)  
#[bench] fn bench_lookup_only()

// Benchmark just consume (1M iterations)
#[bench] fn bench_consume_only()

// Benchmark literal write (1M iterations)
#[bench] fn bench_literal_write()

// Benchmark match copy (1M iterations, various lengths)
#[bench] fn bench_match_copy()
```

### Phase 4: Hardware Performance Counters

**Goal:** Identify microarchitectural bottlenecks.

#### On macOS (M3):
```bash
# Use Instruments with CPU Counters template
xcrun xctrace record --template 'CPU Counters' --launch -- \
    ./target/release/gzippy -d < silesia.tar.gz > /dev/null
```

#### On Linux:
```bash
# Use perf stat for hardware counters
perf stat -e cycles,instructions,L1-dcache-load-misses,branch-misses \
    ./target/release/gzippy -d < silesia.tar.gz > /dev/null
```

**Counters to compare (ASM vs Rust baseline):**
- Instructions per cycle (IPC)
- L1 cache miss rate
- Branch misprediction rate
- Stall cycles (frontend vs backend)

### Phase 5: Minimal Reproducer

**Goal:** Create smallest possible test case.

```rust
// Test decode of 1KB that hits only main-table literals
#[test] fn test_asm_literals_only()

// Test decode of 1KB that hits only main-table lengths
#[test] fn test_asm_lengths_only()

// Test decode that forces subtable hits
#[test] fn test_asm_subtables_only()
```

## Known Issues (UPDATED based on analysis)

### Issue 1: Subtable Fallback (MINOR - was MAJOR)

Current v3 exits to Rust on ANY subtable entry. **However, analysis shows:**
- Only **0.4%** of dynamic block table entries are subtable pointers
- Fixed Huffman has **0%** subtable entries
- This is NOT the cause of the 24% performance gap

**Status:** Low priority - implement for completeness but won't fix the gap.

### Issue 2: Match Copy IS Implemented ✓

The v3 ASM includes full match copy implementation:
- 16-byte copies with `ldp`/`stp` for distance ≥ 16
- 8-byte copies for distance ≥ 8  
- Byte-by-byte for overlapping or remainder

**Status:** Implemented. May need optimization but is not missing.

### Issue 3: The REAL Problem - Instruction Quality

LLVM generates ~1308 instructions, we have ~178. Yet LLVM is faster.
This means our ASM is **per-instruction slower**, likely due to:

1. **Poor instruction scheduling** - Dependencies stall the pipeline
2. **Memory access patterns** - LLVM may prefetch better
3. **SIMD usage** - LLVM uses NEON/AVX more effectively for match copy
4. **Branch prediction** - LLVM knows branch patterns better

### Issue 4: Potential Register Pressure

We use many temporaries (`x10-x17`). LLVM may allocate differently.

## Implementation: Path Coverage Test

```rust
#[test]
fn test_asm_path_coverage() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    static ASM_LITERALS: AtomicUsize = AtomicUsize::new(0);
    static ASM_LENGTHS: AtomicUsize = AtomicUsize::new(0);
    static ASM_SUBTABLES: AtomicUsize = AtomicUsize::new(0);
    static FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);
    
    // Modify decode_huffman_asm_v3 to increment counters
    // Run on silesia
    // Print results
}
```

## Implementation: Instruction Comparison Script

```bash
#!/bin/bash
# compare_asm.sh

# 1. Generate LLVM's assembly
cargo rustc --release -- --emit=asm

# 2. Extract decode_huffman function
# 3. Disassemble our inline asm
# 4. Generate side-by-side diff
```

## Success Criteria

**We know we've found the problem when:**
1. Instruction counts match (±10%) AND performance differs → microarch issue
2. Instruction counts differ significantly → algorithm/implementation issue
3. Path coverage shows >20% fallback → complete ASM implementation needed
4. Hardware counters show high cache misses → data structure issue

## Next Steps (Priority Order - UPDATED)

### High Priority: Understand Why LLVM Wins

1. **Profile with hardware counters** - Get IPC, cache misses, branch mispredictions
   ```bash
   # On Linux
   perf stat -e cycles,instructions,branch-misses,L1-dcache-load-misses ...
   
   # On macOS  
   xcrun xctrace record --template 'CPU Counters' ...
   ```

2. **Analyze LLVM's decode loop** - Understand its instruction scheduling
   - Look at how it interleaves loads with computation
   - Check SIMD usage for match copy
   - Study branch patterns

3. **Benchmark match copy in isolation** - Is this the bottleneck?
   - Create synthetic test with only length/distance codes
   - Compare LLVM's copy loop vs our 16/8/1-byte loops

### Medium Priority: Improve ASM Quality

4. **Instruction scheduling** - Reorder our ASM to hide latency
   - Start next load before consuming current data
   - Interleave independent operations

5. **SIMD match copy** - Use NEON for 32/64-byte copies
   - LLVM's generated code uses NEON; we should too

6. **Reduce register pressure** - Use fewer scratch registers
   - Analyze which registers LLVM uses

### Low Priority: Completeness

7. **Implement subtable handling** - Only 0.4% but for correctness
8. **x86_64 ASM** - Port the optimizations to x86

## Key Insight

> "LLVM has 40+ years of compiler optimization research.
> We can't beat it by writing 'similar' code - we need to do something DIFFERENT."

Options for actually beating LLVM:
1. **Algorithmic innovation** - Multi-symbol decoding, table pre-computation
2. **Microarchitectural exploitation** - Specific M3/x86 tricks LLVM doesn't know
3. **Profile-guided** - Use runtime info LLVM doesn't have
4. **Avoid the problem** - Let LLVM generate the hot loop, optimize elsewhere

## Recommendation

**Consider abandoning hand-written ASM for the decode loop.**

Our analysis shows:
- v3 ASM achieves 76% of Rust baseline
- Subtables are only 0.4% of lookups (not the issue)
- LLVM's 1308 instructions beat our 178 instructions

This suggests LLVM's optimization is fundamentally better for this workload.
Instead of fighting LLVM, focus on:
1. Algorithmic improvements (ISA-L multi-symbol, table caching)
2. Parallel decompression (where we already win)
3. Memory/cache optimizations

The decode hot loop may be a case where letting the compiler win is the right choice.
