# Gap Analysis & Optimization Plan

**Current: 992 MB/s (69.7%) | Target: 1840+ MB/s (130%+)**

---

## Detailed Gap Analysis

### libdeflate vs Us (decompress_template.h)

| Feature | libdeflate | Us | Gap |
|---------|-----------|-----|-----|
| `CAN_CONSUME_AND_THEN_PRELOAD` macro | ✓ Compile-time bit budget | ✗ Runtime checks | 5-10% |
| Triple-literal in fastloop | ✓ 3 fast lits + preload | ✓ 5 lits but nested | ~same |
| `bitsleft -= entry` (full u32) | ✓ No mask | ✓ Done | ~same |
| `REFILL_BITS_IN_FASTLOOP` | ✓ Branchless | ✓ Done | ~same |
| `EXTRACT_VARBITS8` macro | ✓ 8-bit cast optimization | ✗ Not used | 1-2% |
| `out_fastloop_end` pre-calc | ✓ Bounds in registers | ✓ Done | ~same |
| `unlikely()` branch hints | ✓ Everywhere | ✗ Missing | 2-5% |
| Subtable inline in main array | ✓ Cache locality | ✓ Done | ~same |
| BMI2 EXTRACT_VARBITS | ✓ `_bzhi_u64` | ✓ Ready, N/A ARM | N/A |
| Copy with 5-word overwrite | ✓ 40 bytes | ✓ Done | ~same |

### rapidgzip vs Us (HuffmanCodingShortBitsMultiCached.hpp)

| Feature | rapidgzip | Us | Gap |
|---------|----------|-----|-----|
| `CacheEntry` with symbolCount | ✓ 1-2 symbols per lookup | ✗ 1 symbol | 10-20% |
| `needToReadDistanceBits` flag | ✓ Avoid branch | ✗ Check symbol value | 2-5% |
| Pre-computed length in entry | ✓ `readLength()` embedded | ✗ Separate decode | 5-10% |
| `DISTANCE_OFFSET` in symbols | ✓ Length + offset combined | ✗ Separate tables | 5-10% |
| Marker-based parallel | ✓ 16-bit markers | ✓ Similar approach | ~same |
| Shared window deduplication | ✓ WindowMap | ✗ Full copy per chunk | 10-20% parallel |

### What We Have That They Don't

| Feature | Source |
|---------|--------|
| JIT table cache (fingerprint) | Novel |
| Static fixed table (OnceLock) | Novel |
| Pure Rust (no C/assembly deps) | Design |

---

## Optimization Tiers

### Tier 1: Micro-Optimizations (Est: +5-15%)

1. **`#[cold]`/`#[inline(never)]` on error paths** — Move error handling out of hot loop
2. **`likely()`/`unlikely()` intrinsics** — `core::intrinsics::likely` on literal check
3. **`EXTRACT_VARBITS8` pattern** — Cast to u8 before shift to hint 8-bit ops
4. **Register pressure reduction** — Fewer locals in fastloop, use `black_box`
5. **Prefetch next cache line** — `_mm_prefetch` on input + output

### Tier 2: Algorithmic (Est: +15-30%)

6. **Multi-symbol CacheEntry** — Pack 2 literals in single lookup (rapidgzip style)
7. **Length+distance combined entry** — Pre-compute for common short matches  
8. **Compile-time bit budget** — `CAN_CONSUME_AND_THEN_PRELOAD` as const fn
9. **Inline subtables** — Subtable entries immediately follow main entry
10. **Lazy refill** — Only refill when `bitsleft < threshold`, not every iter

### Tier 3: Novel/Exotic (Est: +20-50%)

11. **JIT-compiled decode loop** — Generate machine code for specific Huffman table
    - Fixed table = fixed control flow = no branches
    - Each code length maps to a specific instruction sequence
    - Cranelift/LLVM backend for codegen

12. **SIMD parallel decode** — Decode 4-8 streams in parallel using AVX2/NEON
    - Split input into lanes, decode independently, merge
    - Works best on fixed Huffman (same table per lane)

13. **Speculative decode with rollback** — Assume literal, rollback if wrong
    - Most symbols are literals (70-90%)
    - Write speculatively, revert on misprediction

14. **Huffman → FSM transformation** — Convert code tree to finite state machine
    - Each state = partial code, transitions = bit values
    - Vectorizable state transitions

15. **Table-free fixed Huffman** — Hard-code decode logic for RFC 1951 fixed codes
    - 0-143: 8 bits, 144-255: 9 bits, 256-279: 7 bits, 280-287: 8 bits
    - No table lookup for fixed blocks

16. **Batch output buffering** — Accumulate literals in SIMD register, flush as vector
    - 16-32 byte aligned stores vs byte-at-a-time

17. **Predictive table switching** — Detect block boundary early, pre-build next table
    - Overlap table construction with decode of current block

18. **Memory-mapped I/O** — `mmap` input for zero-copy
    - Kernel prefetch, lazy loading, huge pages

19. **Profile-guided table layout** — Reorder table entries by frequency
    - Most common codes → lowest indices → better cache

20. **Hardware CRC offload** — ARM CRC32 or Intel CRC32C for validation
    - Parallel with decode, not after

---

## Implementation Priority

| # | Optimization | Expected | Effort | Result |
|---|--------------|----------|--------|--------|
| 1 | Multi-symbol CacheEntry | +15% | Medium | Pending - exists, needs integration |
| 6 | `#[cold]` on error paths | +5% | Easy | **REGRESSED 4%** |
| 15 | Table-free fixed Huffman | +20% | Medium | **3.25x SLOWER** |
| - | Unconditional refill | +5% | Easy | **REGRESSED 12%** |
| 11 | JIT decode loop | +30% | Hard | Pending |
| 12 | SIMD parallel decode | +40% | Very Hard | Pending |

### Key Finding

**Simple micro-optimizations REGRESS performance.** The current code is already well-tuned.

Tested optimizations that made things WORSE:
| Optimization | Expected | Actual | Notes |
|--------------|----------|--------|-------|
| `#[cold]` on error paths | +5% | **-4%** | Added function call overhead |
| Table-free fixed Huffman | +20% | **-325%** | Bit reversal kills performance |
| Unconditional refill | +5% | **-12%** | Conditional was better |

What this tells us:
1. **LLVM is already optimizing our code well** - manual "improvements" interfere
2. **L1/L2 cache is critical** - table lookups beat computation
3. **Branch prediction is working** - conditional code isn't the bottleneck

The remaining 30% gap requires **novel approaches** that change the algorithmic structure:
- Precomputed multi-symbol decode (16-bit lookup → 2 symbols)
- SIMD parallel decode (multiple streams in lockstep)
- JIT code generation (no table lookups at all)

---

## Validation

```bash
# Baseline
for i in {1..3}; do cargo test --release bench_cf_silesia -- --nocapture 2>&1 | grep "Our throughput"; done

# After each change
cargo test --release  # Must pass 282 tests
# Re-run baseline
```

---

## Radical Approaches (Extremely Hard)

### 1. Vector Huffman Decode (AVX-512/SVE2) — Est: +100-200%
Split input into 8-16 parallel bit streams at fixed offsets. Decode all simultaneously with SIMD gather. Challenge: variable-length codes desync lanes. Solution: decode in lockstep chunks (e.g., 4 symbols each), resync via prefix-sum of consumed bits. Requires careful lane management and scatter for output.

```
Input:  [========64 bytes (512 bits)========]
Lanes:  [L0][L1][L2][L3][L4][L5][L6][L7]
         ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
        decode simultaneously via VPGATHERDD
         ↓
        scatter to output positions
```

### 2. Precomputed Decode Sequences — Est: +50-100%
For 16-22 bit input patterns, precompute: `(bits) → (out_bytes[], bits_consumed)`. Single lookup decodes 2-4 symbols. Table size: 4MB (22 bits × 8 bytes). Fits in L3 cache. Eliminates per-symbol loop overhead.

```rust
struct PrecomputedEntry {
    out_bytes: [u8; 4],  // Up to 4 decoded literals
    count: u8,          // How many symbols decoded
    bits: u8,           // Total bits consumed
}
// 2^22 entries × 8 bytes = 32MB table
```

### 3. Speculative Multi-Path Decode — Est: +30-50%
Fork execution into 3 paths: assume-literal, assume-match, assume-EOB. SIMD keeps all 3 hot. Commit correct path based on actual entry type. Like CPU branch prediction but at algorithm level. Eliminates branch misprediction entirely.

### 4. JIT State Machine Compiler — Est: +40-60%
Convert Huffman table to native machine code: each codeword → dedicated instruction sequence. No table lookups, no branches. Use computed goto (GNU extension) or tail calls. Requires mmap(PROT_EXEC) and careful code generation.

```
// For codeword 0b1011 (length 4) = literal 'A':
L_1011:
    *out++ = 'A';
    bits >>= 4;
    goto TABLE[bits & MASK];
```

### 5. FPGA/Hardware Accelerator — Est: +500%+
Implement Huffman FSM in Verilog. Use PCIe FPGA board (Xilinx Alveo). Memory-map decode unit. 10GB/s+ possible. Requires hardware design expertise and $500+ FPGA.

### 6. Tensor Core Abuse — Est: +200%+ (theoretical)
Encode Huffman table as sparse matrix. Use WMMA (Tensor Core) for parallel lookup. Matrix multiply = 256 table lookups simultaneously. Only works on NVIDIA GPUs with FP16 Tensor Cores. Completely insane but mathematically sound.

### 7. Two-Pass Table Precompute — Est: +20-40%
**Pass 1**: Scan entire file, fingerprint all Huffman tables, identify repeats
**Pass 2**: Decode with cached tables, parallel per-block
For files with repeated tables (common in BGZF), amortizes table build cost.

### 8. Bit-Parallel Decode via Carry-Less Multiply — Est: +30%
Use PCLMULQDQ to extract multiple bit fields simultaneously. Process 8 symbols' worth of bits in one instruction. Combine with VPSHUFB for table lookup. Requires deep understanding of carry-less polynomial arithmetic.

### 9. Branch-Free Computed Dispatch — Est: +15-25%
Replace all `if` statements with branchless select:
```rust
let is_lit = (entry >> 31) as usize;  // 0 or 1
let handlers = [handle_match, handle_literal];
(handlers[is_lit])(...)  // Indirect call, but predictable
```
Or use CMOV chains for purely arithmetic dispatch.

### 10. Decode-During-DMA Pipeline — Est: +20%
Use io_uring for async I/O. While block N is decoding, DMA block N+1. Overlap memory transfers with computation. Requires careful buffer management and kernel-level async.

### 11. Neural Network Predictor — Est: varies
Train tiny NN to predict next N symbols from context. If prediction correct (>90% for repetitive data), skip decode entirely. For genomics/log data with patterns, could be 5-10x faster.

### 12. Custom ISA Extension (RISC-V) — Est: +300%+
Add Huffman decode instruction to RISC-V: `huffdec rd, rs1, rs2` (rd=symbol, rs1=bitbuf, rs2=table_ptr). Single-cycle decode. Requires custom silicon or FPGA soft-core.

---

## Feasibility Matrix

| Approach | Gain | Effort | Deps |
|----------|------|--------|------|
| Vector Huffman (AVX-512) | +100% | 3 weeks | x86-64 only |
| Precomputed Sequences | +50% | 1 week | 32MB RAM |
| Speculative Multi-Path | +30% | 2 weeks | SIMD |
| JIT State Machine | +40% | 2 weeks | mmap exec |
| Two-Pass Precompute | +25% | 3 days | None |
| Branch-Free Dispatch | +15% | 2 days | None |

---

## Reference Files

| Tool | File | Key Technique |
|------|------|---------------|
| libdeflate | `lib/decompress_template.h:350-500` | Fastloop structure |
| rapidgzip | `huffman/HuffmanCodingShortBitsMultiCached.hpp` | Multi-symbol cache |
| ISA-L | `igzip/igzip_decode_block_stateless.asm` | Assembly decode |
