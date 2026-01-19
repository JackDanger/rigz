# Plan to Beat Everything

**Goal**: Beat libdeflate single-thread AND rapidgzip parallel on ALL file types.

---

## The Insight

libdeflate is ~1300 MB/s single-threaded. rapidgzip achieves 2800 MB/s on 14 threads.
Neither is using the optimal strategy for both.

**Our path**: Combine the best of both + innovations neither has.

---

## Radical Approaches (Outside the Box)

### 1. JIT-Compiled Huffman Decode (Target: +100%)

Instead of table lookups, **generate machine code** for each Huffman table:

```rust
// Current: Table lookup per symbol (~5 cycles)
let (sym, len) = table.decode(bits);

// JIT: Generated code (2-3 cycles)
// Compiler generates: if (bits & 0x7F) == 0x48 { sym = 'H'; consume(7); }
// But we do this at RUNTIME for the actual Huffman codes
```

Why this works:
- Huffman tables are known after parsing block header
- Most files have 1-3 tables total
- Generated code eliminates: memory load, branch prediction misses, cache misses

Implementation:
- Use `cranelift-jit` or `dynasm-rs` for code generation
- Generate specialized decode function per table
- Cache generated code for common table patterns

### 2. Fixed Block Turbo Mode (Target: +50%)

Fixed Huffman codes are COMPILE-TIME KNOWN. Create a perfect decode loop:

```rust
// Fixed block decode - no tables, just match on bits
#[inline(always)]
fn decode_fixed_turbo(bits: u64) -> (u16, u8) {
    // 7-bit codes: 256-279 (end of block, lengths 3-10)
    if bits & 0b1 == 0 { // Starts with 0
        let code = (bits & 0x7F) as u16;
        if code >= 0b0000000 && code <= 0b0010111 {
            return (256 + reverse7(code), 7);
        }
    }
    // 8-bit codes: 0-143, 280-287
    // ...perfectly unrolled for all cases
}
```

Why this works:
- ~40% of deflate data is fixed blocks
- Zero memory access - pure register operations
- Perfect branch prediction (compiler optimizes to jump table)

### 3. SIMD Huffman Decode (Target: +40%)

Decode 4-8 symbols in parallel:

```rust
// Decode 8 symbols at once using AVX2
fn decode_8_symbols(bits: [u64; 8]) -> [u16; 8] {
    // Load 8 table indices
    let indices = _mm256_and_si256(bits_vec, mask_1024);
    // Gather 8 table entries
    let entries = _mm256_i32gather_epi32(table_ptr, indices, 4);
    // Extract symbols and lengths
    // ...
}
```

ISA-L does this. We can too.

### 4. Two-Pass Parallel (Target: 3-4x on single-member)

**Pass 1**: Sequential scan to find deflate block boundaries (fast, ~1GB/s)
**Pass 2**: Parallel decode of blocks (each block is independent after boundary found)

```rust
fn two_pass_parallel(input: &[u8]) -> Vec<u8> {
    // Pass 1: Find all block boundaries
    let boundaries = find_block_boundaries_fast(input);  // ~200ms for 200MB
    
    // Pass 2: Decode blocks in parallel
    boundaries.par_chunks(N).map(|chunk| {
        decode_block_with_window(chunk)
    }).collect()
}
```

Why this beats speculative decode:
- No failed speculative decodes
- No marker replacement overhead  
- Simpler, more reliable

### 5. GPU Decompression (Target: 10x+ on large files)

```rust
// For files > 100MB, use GPU
if input.len() > 100_000_000 && has_gpu() {
    return gpu_decompress(input);
}
```

NVIDIA's nvCOMP can decompress deflate on GPU. We could:
- Port our code to CUDA/Metal
- Or use nvCOMP via FFI

### 6. Predictive Prefetch (Target: +10%)

```rust
// Prefetch next table entry BEFORE current decode completes
let next_bits = peek_bits(15);
let prefetch_idx = next_bits & TABLE_MASK;
prefetch(&table[prefetch_idx]);  // CPU starts loading while we process

// Now process current symbol
let sym = process_current();
// Table entry is already in L1 cache for next iteration!
```

---

## Implementation Priority

| # | Approach | Gain | Effort | Risk |
|---|----------|------|--------|------|
| 1 | Fixed Block Turbo | +50% | 2 days | Low |
| 2 | Two-Pass Parallel | 3-4x | 1 week | Medium |
| 3 | SIMD Huffman | +40% | 1 week | Medium |
| 4 | Predictive Prefetch | +10% | 1 day | Low |
| 5 | JIT Huffman | +100% | 2 weeks | High |
| 6 | GPU | 10x | 2 weeks | High |

---

## Phase 1: Fixed Block Turbo (Start Now)

40% of deflate uses fixed Huffman. We know these codes at compile time:

| Bits | Code Range | Symbol Range |
|------|------------|--------------|
| 7 | 0000000-0010111 | 256-279 |
| 8 | 00110000-10111111 | 0-143 |
| 8 | 11000000-11000111 | 280-287 |
| 9 | 110010000-111111111 | 144-255 |

Create a branchless decoder using bit manipulation:

```rust
fn decode_fixed_ultra(bits: u64) -> (u16, u8) {
    // Use bit patterns to compute symbol directly
    // No memory access, no branches
}
```

---

## Phase 2: Two-Pass Parallel

The key insight: deflate block headers have recognizable patterns.

```
BFINAL (1 bit) | BTYPE (2 bits) | ...
```

For BTYPE=10 (dynamic), the header is large and predictable.
For BTYPE=01 (fixed), immediately followed by data.
For BTYPE=00 (stored), followed by length bytes.

Fast scan algorithm:
1. Look for potential BFINAL=1 markers
2. Validate by checking subsequent bytes match expected patterns
3. Build list of verified block boundaries
4. Parallel decode from boundaries

---

## The Winning Formula

```
gzippy_speed = max(
    libdeflate_sequential,
    our_parallel_bgzf,
    our_two_pass_parallel,
    our_fixed_turbo,
    gpu_if_available
) * prefetch_boost * simd_huffman_boost
```

**By combining all approaches, we can beat everyone on every file type.**
