# Rigz Optimization Opportunities

A comprehensive analysis of optimization opportunities for rigz, organized by implementation complexity and expected impact.

## Current Performance Gap

**Where rigz wins:**
- Level 1, all thread counts: 17-50% faster
- Level 6, single-threaded: 35% faster
- Decompression, all configs: 10-25% faster

**Where pigz wins (the problem):**
- Level 6, max threads: pigz 12% faster
- Level 9, max threads: pigz 32% faster

**Root cause:** At high compression levels with max threads, the overhead of rigz's independent block approach (each block restarts compression with empty dictionary) exceeds the gains from parallelism. pigz uses a pipelined approach where dictionary context flows between blocks.

---

## Architecture Options to Beat pigz Everywhere

### Option A: Pipeline Compression (Like pigz)

**How pigz works:**
```
Thread 1: compress block 0 → extract last 32KB as dict
Thread 2: wait for dict → compress block 1 → extract dict
Thread 3: wait for dict → compress block 2 → ...
```

**Trade-off:** Loses parallel decompression (blocks depend on previous blocks)

**Implementation:**
```rust
fn compress_pipelined(data: &[u8], level: u32, threads: usize) -> Vec<Vec<u8>> {
    let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();
    let mut results = vec![Vec::new(); blocks.len()];
    let mut dict_channel: Vec<Sender<[u8; 32768]>> = Vec::new();
    
    // Pipeline: each thread waits for previous dictionary
    crossbeam::scope(|s| {
        for i in 0..blocks.len() {
            s.spawn(move |_| {
                let dict = if i == 0 { None } else { Some(dict_rx.recv()) };
                let compressed = compress_with_dict(blocks[i], dict, level);
                if i + 1 < blocks.len() {
                    dict_tx.send(extract_last_32kb(blocks[i]));
                }
                results[i] = compressed;
            });
        }
    });
    results
}
```

**Expected gain:** Match pigz at L6/L9 max threads
**Complexity:** Medium
**Downside:** Blocks have ordering dependency, loses independent-block advantage

---

### Option B: Hybrid Strategy (Best of Both)

**Idea:** Use pipelined compression but mark blocks as independently decompressible anyway.

**How:** Each block's dictionary is the *uncompressed* data from previous block (not compressed). Reader can reconstruct by decompressing sequentially, OR if they have the original, they can decompress any block.

**Better idea:** Mark blocks with a flag:
- First block of each "chain" (e.g., every 8th block) is independent
- Other blocks use shared dictionary
- Decompressor can parallel-decompress chains, sequential within chains

**Implementation:**
```rust
const CHAIN_LENGTH: usize = 8;  // Every 8th block is independent

fn compress_hybrid(data: &[u8], level: u32) -> Vec<u8> {
    let blocks: Vec<&[u8]> = data.chunks(BLOCK_SIZE).collect();
    
    blocks.par_chunks(CHAIN_LENGTH).flat_map(|chain| {
        let mut dict = None;
        chain.iter().map(|block| {
            let result = compress_with_dict(block, dict, level);
            dict = Some(&block[block.len().saturating_sub(32768)..]);
            result
        }).collect::<Vec<_>>()
    }).flatten().collect()
}
```

**Expected gain:** 80-90% of pigz's L9 performance, keeps parallel decomp
**Complexity:** Medium-High

---

### Option C: Larger Blocks at High Levels

**Insight:** Block overhead is fixed. At L9, deflate does more work per byte. Larger blocks amortize the overhead.

**Current:** 64KB blocks (BGZF-compatible)
**Proposal:** Level-dependent block sizing

```rust
fn optimal_block_size(level: u32, thread_count: usize) -> usize {
    match (level, thread_count) {
        (1..=3, _) => 64 * 1024,      // Small blocks for fast levels
        (4..=6, 1) => 128 * 1024,     // Medium blocks single-threaded
        (4..=6, _) => 256 * 1024,     // Larger for parallel
        (7..=9, 1) => 256 * 1024,     // Large for slow levels
        (7..=9, _) => 512 * 1024,     // Very large for L9 parallel
        _ => 128 * 1024,
    }
}
```

**Downside:** Larger blocks = fewer parallel units. With 2 cores and 512KB blocks on a 10MB file, only 20 blocks = limited parallelism.

**Expected gain:** 10-20% at L9 max threads
**Complexity:** Low

---

### Option D: Intel ISA-L for Compression

**What:** Intel's ISA-L has AVX-512 optimized DEFLATE that's 2-4x faster than zlib-ng.

**Why this matters:** If compression per block is 2x faster, we can afford the independent-block overhead.

**Challenge:** ISA-L requires autotools to build from source. The `isal-rs` crate fails on systems without autotools.

**Solution:** Vendor a pre-built ISA-L binary or use conditional compilation:

```rust
#[cfg(all(target_arch = "x86_64", feature = "isa-l"))]
fn compress_block_isal(data: &[u8], level: u32) -> Vec<u8> {
    // Use ISA-L when available
    unsafe { isal_deflate(data, level) }
}

#[cfg(not(all(target_arch = "x86_64", feature = "isa-l")))]
fn compress_block_isal(data: &[u8], level: u32) -> Vec<u8> {
    // Fall back to zlib-ng
    flate2_compress(data, level)
}
```

**Alternative:** Use `libdeflate` for compression too (it's faster than zlib-ng at most levels).

**Expected gain:** 50-100% on AVX-512 systems
**Complexity:** High (build system issues)

---

### Option E: libdeflate for Compression

**What:** libdeflate is already used for decompression. It also has compression that's faster than zlib-ng.

**Current stack:** flate2 → zlib-ng (compression) → libdeflate (decompression)
**Proposal:** libdeflate → libdeflate (both directions)

```rust
use libdeflater::{Compressor, CompressionLvl};

fn compress_block_libdeflate(data: &[u8], level: u32) -> Vec<u8> {
    let lvl = CompressionLvl::new(level as i32).unwrap_or(CompressionLvl::default());
    let mut compressor = Compressor::new(lvl);
    
    // libdeflate needs pre-allocated output buffer
    let max_size = compressor.gzip_compress_bound(data.len());
    let mut output = vec![0u8; max_size];
    
    let actual_size = compressor.gzip_compress(data, &mut output).unwrap();
    output.truncate(actual_size);
    output
}
```

**Benchmark needed:** libdeflate vs zlib-ng at L6/L9 with max threads

**Expected gain:** 10-30% (libdeflate is faster but doesn't have shared dictionary)
**Complexity:** Low (already have the crate)

---

### Option F: Zopfli-style Optimal Parsing (Level 10+)

**What:** For maximum compression, use Zopfli's optimal LZ77 parsing.

**Trade-off:** 100x slower but 5-10% better compression.

**Implementation:** Add a `--level 10` or `--best` flag that uses Zopfli algorithm.

```rust
#[cfg(feature = "zopfli")]
fn compress_optimal(data: &[u8]) -> Vec<u8> {
    use zopfli::{Format, Options, compress};
    let options = Options::default();
    compress(&options, &Format::Gzip, data).unwrap()
}
```

**Expected gain:** N/A for speed, but adds a "best compression" option
**Complexity:** Low

---

## Low-Level Optimizations

### 1. Memory and I/O

| Optimization | Description | Expected Gain | Status |
|-------------|-------------|---------------|--------|
| **Thread-local buffers** | Reuse compression buffers | 5-15% | ✅ Done |
| **Cache-aligned buffers** | 64/128-byte alignment | 2-5% | ✅ Done |
| **Huge pages** | Use 2MB pages for large buffers | 5-10% | 🔲 TODO |
| **mmap output** | Write directly to memory-mapped file | 5-15% | 🔲 TODO |
| **io_uring** | Async I/O on Linux | 10-30% (I/O bound) | 🔲 TODO |
| **Vectorized I/O** | write_vectored for multiple blocks | 2-5% | ✅ Done |
| **Prefetching** | madvise(MADV_SEQUENTIAL) | 2-5% | 🔲 TODO |

### 2. CPU Optimization

| Optimization | Description | Expected Gain | Status |
|-------------|-------------|---------------|--------|
| **SIMD header scan** | memchr for gzip magic | 10-50x (scan only) | ✅ Done |
| **Hardware CRC32** | Via libdeflate | ✅ Built-in | ✅ Done |
| **CPU detection** | AVX2/AVX-512/NEON | ✅ Routing | ✅ Done |
| **Core pinning** | Avoid SMT, use P-cores | 5-15% | 🔲 TODO |
| **NUMA awareness** | Allocate on local node | 10-20% (NUMA) | 🔲 TODO |

### 3. Algorithmic

| Optimization | Description | Expected Gain | Status |
|-------------|-------------|---------------|--------|
| **BGZF markers** | Block sizes in FEXTRA | Enables parallel decomp | ✅ Done |
| **Level 1→2 mapping** | Fix zlib-ng L1 RLE issue | 2-5x smaller output | ✅ Done |
| **Adaptive trials** | Run until statistically significant | Better benchmarks | ✅ Done |
| **Content analysis** | Detect compressibility | Skip compression if incompressible | 🔲 TODO |

---

## Platform-Specific Optimizations

### Intel/AMD x86_64

| Feature | How to use | Status |
|---------|-----------|--------|
| AVX2 | zlib-ng uses automatically | ✅ Active |
| AVX-512 | ISA-L required | 🔲 TODO |
| PCLMULQDQ | CRC32 acceleration | ✅ Active (libdeflate) |
| SHA-NI | N/A for gzip | N/A |

### Apple Silicon (M1/M2/M3/M4)

| Feature | How to use | Status |
|---------|-----------|--------|
| NEON | zlib-ng uses automatically | ✅ Active |
| CRC32 instruction | libdeflate uses | ✅ Active |
| 128-byte cache lines | Alignment adjusted | ✅ Done |
| Large L2 cache | Larger blocks possible | 🔲 TODO |
| Unified memory | No optimization needed | N/A |

### ARM Linux (Graviton, etc.)

| Feature | How to use | Status |
|---------|-----------|--------|
| NEON | zlib-ng | ✅ Active |
| SVE/SVE2 | Not yet in zlib-ng | 🔲 Future |
| CRC32 | Via libdeflate | ✅ Active |

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)

1. **Try libdeflate for compression** - Already have the crate, just need to benchmark
2. **Increase block size at L9** - Simple config change
3. **Add content pre-analysis** - Skip incompressible data

### Phase 2: Architectural (3-5 days)

4. **Implement hybrid chain compression** - Get shared dictionary benefits while keeping some parallelism
5. **Add prefetching hints** - madvise() for sequential access
6. **Try huge pages** - For buffers > 2MB

### Phase 3: Platform-Specific (1 week)

7. **Intel ISA-L integration** - Needs build system work
8. **io_uring for Linux** - Async I/O
9. **Core pinning** - Avoid SMT contention

### Phase 4: Deep Optimization (ongoing)

10. **Custom SIMD DEFLATE** - Only if needed after above
11. **Zopfli level 10** - For max compression option

---

## What Doesn't Work

1. **Manual deflate boundary detection** - Deflate streams contain false gzip headers
2. **Dynamic block sizing (by entropy)** - Added complexity, 14% slower in tests
3. **gzp crate** - Threading issues; custom rayon is better
4. **Parallel libdeflate per-member (without markers)** - Boundary detection requires full inflate (2x overhead)
5. **zlib-ng Level 1 directly** - Uses RLE strategy, produces 2-5x larger files on repetitive data
6. **Parallel decompression without BGZF** - 2x overhead finding boundaries
7. **Intel ISA-L integration** - Requires autotools/autoreconf (portability issues)
8. **Full shared dictionary** - Breaks independent-block parallel decompression

---

## Benchmarking Commands

```bash
# Full validation with statistics
make validate

# Quick local benchmark
make quick

# Generate test data
make test-data

# Specific level/thread test
python3 -c "
import subprocess, time, statistics
for i in range(10):
    start = time.perf_counter()
    subprocess.run(['./target/release/rigz', '-9', '-p2', '-c', '/tmp/test.dat'], 
                   stdout=subprocess.DEVNULL)
    print(f'{time.perf_counter() - start:.3f}s')
"
```

---

## References

- [libdeflate](https://github.com/ebiggers/libdeflate) - Fast DEFLATE implementation
- [Intel ISA-L](https://github.com/intel/isa-l) - Hardware-accelerated compression
- [rapidgzip](https://github.com/mxmlnkn/rapidgzip) - Parallel gzip decompression
- [zlib-ng](https://github.com/zlib-ng/zlib-ng) - Modernized zlib
- [pigz source](https://github.com/madler/pigz) - Reference for pipeline compression
- [RFC 1952](https://tools.ietf.org/html/rfc1952) - GZIP file format
- [RFC 1951](https://tools.ietf.org/html/rfc1951) - DEFLATE algorithm
