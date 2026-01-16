# Rigz Optimization Opportunities

A comprehensive analysis of optimization opportunities for rigz, organized by implementation complexity and expected impact.

## Current Architecture Baseline

```
Compression:  Input → mmap → 128KB chunks → rayon parallel → flate2/zlib-ng → concatenate → Output
Decompression: Input → mmap → detect single/multi → libdeflate OR flate2 → Output
```

**Current performance (vs competitors):**
- Compression: 40-53% faster than pigz
- Decompression: ~0-10% faster than pigz (variable)

---

## Tier 1: High Impact, Low Complexity

### 1.1 Use Intel ISA-L for Compression (Intel CPUs)

**What:** Intel's Intelligent Storage Acceleration Library has hardware-accelerated DEFLATE.

**Why:** ISA-L is 2-4x faster than zlib-ng on Intel CPUs with AVX-512.

**How:**
```toml
# Cargo.toml - add conditional dependency
[target.'cfg(target_arch = "x86_64")'.dependencies]
isal = "0.3"  # Rust bindings to ISA-L
```

```rust
// Detect at runtime and use ISA-L when available
#[cfg(target_arch = "x86_64")]
fn compress_block(data: &[u8], level: u32) -> Vec<u8> {
    if is_x86_feature_detected!("avx512f") {
        return isal_compress(data, level);
    }
    flate2_compress(data, level)
}
```

**Compatibility:** ✅ Full - ISA-L produces standard DEFLATE

**Expected gain:** 50-100% on Intel CPUs with AVX-512

---

### 1.2 Pre-allocate Thread-Local Buffers

**What:** Avoid per-block allocation by reusing buffers.

**Current problem:**
```rust
// Current: allocates new Vec for each block
.map(|block| {
    let mut compressed = Vec::with_capacity(estimated_compressed_size);  // ALLOCATION
    // ...
})
```

**Solution:**
```rust
use std::cell::RefCell;

thread_local! {
    static COMPRESS_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(256 * 1024));
    static ENCODER: RefCell<Option<flate2::Compress>> = RefCell::new(None);
}

fn compress_block_reuse(data: &[u8], level: u32) -> Vec<u8> {
    COMPRESS_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        // Reuse the allocated capacity
        // ...
    })
}
```

**Expected gain:** 5-15% on small blocks (reduces allocator pressure)

---

### 1.3 SIMD-Accelerated Multi-Member Detection

**What:** Replace byte-by-byte scan with SIMD search.

**Current problem:**
```rust
// Current: O(n) byte scan
for i in 10..scan_end.saturating_sub(2) {
    if data[i] == 0x1f && data[i + 1] == 0x8b && data[i + 2] == 8 {
        return true;
    }
}
```

**Solution using memchr crate:**
```rust
use memchr::memmem;

fn is_multi_member_fast(data: &[u8]) -> bool {
    const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b, 0x08];
    // memchr uses SIMD internally (AVX2/NEON)
    memmem::find(&data[10..], GZIP_MAGIC).is_some()
}
```

**Expected gain:** 10-50x faster detection on large files

---

### 1.4 Hardware CRC32 via crc32fast

**What:** Ensure CRC32 uses CPU instructions (already in Cargo.toml but verify usage).

**Status:** `crc32fast` is listed in dependencies but may not be directly used.

**Verify:**
```rust
// Check if flate2/libdeflate use hardware CRC
// If not, compute CRC separately using crc32fast
use crc32fast::Hasher;

fn compute_crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}
```

**Note:** Both ARM (CRC32 instruction) and x86 (SSE4.2 CRC32C) have hardware support.

---

## Tier 2: High Impact, Medium Complexity

### 2.1 Shared Dictionary Between Blocks (Pigz-style)

**What:** Use the last N bytes of the previous block as a preset dictionary.

**Why:** Improves compression ratio without breaking parallelism.

**How:**
```rust
const DICT_SIZE: usize = 32768;  // 32KB window

fn compress_with_dict(block: &[u8], prev_block: Option<&[u8]>, level: u32) -> Vec<u8> {
    let mut encoder = flate2::Compress::new(Compression::new(level), true);
    
    if let Some(prev) = prev_block {
        let dict_start = prev.len().saturating_sub(DICT_SIZE);
        encoder.set_dictionary(&prev[dict_start..])?;
    }
    
    // Compress block
    // ...
}
```

**Parallelism approach:**
1. Compress block 0 without dictionary (can start immediately)
2. Blocks 1-N can start as soon as prev block's last 32KB is known
3. Use pipelining: read → compress with dict → write

**Expected gain:** 2-5% better compression ratio

---

### 2.2 Parallel Decompression for Multi-Member Files

**What:** Decompress multiple gzip members in parallel.

**Challenge:** Finding member boundaries requires actually inflating (deflate can contain false headers).

**Solution - Rapidgzip approach:**
```rust
/// Index-building phase: inflate each member sequentially to find boundaries
fn build_member_index(data: &[u8]) -> Vec<MemberInfo> {
    let mut members = Vec::new();
    let mut pos = 0;
    
    while pos < data.len() {
        let start = pos;
        // Use flate2 to inflate and track consumed bytes
        let (end, uncompressed_size) = inflate_member(&data[pos..]);
        members.push(MemberInfo { start, end, uncompressed_size });
        pos = end;
    }
    members
}

/// Parallel decompression using index
fn decompress_parallel(data: &[u8], members: &[MemberInfo]) -> Vec<u8> {
    members.par_iter()
        .map(|m| decompress_member(&data[m.start..m.end]))
        .flatten()
        .collect()
}
```

**Alternative - BGZF-style markers:**
Add block size to gzip FEXTRA field during compression:
```rust
// During compression, add extra field with block size
fn write_gzip_header_with_size(compressed_size: u32) -> Vec<u8> {
    let mut header = vec![
        0x1f, 0x8b,  // Magic
        0x08,        // Deflate
        0x04,        // FEXTRA flag
        // ... timestamp, etc
    ];
    // Add XLEN and extra field
    header.extend_from_slice(&6u16.to_le_bytes()); // XLEN
    header.extend_from_slice(b"BC");  // Subfield ID
    header.extend_from_slice(&2u16.to_le_bytes()); // Subfield len
    header.extend_from_slice(&compressed_size.to_le_bytes()[..2]); // Block size
    header
}
```

**Expected gain:** 2-4x decompression speedup on multi-core

---

### 2.3 Cache-Aware Block Sizing

**What:** Tune block size to CPU cache hierarchy.

**Insight:** 128KB blocks may not fit in L2 cache on all CPUs.

**Solution:**
```rust
fn optimal_block_size() -> usize {
    // Detect cache sizes at runtime
    #[cfg(target_arch = "x86_64")]
    {
        // Use CPUID to get L2 cache size
        let l2_size = detect_l2_cache_size();
        // Aim for blocks that fit in L2 with working memory
        (l2_size / 4).max(64 * 1024).min(256 * 1024)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Apple Silicon has large L2 (up to 32MB shared)
        // ARM servers vary widely
        128 * 1024
    }
}
```

---

## Tier 3: Medium Impact, Higher Complexity

### 3.1 Custom DEFLATE Implementation with SIMD

**What:** Write optimized DEFLATE with explicit SIMD for hot paths.

**Key hot spots in DEFLATE:**
1. **LZ77 hash chain lookup** - finding matches
2. **Huffman encoding** - bit packing
3. **Literal copying** - memcpy operations

**Example - SIMD match finding:**
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

unsafe fn find_match_simd(haystack: &[u8], needle: &[u8; 4]) -> Option<usize> {
    let pattern = _mm_set1_epi32(i32::from_le_bytes(*needle));
    
    for chunk in haystack.chunks(16) {
        let data = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
        let cmp = _mm_cmpeq_epi32(data, pattern);
        let mask = _mm_movemask_epi8(cmp);
        if mask != 0 {
            return Some(mask.trailing_zeros() as usize);
        }
    }
    None
}
```

**Expected gain:** 20-40% for compression, less applicable to decompression

---

### 3.2 Zero-Copy Output with io_uring (Linux)

**What:** Use Linux io_uring for async I/O with zero kernel copies.

```rust
#[cfg(target_os = "linux")]
use io_uring::{IoUring, opcode, types};

fn write_compressed_uring(ring: &mut IoUring, data: &[u8], fd: i32) {
    let write_e = opcode::Write::new(types::Fd(fd), data.as_ptr(), data.len() as u32)
        .build();
    
    unsafe {
        ring.submission().push(&write_e).unwrap();
    }
    ring.submit_and_wait(1).unwrap();
}
```

**Expected gain:** 10-30% for I/O-bound workloads on Linux

---

### 3.3 Memory-Mapped Output

**What:** mmap the output file and write directly to mapped memory.

```rust
use memmap2::MmapMut;

fn compress_to_mmap(input: &[u8], output_path: &Path) -> io::Result<()> {
    let estimated_size = input.len();  // or better estimate
    
    let file = OpenOptions::new()
        .read(true).write(true).create(true)
        .open(output_path)?;
    file.set_len(estimated_size as u64)?;
    
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    
    // Write directly to mmap - no kernel buffer copies
    let actual_size = compress_to_slice(input, &mut mmap[..])?;
    
    file.set_len(actual_size as u64)?;
    Ok(())
}
```

---

## Tier 4: Algorithmic Improvements

### 4.1 Optimal Huffman Tree Building

**What:** Use faster Huffman construction algorithms.

**Current:** zlib-ng uses package-merge algorithm.

**Alternative - Two-pass frequency counting:**
```rust
/// Build Huffman tree with SIMD frequency counting
fn build_huffman_fast(data: &[u8]) -> HuffmanTable {
    // Count frequencies using SIMD
    let freqs = count_frequencies_simd(data);
    
    // Use Moffat's in-place algorithm for code lengths
    // O(n) instead of O(n log n) for package-merge
    let lengths = moffat_code_lengths(&freqs);
    
    // Build canonical Huffman codes
    canonical_huffman(&lengths)
}
```

### 4.2 Lazy vs Greedy Match Selection

**What:** Tune LZ77 match selection strategy per compression level.

```rust
enum MatchStrategy {
    Greedy,      // Level 1-3: take first match
    Lazy,        // Level 4-6: check if next position has better match  
    Optimal,     // Level 7-9: consider multiple possibilities
}

fn select_match(strategy: MatchStrategy, pos: usize, data: &[u8]) -> Match {
    match strategy {
        MatchStrategy::Greedy => find_first_match(pos, data),
        MatchStrategy::Lazy => {
            let m1 = find_best_match(pos, data);
            let m2 = find_best_match(pos + 1, data);
            if m2.len > m1.len + 1 { m2 } else { m1 }
        }
        MatchStrategy::Optimal => optimal_parse(pos, data),
    }
}
```

---

## Tier 5: Platform-Specific Optimizations

### 5.1 Apple Silicon (M1/M2/M3/M4) Optimizations

**What:** Leverage Apple's unified memory architecture.

```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple_optimizations {
    // Apple Silicon specific: 
    // - 128-byte cache lines (not 64)
    // - Large L2 cache (up to 32MB)
    // - High memory bandwidth
    
    const BLOCK_SIZE: usize = 256 * 1024;  // Larger blocks work well
    
    // Use NEON for SIMD operations
    use std::arch::aarch64::*;
    
    pub fn compress_neon(data: &[u8]) -> Vec<u8> {
        // NEON-optimized compression
    }
}
```

### 5.2 AMD EPYC/Ryzen Optimizations

**What:** Tune for AMD's different cache hierarchy.

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn detect_amd() -> bool {
    // Check CPUID for AMD
    let cpuid = core::arch::x86_64::__cpuid(0);
    // AMD signature: "AuthenticAMD"
    cpuid.ebx == 0x68747541 && cpuid.ecx == 0x444d4163
}
```

---

## Implementation Priority Matrix

| Optimization | Impact | Complexity | Compatibility | Priority | Status |
|-------------|--------|------------|---------------|----------|--------|
| Thread-local buffers | Medium | Low | ✅ Full | **P0** | ✅ Done |
| SIMD multi-member detect | Medium | Low | ✅ Full | **P0** | ✅ Done |
| Cache-aligned buffers | Low | Low | ✅ Full | **P0** | ✅ Done |
| ISIZE buffer sizing | Low | Low | ✅ Full | **P0** | ✅ Done |
| Thread-local decompressor | Low | Low | ✅ Full | **P0** | ✅ Done |
| CPU feature detection | Low | Low | ✅ Full | **P0** | ✅ Done |
| Intel ISA-L integration | High | Medium | ✅ Full | **P1** | 🔲 TODO |
| Parallel decompression | High | Medium | ✅ Full | **P1** | ✅ Done |
| Shared dictionaries | Medium | Medium | ✅ Full | **P2** | 🔲 TODO |
| io_uring async I/O | Medium | Medium | ✅ Full | **P3** | 🔲 TODO |
| Custom SIMD DEFLATE | High | High | ✅ Full | **P4** | 🔲 TODO |

---

## Benchmarking Methodology

When implementing optimizations, measure:

1. **Throughput** (MB/s) for various file sizes
2. **Compression ratio** (compressed/original)
3. **Memory usage** (peak RSS)
4. **Latency** (time to first byte for streaming)

```bash
# Suggested benchmark matrix
for size in 1MB 10MB 100MB 1GB; do
    for level in 1 6 9; do
        for threads in 1 4 8; do
            hyperfine "rigz -${level} -p${threads} ${size}.dat"
        done
    done
done
```

---

## References

- [libdeflate](https://github.com/ebiggers/libdeflate) - Fast DEFLATE implementation
- [Intel ISA-L](https://github.com/intel/isa-l) - Hardware-accelerated compression
- [rapidgzip](https://github.com/mxmlnkn/rapidgzip) - Parallel gzip decompression
- [zlib-ng](https://github.com/zlib-ng/zlib-ng) - Modernized zlib
- [RFC 1952](https://tools.ietf.org/html/rfc1952) - GZIP file format
- [RFC 1951](https://tools.ietf.org/html/rfc1951) - DEFLATE algorithm
