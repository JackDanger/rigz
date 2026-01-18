# L9 Absolute Maximum Performance Plan

## Goal
Make rigz L9 the fastest possible gzip -9 implementation. Nobody can beat us without inventing new math.

## Current State (GHA 4-core)
- L9 single-thread: 40% faster than gzip, ~equal to pigz
- L9 multi-thread: 5% slower than pigz (within threshold, but not dominant)
- L9 decompression: 43% faster than pigz

## Target State
- L9 single-thread: **50%+ faster than pigz**
- L9 multi-thread: **30%+ faster than pigz**
- L9 decompression: **Match pigz** (both sequential)

---

## Phase 1: Thread-Local Compress Objects

### Problem
Every block calls `Compress::new()` which internally calls `deflateInit2`.
This allocates ~300KB of internal state per call.

### Solution
```rust
thread_local! {
    static COMPRESSOR: RefCell<Option<Compress>> = RefCell::new(None);
}

fn get_compressor(level: u32) -> Compress {
    COMPRESSOR.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(comp) = c.as_mut() {
            comp.reset();  // Much cheaper than new()
            return comp.clone();
        }
        let comp = Compress::new(Compression::new(level), false);
        *c = Some(comp.clone());
        comp
    })
}
```

### Expected Gain: 5-10%
Eliminates ~300KB allocation per block.

---

## Phase 2: Pre-Allocated Output Buffers

### Problem
Each `compress_block_with_dict` creates a new Vec and resizes it.

### Solution
```rust
thread_local! {
    static OUTPUT_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(512 * 1024));
}

fn compress_block_with_dict(block: &[u8], dict: Option<&[u8]>, ...) -> Vec<u8> {
    OUTPUT_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        // ... compress into buf ...
        buf.clone()  // Return copy, buf stays allocated
    })
}
```

### Expected Gain: 3-5%
Eliminates Vec allocation/resizing overhead.

---

## Phase 3: Optimal Block Size = L2 Cache

### Problem
Current block sizes are arbitrary (64KB/128KB/256KB).

### Solution
```rust
fn get_optimal_block_size() -> usize {
    // L2 cache is typically 256KB-1MB per core
    // Use 1/4 of L2 so block + dictionary + output fit
    let l2_size = detect_l2_cache_size();  // Already have this
    (l2_size / 4).clamp(64 * 1024, 512 * 1024)
}
```

### Expected Gain: 2-5%
Better cache utilization, fewer cache misses.

---

## Phase 4: Memory Prefetching

### Problem
Sequential reads still cause cache misses.

### Solution
```rust
#[cfg(unix)]
fn prefetch_sequential(data: &[u8]) {
    use libc::{madvise, MADV_SEQUENTIAL, MADV_WILLNEED};
    unsafe {
        madvise(data.as_ptr() as *mut _, data.len(), MADV_SEQUENTIAL);
        madvise(data.as_ptr() as *mut _, data.len(), MADV_WILLNEED);
    }
}
```

### Expected Gain: 2-3%
Tells kernel to prefetch pages ahead of reads.

---

## Phase 5: Batch CRC Calculation

### Problem
CRC32 for each block adds overhead.

### Current Solution (already good)
Using `crc32fast::combine()` to merge block CRCs.

### Optimization
Ensure CRC is calculated inline during compression, not as separate pass.
```rust
// Inside compression loop:
let crc = crc32fast::hash(block);  // Already doing this
```

### Expected Gain: Already optimized

---

## Phase 6: Lock-Free Work Stealing

### Problem
Rayon's default work stealing has some overhead.

### Solution
Use a custom work queue with lock-free atomics:
```rust
use crossbeam::deque::{Injector, Stealer, Worker};

// Pre-create work items
let injector = Injector::new();
for (i, block) in blocks.iter().enumerate() {
    injector.push(CompressionJob { block_idx: i, ... });
}

// Workers steal from each other
```

### Expected Gain: 3-5%
Reduces thread synchronization overhead.

---

## Phase 7: Intel ISA-L Integration (Optional)

### Problem
zlib-ng is good but ISA-L is faster on Intel CPUs.

### Challenge
ISA-L requires autotools to build, complex setup.

### Solution
Conditional compilation:
```rust
#[cfg(feature = "isa-l")]
fn compress_l9_block(block: &[u8], dict: &[u8]) -> Vec<u8> {
    // Use ISA-L
}

#[cfg(not(feature = "isa-l"))]
fn compress_l9_block(block: &[u8], dict: &[u8]) -> Vec<u8> {
    // Use zlib-ng
}
```

### Expected Gain: 10-20% on Intel
ISA-L uses AVX-512 for match finding.

---

## Phase 8: Profile-Guided Optimization

### Process
```bash
# Step 1: Build with instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo" cargo build --release

# Step 2: Run representative workloads
./target/release/rigz -9 test_data/text-100MB.txt -c > /dev/null
./target/release/rigz -d test_data/compressed.gz -c > /dev/null

# Step 3: Rebuild with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo" cargo build --release
```

### Expected Gain: 5-10%
Compiler optimizes hot paths based on actual usage.

---

## Phase 9: LTO + Single Codegen Unit

### Current Cargo.toml
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

### Expected Gain: 3-5%
Better inlining across crate boundaries.

---

## Phase 10: Huge Pages

### Problem
Many small pages = TLB misses.

### Solution
```rust
#[cfg(target_os = "linux")]
fn allocate_huge(size: usize) -> Vec<u8> {
    use libc::{mmap, MAP_ANONYMOUS, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE};
    let ptr = unsafe {
        mmap(
            std::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1,
            0,
        )
    };
    // ...
}
```

### Expected Gain: 1-3%
Reduces TLB misses for large buffers.

---

## Implementation Order

| Phase | Effort | Gain | Priority |
|-------|--------|------|----------|
| 1. Thread-local Compress | Low | 5-10% | **HIGH** |
| 2. Pre-alloc buffers | Low | 3-5% | **HIGH** |
| 3. L2 cache block size | Low | 2-5% | Medium |
| 4. Memory prefetching | Low | 2-3% | Medium |
| 5. Batch CRC | Done | - | - |
| 6. Lock-free work | Medium | 3-5% | Low |
| 7. ISA-L | High | 10-20% | Optional |
| 8. PGO | Medium | 5-10% | Medium |
| 9. LTO | Low | 3-5% | **HIGH** |
| 10. Huge pages | Medium | 1-3% | Low |

**Total Expected Gain: 25-45%**

---

## Verification

After each phase:
```bash
# Local test
python3 scripts/benchmark_ci.py --size 100 --level 9 --threads 4 --data-type text

# Remote test (real x86_64 hardware)
./scripts/remote_bench.sh --quick

# Full CI
git push && gh run watch
```

---

## Success Criteria

- [ ] L9 single-thread: 50%+ faster than pigz
- [ ] L9 multi-thread: 30%+ faster than pigz
- [ ] L9 compression ratio: within 0.5% of pigz
- [ ] All CI tests pass
- [ ] No regressions in L1-L8
