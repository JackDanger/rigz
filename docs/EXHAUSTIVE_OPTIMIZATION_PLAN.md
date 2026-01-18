# Exhaustive Optimization Plan

## Goal
**Nobody can beat rigz under any conditions unless they invent new mathematics.**

---

## Current State Audit

### What L1-L6 (parallel/libdeflate) Has
- [x] libdeflate compression (faster than zlib-ng)
- [x] Thread-local output buffers (`COMPRESS_BUF`)
- [x] Parallel block compression with rayon
- [x] BGZF markers for parallel decompression
- [x] Dynamic block sizing by file size
- [x] mmap for zero-copy file access

### What L1-L6 is MISSING
- [ ] Thread-local libdeflate `Compressor` reuse
- [ ] Pre-computed CRC tables (libdeflate may already do this)

### What L7-L9 (pipelined/zlib-ng) Has
- [x] Dictionary sharing between blocks
- [x] Parallel block compression (pigz-style pipelining)
- [x] Parallel CRC calculation with combine
- [x] Dynamic block sizing by file size
- [x] mmap for zero-copy file access

### What L7-L9 is MISSING
- [ ] Thread-local zlib `Compress` reuse
- [ ] Thread-local output buffers
- [ ] Thread-local dictionary buffers?

### What Decompression Has
- [x] libdeflate for all decompression
- [x] Parallel decompression for BGZF-marked files
- [x] Sequential fallback for standard gzip
- [x] SIMD-accelerated multi-member detection

### What Decompression is MISSING
- [ ] Thread-local libdeflate `Decompressor` reuse
- [ ] Pre-allocated decompression buffers

### What Build System Has
- [x] LTO enabled (`lto = true`)
- [x] opt-level = 3

### What Build System is MISSING
- [ ] Fat LTO (`lto = "fat"`)
- [ ] Single codegen unit (`codegen-units = 1`)
- [ ] `panic = "abort"` (smaller binary, slight speedup)
- [ ] PGO (profile-guided optimization)

---

## Cross-Pollination Opportunities

### From L1-L6 → L7-L9
| Optimization | In L1-L6? | In L7-L9? | Port? |
|-------------|-----------|-----------|-------|
| Thread-local output buffers | ✅ Yes | ❌ No | ✅ PORT |
| libdeflate compression | ✅ Yes | ❌ No (needs dict) | ❌ Can't |

### From L7-L9 → L1-L6
| Optimization | In L7-L9? | In L1-L6? | Port? |
|-------------|-----------|-----------|-------|
| Dynamic block by file size | ✅ Yes | ✅ Yes | ✅ Already done |

### From Compression → Decompression
| Optimization | In Compression? | In Decompression? | Port? |
|-------------|-----------------|-------------------|-------|
| Thread-local Compressor/Decompressor | Adding | ❌ No | ✅ PORT |
| Thread-local buffers | Adding | ❌ No | ✅ PORT |

---

## Complete Implementation Matrix

### Compression Optimizations

| # | Optimization | L1-L6 1t | L1-L6 Mt | L7-L9 1t | L7-L9 Mt | Status |
|---|-------------|----------|----------|----------|----------|--------|
| C1 | TL libdeflate Compressor | ✅ | ✅ | N/A | N/A | TODO |
| C2 | TL zlib Compress | N/A | N/A | ✅ | ✅ | TODO |
| C3 | TL output buffers (L1-L6) | ✅ | ✅ | N/A | N/A | DONE |
| C4 | TL output buffers (L7-L9) | N/A | N/A | ✅ | ✅ | TODO |
| C5 | Fat LTO | ✅ | ✅ | ✅ | ✅ | TODO |
| C6 | codegen-units=1 | ✅ | ✅ | ✅ | ✅ | TODO |
| C7 | panic=abort | ✅ | ✅ | ✅ | ✅ | TODO |

### Decompression Optimizations

| # | Optimization | L1-L6 | L7-L9 | Status |
|---|-------------|-------|-------|--------|
| D1 | TL libdeflate Decompressor | ✅ | ✅ | TODO |
| D2 | TL decompression buffers | ✅ | ✅ | TODO |

### Single-Thread Specific

| # | Optimization | Compression | Decompression | Status |
|---|-------------|-------------|---------------|--------|
| S1 | Skip thread pool init | ✅ | ✅ | CHECK |
| S2 | Direct I/O (no buffering) | ✅ | ✅ | CHECK |

---

## Detailed Analysis of Each

### C1: Thread-Local libdeflate Compressor (L1-L6)

**Current code:**
```rust
fn compress_block_bgzf_libdeflate(output: &mut Vec<u8>, block: &[u8], level: u32) {
    let lvl = CompressionLvl::new(level as i32).unwrap();
    let mut compressor = Compressor::new(lvl);  // NEW EACH TIME
    // ...
}
```

**Proposed:**
```rust
thread_local! {
    static LIBDEFLATE_COMPRESSOR: RefCell<Option<(i32, Compressor)>> = RefCell::new(None);
}

fn get_libdeflate_compressor(level: i32) -> Compressor {
    LIBDEFLATE_COMPRESSOR.with(|c| {
        let mut c = c.borrow_mut();
        if let Some((cached_level, comp)) = c.as_ref() {
            if *cached_level == level {
                // libdeflate Compressor is stateless between compressions
                // so we can just return a reference... but we need to own it
                // Actually, libdeflate Compressor doesn't have reset()
                // We might need to just cache by level
            }
        }
        let comp = Compressor::new(CompressionLvl::new(level).unwrap());
        *c = Some((level, comp.clone()));
        comp
    })
}
```

**Issue:** libdeflate's Rust bindings don't expose a way to reuse Compressor.
Looking at libdeflater source... `Compressor` wraps a raw pointer.
We can cache it by level.

**Expected gain:** 2-3% for L1-L6 multi-thread

---

### C2: Thread-Local zlib Compress (L7-L9)

**Current code:**
```rust
fn compress_block_with_dict(block: &[u8], dict: Option<&[u8]>, ...) -> Vec<u8> {
    let mut compress = Compress::new(Compression::new(level), false);  // NEW EACH TIME
    if let Some(d) = dict {
        compress.set_dictionary(d);
    }
    // ...
}
```

**Proposed:**
```rust
thread_local! {
    static ZLIB_COMPRESS: RefCell<Option<Compress>> = RefCell::new(None);
}

fn get_zlib_compress(level: u32) -> Compress {
    ZLIB_COMPRESS.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(comp) = c.as_mut() {
            comp.reset();  // Reset internal state
            return comp.clone();  // Hmm, Clone might defeat the purpose
        }
        let comp = Compress::new(Compression::new(level), false);
        *c = Some(comp.clone());
        comp
    })
}
```

**Issue:** flate2's `Compress` implements Clone by... creating a new one!
We need to use a different pattern - pass the Compress into the function.

**Better approach:**
```rust
fn compress_blocks_parallel(blocks: &[&[u8]], level: u32) -> Vec<Vec<u8>> {
    // Each thread gets ONE Compress for ALL its blocks
    blocks.par_chunks(blocks_per_thread).map(|chunk| {
        let mut compress = Compress::new(...);  // One per thread
        chunk.iter().map(|block| {
            compress.reset();
            compress_single_block(&mut compress, block)
        }).collect()
    }).flatten().collect()
}
```

**Expected gain:** 5-10% for L7-L9 multi-thread

---

### C4: Thread-Local Output Buffers (L7-L9)

**Current code:**
```rust
fn compress_block_with_dict(...) -> Vec<u8> {
    let mut output = vec![0u8; block.len() + block.len() / 10 + 128];  // NEW ALLOC
    // ...
    output
}
```

**Proposed:**
```rust
thread_local! {
    static PIPELINED_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(512 * 1024));
}

fn compress_block_with_dict(...) -> Vec<u8> {
    PIPELINED_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        buf.reserve(block.len() + block.len() / 10 + 128);
        // ... compress into buf ...
        buf.clone()  // Return copy, buf stays allocated
    })
}
```

**Expected gain:** 3-5% for L7-L9

---

### D1 & D2: Decompression Optimizations

**Current code in decompression.rs:**
```rust
pub fn decompress_gzip(input: &[u8]) -> io::Result<Vec<u8>> {
    let mut decompressor = Decompressor::new();  // NEW EACH TIME
    let mut output = Vec::with_capacity(estimated_size);  // NEW ALLOC
    // ...
}
```

**Proposed:**
```rust
thread_local! {
    static DECOMPRESSOR: RefCell<Decompressor> = RefCell::new(Decompressor::new());
    static DECOMPRESS_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(1024 * 1024));
}
```

**Expected gain:** 2-5% for decompression

---

### S1: Skip Thread Pool Init for Single-Thread

**Check:** Does rayon initialize a thread pool even for single-threaded?

Looking at our code:
```rust
// In compress_parallel:
if self.num_threads == 1 {
    // Does this still touch rayon?
}
```

We should ensure single-threaded path completely bypasses rayon.

---

## Final Exhaustive Plan

### Phase 1: Build Optimizations (Universal)
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

### Phase 2: L7-L9 Compression (Main Target)
1. Thread-local zlib Compress objects
2. Thread-local output buffers

### Phase 3: L1-L6 Compression (Bonus)
3. Thread-local libdeflate Compressor

### Phase 4: Decompression (All Levels)
4. Thread-local libdeflate Decompressor
5. Thread-local decompression buffers

### Phase 5: Single-Thread Path
6. Verify no unnecessary thread pool initialization
7. Direct I/O path (bypass buffering for large files)

---

## Expected Total Gains

| Path | Current vs pigz | After Optimization | Gain |
|------|-----------------|-------------------|------|
| L1 multi-thread | 30% faster | 35% faster | +5% |
| L6 multi-thread | 54% faster | 60% faster | +6% |
| L9 single-thread | ~equal | 15% faster | +15% |
| L9 multi-thread | 5% slower | 15% faster | +20% |
| Decompression (all) | 40% faster | 45% faster | +5% |

---

## Verification Checklist

For EACH optimization:
- [ ] Benchmark before
- [ ] Implement
- [ ] Verify correctness (output matches)
- [ ] Benchmark after
- [ ] Commit if improvement
- [ ] Revert if regression

---

## What Would Be Left for Competitors?

After implementing ALL of the above:

### We've Done
- Best available compression libraries (libdeflate, zlib-ng)
- All thread-local optimizations
- All build optimizations
- Optimal parallelization strategy

### What's Left (Requires New Math/Hardware)
1. **Custom DEFLATE implementation** - Possible 10% more, but years of work
2. **GPU acceleration** - Possible for huge files, complex
3. **Hardware deflate (Intel QAT)** - Specialized hardware
4. **New compression algorithm** - Not gzip-compatible

**Conclusion:** After this plan, the only way to beat rigz is:
- Custom DEFLATE (massive effort)
- Hardware acceleration (specialized)
- New algorithms (not gzip-compatible)

This is the theoretical maximum within the constraints.
