# Optimization Applicability Matrix

## Current Architecture Recap

| Level | Compressor | Strategy | Decompressor |
|-------|------------|----------|--------------|
| L1-L6 | libdeflate | Parallel blocks | libdeflate (parallel) |
| L7-L9 | zlib-ng | Pipelined (dict sharing) | libdeflate (sequential) |

---

## Phase 1: Thread-Local Compress Objects

### What it does
Reuses zlib `Compress` struct instead of creating new one per block.

### Applicability by Level

| Level | Uses zlib Compress? | Benefit? |
|-------|---------------------|----------|
| L1-L6 | ❌ No (libdeflate) | ❌ None |
| L7-L9 | ✅ Yes (zlib-ng) | ✅ Yes |

**L1-L6 uses libdeflate which has its own `Compressor` struct.**
Looking at the code:

```rust
// parallel_compress.rs (L1-L6)
fn compress_block_bgzf_libdeflate(...) {
    let mut compressor = Compressor::new(lvl);  // libdeflate, not zlib
    // ...
}
```

```rust
// pipelined_compress.rs (L7-L9)
fn compress_block_with_dict(...) {
    let mut compress = Compress::new(...);  // zlib-ng
    compress.set_dictionary(dict);
    // ...
}
```

### Should we also do thread-local libdeflate Compressor?

libdeflate `Compressor::new()` is much lighter than zlib's `deflateInit2()`:
- libdeflate: ~50KB state
- zlib-ng: ~300KB state

But libdeflate still allocates. Let me check if it has reset...

```rust
// libdeflater crate doesn't expose reset()
// We'd need to reuse the Compressor object directly
```

### Applicability by Thread Count

| Threads | L1-L6 Benefit | L7-L9 Benefit |
|---------|---------------|---------------|
| 1 | ❌ None (single compressor anyway) | ⚠️ Small (one compressor) |
| 2+ | ⚠️ Maybe (libdeflate) | ✅ Yes (one per thread) |

For single-threaded, there's only one compressor so reuse happens naturally.
For multi-threaded, each thread creates its own compressor per block.

### Applicability by Content Type

| Content | Benefit? |
|---------|----------|
| Text | ✅ Same |
| Random | ✅ Same |
| Binary | ✅ Same |

Content type doesn't affect compressor allocation overhead.

### VERDICT

| Combination | Helps? | Why |
|-------------|--------|-----|
| L1-L6, any threads | ⚠️ Maybe | libdeflate Compressor reuse (smaller gain) |
| L7-L9, 1 thread | ⚠️ Small | Only one compressor anyway |
| L7-L9, 2+ threads | ✅ YES | Main target - 300KB × blocks saved |

**Optimization is L7-L9 multi-threaded specific, but we COULD also add libdeflate reuse for L1-L6.**

---

## Phase 2: Pre-Allocated Output Buffers

### What it does
Reuses Vec<u8> output buffer instead of allocating new one per block.

### Current State

```rust
// parallel_compress.rs (L1-L6) - ALREADY HAS IT
thread_local! {
    static COMPRESS_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(256 * 1024));
}
```

```rust
// pipelined_compress.rs (L7-L9) - DOES NOT HAVE IT
fn compress_block_with_dict(...) -> Vec<u8> {
    let mut output = vec![0u8; estimated_size];  // New allocation!
    // ...
}
```

### Applicability by Level

| Level | Has thread-local buffer? | Benefit from adding? |
|-------|--------------------------|----------------------|
| L1-L6 | ✅ Already has | ❌ None (done) |
| L7-L9 | ❌ Does not have | ✅ Yes |

### Applicability by Thread Count

| Threads | L1-L6 Benefit | L7-L9 Benefit |
|---------|---------------|---------------|
| 1 | ❌ Already done | ⚠️ Small (one buffer anyway) |
| 2+ | ❌ Already done | ✅ Yes (one per thread) |

### Applicability by Content Type

| Content | Benefit? |
|---------|----------|
| Text | ✅ Same |
| Random | ✅ Same |
| Binary | ✅ Same |

Content doesn't affect buffer allocation overhead.

### VERDICT

| Combination | Helps? | Why |
|-------------|--------|-----|
| L1-L6, any | ❌ No | Already implemented |
| L7-L9, 1 thread | ⚠️ Small | Single buffer, less churn |
| L7-L9, 2+ threads | ✅ YES | Avoids per-block allocation |

**Optimization is L7-L9 specific (L1-L6 already has it).**

---

## Phase 3: Fat LTO + codegen-units=1

### What it does
Enables aggressive whole-program optimization and inlining.

### Applicability by Level

| Level | Benefit? | Why |
|-------|----------|-----|
| L1-L6 | ✅ Yes | Better inlining of libdeflate calls |
| L7-L9 | ✅ Yes | Better inlining of zlib-ng calls |

LTO helps everything by enabling cross-crate inlining.

### Applicability by Thread Count

| Threads | Benefit? |
|---------|----------|
| 1 | ✅ Yes |
| 2+ | ✅ Yes |

Threading code also benefits from better inlining.

### Applicability by Content Type

| Content | Benefit? |
|---------|----------|
| Text | ✅ Same |
| Random | ✅ Same |
| Binary | ✅ Same |

### VERDICT

| Combination | Helps? | Why |
|-------------|--------|-----|
| All levels | ✅ YES | Universal optimization |
| All threads | ✅ YES | Universal optimization |
| All content | ✅ YES | Universal optimization |

**This is a universal optimization that helps everything.**

---

## Summary Matrix

```
                        L1-L6           L7-L9
                     1t    2+t       1t    2+t
                    ─────────────────────────────
Phase 1 (TL Comp)   ⚠️     ⚠️        ⚠️     ✅
Phase 2 (TL Buf)    ❌     ❌        ⚠️     ✅
Phase 3 (Fat LTO)   ✅     ✅        ✅     ✅

✅ = Significant benefit
⚠️ = Small/possible benefit  
❌ = No benefit (already done or N/A)
```

---

## Refined Implementation Plan

### Universal Optimizations (Do First)
1. **Fat LTO + codegen-units=1** - Helps everything, easy to add

### L7-L9 Multi-Thread Specific (Main Target)
2. **Thread-local Compress objects** - Biggest win for L9 multi-thread
3. **Thread-local output buffers** - Good win for L9 multi-thread

### Optional: L1-L6 Enhancement
4. **Thread-local libdeflate Compressor** - Smaller gain, but could add

---

## Recommendation

Given the goal is L9 optimization:

| Step | What | Universal? | L9 Specific? |
|------|------|------------|--------------|
| 1 | Fat LTO | ✅ Yes | Also helps |
| 2 | TL Compress (zlib) | No | ✅ L7-L9 multi-thread |
| 3 | TL Buffers (pipelined) | No | ✅ L7-L9 multi-thread |
| 4 | TL libdeflate Compressor | ✅ L1-L6 | No |

**Do steps 1-3 for L9 focus. Step 4 is optional bonus for L1-L6.**

---

## Expected Gains by Combination

After implementing steps 1-3:

| Combination | Current | Expected | Gain |
|-------------|---------|----------|------|
| L9, 1 thread | ~equal to pigz | 5-10% faster | +5-10% |
| L9, 4 threads | 5% slower | 10-15% faster | +15-20% |
| L1, 4 threads | 30% faster | 33% faster | +3% (LTO only) |
| L6, 4 threads | 54% faster | 57% faster | +3% (LTO only) |

The L9 multi-thread combination sees the biggest gain because:
1. It's the only one using per-block zlib Compress allocation
2. It's the only one without thread-local output buffers
3. LTO helps it too
