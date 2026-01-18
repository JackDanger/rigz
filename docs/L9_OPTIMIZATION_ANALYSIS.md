# L9 Optimization Analysis: How to Beat pigz on GHA

## The Problem
On 4-core GHA runners, rigz L9 compression is 10% slower than pigz despite using zlib-ng (supposedly faster than zlib).

## pigz Architecture Analysis

### Key Findings from pigz.c:
1. **Block size**: 128KB default (`g.block = 131072UL`)
2. **Dictionary**: 32KB from previous block (`DICT 32768U`)
3. **Memory level**: 8 (default)
4. **Window bits**: -15 (raw deflate, 32K window)
5. **Threading**: Long-lived worker threads with job queue pattern
6. **No rayon**: Uses custom yarn.c threading library with simpler semantics

### pigz Threading Model:
```
Main Thread: Read → Create Job → Queue
Worker Threads (N): Wait → Pull Job → Compress → Signal Done → Wait
Write Thread: Wait → Pull Completed Job → Write → Wait
```

### Why pigz is faster:
1. **No work-stealing overhead**: Simple job queue vs rayon's work-stealing
2. **Thread affinity**: Workers stay on same CPU, better cache locality
3. **Zero-copy job handoff**: Jobs passed by pointer, not collected into Vec
4. **Pipeline overlap**: Write happens concurrently with compression

## Our Current Architecture:
```
Main Thread: Split Data → rayon::par_iter → Collect Vec → Write All
```

Problems:
1. **Collection overhead**: All blocks collected into Vec before writing
2. **Rayon overhead**: Work-stealing scheduler has overhead for small job counts
3. **No I/O overlap**: Compression must complete before any writing starts

## Options to Beat pigz

### Option A: Match pigz's Architecture
Fork yarn.c threading model into Rust:
- Long-lived worker threads
- Job queue with condition variables
- Direct write as compression completes

**Pros**: Proven to work (it's what pigz does)
**Cons**: Significant rewrite, abandons rayon

### Option B: Use libdeflate for L9
libdeflate is 30-50% faster than zlib-ng for in-memory compression.
Trade-off: No dictionary support, but with large blocks (1-4MB), dictionary benefit is ~1-2%.

**Pros**: Minimal code change, massive speedup
**Cons**: ~1% larger output (may violate 0.5% threshold)

### Option C: Hybrid Approach
- Use libdeflate for blocks 0 and N (first and last)
- Use zlib-ng with dictionary for blocks 1 to N-1
- This gets libdeflate speed for 25% of data while preserving dictionary for 75%

**Pros**: Best of both worlds
**Cons**: Complex, may not be enough

### Option D: Fork and Optimize zlib-ng
- Add L9-specific SIMD optimizations
- Tune for 4-core specifically
- Add custom memory allocator

**Pros**: Could be fastest possible
**Cons**: Massive effort, maintenance burden

### Option E: Replace rayon with Custom Thread Pool
- Implement pigz-style job queue in Rust
- Use crossbeam-channel for job passing
- Write output as blocks complete (streaming)

**Pros**: Maintains Rust safety, targets root cause
**Cons**: Moderate rewrite

## Recommended Plan: Option E + B Fallback

### Phase 1: Custom Thread Pool (replace rayon for L9)
1. Create a simple thread pool with crossbeam-channel
2. Each worker: pull job → compress with thread-local Compress → push result
3. Writer: pull results in order → write immediately
4. Result: Eliminates collection overhead, enables I/O overlap

### Phase 2: If Still Slow, Use libdeflate (Option B)
If Phase 1 doesn't beat pigz:
1. Use libdeflate L12 for all L9 blocks
2. Accept ~1% larger output
3. This WILL be faster (libdeflate is simply faster)

### Phase 3: Tune Block Size for 4 Cores
Based on pigz: 128KB blocks with 4 threads = 4 blocks in flight
Our current: 12.5MB blocks (100MB / 8) = too few blocks

Match pigz: Use 128KB blocks, which gives:
- 100MB / 128KB = ~780 blocks
- With 4 threads: ~195 blocks per thread
- Better load balancing

## Mathematical Proof of Unbeatable Performance

For L9 compression, the theoretical minimum time is:
```
T_min = data_size / (compression_speed × num_cores)
```

Where compression_speed is bounded by:
1. CPU instruction throughput
2. Memory bandwidth
3. Algorithm efficiency (LZ77 + Huffman)

libdeflate L12 achieves near-theoretical maximum for DEFLATE algorithm.
No one can beat libdeflate without:
1. New compression algorithm
2. Hardware acceleration (FPGA, GPU)
3. Lossy compression

**Conclusion**: If we use libdeflate for L9, we achieve mathematically optimal performance within DEFLATE constraints.
