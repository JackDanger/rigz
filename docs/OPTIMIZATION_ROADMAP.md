# Optimization Roadmap: Surpassing libdeflate

**Goal**: Exceed libdeflate's decompression speed in ALL scenarios.

**Current Status** (Jan 2026 - UPDATED):
- Simple data: 50% of libdeflate (614 vs 1,228 MB/s on silesia)
- Complex data: 50% of libdeflate (similar to silesia benchmark)
- BGZF parallel: **2,541 MB/s with 8 threads** (3.86x speedup)

**Optimizations Implemented**:
1. 3-literal decode chain with entry preloading (libdeflate-style)
2. TurboBits with overlapping load technique (libdeflate-style)
3. PackedLUT for bitsleft -= entry optimization
4. Refactored ASM decode for runtime BMI2 detection (#[target_feature(enable = "bmi2")])

**Remaining Gap Analysis**:
The 50% gap to libdeflate comes from:
1. libdeflate's entry format encodes length base + extra bit count in one u32
2. libdeflate extracts extra bits with `EXTRACT_VARBITS8(saved_bitbuf, entry)`
3. libdeflate pre-computes distance into entry when code fits in LUT bits
4. libdeflate's C loop has tighter register allocation than Rust's match

---

## Status: Turbo Path INTEGRATED (Jan 2026)

**TurboBits + PackedLUT now in production hot path!**

| Function | Location | Status |
|----------|----------|--------|
| `decode_huffman_turbo` | `bgzf.rs` | ✅ **INTEGRATED** - Now the default decode path |
| `inflate_into_turbo` | `bgzf.rs` | ✅ **INTEGRATED** - Called by `inflate_into()` |
| `TurboBits` | `two_level_table.rs` | ✅ Fixed - libdeflate-style overlapping loads |
| `PackedLUT` | `packed_lut.rs` | ✅ Ready - libdeflate-style entries |
| `decode_huffman_asm_x64` | `bgzf.rs` | ⏳ Pending - Would give additional gains |

**Bug Fixed (Jan 2026)**:
- `TurboBits::refill_branchless()` - Fixed overlapping load technique
- The issue was that pos tracked next unread byte, but libdeflate re-reads overlapping bytes
- Fix: `pos += 7 - ((bits >> 3) & 7)` instead of `pos += (64 - bits) / 8`

**Current Performance**:
- All 163 tests pass
- Turbo path matches ultra_fast_inflate speed (~570 MB/s on silesia)
- No regression from standard path

---

## Phase 1: Integration (COMPLETED - Jan 2026)

### 1.1 Integrate TurboBits into main decode path ✅
- [x] Created `TurboBits` with branchless refill: `bits |= 56` trick
- [x] Added `TurboBits::align()` for stored block support
- [x] Infrastructure in `inflate_into_turbo()` is ready
- [x] Fixed refill bug: use libdeflate's overlapping load technique

### 1.2 Integrate PackedLUT for multi-symbol decode ✅
- [x] Created `PackedLUT` in turbo decode path
- [x] Implemented `bitsleft -= entry` via `consume_entry()`
- [x] Multi-literal decode (2+ literals per iteration)
- [x] All 163 tests pass

### 1.3 Activate ASM decode loop on x86_64 ⏳
- [ ] Call `decode_huffman_asm_x64` for dynamic blocks on x86_64+BMI2
- [ ] Runtime feature detection: `if is_x86_feature_detected!("bmi2")`
- [ ] Fallback to `decode_huffman_turbo` otherwise

### 1.4 Integrate triple-symbol decode (ISA-L's secret)
- [x] Current implementation decodes 2+ literals per iteration
- [ ] Extend to 3 literals using packed entry format
- [ ] Modify entry format: `[count:2][sym2:9][sym1:9][bits:8]`

---

## Phase 2: Deep Optimizations (Target: 95%)

### 2.1 Speculative symbol precomputation
- [ ] Precompute TWO future table entries while match copy in flight
- [ ] Hides two memory latencies instead of one
```rust
let future0 = table[(bits >> 0) & MASK];
let future1 = table[(bits >> 11) & MASK];
copy_match(...);  // While these prefetch
// Choose correct based on actual bits consumed
```

### 2.2 Branch-free distance decode
- [ ] Remove branch for subtable lookup
- [ ] Use computed offset: `subtable_offset = (entry >> 16) + secondary_bits`
- [ ] Single memory access for all distance codes

### 2.3 Window double-buffer (no wrap checks)
- [ ] Allocate 64KB window instead of 32KB circular
- [ ] Copy window[0:32KB] → window[32KB:64KB] after each 32KB
- [ ] Eliminates ALL modulo operations in match copy

### 2.4 SIMD match copy improvements
- [ ] Use `rep movsb` for matches > 64 bytes on x86_64 (ERMS)
- [ ] AVX-512 64-byte copies for large matches
- [ ] NEON ldp/stp for ARM64 (already partially done)

### 2.5 CRC32 overlap with decode
- [ ] Compute CRC32 using hardware instructions (x86 `_mm_crc32_u64`, ARM CRC)
- [ ] Interleave CRC with literal writes (hide latency)
- [ ] For BGZF: parallel per-block CRC verification

---

## Phase 3: Architecture-Specific (Target: 105%)

### 3.1 x86_64 with AVX2
- [ ] SIMD Huffman decode: 8-way parallel gather (`vpgatherdd`)
- [ ] Batch decode 8 literals simultaneously
- [ ] Use BMI2 `_bzhi_u64`, `_shrx_u64` for bit extraction

### 3.2 x86_64 with AVX-512
- [ ] 16-way parallel symbol decode
- [ ] `vpmovzxbd` for byte-to-dword expansion
- [ ] `vpscatterdd` for parallel output writes

### 3.3 Apple Silicon (M1/M2/M3)
- [ ] 128-byte cache line awareness (vs 64-byte x86)
- [ ] NEON `vld1q_u8` / `vst1q_u8` for 16-byte copies
- [ ] ARM CRC32 instructions for checksum

### 3.4 AMD Zen4
- [ ] Tune for Zen's different branch predictor
- [ ] Optimize for L3 cache latency patterns
- [ ] Consider `rep stosb` vs SIMD for memset

---

## Phase 4: Algorithmic Innovations (Target: 115%+)

### 4.1 JIT-Generated Decode Paths
- [ ] Build per-Huffman-tree specialized decoder at runtime
- [ ] Use `dynasmrt` or `cranelift` for runtime codegen
- [ ] Cache generated functions for repeated tables
- [ ] Eliminates ALL table lookups for known tree shapes

Implementation sketch:
```rust
fn jit_decoder_for_tree(tree: &HuffmanTree) -> CompiledDecoder {
    let mut asm = Assembler::new();
    // Generate decision tree as native branches
    for code in tree.codes() {
        // emit: test bits, branch, write symbol
    }
    asm.finalize()
}
```

### 4.2 GDeflate substream decode
- [ ] Detect GDeflate format (NVIDIA's parallel deflate)
- [ ] Decode 32 interleaved substreams in parallel
- [ ] Even without GPU: use AVX-512 for 16-way parallel

### 4.3 Adaptive symbol width
- [ ] Detect data characteristics in first N blocks
- [ ] Select optimized path: SINGLE_SYM, DOUBLE_SYM, TRIPLE_SYM
- [ ] Like ISA-L's adaptive mode selection

### 4.4 Speculative parallel single-member (rapidgzip-style)
- [ ] Partition large files at 4MB boundaries
- [ ] Decode speculatively with markers
- [ ] Resolve markers with window propagation
- [ ] Already have `marker_decode.rs` - INTEGRATE IT

### 4.5 Profile-Guided Optimization (PGO)
- [ ] Run `scripts/pgo-build.sh` in CI for release builds
- [ ] Profile with representative test corpus
- [ ] Expected gain: 5-15% from compiler optimizations

---

## Phase 5: Extreme Optimizations (Research)

### 5.1 Hardware acceleration hooks
- [ ] Detect IBM POWER NX / z15 NXU deflate units
- [ ] Intel QAT integration for supported platforms
- [ ] FPGA acceleration path for cloud deployments

### 5.2 Memory-mapped streaming
- [ ] mmap input file, prefetch 64KB ahead
- [ ] mmap output file for zero-copy writes
- [ ] Use `madvise(WILLNEED)` for prefetch hints

### 5.3 NUMA-aware parallel decode
- [ ] Pin worker threads to NUMA nodes
- [ ] Allocate output buffers on correct node
- [ ] Reduce cross-socket memory traffic

### 5.4 GPU offload (optional)
- [ ] CUDA/OpenCL path for decompression
- [ ] Useful for batch processing (many files)
- [ ] GDeflate native GPU decode

---

## Realistic Implementation Order

**Week 1: Integration (10-20% gain)**
1. Wire up `decode_huffman_turbo` → main path
2. Wire up `decode_huffman_asm_x64` → x86_64 path  
3. Replace `FastBits` → `TurboBits`
4. Replace `CombinedLUT` → `PackedLUT`
5. Benchmark after each change

**Week 2: Deep Opts (10-15% gain)**
1. Triple-symbol decode
2. Branch-free distance
3. Window double-buffer
4. CRC32 overlap

**Week 3: Architecture (5-10% gain)**
1. AVX2 SIMD Huffman
2. Apple Silicon tune
3. AMD Zen4 tune

**Week 4: JIT + Advanced (10-20% gain)**
1. JIT decoder prototype
2. GDeflate detection
3. PGO build in CI

---

## Benchmarking Protocol

After each optimization:

```bash
# Quick benchmark
cargo bench --bench decompress -- --warm-up-time 2

# Full comparison
GZIPPY_DEBUG=1 ./target/release/gzippy -d benchmark_data/silesia-gzippy.tar.gz -c > /dev/null

# Compare to libdeflate
hyperfine --warmup 3 \
  'gzippy -d silesia.gz -c > /dev/null' \
  'libdeflate-gunzip -c silesia.gz > /dev/null'
```

---

## Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Simple data | 62% | 110% | JIT + triple-sym |
| Complex data | 45% | 106% | ASM loop + triple-sym |
| BGZF 8 threads | 3600 MB/s | 5000 MB/s | Better work distribution |
| Multi-member | 2800 MB/s | 4000 MB/s | Lock-free output |
| Single-member | libdeflate | 106% | JIT + all optimizations |

---

## References

- libdeflate: `libdeflate/lib/deflate_decompress.c`
- ISA-L: `isa-l/igzip/igzip_decode_block_stateless.asm` (triple-symbol)
- rapidgzip: `rapidgzip/librapidarchive/src/rapidgzip/` (marker decode)
- GDeflate: IETF draft `draft-uralsky-gdeflate-00`
