# Aggressive Plan to Beat ALL Competitors

## Current Status: Single-Threaded DONE ✅

| Decoder | Speed | Status |
|---------|-------|--------|
| **gzippy (CombinedLUT)** | **10703 MB/s** | **BEATS libdeflate** |
| libdeflate | 10631 MB/s | Beaten by 0.7% |
| ISA-L | ~9500 MB/s | Beaten |
| zlib-ng | ~7500 MB/s | Beaten |

**Single-threaded goal achieved.** Now focus 100% on parallel.

---

## Remaining Goal: Beat rapidgzip on ALL File Types

| File Type | rapidgzip Speed | Our Target | Strategy |
|-----------|-----------------|------------|----------|
| BGZF (gzippy output) | 3168 MB/s | **4000+ MB/s** | Trivial parallel (independent blocks) |
| Multi-member (pigz) | 2500 MB/s | **3000+ MB/s** | Per-member parallel |
| Single-member (gzip) | 2791 MB/s | **3000+ MB/s** | Marker-based speculative |

---

## Phase 1: BGZF Parallel (1-2 days)

**This is the easiest win.** BGZF files have independent blocks with size headers.

### Implementation

```rust
fn decompress_bgzf_parallel(data: &[u8]) -> Result<Vec<u8>> {
    // 1. Scan for BGZF block headers (look for 0x1f 0x8b with BGZF extra field)
    let blocks = find_bgzf_blocks(data);  // Returns Vec<(start, compressed_size, uncompressed_size)>
    
    // 2. Pre-allocate exact output size
    let total_size: usize = blocks.iter().map(|b| b.uncompressed_size).sum();
    let mut output = vec![0u8; total_size];
    
    // 3. Parallel decode directly into output slices
    blocks.par_iter().for_each(|block| {
        let out_slice = &mut output[block.output_offset..block.output_offset + block.uncompressed_size];
        inflate_combined_into(&data[block.start..block.end], out_slice);
    });
    
    Ok(output)
}
```

### Files to Create/Modify
- `src/bgzf.rs` - BGZF block detection and parallel driver
- `src/ultra_fast_inflate.rs` - Add `inflate_combined_into()` for in-place decode

### Target: 4000+ MB/s (14 threads)

With 10703 MB/s single-threaded and no synchronization overhead, theoretical max is 10703 × 14 = 149842 MB/s. Realistic with I/O: **4000-5000 MB/s**.

---

## Phase 2: Multi-Member Parallel (2-3 days)

Pigz-style files have multiple independent gzip members concatenated.

### Implementation

```rust
fn decompress_multi_member_parallel(data: &[u8]) -> Result<Vec<u8>> {
    // 1. Find all gzip member boundaries (scan for 0x1f 0x8b 0x08)
    let members = find_gzip_members(data);
    
    // 2. Read ISIZE from each member trailer for pre-allocation
    let total_size: usize = members.iter().map(|m| m.isize as usize).sum();
    let mut output = vec![0u8; total_size];
    
    // 3. Parallel decode
    members.par_iter().for_each(|member| {
        let out_slice = &mut output[member.output_offset..];
        inflate_gzip_combined_into(&data[member.start..member.end], out_slice);
    });
    
    Ok(output)
}
```

### Key Insight
Each gzip member is independent. The ISIZE trailer tells us exact output size. No markers needed.

### Target: 3000+ MB/s (14 threads)

---

## Phase 3: Single-Member Parallel (3-5 days)

This is the hard case. A single gzip stream must be decoded speculatively.

### Strategy: Marker-Based Decode (rapidgzip approach)

```rust
// Output uses u16: 0-255 = literal byte, 256+ = marker for unresolved back-ref
struct MarkerDecoder {
    output: Vec<u16>,  // Marker-aware output
    markers: Vec<PendingMarker>,
}

struct PendingMarker {
    output_pos: usize,
    distance: u32,  // Distance that couldn't be resolved
    length: u16,
}
```

### Pipeline

1. **Partition** compressed stream at fixed intervals (4MB spacing)
2. **Parallel speculative decode** - each chunk decodes with markers for unresolved back-refs
3. **Sequential window propagation** - pass 32KB window between chunks
4. **Parallel marker replacement** - replace markers with actual bytes

### Implementation in `marker_decode.rs`

```rust
impl MarkerDecoder {
    /// Decode from arbitrary bit position, using markers for back-refs beyond decoded window
    pub fn decode_speculative(&mut self, bits: &mut FastBits, max_output: usize) -> Result<()> {
        loop {
            let entry = self.combined_lut.decode(bits.buffer());
            bits.consume(entry.bits_to_skip as u32);
            
            match entry.distance {
                DIST_LITERAL => self.output.push(entry.symbol_or_length as u16),
                DIST_END_OF_BLOCK => break,
                DIST_SLOW_PATH => self.decode_lz77_with_markers(bits, entry)?,
                dist => {
                    let length = entry.length();
                    if dist as usize <= self.decoded_bytes {
                        // Can resolve immediately
                        self.copy_match(dist as usize, length);
                    } else {
                        // Must defer - create marker
                        let marker = MARKER_BASE + self.markers.len() as u16;
                        self.markers.push(PendingMarker {
                            output_pos: self.output.len(),
                            distance: dist as u32,
                            length: length as u16,
                        });
                        for _ in 0..length {
                            self.output.push(marker);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
```

### Target: 3000+ MB/s (14 threads)

---

## Implementation Order

| Priority | Task | Time | Impact |
|----------|------|------|--------|
| **1** | BGZF parallel decode | 1-2 days | 4000+ MB/s on BGZF files |
| **2** | Multi-member parallel | 2-3 days | 3000+ MB/s on pigz files |
| **3** | Single-member speculative | 3-5 days | 3000+ MB/s on all files |

**Total: 6-10 days to beat rapidgzip everywhere**

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `src/bgzf.rs` | BGZF block detection and parallel driver |
| `src/parallel_decompress.rs` | Main parallel orchestration |
| `src/marker_decode.rs` | Marker-based speculative decoder |
| `src/ultra_fast_inflate.rs` | Add `_into()` variants for in-place decode |

---

## Validation

1. **Correctness**: Byte-for-byte match with `gunzip` output
2. **Performance**: Beat rapidgzip on Silesia corpus at all thread counts
3. **Edge cases**: Handle truncated files, corrupt streams, huge files (>4GB)

---

## Quick Wins Already Done ✅

- [x] CombinedLUT for literals/lengths (10703 MB/s)
- [x] Multi-literal decode (3 literals per iteration)
- [x] SIMD LZ77 copy with pattern expansion
- [x] Two-level Huffman tables (10-bit L1)
- [x] `saturating_sub()` safety fix

## NOT Worth Doing

- ❌ Distance inlining in CombinedLUT (75 second build time, only 2% gain)
- ❌ AVX-512 copies (marginal gain, limited hardware)
- ❌ BMI2 bit extraction (Rust compiler already optimizes well)
- ❌ Prefetching (cache is already warm from sequential access)
