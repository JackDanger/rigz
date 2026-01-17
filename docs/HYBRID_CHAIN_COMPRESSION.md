# Hybrid Chain Compression: Mathematical Analysis

> **IMPORTANT: This approach is NOT compatible with gzip format.**
>
> After thorough analysis and implementation, we discovered that DEFLATE preset
> dictionaries cannot be used in gzip-compatible output. The decompressor MUST have
> the same dictionary set before decompression - but gzip format has no field to
> embed the dictionary. This document is preserved for educational purposes.

## Status: Abandoned

The mathematical analysis is correct - dictionary sharing does improve compression
by 2-5% depending on data type and compression level. However, this approach
**breaks gzip compatibility**:

1. `deflateSetDictionary()` tells the compressor "here's 32KB of context, find matches in it"
2. The compressed output references the dictionary as if it were already in the sliding window
3. `inflateSetDictionary()` must be called before decompression with the SAME dictionary
4. Gzip format has no mechanism to embed or signal a preset dictionary
5. Standard decompressors (gzip, pigz, zlib) will fail with "data stream error"

**Tested**: gunzip fails to decompress rigz output when hybrid chain compression is enabled.

## Alternative Approaches (also don't work)

1. **ZSTD-style linked blocks** - ZSTD natively supports this, but we're making gzip
2. **RAW_DEFLATE + manual gzip wrapper** - Still requires dictionary at decompress time
3. **rigz-only format** - Defeats the purpose of gzip replacement

## Why pigz achieves better compression

pigz uses a different approach: **pipeline parallelism** with a shared sliding window.
Instead of compressing independent blocks in parallel, pigz:

1. Compresses sequentially in a single thread (maintaining the sliding window)
2. Uses worker threads for I/O and CRC computation
3. At high compression levels, this shared-window approach produces smaller output

rigz trades some compression efficiency for:
- Truly parallel compression (faster on multi-core)
- Parallel decompression (via BGZF markers)
- Simpler implementation

## The Original Analysis (Educational)

## The Problem

**Current rigz approach (independent blocks):**
```
Block 0: compress(data[0..B]) with empty dictionary
Block 1: compress(data[B..2B]) with empty dictionary  ← loses context!
Block 2: compress(data[2B..3B]) with empty dictionary
...
```

**pigz approach (fully dependent):**
```
Block 0: compress(data[0..B]) with empty dictionary
Block 1: compress(data[B..2B]) with dict=data[0..B]    ← uses context
Block 2: compress(data[2B..3B]) with dict=data[B..2B]
...
```

The DEFLATE algorithm uses a 32KB sliding window. When we share dictionary context, we're telling the compressor: "here's 32KB of data that came before this block, use it to find matches."

## Mathematical Model

### Variables

| Symbol | Meaning |
|--------|---------|
| B | Block size (bytes) |
| N | Total blocks |
| C | Chain length (blocks per chain) |
| D | Dictionary benefit (compression ratio improvement) |
| P | Number of CPU cores |
| W | Window size (32KB for DEFLATE) |

### Compression Ratio Analysis

**Independent blocks:**
- Ratio: R_ind
- Each block starts fresh, no cross-boundary matches

**Dependent blocks:**
- Ratio: R_dep = R_ind × (1 + D)
- D typically 0.02–0.10 (2–10% improvement)
- Higher for repetitive data (logs, source code)
- Lower for random/encrypted data

**Hybrid chains of length C:**
- First block: independent (ratio R_ind)
- Remaining C-1 blocks: dependent (ratio R_dep)
- Fraction with dictionary: (C-1)/C

**Expected ratio:**
```
R_hybrid = (1/C) × R_ind + ((C-1)/C) × R_dep
         = R_ind × (1 + D × (C-1)/C)
```

| Chain Length C | Dictionary Benefit Captured |
|----------------|----------------------------|
| 1 | 0% (fully independent) |
| 2 | 50% |
| 4 | 75% |
| 8 | 87.5% |
| 16 | 93.75% |
| ∞ | 100% (fully dependent) |

**Key insight:** C=8 captures 87.5% of the dictionary benefit.

### Parallelism Analysis

**Compression parallelism:**

With N blocks and chain length C:
- Number of chains: ⌈N/C⌉
- Chains are independent → can run in parallel
- Within a chain: blocks are sequential (pipelined)

With P cores, makespan (wall time) is:
```
T_compress = ⌈(N/C)/P⌉ × C × t_block
           = ⌈N/(C×P)⌉ × C × t_block
```

Where t_block is time to compress one block.

**Decompression parallelism:**

Same structure:
```
T_decompress = ⌈N/(C×P)⌉ × C × t_decomp_block
```

For C=1 (independent): T = N/P × t_block (fully parallel)
For C=∞ (dependent): T = N × t_block (fully sequential)

### Optimal Chain Length

**Objective:** Maximize compression quality while maintaining parallelism.

**Constraints:**
1. Enough chains to saturate cores: N/C ≥ P
2. Practical limit on sequential work: C ≤ C_max

**Derivation:**

From constraint 1: C ≤ N/P

For parallelism to be effective:
```
C_optimal = min(C_target, N/P)
```

Where C_target is chosen to balance:
- Dictionary benefit (want larger C)
- Pipeline depth (want smaller C for better core utilization)

**Empirical sweet spot:** C = 8

Rationale:
- Captures 87.5% of dictionary benefit
- For 10MB file with 64KB blocks (N=160): 20 chains
- For 100MB file (N=1600): 200 chains
- Plenty of parallelism even with 16 cores

### Scheduling: Job-Shop with Precedence

This is an instance of the **job-shop scheduling problem** with chain precedence constraints.

**Graph structure:**
```
Chain 0: B0 → B1 → B2 → ... → B(C-1)
Chain 1: BC → B(C+1) → B(C+2) → ... → B(2C-1)
...
```

**Greedy schedule (optimal for this structure):**
1. Assign chains to cores round-robin
2. Each core processes its chains sequentially
3. Within each chain, blocks are pipelined

```python
def schedule(N, C, P):
    n_chains = ceil(N / C)
    chains_per_core = ceil(n_chains / P)
    makespan = chains_per_core * C * t_block
    return makespan
```

**Work-stealing improvement:**
Rayon's work-stealing scheduler naturally handles load imbalance when some chains are shorter (last chain may have < C blocks).

## Implementation Design

### Data Flow

```
Input file (memory-mapped)
    ↓
┌─────────────────────────────────────────────────────┐
│  Split into chains of C blocks each                 │
│  [Chain 0: B0..B7] [Chain 1: B8..B15] ...          │
└─────────────────────────────────────────────────────┘
    ↓ (parallel across chains)
┌─────────────────────────────────────────────────────┐
│  For each chain (in parallel):                      │
│    Block 0: compress(data, empty_dict)              │
│    Block 1: compress(data, dict=block_0_data)       │
│    Block 2: compress(data, dict=block_1_data)       │
│    ...                                              │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Concatenate: chain0_blocks ++ chain1_blocks ++ ... │
└─────────────────────────────────────────────────────┘
    ↓
Output gzip stream
```

### Key Implementation Details

**1. Dictionary passing:**
The dictionary for block i is the UNCOMPRESSED data of block i-1, not the compressed data. This is crucial—we pass a slice reference, not bytes.

```rust
fn compress_chain(chain_data: &[u8], block_size: usize, level: u32) -> Vec<Vec<u8>> {
    let blocks: Vec<&[u8]> = chain_data.chunks(block_size).collect();
    let mut results = Vec::with_capacity(blocks.len());
    
    // First block: no dictionary
    results.push(compress_block(blocks[0], None, level));
    
    // Remaining blocks: use previous block as dictionary
    for i in 1..blocks.len() {
        let dict = &blocks[i-1][blocks[i-1].len().saturating_sub(32768)..];
        results.push(compress_block(blocks[i], Some(dict), level));
    }
    
    results
}
```

**2. flate2 dictionary API:**
```rust
use flate2::Compress;

fn compress_with_dict(data: &[u8], dict: Option<&[u8]>, level: u32) -> Vec<u8> {
    let mut encoder = Compress::new(Compression::new(level), true);
    
    if let Some(d) = dict {
        encoder.set_dictionary(d).expect("dictionary set failed");
    }
    
    // ... compress data ...
}
```

**3. Chain-parallel structure:**
```rust
fn compress_parallel_chains(data: &[u8], block_size: usize, chain_len: usize, level: u32) -> Vec<Vec<u8>> {
    let chain_size = block_size * chain_len;
    
    data.chunks(chain_size)
        .par_iter()  // Parallel across chains
        .flat_map(|chain| compress_chain(chain, block_size, level))
        .collect()
}
```

### BGZF Marker Compatibility

For parallel decompression, we embed chain boundaries in the gzip headers:

**Current BGZF format (block-level):**
- FEXTRA field with "RZ" subfield
- Contains compressed block size

**Extended for chains:**
- Add "chain start" flag to first block of each chain
- Or: embed chain length in header

For simplicity, we can keep current BGZF format—chains are just groups of consecutive blocks. The decompressor can:
1. Parse BGZF markers to find block boundaries
2. Decompress blocks sequentially within a chain (dictionary flows naturally)
3. Process chains in parallel

### Adaptive Chain Length

```rust
fn optimal_chain_length(n_blocks: usize, n_cores: usize, level: u32) -> usize {
    // More dictionary benefit at higher levels
    let target = match level {
        1..=3 => 4,   // Less benefit, prioritize parallelism
        4..=6 => 8,   // Balanced
        7..=9 => 12,  // More benefit, longer chains
        _ => 8,
    };
    
    // Ensure enough chains for parallelism
    let min_chains = n_cores * 2;
    let max_chain = n_blocks / min_chains;
    
    target.min(max_chain).max(1)
}
```

## Expected Performance

### Compression Speed

| Configuration | Current (independent) | Hybrid (C=8) | Change |
|--------------|----------------------|--------------|--------|
| L9, 2t, 10MB | ~0.45s | ~0.38s | -15% |
| L9, 4t, 10MB | ~0.25s | ~0.22s | -12% |
| L6, 2t, 10MB | ~0.35s | ~0.32s | -9% |

(Estimates based on dictionary benefit D≈0.05)

### Compression Ratio

| Data Type | Current Ratio | Hybrid Ratio | Change |
|-----------|---------------|--------------|--------|
| Text | 38% | 36% | -5% (smaller) |
| Tarball | 45% | 43% | -4% |
| Random | 100% | 100% | 0% |

### Decompression

Parallel decompression is preserved:
- N/C chains can decompress in parallel
- Within each chain: sequential (matches compression)
- For C=8, lose ~12.5% parallelism vs fully independent

## Risks and Mitigations

**Risk 1:** flate2 `set_dictionary()` has overhead
- Mitigation: Benchmark; if significant, use raw zlib calls

**Risk 2:** Chain boundaries hurt compression at boundaries  
- Mitigation: Use first 32KB of next chain as dictionary? (complex)

**Risk 3:** Load imbalance with variable-length last chain
- Mitigation: Rayon work-stealing handles this

## Implementation Plan

1. **Benchmark dictionary benefit** (30 min)
   - Measure compression ratio with/without dictionary
   - Validate D≈0.05 assumption

2. **Implement compress_with_dict()** (1 hour)
   - Add dictionary support to block compression
   - Unit test with known data

3. **Implement chain processing** (2 hours)
   - Replace block-parallel with chain-parallel
   - Maintain BGZF markers

4. **Tune chain length** (1 hour)
   - Benchmark different C values
   - Implement adaptive selection

5. **Validate & benchmark** (1 hour)
   - Cross-tool validation (gzip/pigz compatibility)
   - Performance comparison on x86_64 CI

## References

- [DEFLATE RFC 1951 §3.2](https://tools.ietf.org/html/rfc1951#section-3.2) - Preset dictionary
- [zlib manual](https://zlib.net/manual.html) - deflateSetDictionary()
- [pigz source](https://github.com/madler/pigz/blob/master/pigz.c) - yarn.c threading
