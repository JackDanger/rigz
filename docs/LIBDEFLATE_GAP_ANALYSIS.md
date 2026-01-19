# gzippy vs libdeflate: Gap Analysis

## Current Performance (Measured Jan 2026)

| Data Type | gzippy | libdeflate | Ratio |
|-----------|--------|------------|-------|
| Simple 1MB | 17,000 MB/s | 24,000 MB/s | **71%** |
| Complex silesia | 667 MB/s | 1,279 MB/s | **52%** |
| BGZF parallel 8T | 3,893 MB/s | - | **2.9x libdeflate 1T** ✅ |

**Key insight**: Our parallel implementation beats libdeflate on BGZF.
For single-member complex files, we're at 52% - closing this is hard.

**Goal**: Close the 48% gap on complex data (nice-to-have, not blocking).

---

## Root Cause Analysis

### Where Time Goes (complex data)

| Operation | gzippy | libdeflate | Issue |
|-----------|--------|------------|-------|
| Table lookup | 2 lookups (primary + L2) | 1 lookup (11-bit) | **+1 branch** |
| Bit consume | `bits >>= n; bitsleft -= n` | `bitsleft -= entry` | **+1 op** |
| Extra bits | Separate read after decode | Packed in entry | **+1 lookup** |
| Refill | Always check `bitsleft < 56` | `(u8)bitsleft < N` | **+1 branch** |
| Next entry | After current symbol done | **Preloaded during copy** | **+latency** |

### libdeflate's Key Techniques

1. **Packed Entry Format** (lines 495-496):
   ```c
   length = entry >> 16;
   length += EXTRACT_VARBITS8(saved_bitbuf, entry) >> (u8)(entry >> 8);
   ```
   - High 16 bits: base value
   - Bits 8-13: extra bits position  
   - Low 8 bits: total bits to consume
   - **One entry = complete decode info**

2. **`bitsleft -= entry`** (line 545):
   ```c
   bitsleft -= entry;  // subtracts whole u32, only low byte matters
   ```
   - Avoids masking out code length
   - Compiler generates faster code

3. **Preload During Copy** (lines 571-572):
   ```c
   entry = d->u.litlen_decode_table[bitbuf & litlen_tablemask];
   REFILL_BITS_IN_FASTLOOP();
   // then do copy...
   ```
   - Memory fetch happens while copy executes
   - Hides ~100 cycle L2 latency

4. **Multi-Literal Fast Path** (lines 400-417):
   ```c
   if (entry & HUFFDEC_LITERAL) {
       lit = entry >> 16;  // second literal packed in same entry!
       entry = table[bitbuf & mask];
       *out_next++ = lit;
       continue;
   }
   ```
   - Two literals decoded from one table entry
   - Only works when codes align

---

## Implementation Plan

### Phase 1: Packed Entry Format (Target: +20%)

```rust
// Current: 2 reads
let (symbol, code_len) = table.decode(bits.buffer());
let extra = bits.read(EXTRA_BITS[symbol]);

// Target: 1 read
let entry = table[bits.buffer() & MASK];
let symbol = (entry >> 16) as u16;
let extra = (saved_buf >> ((entry >> 8) & 0x3F)) & ((1 << (entry & 0xF)) - 1);
bits.consume_entry(entry);  // bitsleft -= entry
```

**Files to modify**:
- `src/packed_decode.rs` (new) - packed entry table + decode loop
- `src/ultra_fast_inflate.rs` - integrate packed decode

### Phase 2: Entry Preload (Target: +15%)

```rust
// Current: sequential
let entry = table.decode(bits.buffer());
copy_match(dst, src, length);
// then decode next...

// Target: overlapped
let next_entry = table.decode(bits.buffer());  // preload
copy_match(dst, src, length);                   // execute during fetch
// use next_entry...
```

**Files to modify**:
- `src/packed_decode.rs` - preload in match copy path

### Phase 3: Branchless Refill (Target: +5%)

```rust
// Current
if self.bitsleft < 56 { self.refill(); }

// Target (libdeflate style)
if (self.bitsleft as u8) < NEEDED_BITS - PRELOAD_SLACK {
    // only refill when truly needed
}
```

---

## Experimental Results (Jan 2026)

### Phase 1: Packed Entry Format - IMPLEMENTED

Created `packed_decode.rs` with libdeflate-style packed entries:
- 12-bit primary table (4096 entries)
- Packed u32 entries with base, extra bits count, total bits
- `consume_entry()` using low byte for bit count
- TwoLevelTable fallback for codes > 12 bits

**Result**: No significant improvement over existing TwoLevelTable.
- Simple data: 17,500 MB/s (same as TwoLevelTable)
- Complex data: Not fully working yet (edge cases with distance codes)

**Why it didn't help**: 
1. TwoLevelTable is already fast (one primary lookup + rare secondary)
2. Extra bits extraction from saved_buf adds complexity
3. Rust compiler optimizes our existing code well

### Key Finding

The remaining gap between us and libdeflate is NOT in the table format.
It's in the tight C decode loop that libdeflate uses:
- Hand-optimized for register allocation
- Inline assembly for critical paths on some platforms
- Decades of micro-optimization

**Recommendation**: Accept 52% on complex single-member files.
Focus on parallel paths where we beat libdeflate by 2.9x.

---

## Non-Goals

- Matching libdeflate on complex single-member (diminishing returns)
- Assembly code (maintain pure Rust for portability)
- Breaking existing API
