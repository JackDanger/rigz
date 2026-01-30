# LLVM-Parity Assembly Implementation Plan

## Goal

Achieve 100% parity with LLVM-generated assembly for the Huffman decode hot loop,
then iterate to exceed it.

## Current State (Updated Jan 2026)

### v4 Implementation Complete
- **v4 ASM:** 70-77% of Rust baseline (~1000 MB/s vs ~1300 MB/s)
- **v4 is correct:** Full SILESIA corpus (212M bytes) decodes correctly
- **v4 handles:** Literals, lengths, distances, subtables, SIMD match copy, 4-literal batching

### Performance Comparison
| Decoder | MB/s | % of Rust |
|---------|------|-----------|
| Rust (LLVM) | 1356 | 100% |
| v4 (LLVM-parity) | 954 | 70% |
| v3 (pure ASM) | 958 | 71% |
| v2 (macros) | 865 | 64% |
| v1 (primitives) | 676 | 50% |

### Key Bugs Fixed in v4
1. **Entry lookup order:** Must lookup AFTER refill, not before
2. **Check before consume:** Check entry type before consuming bits
3. **Distance subtable:** Must handle in ASM to avoid mid-decode exit
4. **Double lookup:** Was looking up entry twice per iteration

## Gap Analysis: Why LLVM is 30% Faster

### What LLVM Does Better

1. **Instruction Scheduling**
   - LLVM interleaves independent operations to hide latency
   - Loads are issued early, computation happens while waiting
   - Our v4 has sequential dependencies

2. **Register Allocation**
   - LLVM uses more registers and avoids spills
   - Our v4 is limited by inline asm register constraints

3. **Entry Preloading**
   - LLVM preloads the next entry BEFORE consuming current
   - This overlaps table lookup with current entry processing

4. **Bitbuf Handling**
   - LLVM uses the "subtract full entry" pattern more aggressively
   - Avoids extra masking operations

### What v4 Does Well

1. **SIMD match copy** - Uses LDP/STP for 32-byte copies
2. **Literal batching** - Decodes up to 4 literals before loop check
3. **Correct handling** - All paths work correctly

### Remaining Optimizations to Try

1. **Preload pattern:** Lookup next entry before writing current literal
2. **8-literal batching:** Decode 8 literals at once
3. **Packed literal writes:** Combine multiple literals into one u32/u64 store
4. **Better refill:** Only refill when absolutely necessary
5. **Profile-guided:** Use hardware counters to find specific bottlenecks

## Strategy

We will:
1. Extract LLVM's exact assembly for the hot loop
2. Translate it instruction-by-instruction to our inline ASM
3. Verify identical behavior
4. Measure performance parity
5. Then add optimizations LLVM can't do

---

## Phase 1: Extract and Analyze LLVM's Hot Loop

### Step 1.1: Generate Clean Assembly

```bash
cd /Users/jackdanger/www/gzippy
RUSTFLAGS="--emit asm" cargo build --release
```

### Step 1.2: Extract the Decode Function

The function `decode_huffman_libdeflate_style` starts at line 69512 in:
`target/release/deps/gzippy-*.s`

Key labels in LLVM's code:
- `LBB250_7`: Main loop start (refill check)
- `LBB250_9`: After refill, check entry type
- `LBB250_12`: No-refill path
- `LBB250_13`: Literal decode
- `LBB250_16`: Subtable/exceptional path
- `LBB250_21-23`: Distance decode
- `LBB250_27-39`: Match copy

### Step 1.3: Document LLVM's Register Allocation

From LLVM's generated code:
```
x0  = bits struct pointer
x1  = output pointer  
x2  = output length
x3  = out_pos
x4  = litlen table pointer
x5  = (unused by LLVM in hot path)
x6  = dist table pointer
x8  = input data pointer
x9  = in_fastloop_end
x10 = in_pos
x11 = bitbuf
x12-x19 = scratch
x20-x28 = callee-saved (loop constants)
```

---

## Phase 2: Create LLVM-Matched ASM Decoder

### Step 2.1: New Function - `decode_huffman_asm_llvm`

This will be a new decoder that exactly mirrors LLVM's structure.

```rust
// File: src/asm_decode.rs

/// LLVM-parity decoder: matches LLVM's generated assembly exactly
/// 
/// This decoder mirrors the structure, register usage, and instruction
/// ordering of LLVM's compiled decode_huffman_libdeflate_style.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn decode_huffman_asm_llvm(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    const FASTLOOP_MARGIN: usize = 320;
    const LITLEN_TABLEMASK: u64 = (1u64 << LitLenTable::TABLE_BITS) - 1;
    
    let out_ptr = output.as_mut_ptr();
    let out_end = output.len();
    let litlen_ptr = litlen.entries_ptr();
    let dist_ptr = dist.entries_ptr();
    
    let mut bitbuf = bits.bitbuf;
    let mut bitsleft = bits.bitsleft;
    let mut in_pos = bits.pos;
    let in_data = bits.data;
    let in_ptr = in_data.as_ptr();
    let in_fastloop_end = in_data.len().saturating_sub(32);
    let out_fastloop_end = out_end.saturating_sub(FASTLOOP_MARGIN);
    
    // Initial refill to ensure we have bits
    if (bitsleft as u8) < 56 && in_pos + 8 <= in_data.len() {
        unsafe {
            let bits_u8 = bitsleft as u8;
            let word = (in_ptr.add(in_pos) as *const u64).read_unaligned();
            bitbuf |= u64::from_le(word) << bits_u8;
            in_pos += (7 - ((bits_u8 >> 3) & 7)) as usize;
            bitsleft = (bits_u8 as u32) | 56;
        }
    }
    
    // Initial lookup
    let mut entry = unsafe { 
        (*litlen_ptr.add((bitbuf & LITLEN_TABLEMASK) as usize)).raw() 
    };
    
    // LLVM-matched fast loop
    unsafe {
        std::arch::asm!(
            // ============================================================
            // LLVM-MATCHED FAST LOOP
            // This mirrors LLVM's LBB250_7 through LBB250_39
            // ============================================================
            
            // LBB250_7: Loop entry - check if refill needed
            "2:",  // Our label for loop start (matches LBB250_7)
            
            // Bounds check
            "cmp {in_pos}, {in_end}",
            "b.hs 99f",
            "cmp {out_pos}, {out_end}",
            "b.hs 99f",
            
            // LLVM's refill check: and w14, w21, #0xff; cmp w14, #47
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #47",
            "b.hi 12f",  // Skip refill if > 47 bits (LLVM's LBB250_12)
            
            // Refill path (LLVM's inline in LBB250_7)
            "ldr x19, [{in_ptr}, {in_pos}]",
            "lsl x19, x19, {bitsleft}",
            "orr x23, x19, {bitbuf}",  // x23 = new bitbuf
            "mov w11, #7",
            "sub w11, w11, w14, lsr #3",
            "add {in_pos}, {in_pos}, x11",
            "mov w11, #56",
            "bfxil w11, {bitsleft:w}, #0, #3",
            "mov {bitsleft:x}, x11",
            
            // Consume entry and check type
            "lsr {bitbuf}, x23, {entry}",
            "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
            "tbnz {entry:w}, #31, 13f",  // Literal (LBB250_13)
            "b 9f",  // Not literal, go to LBB250_9
            
            // LBB250_12: No refill needed
            "12:",
            "mov x23, {bitbuf}",
            "lsr {bitbuf}, {bitbuf}, {entry}",
            "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
            "tbz {entry:w}, #31, 9f",  // Not literal
            
            // LBB250_13: Literal decode
            "13:",
            // Lookup next entry
            "and x14, {bitbuf}, {tablemask}",
            "ldr w24, [{litlen_ptr}, x14, lsl #2]",
            "tbnz w24, #31, 48f",  // Next is also literal
            
            // Single literal - write and loop back
            "lsr w14, {entry:w}, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            
            // Refill before next iteration
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #32",
            "b.hs 51f",
            
            // Inline refill
            "ldr x19, [{in_ptr}, {in_pos}]",
            "and w22, {bitsleft:w}, #0x1f",
            "lsl x19, x19, x22",
            "orr {bitbuf}, x19, {bitbuf}",
            "mov w11, #7",
            "sub w14, w11, w14, lsr #3",
            "add {in_pos}, {in_pos}, x14",
            "mov w14, #56",
            "bfxil w14, {bitsleft:w}, #0, #3",
            
            "51:",
            "mov {entry:w}, w24",
            "mov w23, #1",
            "add {out_pos}, {out_pos}, x23",
            "b 2b",
            
            // LBB250_48: Next entry is also literal - batch 2
            "48:",
            "lsr w14, {entry:w}, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            
            // Consume second literal
            "lsr {bitbuf}, {bitbuf}, w24",
            "sub {bitsleft:w}, {bitsleft:w}, w24",
            
            // Write second literal
            "lsr w14, w24, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            
            // Lookup and loop
            "and x14, {bitbuf}, {tablemask}",
            "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
            "b 2b",
            
            // LBB250_9: Not a literal - check exceptional
            "9:",
            "tbnz {entry:w}, #15, 16f",  // Exceptional (subtable/EOB)
            
            // Length code from main table
            // Decode length
            "and w14, {entry:w}, #0x3f",
            "mov x19, #-1",
            "lsl x14, x19, x14",
            "bic x14, x23, x14",
            "lsr w19, {entry:w}, #8",
            "lsr x14, x14, x19",
            "add w26, w14, {entry:w}, lsr #16",
            
            // Validate length
            "cmp w26, #0",
            "ccmp {out_pos}, x26, #0, ne",
            "b.lo 99f",
            
            // Refill for distance
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #32",
            "b.hs 21f",
            
            "ldr x19, [{in_ptr}, {in_pos}]",
            "and w24, {bitsleft:w}, #0x1f",
            "lsl x19, x19, x24",
            "orr x24, x19, {bitbuf}",
            "mov w11, #7",
            "sub w14, w11, w14, lsr #3",
            "add {in_pos}, {in_pos}, x14",
            "mov w14, #56",
            "bfxil w14, {bitsleft:w}, #0, #3",
            "mov {bitsleft:x}, x14",
            
            // Distance lookup
            "and x11, x24, #0xff",
            "ldr w11, [{dist_ptr}, x11, lsl #2]",
            "tbz w11, #14, 23f",
            "b 22f",
            
            "21:",
            "mov x24, {bitbuf}",
            "and x11, {bitbuf}, #0xff",
            "ldr w11, [{dist_ptr}, x11, lsl #2]",
            "tbz w11, #14, 23f",
            
            // LBB250_22: Distance subtable
            "22:",
            "lsr x24, x24, #8",
            "sub {bitsleft:w}, {bitsleft:w}, #8",
            "ubfx x14, x11, #8, #4",
            "mov x19, #-1",
            "lsl x14, x19, x14",
            "bic x14, x24, x14",
            "add x11, x14, x11, lsr #16",
            "ldr w11, [{dist_ptr}, x11, lsl #2]",
            
            // LBB250_23: Distance decode
            "23:",
            "mov x14, x11",
            "lsr {bitbuf}, x24, x11",
            "sub {bitsleft:w}, {bitsleft:w}, w14",
            "mov x19, #-1",
            "lsl x19, x19, x14",
            "bic x19, x24, x19",
            "lsr w24, w14, #8",
            "lsr x19, x19, x24",
            "add w27, w19, w14, lsr #16",  // w27 = distance
            
            // Validate distance
            "cbz w27, 99f",
            "cmp {out_pos}, x27",
            "b.lo 99f",
            
            // Match copy setup
            "neg x28, x27",  // x28 = -distance
            "add x25, {out_ptr}, {out_pos}",  // x25 = dst
            "add x24, x25, x26",  // x24 = dst + length
            
            // Choose copy strategy based on length and distance
            "cmp w26, #64",
            "b.lo 39f",
            "cmp w27, #32",
            "b.lo 39f",
            
            // LBB250_27: Fast 32-byte copy loop
            "27:",
            "add x14, x25, x28",  // src = dst - distance
            "ldp q0, q1, [x14]",
            "stp q0, q1, [x25]",
            "add x25, x25, #32",
            "cmp x25, x24",
            "b.lo 27b",
            
            // Update out_pos
            "sub {out_pos}, x24, {out_ptr}",
            
            // Refill and next lookup
            "ldr x14, [{in_ptr}, {in_pos}]",
            "lsl x14, x14, {bitsleft}",
            "orr {bitbuf}, x14, {bitbuf}",
            "mov w11, #7",
            "and w14, {bitsleft:w}, #0xff",
            "bic w11, w11, w14, lsr #3",
            "add {in_pos}, {in_pos}, x11",
            "orr {bitsleft:w}, w14, #0x38",
            
            "and x14, {bitbuf}, {tablemask}",
            "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
            "b 2b",
            
            // LBB250_39: Small/overlapping copy
            "39:",
            "cmp w27, #8",
            "b.lo 35f",
            
            // 8-byte copy loop
            "36:",
            "add x14, x25, x28",
            "ldr x19, [x14]",
            "str x19, [x25]",
            "add x25, x25, #8",
            "cmp x25, x24",
            "b.lo 36b",
            "b 38f",
            
            // Byte-by-byte copy for overlap
            "35:",
            "add x14, x25, x28",
            "ldrb w19, [x14]",
            "strb w19, [x25]",
            "add x25, x25, #1",
            "cmp x25, x24",
            "b.lo 35b",
            
            "38:",
            "sub {out_pos}, x25, {out_ptr}",
            
            // Refill
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #48",
            "b.hs 34f",
            "ldr x11, [{in_ptr}, {in_pos}]",
            "lsl x11, x11, x14",
            "orr {bitbuf}, {bitbuf}, x11",
            "mov w11, #7",
            "lsr w19, w14, #3",
            "and w19, w19, #7",
            "sub w11, w11, w19",
            "add {in_pos}, {in_pos}, x11",
            "orr {bitsleft:w}, w14, #56",
            "34:",
            
            "and x14, {bitbuf}, {tablemask}",
            "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
            "b 2b",
            
            // LBB250_16: Exceptional (subtable or EOB)
            "16:",
            "tbnz {entry:w}, #13, 99f",  // EOB - exit
            
            // Subtable lookup
            "lsr w14, {entry:w}, #16",
            "ubfx w19, {entry:w}, #8, #5",
            "mov x22, #-1",
            "lsl x19, x22, x19",
            "bic x19, {bitbuf}, x19",
            "add x19, {litlen_ptr}, x19, lsl #2",
            "ldr {entry:w}, [x19, w14, uxtw #2]",
            
            // Consume subtable bits
            "lsr x19, {bitbuf}, {entry}",
            "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",
            "mov {bitbuf}, x19",
            
            // Check subtable entry type
            "tbnz {entry:w}, #31, 52f",  // Literal from subtable
            "tbnz {entry:w}, #13, 99f",  // EOB from subtable
            
            // Length from subtable - continue to distance decode
            // (reuse the length decode path above)
            "b 9f",
            
            // Literal from subtable
            "52:",
            "lsr w14, {entry:w}, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            
            // Refill and next lookup
            "and w14, {bitsleft:w}, #0xff",
            "cmp w14, #32",
            "b.hs 53f",
            "ldr x19, [{in_ptr}, {in_pos}]",
            "lsl x19, x19, w14",
            "orr {bitbuf}, {bitbuf}, x19",
            "mov w11, #7",
            "lsr w19, w14, #3",
            "bic w11, w11, w19",
            "add {in_pos}, {in_pos}, x11",
            "orr {bitsleft:w}, w14, #56",
            "53:",
            
            "and x14, {bitbuf}, {tablemask}",
            "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
            "b 2b",
            
            // EXIT
            "99:",
            
            // Register bindings - matching LLVM's allocation
            bitbuf = inout(reg) bitbuf,
            bitsleft = inout(reg) bitsleft,
            in_pos = inout(reg) in_pos,
            out_pos = inout(reg) out_pos,
            entry = inout(reg) entry,
            in_ptr = in(reg) in_ptr,
            out_ptr = in(reg) out_ptr,
            litlen_ptr = in(reg) litlen_ptr,
            dist_ptr = in(reg) dist_ptr,
            in_end = in(reg) in_fastloop_end,
            out_end = in(reg) out_fastloop_end,
            tablemask = in(reg) LITLEN_TABLEMASK,
            
            // Scratch registers
            out("x11") _,
            out("x14") _,
            out("x19") _,
            out("x22") _,
            out("x23") _,
            out("x24") _,
            out("x25") _,
            out("x26") _,
            out("x27") _,
            out("x28") _,
            out("q0") _,
            out("q1") _,
            
            options(nostack),
        );
    }
    
    // Sync state back
    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft;
    bits.pos = in_pos;
    
    // Fallback to standard decoder for remainder
    crate::consume_first_decode::decode_huffman_cf_pub(bits, output, out_pos, litlen, dist)
}
```

---

## Phase 3: Verify Correctness

### Step 3.1: Add Test Function

```rust
#[test]
fn test_asm_llvm_parity_correctness() {
    use std::fs;
    use flate2::read::GzDecoder;
    use std::io::Read;
    
    let data = match fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping - no SILESIA data file");
            return;
        }
    };
    
    let mut decoder = GzDecoder::new(&data[..]);
    let mut expected = Vec::new();
    decoder.read_to_end(&mut expected).unwrap();
    
    let deflate_data = &data[10..data.len() - 8];
    
    // Decode with LLVM-parity ASM
    let mut output = vec![0u8; expected.len() + 4096];
    let result = crate::consume_first_decode::inflate_with_asm_llvm(
        deflate_data, &mut output);
    
    match result {
        Ok(len) => {
            assert_eq!(len, expected.len(), "Length mismatch");
            for i in 0..len {
                if output[i] != expected[i] {
                    panic!("Mismatch at byte {}: got {:02x}, expected {:02x}",
                           i, output[i], expected[i]);
                }
            }
            eprintln!("LLVM-parity ASM: CORRECT ({} bytes)", len);
        }
        Err(e) => panic!("LLVM-parity ASM failed: {}", e),
    }
}
```

### Step 3.2: Add Benchmark Function

```rust
#[test]
fn bench_asm_llvm_parity() {
    use std::fs;
    use std::time::Instant;
    use flate2::read::GzDecoder;
    use std::io::Read;
    
    let data = match fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping - no SILESIA data file");
            return;
        }
    };
    
    let mut decoder = GzDecoder::new(&data[..]);
    let mut expected = Vec::new();
    decoder.read_to_end(&mut expected).unwrap();
    
    let deflate_data = &data[10..data.len() - 8];
    let output_size = expected.len();
    let iterations = 10;
    
    eprintln!("\n=== LLVM-PARITY ASM BENCHMARK ===\n");
    
    let mut output = vec![0u8; output_size + 4096];
    
    // Warmup
    for _ in 0..3 {
        let _ = crate::bgzf::inflate_into_pub(deflate_data, &mut output);
        let _ = crate::consume_first_decode::inflate_with_asm_llvm(deflate_data, &mut output);
    }
    
    // Benchmark Rust baseline
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = crate::bgzf::inflate_into_pub(deflate_data, &mut output);
    }
    let rust_elapsed = start.elapsed();
    let rust_mb_s = (output_size as f64 * iterations as f64) 
        / rust_elapsed.as_secs_f64() / 1_000_000.0;
    
    // Benchmark LLVM-parity ASM
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = crate::consume_first_decode::inflate_with_asm_llvm(deflate_data, &mut output);
    }
    let asm_elapsed = start.elapsed();
    let asm_mb_s = (output_size as f64 * iterations as f64) 
        / asm_elapsed.as_secs_f64() / 1_000_000.0;
    
    eprintln!("Rust (LLVM-generated):  {:.0} MB/s", rust_mb_s);
    eprintln!("LLVM-parity ASM:        {:.0} MB/s", asm_mb_s);
    eprintln!("Ratio: {:.1}%", asm_mb_s / rust_mb_s * 100.0);
    
    if asm_mb_s >= rust_mb_s * 0.95 {
        eprintln!("\n✓ PARITY ACHIEVED (within 5%)");
    } else {
        eprintln!("\n✗ Not at parity yet - continue optimization");
    }
}
```

---

## Phase 4: Iterate to Parity

If we don't achieve parity immediately, analyze differences:

### Step 4.1: Instruction-Level Comparison

Create a test that dumps both LLVM's and our ASM side-by-side:

```rust
#[test]
fn analyze_instruction_differences() {
    // This is a documentation test - read the generated .s file
    // and our inline asm to compare instruction sequences
    
    eprintln!("=== INSTRUCTION COMPARISON GUIDE ===\n");
    eprintln!("1. Generate LLVM assembly:");
    eprintln!("   RUSTFLAGS=\"--emit asm\" cargo build --release\n");
    eprintln!("2. Find decode function:");
    eprintln!("   grep -n 'decode_huffman_libdeflate_style' target/release/deps/*.s\n");
    eprintln!("3. Compare key sections:\n");
    
    eprintln!("   REFILL:");
    eprintln!("   LLVM: ldr x19, [x8, x10]; lsl x19, x19, x21; orr x23, x19, x11");
    eprintln!("   OURS: ldr x19, [in_ptr, in_pos]; lsl x19, x19, bitsleft; orr x23, x19, bitbuf\n");
    
    eprintln!("   LITERAL:");
    eprintln!("   LLVM: lsr w14, w22, #16; strb w14, [x1, x3]");
    eprintln!("   OURS: lsr w14, entry, #16; strb w14, [out_ptr, out_pos]\n");
    
    eprintln!("   If instructions match but performance differs,");
    eprintln!("   the issue is scheduling or register pressure.\n");
}
```

### Step 4.2: Microarchitectural Analysis

```rust
#[test]
fn guide_hardware_profiling() {
    eprintln!("=== HARDWARE PROFILING GUIDE ===\n");
    
    eprintln!("On macOS (M3):");
    eprintln!("1. Build release binary:");
    eprintln!("   cargo build --release\n");
    
    eprintln!("2. Profile with Instruments:");
    eprintln!("   xcrun xctrace record --template 'CPU Counters' \\");
    eprintln!("     --launch -- ./target/release/gzippy -d < silesia.tar.gz > /dev/null\n");
    
    eprintln!("3. Key metrics to compare (ASM vs Rust):");
    eprintln!("   - Instructions Per Cycle (IPC)");
    eprintln!("   - L1 Data Cache Misses");
    eprintln!("   - Branch Mispredictions");
    eprintln!("   - Stall Cycles (Frontend/Backend)\n");
    
    eprintln!("If IPC is lower for ASM:");
    eprintln!("  -> Instruction scheduling issue (dependencies stalling pipeline)\n");
    
    eprintln!("If cache misses are higher:");
    eprintln!("  -> Memory access pattern issue\n");
    
    eprintln!("If branch mispredictions are higher:");
    eprintln!("  -> Need different branch patterns\n");
}
```

---

## Phase 5: Post-Parity Optimizations

Once we achieve parity, add optimizations LLVM can't do:

### Step 5.1: 4-Way Literal Unrolling

```asm
// Decode 4 literals in a row without checking entry type
// Only valid when we know next 4 entries are literals
"lit4:",
// Check if we have enough bits for 4 literals (4 * 9 = 36 bits max)
"and w14, {bitsleft:w}, #0xff",
"cmp w14, #36",
"b.lo 2b",  // Not enough, go back to normal loop

// Speculatively load 4 entries
"and x14, {bitbuf}, {tablemask}",
"ldr w24, [{litlen_ptr}, x14, lsl #2]",
"lsr x19, {bitbuf}, w24",
"and x14, x19, {tablemask}",
"ldr w25, [{litlen_ptr}, x14, lsl #2]",
"lsr x19, x19, w25",
"and x14, x19, {tablemask}",
"ldr w26, [{litlen_ptr}, x14, lsl #2]",
"lsr x19, x19, w26",
"and x14, x19, {tablemask}",
"ldr w27, [{litlen_ptr}, x14, lsl #2]",

// Check all 4 are literals
"and x14, x24, x25",
"and x14, x14, x26",
"and x14, x14, x27",
"tbz x14, #31, 2b",  // Not all literals

// Write 4 literals as u32
"lsr w24, w24, #16",
"lsr w25, w25, #16",
"lsr w26, w26, #16",
"lsr w27, w27, #16",
"orr w24, w24, w25, lsl #8",
"orr w24, w24, w26, lsl #16",
"orr w24, w24, w27, lsl #24",
"str w24, [{out_ptr}, {out_pos}]",
"add {out_pos}, {out_pos}, #4",

// Update bitbuf/bitsleft
"..." // Calculate total bits consumed
```

### Step 5.2: Prefetching

```asm
// Prefetch next cache line of input
"prfm pldl1strm, [{in_ptr}, {in_pos}, lsl #0]",
"add x14, {in_pos}, #64",
"prfm pldl1strm, [{in_ptr}, x14]",
```

### Step 5.3: Branch Hints

ARM64 doesn't have explicit branch hints, but we can optimize
branch patterns based on profiling data.

---

## Implementation Checklist

- [ ] Phase 1: Extract LLVM assembly
- [ ] Phase 2: Create `decode_huffman_asm_llvm` function
- [ ] Phase 3.1: Add correctness test
- [ ] Phase 3.2: Add benchmark test
- [ ] Phase 4: Iterate until parity (95%+)
- [ ] Phase 5.1: 4-way literal unrolling
- [ ] Phase 5.2: Prefetching
- [ ] Phase 5.3: Branch optimization

## Success Criteria

1. **Correctness:** Byte-for-byte match with Rust baseline on all test files
2. **Performance:** 95%+ of Rust baseline = parity
3. **Performance:** 105%+ of Rust baseline = exceeds LLVM

---

## Commands to Run

```bash
# Generate LLVM assembly for reference
RUSTFLAGS="--emit asm" cargo build --release

# Run correctness test
cargo test --release test_asm_llvm_parity_correctness -- --nocapture

# Run benchmark
cargo test --release bench_asm_llvm_parity -- --nocapture

# Full test suite
cargo test --release
```
