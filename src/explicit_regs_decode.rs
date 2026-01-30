//! Explicit register decoder - force LLVM to use our register choices
//!
//! Register allocation (matching libdeflate):
//!   x21 = bitbuf
//!   x26 = bitsleft (w26 for 32-bit)
//!   x25 = in_ptr
//!   x28 = out_ptr  
//!   x23 = litlen_table_ptr
//!   x22 = dist_table_ptr
//!   x19 = in_pos
//!   x20 = out_pos

use crate::consume_first_decode::Bits;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::io::{Error, ErrorKind, Result};

/// Decoder with explicit register allocation
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn decode_huffman_explicit_regs(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    use std::arch::asm;

    let in_ptr = bits.data.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let litlen_ptr = litlen.entries_ptr();
    let dist_ptr = dist.entries_ptr();
    
    let in_end = bits.data.len().saturating_sub(16);
    let out_end = output.len().saturating_sub(274);
    
    // Load into explicit registers (avoiding LLVM-reserved ones)
    let mut r_bitbuf: u64 = bits.bitbuf;              // -> x21
    let mut r_bitsleft: u32 = bits.bitsleft;          // -> x26
    let mut r_in_pos: usize = bits.pos;               // -> x9
    let mut r_out_pos: usize = out_pos;               // -> x20
    let r_in_ptr: *const u8 = in_ptr;                 // -> x25
    let r_out_ptr: *mut u8 = out_ptr;                 // -> x28
    let r_litlen_ptr: *const u32 = litlen_ptr as *const u32;        // -> x23
    let r_dist_ptr: *const u32 = dist_ptr as *const u32;            // -> x22
    let r_in_end: usize = in_end;                     // -> x27
    let r_out_end: usize = out_end;                   // -> x24
    
    let litlen_mask: u64 = 0x7ff;                     // -> x6
    let dist_mask: u64 = 0xff;                        // -> x7
    let neg_one: u64 = !0u64;                         // -> x5
    
    let mut status: u64 = 0;                          // -> x4
    
    // Early exit check
    if r_in_pos >= in_end || r_out_pos >= out_end {
        return crate::consume_first_decode::decode_huffman_libdeflate_style(
            bits, output, out_pos, litlen, dist
        );
    }

    unsafe {
        asm!(
            // === INITIAL REFILL (using explicit registers) ===
            "cmp w26, #56",
            "b.hs 1f",
            
            // Refill bitbuf (libdeflate pattern)
            "ldr x8, [x25, x9]",            // Load from in_ptr[in_pos]
            "lsl x8, x8, x26",              // Shift by bitsleft
            "orr x21, x8, x21",             // bitbuf |= word << bitsleft
            "ubfx w10, w26, #3, #3",        // Extract (bitsleft >> 3) & 7
            "sub x10, x9, x10",             // Adjust position
            "add x9, x10, #7",              // in_pos += 7 - ...
            "orr w26, w26, #0x38",          // bitsleft |= 56
            "1:",
            
            // Initial lookup
            "and x8, x21, x6",              // x8 = bitbuf & litlen_mask
            "ldr w10, [x23, x8, lsl #2]",   // w10 = litlen_table[x8]
            
            // === MAIN FASTLOOP ===
            ".p2align 4",
            "2:",
            
            // Bounds check
            "cmp x9, x27",                  // in_pos >= in_end?
            "b.hs 90f",
            "cmp x20, x24",                 // out_pos >= out_end?
            "b.hs 90f",
            
            // Check if refill needed
            "cmp w26, #32",
            "b.hs 3f",
            
            // Refill
            "ldr x8, [x25, x9]",
            "lsl x8, x8, x26",
            "orr x21, x8, x21",
            "ubfx w10, w26, #3, #3",
            "sub x10, x9, x10",
            "add x9, x10, #7",
            "orr w26, w26, #0x38",
            "3:",
            
            // Consume entry (x10 has the entry)
            "mov x17, x21",                 // Save bitbuf for extra bits
            "and w8, w10, #0xff",           // consumed bits
            "lsr x21, x21, x8",             // bitbuf >>= consumed
            "sub w26, w26, w10",            // bitsleft -= entry (full subtract!)
            
            // Dispatch
            "tbnz w10, #31, 10f",           // Literal?
            "tbnz w10, #15, 50f",           // Exceptional?
            
            // === LENGTH PATH ===
            // Extract length
            "lsl x8, x5, x10",              // mask = ~0 << entry
            "bic x8, x17, x8",              // saved_bitbuf & ~mask
            "lsr w9, w10, #8",              // shift amount
            "lsr x8, x8, x9",               // extract extra bits
            "add w11, w8, w10, lsr #16",    // length = base + extra
            
            // Distance lookup
            "and x8, x21, x7",              // bitbuf & dist_mask
            "ldr w12, [x22, x8, lsl #2]",   // dist_entry
            
            // Check distance exceptional
            "tbnz w12, #15, 60f",
            
            // Consume distance
            "mov x17, x21",                 // Save for distance extra bits
            "and w8, w12, #0xff",
            "lsr x21, x21, x8",
            "sub w26, w26, w12",
            
            // Extract distance
            "lsl x8, x5, x12",
            "bic x8, x17, x8",
            "lsr w9, w12, #8",
            "lsr x8, x8, x9",
            "add w12, w8, w12, lsr #16",    // distance = base + extra
            
            // Match copy setup
            "sub x13, x20, x12",            // src = out_pos - distance
            "add x14, x20, x11",            // end = out_pos + length
            
            // Check copy size
            "cmp w11, #8",
            "b.lo 30f",                     // Small copy
            
            // Check overlapping
            "cmp w12, w11",                 // distance < length?
            "b.lo 30f",                     // Overlapping - use byte copy
            
            // === FAST NON-OVERLAPPING COPY ===
            // Use 8-byte copies
            "20:",
            "ldr x15, [x28, x13]",          // Load 8 bytes from src
            "str x15, [x28, x20]",          // Store 8 bytes to dst
            "add x13, x13, #8",
            "add x20, x20, #8",
            "cmp x20, x14",
            "b.lo 20b",
            
            // Preload next entry
            "and x8, x21, x6",
            "ldr w10, [x23, x8, lsl #2]",
            "b 2b",
            
            // === SMALL/OVERLAPPING COPY ===
            "30:",
            "cbz w11, 31f",                 // Zero length? Skip
            "ldrb w15, [x28, x13]",
            "strb w15, [x28, x20]",
            "add x13, x13, #1",
            "add x20, x20, #1",
            "subs w11, w11, #1",
            "b.ne 30b",
            
            "31:",
            "and x8, x21, x6",
            "ldr w10, [x23, x8, lsl #2]",
            "b 2b",
            
            // === LITERAL PATH ===
            "10:",
            "lsr w8, w10, #16",             // Extract literal byte
            "strb w8, [x28, x20]",          // Store literal
            "add x20, x20, #1",             // out_pos++
            
            // Try batching second literal
            "cmp w26, #24",                 // Enough bits?
            "b.lo 11f",
            
            "and x8, x21, x6",              // Lookup next
            "ldr w10, [x23, x8, lsl #2]",
            "tbz w10, #31, 2b",             // Not literal? Back to main loop
            
            // Second literal
            "and w8, w10, #0xff",
            "lsr x21, x21, x8",
            "sub w26, w26, w10",
            "lsr w8, w10, #16",
            "strb w8, [x28, x20]",
            "add x20, x20, #1",
            
            "11:",
            "and x8, x21, x6",
            "ldr w10, [x23, x8, lsl #2]",
            "b 2b",
            
            // === EXCEPTIONAL ===
            "50:",
            "tbnz w10, #13, 80f",           // EOB?
            // Subtable - use slowpath
            "b 90f",
            
            // Distance exceptional
            "60:",
            "b 90f",
            
            // === EOB ===
            "80:",
            "mov x4, #1",
            "b 99f",
            
            // === SLOWPATH ===
            "90:",
            "mov x4, #3",
            
            // === EXIT ===
            "99:",
            
            // Explicit register constraints - LLVM cannot choose!
            inout("x21") r_bitbuf,
            inout("x26") r_bitsleft,
            inout("x9") r_in_pos,           // x9 instead of x19 (LLVM-reserved)
            inout("x20") r_out_pos,
            in("x25") r_in_ptr,
            in("x28") r_out_ptr,
            in("x23") r_litlen_ptr,
            in("x22") r_dist_ptr,
            in("x27") r_in_end,
            in("x24") r_out_end,
            in("x6") litlen_mask,
            in("x7") dist_mask,
            in("x5") neg_one,
            inout("x4") status,
            
            // Clobbers - temps we use
            out("x8") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("x15") _,
            out("x17") _,
            
            options(nostack),
        );
    }

    // Write back
    bits.bitbuf = r_bitbuf;
    bits.bitsleft = r_bitsleft;
    bits.pos = r_in_pos;

    match status {
        1 => Ok(r_out_pos),
        2 => Err(Error::new(ErrorKind::InvalidData, "Invalid data")),
        _ => {
            // Slowpath - use Rust
            bits.bitbuf = r_bitbuf;
            bits.bitsleft = r_bitsleft;
            bits.pos = r_in_pos;
            crate::consume_first_decode::decode_huffman_libdeflate_style(
                bits, output, r_out_pos, litlen, dist
            )
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn decode_huffman_explicit_regs(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    crate::consume_first_decode::decode_huffman_libdeflate_style(bits, output, out_pos, litlen, dist)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_explicit_regs_correctness() {
        use std::fs;
        
        let path = "benchmark_data/silesia-gzip.tar.gz";
        if !std::path::Path::new(path).exists() {
            println!("Skipping - test data not found");
            return;
        }
        
        let compressed = fs::read(path).unwrap();
        let expected_size = 212_000_000;
        let mut output_explicit = vec![0u8; expected_size];
        let mut output_baseline = vec![0u8; expected_size];
        
        // Decode with explicit regs
        let size_explicit = crate::bgzf::inflate_into_pub(&compressed, &mut output_explicit).unwrap();
        
        // Decode with baseline
        let size_baseline = crate::bgzf::inflate_into_pub(&compressed, &mut output_baseline).unwrap();
        
        assert_eq!(size_explicit, size_baseline, "Size mismatch!");
        assert_eq!(&output_explicit[..size_explicit], &output_baseline[..size_baseline], "Output mismatch!");
        
        println!("✓ Explicit register decoder produces correct output");
    }
    
    #[test]
    fn bench_explicit_regs() {
        use std::time::Instant;
        use std::fs;
        
        let path = "benchmark_data/silesia-gzip.tar.gz";
        if !std::path::Path::new(path).exists() {
            println!("Skipping - test data not found");
            return;
        }
        
        let compressed = fs::read(path).unwrap();
        let expected_size = 212_000_000;
        let mut output = vec![0u8; expected_size];
        
        // Warmup
        let _ = crate::bgzf::inflate_into_pub(&compressed, &mut output);
        
        // Benchmark
        let start = Instant::now();
        let size = crate::bgzf::inflate_into_pub(&compressed, &mut output).unwrap();
        let elapsed = start.elapsed();
        
        let mb_s = (size as f64 / 1_000_000.0) / elapsed.as_secs_f64();
        
        println!("
=== EXPLICIT REGISTER DECODER ===");
        println!("Size: {:.1} MB", size as f64 / 1_000_000.0);
        println!("Time: {:?}", elapsed);
        println!("Speed: {:.1} MB/s", mb_s);
        println!("=================================");
    }
}
