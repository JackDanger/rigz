//! Exact libdeflate register assignments
//!
//! This decoder uses EXPLICIT register assignments matching libdeflate-C's
//! compiled output. We pin registers to the same choices Clang made.
//!
//! libdeflate register assignments (from compiled output):
//!   x21 = bitbuf
//!   w26 = bitsleft  
//!   x25 = in_ptr current position
//!   x28 = out_ptr current position
//!   x23 = litlen_table
//!   x22 = dist_table
//!   x8  = litlen_mask (0x7ff)
//!   x24 = neg_one for bit extraction

use crate::consume_first_decode::Bits;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::io::{Error, ErrorKind, Result};

/// Decoder with EXACT libdeflate register assignments
#[cfg(target_arch = "aarch64")]
pub fn decode_huffman_exact_regs(
    bits: &mut Bits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    use std::arch::asm;

    let out_ptr = output.as_mut_ptr();
    let out_len = output.len();
    let litlen_ptr = litlen.entries_ptr();
    let dist_ptr = dist.entries_ptr();

    let mut bitbuf: u64 = bits.bitbuf;
    let mut bitsleft: u32 = bits.bitsleft;
    let mut in_pos: usize = bits.pos;
    let in_ptr = bits.data.as_ptr();
    
    let in_end: usize = bits.data.len().saturating_sub(16);
    let out_end: usize = out_len.saturating_sub(274);

    let litlen_mask: u64 = 0x7ff;
    let dist_mask: u64 = 0xff;
    let neg_one: u64 = !0u64;
    
    let mut status: u64 = 0;

    // Early exit for boundary conditions
    if in_pos >= in_end || out_pos >= out_end {
        return crate::consume_first_decode::decode_huffman_libdeflate_style(
            bits, output, out_pos, litlen, dist
        );
    }

    unsafe {
        asm!(
            // === REFILL (libdeflate pattern with EXACT registers) ===
            // x21 = bitbuf, w26 = bitsleft, x25 = in position
            "ldr x9, [{in_ptr}, {in_pos}]",
            "lsl x9, x9, x26",
            "orr x21, x9, x21",
            "ubfx w9, w26, #3, #3",
            "sub x9, {in_pos}, x9",
            "add {in_pos}, x9, #7",
            "orr w26, w26, #0x38",
            
            // Initial lookup
            "and x9, x21, x8",
            "ldr w10, [x23, x9, lsl #2]",
            
            // === MAIN FASTLOOP ===
            ".p2align 4",
            "2:",
            
            // Bounds check
            "cmp {in_pos}, {in_end}",
            "b.hs 90f",
            "cmp {out_pos}, {out_end}",
            "b.hs 90f",
            
            // Consume (libdeflate pattern: lsr then sub)
            "lsr x12, x21, x10",
            "sub w9, w26, w10",
            
            // Check literal (bit 31)
            "tbnz w10, #31, 10f",
            
            // Save entry and shifted bitbuf
            "mov x11, x10",
            "mov x10, x12",
            
            // Check exceptional (bit 15)
            "tbnz w11, #15, 50f",
            
            // === LENGTH PATH ===
            "and w12, w11, #0xff",
            "and x13, x10, x7",       // x7 = dist_mask
            "ldr w13, [x22, x13, lsl #2]",
            "and w14, w9, #0xff",
            
            // Check distance subtable
            "tbnz w13, #15, 60f",
            
            // Check bits available
            "cmp w14, #30",
            "b.ls 90f",
            
            // Extract distance (lines 463-467 pattern)
            "mov x14, x13",
            "lsl x13, x24, x14",
            "bic x13, x10, x13",
            "lsr w15, w14, #8",
            "lsr x13, x13, x15",
            "add w13, w13, w14, lsr #16",
            
            // Validate distance
            "sub x15, {out_pos}, x20",
            "cmp x15, x13",
            "b.lt 90f",
            
            // Extract length (lines 472-476 pattern)
            "lsl x12, x24, x12",
            "bic x12, x21, x12",
            "lsr w15, w11, #8",
            "lsr x12, x12, x15",
            "add w12, w12, w11, lsr #16",
            
            // Consume distance
            "and w11, w14, #0xff",
            "sub w9, w9, w14",
            
            // Refill (lines 479-490 pattern)
            "ldr x14, [{in_ptr}, {in_pos}]",
            "lsl x15, x14, x9",
            "ubfx w16, w9, #3, #3",
            "and w14, w20, #0xff",
            "lsr x17, x10, x11",
            "sub x14, {out_pos}, x13",      // src = out_pos - distance
            "add x11, {out_pos}, w12, uxtw", // end = out_pos + length
            "and x10, x17, x8",
            "ldr w10, [x23, x10, lsl #2]",
            "orr x21, x15, x17",
            "sub x15, {in_pos}, x16",
            "add {in_pos}, x15, #7",
            
            // Copy check
            "cmp w13, #8",
            "b.lo 30f",
            
            // Fast copy (8+ bytes)
            "25:",
            "ldr x15, [{out_ptr}, x14]",
            "str x15, [{out_ptr}, {out_pos}]",
            "add x14, x14, #8",
            "add {out_pos}, {out_pos}, #8",
            "cmp {out_pos}, x11",
            "b.lo 25b",
            "mov w26, w9",
            "b 2b",
            
            // Byte copy
            "30:",
            "ldrb w15, [{out_ptr}, x14]",
            "strb w15, [{out_ptr}, {out_pos}]",
            "add x14, x14, #1",
            "add {out_pos}, {out_pos}, #1",
            "cmp {out_pos}, x11",
            "b.lo 30b",
            "mov w26, w9",
            "b 2b",
            
            // === LITERAL PATH ===
            "10:",
            "lsr w14, w10, #16",
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            "mov x21, x12",
            "mov w26, w9",
            "and x9, x21, x8",
            "ldr w10, [x23, x9, lsl #2]",
            "b 2b",
            
            // === EXCEPTIONAL ===
            "50:",
            "tbnz w11, #13, 80f",
            "b 90f",
            
            // Distance subtable
            "60:",
            "b 90f",
            
            // === EOB ===
            "80:",
            "mov {status}, #1",
            "b 99f",
            
            // === SLOWPATH ===
            "90:",
            "mov x21, {bitbuf}",
            "mov w26, {bitsleft:w}",
            "mov {status}, #3",
            
            // === EXIT ===
            "99:",
            "mov {bitbuf}, x21",
            "mov {bitsleft:w}, w26",
            
            // Use EXPLICIT registers matching libdeflate!
            bitbuf = inout(reg) bitbuf,
            bitsleft = inout(reg) bitsleft,
            in_pos = inout(reg) in_pos,
            out_pos = inout(reg) out_pos,
            status = inout(reg) status,
            
            in_ptr = in(reg) in_ptr,
            in_end = in(reg) in_end,
            out_ptr = in(reg) out_ptr,
            out_end = in(reg) out_end,
            
            // Explicit register bindings for libdeflate compatibility
            in("x8") litlen_mask,     // libdeflate uses x8 for mask
            in("x7") dist_mask,       // We use x7 for dist_mask
            in("x20") 0u64,           // Base output position
            in("x21") bitbuf,         // libdeflate uses x21 for bitbuf
            in("x22") dist_ptr,       // libdeflate uses x22 for dist table
            in("x23") litlen_ptr,     // libdeflate uses x23 for litlen table
            in("x24") neg_one,        // For bit extraction
            
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("x15") _,
            out("x16") _,
            out("x17") _,
            out("w26") _,
            
            options(nostack),
        );
    }

    bits.bitbuf = bitbuf;
    bits.bitsleft = bitsleft;
    bits.pos = in_pos;

    match status {
        1 => Ok(out_pos),
        2 => Err(Error::new(ErrorKind::InvalidData, "Invalid data")),
        _ => crate::consume_first_decode::decode_huffman_libdeflate_style(
            bits, output, out_pos, litlen, dist
        ),
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn decode_huffman_exact_regs(
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
    #[test]
    fn test_explicit_regs_compile() {
        println!("Explicit register decoder compiled!");
    }
}
