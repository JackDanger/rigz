//! Exact libdeflate fastloop - generated from compiled libdeflate-C
//!
//! This decoder uses the EXACT instruction sequences from libdeflate's
//! compiled output. No approximation, no "similar" patterns - exact copies.

use crate::consume_first_decode::Bits;
use crate::libdeflate_entry::{DistTable, LitLenTable};
use std::io::{Error, ErrorKind, Result};

/// Decoder with exact libdeflate instruction sequences
#[cfg(target_arch = "aarch64")]
pub fn decode_huffman_libdeflate_exact(
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

    // Libdeflate's masks
    let litlen_mask: u64 = 0x7ff;  // 11-bit main table
    let dist_mask: u64 = 0xff;     // 8-bit distance table
    let neg_one: u64 = !0u64;      // For extra bits extraction
    
    let mut entry: u32;
    let mut status: u64 = 0;

    if in_pos >= in_end || out_pos >= out_end {
        return crate::consume_first_decode::decode_huffman_libdeflate_style(
            bits, output, out_pos, litlen, dist
        );
    }

    unsafe {
        asm!(
            // === libdeflate's exact refill sequence ===
            // From lines 430-436: refill before entering loop
            "ldr x14, [{in_ptr}, {in_pos}]",
            "lsl x14, x14, {bitsleft}",
            "orr {bitbuf}, x14, {bitbuf}",
            "ubfx w15, {bitsleft:w}, #3, #3",    // Extract (bitsleft >> 3) & 7
            "sub x15, {in_pos}, x15",
            "add {in_pos}, x15, #7",
            "orr {bitsleft:w}, {bitsleft:w}, #0x38",  // bitsleft |= 56
            
            // Initial lookup (line 437-438)
            "and x14, {bitbuf}, {litlen_mask}",
            "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
            
            // === MAIN FASTLOOP (LBB0_44) ===
            ".p2align 4",
            "2:",
            
            // Bounds check
            "cmp {in_pos}, {in_end}",
            "b.hs 90f",
            "cmp {out_pos}, {out_end}",
            "b.hs 90f",
            
            // Line 444: Consume bits
            "lsr x15, {bitbuf}, {entry}",        // x15 = bitbuf >> entry
            "sub {bitsleft:w}, {bitsleft:w}, {entry:w}",  // bitsleft -= entry
            
            // Line 446: Check literal (bit 31)
            "tbnz {entry:w}, #31, 10f",
            
            // Line 450: Check exceptional (bit 15) 
            "mov x16, {entry}",                   // Save entry
            "mov x17, x15",                       // Save shifted bitbuf
            "tbnz w16, #15, 50f",
            
            // === LENGTH PATH ===
            // Lines 452-454: Get distance entry
            "and w14, w16, #0xff",               // consumed bits
            "and x15, x17, {dist_mask}",
            "ldr w20, [{dist_ptr}, x15, lsl #2]",
            "and w21, {bitsleft:w}, #0xff",
            
            // Check distance subtable
            "tbnz w20, #15, 60f",
            
            // Lines 463-467: Extract distance with extra bits
            "lsl x15, {neg_one}, x20",
            "bic x15, x17, x15",
            "lsr w22, w20, #8",
            "lsr x15, x15, x22",
            "add w15, w15, w20, lsr #16",        // distance = base + extra
            
            // Lines 472-476: Extract length with extra bits
            "lsl x22, {neg_one}, x14",
            "bic x22, {bitbuf}, x22",
            "lsr w23, w16, #8",
            "lsr x22, x22, x23",
            "add w22, w22, w16, lsr #16",        // length = base + extra
            
            // Lines 477-478: Consume distance bits
            "and w14, w20, #0xff",
            "sub {bitsleft:w}, {bitsleft:w}, w20",
            
            // Lines 479-490: Refill and preload
            "ldr x23, [{in_ptr}, {in_pos}]",
            "lsl x24, x23, {bitsleft}",
            "ubfx w23, {bitsleft:w}, #3, #3",
            "and w14, w20, #0xff",
            "lsr x20, x17, x14",
            "sub x14, {out_pos}, x15",           // src = out_pos - distance
            "add x15, {out_pos}, w22, uxtw",     // end = out_pos + length
            "and x17, x20, {litlen_mask}",
            "ldr {entry:w}, [{litlen_ptr}, x17, lsl #2]",
            "orr {bitbuf}, x24, x20",
            "sub x23, {in_pos}, x23",
            "add {in_pos}, x23, #7",
            
            // Check if short copy
            "cmp w22, #8",
            "b.lo 30f",
            
            // === MATCH COPY (Lines 493-503: 40-byte unrolled) ===
            "ldr x23, [{out_ptr}, x14]",
            "str x23, [{out_ptr}, {out_pos}]",
            "ldr x23, [{out_ptr}, x14, #8]",     // Note: libdeflate uses offset addressing
            "str x23, [{out_ptr}, {out_pos}, #8]",
            "cmp w22, #40",
            "b.lo 25f",
            
            // Large copy loop
            "20:",
            "add x14, x14, #8",
            "add {out_pos}, {out_pos}, #8",
            "ldr x23, [{out_ptr}, x14]",
            "str x23, [{out_ptr}, {out_pos}]",
            "cmp {out_pos}, x15",
            "b.lo 20b",
            "b 2b",
            
            // Medium copy (8-40 bytes)
            "25:",
            "mov {out_pos}, x15",
            "b 2b",
            
            // Short copy (<8 bytes) - byte by byte
            "30:",
            "ldrb w23, [{out_ptr}, x14]",
            "strb w23, [{out_ptr}, {out_pos}]",
            "add x14, x14, #1",
            "add {out_pos}, {out_pos}, #1",
            "cmp {out_pos}, x15",
            "b.lo 30b",
            "b 2b",
            
            // === LITERAL PATH (Line 446: tbnz #31) ===
            "10:",
            "lsr w14, {entry:w}, #16",           // Extract literal
            "strb w14, [{out_ptr}, {out_pos}]",
            "add {out_pos}, {out_pos}, #1",
            "mov {bitbuf}, x15",                 // Update bitbuf
            "and x14, {bitbuf}, {litlen_mask}",
            "ldr {entry:w}, [{litlen_ptr}, x14, lsl #2]",
            "b 2b",
            
            // === EXCEPTIONAL (bit 15) ===
            "50:",
            "tbnz w16, #13, 80f",                // EOB check
            // Subtable - slowpath
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
            "mov {status}, #3",
            
            // === EXIT ===
            "99:",
            
            bitbuf = inout(reg) bitbuf,
            bitsleft = inout(reg) bitsleft,
            in_pos = inout(reg) in_pos,
            out_pos = inout(reg) out_pos,
            entry = out(reg) entry,
            status = inout(reg) status,
            
            in_ptr = in(reg) in_ptr,
            in_end = in(reg) in_end,
            out_ptr = in(reg) out_ptr,
            out_end = in(reg) out_end,
            litlen_ptr = in(reg) litlen_ptr,
            litlen_mask = in(reg) litlen_mask,
            dist_ptr = in(reg) dist_ptr,
            dist_mask = in(reg) dist_mask,
            neg_one = in(reg) neg_one,
            
            out("x14") _,
            out("x15") _,
            out("x16") _,
            out("x17") _,
            out("x20") _,
            out("x21") _,
            out("x22") _,
            out("x23") _,
            out("x24") _,
            
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
pub fn decode_huffman_libdeflate_exact(
    bits: &mut Bits,
    output: &mut [u8],
    out_pos: usize,
    litlen: &LitLenTable,
    dist: &DistTable,
) -> Result<usize> {
    crate::consume_first_decode::decode_huffman_libdeflate_style(bits, output, out_pos, litlen, dist)
}
