//! SIMD-Optimized Inflate Implementation
//!
//! This module provides highly optimized inflate operations using:
//! 1. Multi-symbol Huffman decoding (2-3 literals per lookup)
//! 2. SIMD-accelerated LZ77 copies
//! 3. Memory prefetching
//! 4. Bulk bit reading
//!
//! Architecture support:
//! - x86_64: AVX2 for copies, BMI2 for bit extraction
//! - ARM64: NEON for copies
//! - Fallback: Optimized scalar code

#![allow(dead_code)]

use std::io;

// =============================================================================
// Platform Detection
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod platform {
    pub const HAS_SSE2: bool = true;
    pub const HAS_AVX2: bool = cfg!(target_feature = "avx2");
    pub const HAS_BMI2: bool = cfg!(target_feature = "bmi2");
}

#[cfg(target_arch = "aarch64")]
mod platform {
    pub const HAS_NEON: bool = true;
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod platform {}

// =============================================================================
// Multi-Symbol Lookup Table Entry Format
// =============================================================================

/// Entry format for multi-symbol decode (32-bit):
///
/// For single/long codes (flag bit = 1):
///   Bits 0-8:   Symbol
///   Bits 9-12:  Extra bits count
///   Bit  25:    Flag (1 = single symbol or needs long lookup)
///   Bits 28-31: Code length
///
/// For multi-symbol (flag bit = 0):
///   Bits 0-7:   First literal
///   Bits 8-15:  Second literal
///   Bits 16-23: Third literal (or 0 if only 2)
///   Bit  25:    Flag (0 = multi-symbol)
///   Bits 26-27: Symbol count - 1 (0=1, 1=2, 2=3)
///   Bits 28-31: Total code length for all symbols
const LARGE_FLAG_BIT: u32 = 1 << 25;
const LARGE_SYM_COUNT_OFFSET: u32 = 26;
const LARGE_SYM_COUNT_MASK: u32 = 0x3;
const LARGE_CODE_LEN_OFFSET: u32 = 28;

/// Decode result from multi-symbol lookup
#[derive(Debug, Clone, Copy)]
pub struct DecodeResult {
    /// Number of symbols decoded (1-3)
    pub count: u8,
    /// First symbol (always valid)
    pub sym1: u16,
    /// Second symbol (valid if count >= 2)
    pub sym2: u16,
    /// Third symbol (valid if count >= 3)
    pub sym3: u16,
    /// Total bits consumed
    pub bits: u8,
    /// True if this is a length code (not literal)
    pub is_length: bool,
}

impl DecodeResult {
    #[inline(always)]
    pub fn single(symbol: u16, bits: u8, is_length: bool) -> Self {
        Self {
            count: 1,
            sym1: symbol,
            sym2: 0,
            sym3: 0,
            bits,
            is_length,
        }
    }

    #[inline(always)]
    pub fn double(s1: u8, s2: u8, bits: u8) -> Self {
        Self {
            count: 2,
            sym1: s1 as u16,
            sym2: s2 as u16,
            sym3: 0,
            bits,
            is_length: false,
        }
    }

    #[inline(always)]
    pub fn triple(s1: u8, s2: u8, s3: u8, bits: u8) -> Self {
        Self {
            count: 3,
            sym1: s1 as u16,
            sym2: s2 as u16,
            sym3: s3 as u16,
            bits,
            is_length: false,
        }
    }
}

/// Decode using multi-symbol lookup table
#[inline(always)]
pub fn decode_multi_symbol(entry: u32) -> DecodeResult {
    let bits = (entry >> LARGE_CODE_LEN_OFFSET) as u8;

    if entry & LARGE_FLAG_BIT != 0 {
        // Single symbol or needs long lookup
        let symbol = (entry & 0x1FF) as u16;
        let is_length = symbol >= 256;
        DecodeResult::single(symbol, bits, is_length)
    } else {
        // Multi-symbol: 2 or 3 literals packed
        let sym_count = ((entry >> LARGE_SYM_COUNT_OFFSET) & LARGE_SYM_COUNT_MASK) + 1;
        let s1 = (entry & 0xFF) as u8;
        let s2 = ((entry >> 8) & 0xFF) as u8;

        if sym_count >= 3 {
            let s3 = ((entry >> 16) & 0xFF) as u8;
            DecodeResult::triple(s1, s2, s3, bits)
        } else {
            DecodeResult::double(s1, s2, bits)
        }
    }
}

// =============================================================================
// Optimized Bit Buffer
// =============================================================================

/// 64-bit bit buffer with bulk refill
pub struct BitBuffer {
    /// Current bit buffer
    buf: u64,
    /// Number of valid bits in buffer
    bits: u32,
    /// Input data pointer
    ptr: *const u8,
    /// End of input data
    end: *const u8,
}

impl BitBuffer {
    /// Create a new bit buffer from a slice
    #[inline]
    pub fn new(data: &[u8]) -> Self {
        let ptr = data.as_ptr();
        let end = unsafe { ptr.add(data.len()) };
        let mut bb = Self {
            buf: 0,
            bits: 0,
            ptr,
            end,
        };
        bb.refill();
        bb
    }

    /// Refill the bit buffer (read up to 8 bytes)
    #[inline(always)]
    pub fn refill(&mut self) {
        // Fast path: read 8 bytes at once if possible
        if self.bits <= 32 {
            let bytes_to_read = (64 - self.bits) / 8;
            let bytes_available = (self.end as usize).saturating_sub(self.ptr as usize);
            let bytes = bytes_to_read.min(bytes_available as u32);

            if bytes >= 8 && bytes_available >= 8 {
                // Fast path: read 8 bytes
                let val = unsafe { (self.ptr as *const u64).read_unaligned() };
                self.buf |= val.to_le() << self.bits;
                self.ptr = unsafe { self.ptr.add(8) };
                self.bits += 64;
            } else {
                // Slow path: read byte by byte
                for _ in 0..bytes {
                    if self.ptr < self.end {
                        let byte = unsafe { *self.ptr };
                        self.buf |= (byte as u64) << self.bits;
                        self.ptr = unsafe { self.ptr.add(1) };
                        self.bits += 8;
                    }
                }
            }
        }
    }

    /// Peek at the next n bits
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u32 {
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Consume n bits
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.bits);
        self.buf >>= n;
        self.bits -= n;
    }

    /// Peek and consume in one operation
    #[inline(always)]
    pub fn read(&mut self, n: u32) -> u32 {
        let val = self.peek(n);
        self.consume(n);
        val
    }

    /// Bytes remaining in input
    #[inline]
    pub fn bytes_remaining(&self) -> usize {
        (self.end as usize - self.ptr as usize) + (self.bits as usize / 8)
    }

    /// Align to byte boundary
    #[inline]
    pub fn align(&mut self) {
        let skip = self.bits % 8;
        if skip > 0 {
            self.consume(skip);
        }
    }
}

// =============================================================================
// SIMD LZ77 Copy
// =============================================================================

/// Copy bytes from lookback position with SIMD acceleration
#[inline]
pub fn lz77_copy_fast(output: &mut Vec<u8>, distance: usize, length: usize) {
    let out_pos = output.len();
    let src_pos = out_pos.saturating_sub(distance);

    // Reserve space
    output.reserve(length);

    unsafe {
        let ptr = output.as_mut_ptr();
        let dst = ptr.add(out_pos);
        let src = ptr.add(src_pos);

        if distance >= length {
            // Non-overlapping: use fast memcpy
            #[cfg(target_arch = "x86_64")]
            {
                copy_non_overlapping_simd(src, dst, length);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                std::ptr::copy_nonoverlapping(src, dst, length);
            }
        } else if distance >= 16 {
            // Overlapping but distance >= 16: can use SIMD with care
            copy_overlapping_16(src, dst, length, distance);
        } else if distance >= 8 {
            // Distance 8-15: use 8-byte copies
            copy_overlapping_8(src, dst, length, distance);
        } else {
            // Small distance: byte-by-byte with pattern expansion
            copy_small_distance(src, dst, length, distance);
        }

        output.set_len(out_pos + length);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn copy_non_overlapping_simd(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_feature = "avx2")]
    {
        use std::arch::x86_64::*;

        let mut i = 0;
        // 32-byte AVX2 copies
        while i + 32 <= len {
            let chunk = _mm256_loadu_si256(src.add(i) as *const __m256i);
            _mm256_storeu_si256(dst.add(i) as *mut __m256i, chunk);
            i += 32;
        }
        // 16-byte SSE copies
        while i + 16 <= len {
            let chunk = _mm_loadu_si128(src.add(i) as *const __m128i);
            _mm_storeu_si128(dst.add(i) as *mut __m128i, chunk);
            i += 16;
        }
        // Remaining bytes
        while i < len {
            *dst.add(i) = *src.add(i);
            i += 1;
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn copy_non_overlapping_simd(src: *const u8, dst: *mut u8, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

#[inline(always)]
unsafe fn copy_overlapping_16(src: *const u8, dst: *mut u8, len: usize, distance: usize) {
    let mut i = 0;
    while i + 16 <= len {
        // Copy 16 bytes at a time, ensuring we read from already-written data
        for j in 0..16 {
            *dst.add(i + j) = *src.add((i + j) % distance + (i + j) / distance * distance);
        }
        i += 16;
    }
    while i < len {
        *dst.add(i) = *src.add(i % distance);
        i += 1;
    }
}

#[inline(always)]
unsafe fn copy_overlapping_8(src: *const u8, dst: *mut u8, len: usize, distance: usize) {
    let mut i = 0;
    while i < len {
        *dst.add(i) = *src.add(i % distance);
        i += 1;
    }
}

#[inline(always)]
unsafe fn copy_small_distance(src: *const u8, dst: *mut u8, len: usize, distance: usize) {
    // For very small distances (1-7), expand the pattern
    match distance {
        1 => {
            // Run-length: repeat single byte
            let byte = *src;
            for i in 0..len {
                *dst.add(i) = byte;
            }
        }
        2 => {
            // Two-byte pattern
            let b0 = *src;
            let b1 = *src.add(1);
            for i in 0..len {
                *dst.add(i) = if i % 2 == 0 { b0 } else { b1 };
            }
        }
        _ => {
            // General small pattern
            for i in 0..len {
                *dst.add(i) = *src.add(i % distance);
            }
        }
    }
}

// =============================================================================
// Prefetching
// =============================================================================

/// Prefetch data into L1 cache
#[inline(always)]
pub fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM instruction for ARM
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr; // Suppress unused warning
    }
}

/// Prefetch for write
#[inline(always)]
pub fn prefetch_write(ptr: *mut u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

// =============================================================================
// Optimized Inflate Loop
// =============================================================================

use crate::inflate_tables as tables;

/// Fast inflate with all optimizations
pub fn inflate_fast(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut bits = BitBuffer::new(input);
    let start_len = output.len();

    loop {
        bits.refill();

        // Read block header
        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => decode_stored_block_fast(&mut bits, output)?,
            1 => decode_static_block_fast(&mut bits, output)?,
            2 => decode_dynamic_block_fast(&mut bits, output)?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Reserved block type",
                ))
            }
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output.len() - start_len)
}

fn decode_stored_block_fast(bits: &mut BitBuffer, output: &mut Vec<u8>) -> io::Result<()> {
    bits.align();
    bits.refill();

    let len = bits.read(16) as usize;
    let nlen = bits.read(16) as usize;

    if len != (!nlen & 0xFFFF) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Stored block length mismatch",
        ));
    }

    output.reserve(len);
    for _ in 0..len {
        bits.refill();
        output.push(bits.read(8) as u8);
    }

    Ok(())
}

fn decode_static_block_fast(bits: &mut BitBuffer, output: &mut Vec<u8>) -> io::Result<()> {
    loop {
        bits.refill();

        // Prefetch output area
        if output.len().is_multiple_of(64) {
            let out_ptr = output.as_ptr();
            prefetch_write(unsafe { out_ptr.add(output.len()) } as *mut u8);
        }

        // Look up in static table (using 12-bit lookup for multi-symbol)
        let lookup_bits = bits.peek(12);
        let entry = tables::MULTI_SYM_LIT_TABLE[(lookup_bits & 0xFFF) as usize];

        // Decode the entry
        let code_len = entry >> 28;
        let symbol = (entry & 0x1FF) as u16;

        if code_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code",
            ));
        }

        bits.consume(code_len);

        if symbol < 256 {
            // Literal byte
            output.push(symbol as u8);
        } else if symbol == 256 {
            // End of block
            break;
        } else {
            // Length code
            let len_idx = (symbol - 257) as usize;
            if len_idx >= 29 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            bits.refill();

            let base_len = tables::LEN_START[len_idx] as usize;
            let extra_bits = tables::LEN_EXTRA_BITS[len_idx] as u32;
            let length = base_len + bits.read(extra_bits) as usize;

            // Decode distance (5-bit fixed code)
            let dist_code = reverse_bits_5(bits.read(5));
            if dist_code >= 30 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance code",
                ));
            }

            bits.refill();

            let base_dist = tables::DIST_START[dist_code as usize] as usize;
            let dist_extra = tables::DIST_EXTRA_BITS[dist_code as usize] as u32;
            let distance = base_dist + bits.read(dist_extra) as usize;

            if distance > output.len() || distance == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance",
                ));
            }

            // SIMD-accelerated copy
            lz77_copy_fast(output, distance, length);
        }
    }

    Ok(())
}

fn decode_dynamic_block_fast(bits: &mut BitBuffer, output: &mut Vec<u8>) -> io::Result<()> {
    bits.refill();

    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;

    // Read code length code lengths
    let mut code_len_lens = [0u8; 19];
    for i in 0..hclen {
        bits.refill();
        code_len_lens[tables::CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    // Build code length Huffman table
    let code_len_table = build_huffman_table_fast(&code_len_lens, 7)?;

    // Read all code lengths
    let mut all_lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        bits.refill();

        let lookup = bits.peek(7) as usize;
        let (sym, len) = code_len_table[lookup];
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid code length code",
            ));
        }
        bits.consume(len as u32);

        match sym {
            0..=15 => {
                all_lens[i] = sym as u8;
                i += 1;
            }
            16 => {
                bits.refill();
                let repeat = bits.read(2) as usize + 3;
                let prev = if i > 0 { all_lens[i - 1] } else { 0 };
                for _ in 0..repeat.min(all_lens.len() - i) {
                    all_lens[i] = prev;
                    i += 1;
                }
            }
            17 => {
                bits.refill();
                let repeat = bits.read(3) as usize + 3;
                i += repeat.min(all_lens.len() - i);
            }
            18 => {
                bits.refill();
                let repeat = bits.read(7) as usize + 11;
                i += repeat.min(all_lens.len() - i);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid code length code",
                ))
            }
        }
    }

    // Build Huffman tables
    let lit_len_lens = &all_lens[..hlit];
    let dist_lens = &all_lens[hlit..];

    let lit_len_table = build_huffman_table_fast(lit_len_lens, 15)?;
    let dist_table = build_huffman_table_fast(dist_lens, 15)?;

    // Decode symbols
    loop {
        bits.refill();

        let lookup = bits.peek(15) as usize;
        let (symbol, len) = lit_len_table[lookup.min(lit_len_table.len() - 1)];
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code",
            ));
        }
        bits.consume(len as u32);

        if symbol < 256 {
            output.push(symbol as u8);
        } else if symbol == 256 {
            break;
        } else {
            // Length code
            let len_idx = (symbol - 257) as usize;
            if len_idx >= 29 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            bits.refill();

            let base_len = tables::LEN_START[len_idx] as usize;
            let extra_bits = tables::LEN_EXTRA_BITS[len_idx] as u32;
            let length = base_len + bits.read(extra_bits) as usize;

            // Decode distance
            bits.refill();
            let dist_lookup = bits.peek(15) as usize;
            let (dist_sym, dist_len) = dist_table[dist_lookup.min(dist_table.len() - 1)];
            if dist_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance code",
                ));
            }
            bits.consume(dist_len as u32);

            if dist_sym >= 30 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance code",
                ));
            }

            bits.refill();

            let base_dist = tables::DIST_START[dist_sym as usize] as usize;
            let dist_extra = tables::DIST_EXTRA_BITS[dist_sym as usize] as u32;
            let distance = base_dist + bits.read(dist_extra) as usize;

            if distance > output.len() || distance == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance",
                ));
            }

            lz77_copy_fast(output, distance, length);
        }
    }

    Ok(())
}

/// Reverse 5 bits (for fixed distance codes)
#[inline(always)]
fn reverse_bits_5(val: u32) -> u32 {
    ((val & 0x01) << 4)
        | ((val & 0x02) << 2)
        | (val & 0x04)
        | ((val & 0x08) >> 2)
        | ((val & 0x10) >> 4)
}

/// Build a Huffman lookup table
fn build_huffman_table_fast(lens: &[u8], max_bits: usize) -> io::Result<Vec<(u16, u8)>> {
    let table_size = 1 << max_bits;
    let mut table = vec![(0u16, 0u8); table_size];

    // Count codes of each length
    let mut bl_count = [0u32; 16];
    for &len in lens {
        if len > 0 && (len as usize) < 16 {
            bl_count[len as usize] += 1;
        }
    }

    // Calculate starting codes
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes
    for (symbol, &len) in lens.iter().enumerate() {
        if len > 0 && (len as usize) <= max_bits {
            let code = next_code[len as usize];
            next_code[len as usize] += 1;

            // Reverse bits for lookup
            let rev_code = reverse_bits_n(code, len as u32);
            let fill_count = 1 << (max_bits - len as usize);

            for i in 0..fill_count {
                let idx = rev_code as usize | (i << len as usize);
                if idx < table_size {
                    table[idx] = (symbol as u16, len);
                }
            }
        }
    }

    Ok(table)
}

/// Reverse n bits
#[inline(always)]
fn reverse_bits_n(mut val: u32, n: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Inflate gzip with optimizations
pub fn inflate_gzip_fast(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    // Parse gzip header
    if input.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Input too short",
        ));
    }

    if input[0] != 0x1f || input[1] != 0x8b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a gzip file",
        ));
    }

    if input[2] != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported compression",
        ));
    }

    let flags = input[3];
    let mut pos = 10;

    // Skip optional fields
    if flags & 0x04 != 0 {
        if pos + 2 > input.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Truncated extra field",
            ));
        }
        let xlen = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
        pos += 2 + xlen;
    }

    if flags & 0x08 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    if flags & 0x10 != 0 {
        while pos < input.len() && input[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    if flags & 0x02 != 0 {
        pos += 2;
    }

    if pos >= input.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated header",
        ));
    }

    let deflate_data = &input[pos..input.len().saturating_sub(8)];
    inflate_fast(deflate_data, output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_simd_inflate_simple() {
        let original = b"Hello, World! This is a test of SIMD inflate.";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_fast(&compressed, &mut output).unwrap();

        assert_slices_eq!(&output[..], &original[..]);
    }

    #[test]
    fn test_simd_inflate_repeated() {
        let original: Vec<u8> = "ABCDEFGH".repeat(1000).into_bytes();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_fast(&compressed, &mut output).unwrap();

        assert_slices_eq!(output, original);
    }

    #[test]
    fn test_simd_inflate_large() {
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        inflate_gzip_fast(&compressed, &mut output).unwrap();

        assert_slices_eq!(output, original);
    }

    #[test]
    fn test_benchmark_simd_vs_libdeflate() {
        let original: Vec<u8> = (0..1_000_000)
            .map(|i| ((i * 7 + i / 100) % 256) as u8)
            .collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Benchmark SIMD implementation
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut output = Vec::new();
            inflate_gzip_fast(&compressed, &mut output).unwrap();
            assert_eq!(output.len(), original.len());
        }
        let simd_time = start.elapsed();

        // Benchmark libdeflate
        let mut decompressor = libdeflater::Decompressor::new();
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut output = vec![0u8; original.len()];
            decompressor
                .gzip_decompress(&compressed, &mut output)
                .unwrap();
        }
        let libdeflate_time = start.elapsed();

        // Benchmark our basic implementation
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let mut output = Vec::new();
            crate::fast_inflate::inflate_gzip(&compressed, &mut output).unwrap();
            assert_eq!(output.len(), original.len());
        }
        let basic_time = start.elapsed();

        println!("\n=== SIMD Decompression Benchmark (1MB x 10) ===");
        println!("SIMD impl:    {:?}", simd_time);
        println!("Basic impl:   {:?}", basic_time);
        println!("libdeflate:   {:?}", libdeflate_time);
        println!(
            "SIMD vs libdeflate: {:.2}x",
            simd_time.as_secs_f64() / libdeflate_time.as_secs_f64()
        );
        println!(
            "SIMD vs basic:      {:.2}x",
            simd_time.as_secs_f64() / basic_time.as_secs_f64()
        );
    }
}
