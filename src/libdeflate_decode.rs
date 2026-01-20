//! Libdeflate-compatible Decode Loop
//!
//! This module implements a decode loop that matches libdeflate's structure
//! and performance characteristics. Key optimizations:
//!
//! 1. **Signed literal check**: `(entry as i32) < 0` for fast literal detection
//! 2. **Single entry consumption**: `bitsleft -= entry.raw()` (low 5 bits)
//! 3. **Preload next entry**: Look ahead before completing current operation
//! 4. **saved_bitbuf pattern**: Extract extra bits from saved state
//! 5. **Fastloop/generic split**: Fast path with margins, safe fallback
//! 6. **BMI2 intrinsics**: Use `_bzhi_u64` for bit extraction on x86_64

#![allow(dead_code)]

use crate::libdeflate_entry::{DistTable, LitLenTable};

/// Check if BMI2 is available at runtime
#[cfg(target_arch = "x86_64")]
fn has_bmi2() -> bool {
    #[cfg(target_feature = "bmi2")]
    {
        true
    }
    #[cfg(not(target_feature = "bmi2"))]
    {
        is_x86_feature_detected!("bmi2")
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn has_bmi2() -> bool {
    false
}

/// Extract low n bits using BMI2 bzhi instruction
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn extract_bits_bmi2(val: u64, n: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_bzhi_u64(val, n)
    }
}

/// Fallback bit extraction
#[inline(always)]
fn extract_bits(val: u64, n: u32) -> u64 {
    val & ((1u64 << n) - 1)
}
use std::io::{Error, ErrorKind, Result};

/// Bit reader optimized for libdeflate-style decode
#[derive(Debug)]
pub struct LibdeflateBits<'a> {
    /// Input buffer
    data: &'a [u8],
    /// Current position in input
    pos: usize,
    /// Bit buffer (64-bit)
    bitbuf: u64,
    /// Number of valid bits (allow garbage in high bits per libdeflate)
    bitsleft: u32,
    /// Count of bytes read past input end
    overread_count: u32,
}

impl<'a> LibdeflateBits<'a> {
    /// Create a new bit reader
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut bits = Self {
            data,
            pos: 0,
            bitbuf: 0,
            bitsleft: 0,
            overread_count: 0,
        };
        bits.refill_branchless();
        bits
    }

    /// Maximum consumable bits (63 for branchless refill)
    const MAX_BITSLEFT: u32 = 63;

    /// Minimum bits guaranteed after refill
    const CONSUMABLE_NBITS: u32 = Self::MAX_BITSLEFT - 7; // 56

    /// Branchless refill matching libdeflate's REFILL_BITS_BRANCHLESS
    /// 
    /// From libdeflate:
    /// ```c
    /// bitbuf |= get_unaligned_leword(in_next) << (u8)bitsleft;
    /// in_next += sizeof(bitbuf_t) - 1;
    /// in_next -= (bitsleft >> 3) & 0x7;
    /// bitsleft |= MAX_BITSLEFT & ~7;
    /// ```
    #[inline(always)]
    pub fn refill_branchless(&mut self) {
        if self.pos + 8 <= self.data.len() {
            // Fast path: unaligned 8-byte load (avoids slice overhead)
            let word = unsafe {
                (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned()
            };
            let word = u64::from_le(word);
            self.bitbuf |= word << (self.bitsleft as u8);
            self.pos += 7;
            self.pos -= ((self.bitsleft >> 3) & 0x7) as usize;
            self.bitsleft |= Self::MAX_BITSLEFT & !7;
        } else {
            // Slow path: near end of input
            self.refill_slow();
        }
    }

    /// Slow refill for near end of input
    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.bitsleft <= Self::MAX_BITSLEFT - 8 {
            if self.pos < self.data.len() {
                self.bitbuf |= (self.data[self.pos] as u64) << self.bitsleft;
                self.pos += 1;
            } else {
                // Reading past end - track overread
                self.overread_count += 1;
            }
            self.bitsleft += 8;
        }
    }

    /// Peek at bits without consuming
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u64 {
        self.bitbuf & ((1u64 << n) - 1)
    }

    /// Peek at all available bits (for table lookup)
    #[inline(always)]
    pub fn peek_bits(&self) -> u64 {
        self.bitbuf
    }

    /// Consume bits (libdeflate style: bitsleft -= entry is done externally)
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bitbuf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    /// Consume bits using entry's raw value (low 5 bits)
    /// This matches libdeflate's `bitsleft -= entry` optimization
    #[inline(always)]
    pub fn consume_entry(&mut self, entry_raw: u32) {
        let n = entry_raw & 0x1F;
        self.bitbuf >>= n;
        self.bitsleft = self.bitsleft.wrapping_sub(n);
    }

    /// Available consumable bits
    #[inline(always)]
    pub fn available(&self) -> u32 {
        self.bitsleft & 0xFF // Mask to 8 bits per libdeflate
    }

    /// Check if we have enough bits
    #[inline(always)]
    pub fn has_bits(&self, n: u32) -> bool {
        (self.bitsleft & 0xFF) >= n
    }

    /// Get current position
    #[inline(always)]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Check overread status
    #[inline(always)]
    pub fn is_overread(&self) -> bool {
        self.overread_count > 8 // Allow some overread for bit buffer
    }
}

/// Copy a match (LZ77 back-reference) to output
/// UNSAFE version for maximum performance
/// Optimized for common patterns:
/// - distance=1 (RLE): memset
/// - distance>=8: word-at-a-time copy with overlapping writes
/// - distance 2-7: pattern expansion
#[inline(always)]
fn copy_match(output: &mut [u8], out_pos: usize, distance: u32, length: u32) -> usize {
    let dist = distance as usize;
    let len = length as usize;
    
    unsafe {
        let out_ptr = output.as_mut_ptr();
        let dst = out_ptr.add(out_pos);
        let src = out_ptr.add(out_pos - dist);
        
        if distance == 1 {
            // RLE: memset
            let byte = *src;
            std::ptr::write_bytes(dst, byte, len);
        } else if dist >= 8 {
            // Fast path: 8-byte chunks with overlapping writes
            // This works because we're always writing forward
            let mut s = src;
            let mut d = dst;
            let end = dst.add(len);
            
            while d < end {
                let chunk = (s as *const u64).read_unaligned();
                (d as *mut u64).write_unaligned(chunk);
                s = s.add(8);
                d = d.add(8);
            }
        } else if dist >= 4 {
            // 4-byte chunks for medium distances
            let mut s = src;
            let mut d = dst;
            let end = dst.add(len);
            
            while d < end {
                let chunk = (s as *const u32).read_unaligned();
                (d as *mut u32).write_unaligned(chunk);
                s = s.add(4);
                d = d.add(4);
            }
        } else {
            // Byte-by-byte for very short distances (2-3)
            for i in 0..len {
                *dst.add(i) = *src.add(i % dist);
            }
        }
    }

    out_pos + len
}

/// Decode a deflate stream using libdeflate-compatible algorithm
pub fn decode_libdeflate(
    input: &[u8],
    output: &mut [u8],
) -> Result<usize> {
    let mut bits = LibdeflateBits::new(input);
    let mut out_pos = 0;

    loop {
        // Ensure we have bits for block header
        if !bits.has_bits(3) {
            bits.refill_branchless();
        }

        let bfinal = bits.peek(1) != 0;
        let btype = ((bits.peek(3) >> 1) & 0x3) as u8;
        bits.consume(3);

        match btype {
            0 => {
                // Stored block
                out_pos = decode_stored(&mut bits, output, out_pos)?;
            }
            1 => {
                // Fixed Huffman
                out_pos = decode_fixed(&mut bits, output, out_pos)?;
            }
            2 => {
                // Dynamic Huffman
                out_pos = decode_dynamic(&mut bits, output, out_pos)?;
            }
            _ => {
                return Err(Error::new(ErrorKind::InvalidData, "Reserved block type"));
            }
        }

        if bfinal {
            break;
        }
    }

    if bits.is_overread() {
        return Err(Error::new(ErrorKind::InvalidData, "Input overread"));
    }

    Ok(out_pos)
}

/// Decode stored block
fn decode_stored(
    bits: &mut LibdeflateBits,
    output: &mut [u8],
    mut out_pos: usize,
) -> Result<usize> {
    // Align to byte boundary
    let extra = bits.available() % 8;
    if extra > 0 {
        bits.consume(extra);
    }

    bits.refill_branchless();

    // Read LEN and NLEN
    let len = bits.peek(16) as u16;
    bits.consume(16);
    bits.refill_branchless();
    let nlen = bits.peek(16) as u16;
    bits.consume(16);

    if len != !nlen {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid stored block length"));
    }

    // Copy bytes
    let len = len as usize;
    if out_pos + len > output.len() {
        return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
    }

    // Read from bit buffer and input
    bits.refill_branchless();
    for _ in 0..len {
        if bits.available() < 8 {
            bits.refill_branchless();
        }
        output[out_pos] = bits.peek(8) as u8;
        bits.consume(8);
        out_pos += 1;
    }

    Ok(out_pos)
}

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};
use std::sync::Arc;

/// Cached fixed Huffman tables
static FIXED_TABLES: OnceLock<(LitLenTable, DistTable)> = OnceLock::new();

use std::collections::HashMap;

/// JIT table cache for dynamic Huffman tables (simple HashMap, max 64 entries)
type TableCache = Mutex<HashMap<u64, Arc<(LitLenTable, DistTable)>>>;
static DYNAMIC_TABLE_CACHE: OnceLock<TableCache> = OnceLock::new();

/// Get or create the table cache
fn get_table_cache() -> &'static TableCache {
    DYNAMIC_TABLE_CACHE.get_or_init(|| Mutex::new(HashMap::with_capacity(64)))
}

/// Hash code lengths for cache lookup
fn hash_code_lengths(litlen_lengths: &[u8], dist_lengths: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    litlen_lengths.hash(&mut hasher);
    dist_lengths.hash(&mut hasher);
    hasher.finish()
}

/// Get or build dynamic Huffman tables with caching
#[allow(dead_code)]
fn get_or_build_dynamic_tables(
    litlen_lengths: &[u8],
    dist_lengths: &[u8],
) -> Result<Arc<(LitLenTable, DistTable)>> {
    let hash = hash_code_lengths(litlen_lengths, dist_lengths);
    
    // Try cache first
    {
        let cache = get_table_cache().lock().unwrap();
        if let Some(tables) = cache.get(&hash) {
            return Ok(Arc::clone(tables));
        }
    }
    
    // Build new tables
    let litlen = LitLenTable::build(litlen_lengths)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Failed to build litlen table"))?;
    let dist = DistTable::build(dist_lengths)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Failed to build dist table"))?;
    
    let tables = Arc::new((litlen, dist));
    
    // Cache it (simple eviction: clear if too large)
    {
        let mut cache = get_table_cache().lock().unwrap();
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(hash, Arc::clone(&tables));
    }
    
    Ok(tables)
}

/// Build fixed Huffman tables (cached)
fn get_fixed_tables() -> &'static (LitLenTable, DistTable) {
    FIXED_TABLES.get_or_init(|| {
        // Fixed literal/length table
        let mut litlen_lengths = [0u8; 288];
        for i in 0..144 {
            litlen_lengths[i] = 8;
        }
        for i in 144..256 {
            litlen_lengths[i] = 9;
        }
        for i in 256..280 {
            litlen_lengths[i] = 7;
        }
        for i in 280..288 {
            litlen_lengths[i] = 8;
        }

        // Fixed distance table (all 5-bit codes)
        let dist_lengths = [5u8; 32];

        let litlen = LitLenTable::build(&litlen_lengths).unwrap();
        let dist = DistTable::build(&dist_lengths).unwrap();

        (litlen, dist)
    })
}

/// Decode fixed Huffman block
fn decode_fixed(
    bits: &mut LibdeflateBits,
    output: &mut [u8],
    out_pos: usize,
) -> Result<usize> {
    let (litlen_table, dist_table) = get_fixed_tables();
    let (litlen_table, dist_table) = (litlen_table, dist_table);
    decode_huffman(bits, output, out_pos, &litlen_table, &dist_table)
}

/// Decode dynamic Huffman block
fn decode_dynamic(
    bits: &mut LibdeflateBits,
    output: &mut [u8],
    out_pos: usize,
) -> Result<usize> {
    bits.refill_branchless();

    // Read header
    let hlit = (bits.peek(5) as usize) + 257;
    bits.consume(5);
    let hdist = (bits.peek(5) as usize) + 1;
    bits.consume(5);
    let hclen = (bits.peek(4) as usize) + 4;
    bits.consume(4);

    // Read code length code lengths
    const CODE_LENGTH_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        bits.refill_branchless();
        cl_lengths[CODE_LENGTH_ORDER[i]] = bits.peek(3) as u8;
        bits.consume(3);
    }

    // Build code length table
    let cl_table = build_code_length_table(&cl_lengths)?;

    // Read literal/length and distance code lengths
    let mut all_lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        bits.refill_branchless();
        let entry = cl_table[bits.peek(7) as usize & 0x7F];
        let symbol = (entry >> 8) as u8;
        let nbits = (entry & 0xFF) as u32;
        bits.consume(nbits);

        match symbol {
            0..=15 => {
                all_lengths[i] = symbol;
                i += 1;
            }
            16 => {
                bits.refill_branchless();
                let repeat = 3 + bits.peek(2) as usize;
                bits.consume(2);
                let prev = if i > 0 { all_lengths[i - 1] } else { 0 };
                for _ in 0..repeat {
                    all_lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                bits.refill_branchless();
                let repeat = 3 + bits.peek(3) as usize;
                bits.consume(3);
                for _ in 0..repeat {
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                bits.refill_branchless();
                let repeat = 11 + bits.peek(7) as usize;
                bits.consume(7);
                for _ in 0..repeat {
                    all_lengths[i] = 0;
                    i += 1;
                }
            }
            _ => {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid code length code"));
            }
        }
    }

    // Build tables
    let litlen_table = LitLenTable::build(&all_lengths[..hlit])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Failed to build litlen table"))?;
    let dist_table = DistTable::build(&all_lengths[hlit..])
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Failed to build dist table"))?;

    decode_huffman(bits, output, out_pos, &litlen_table, &dist_table)
}

/// Build simple code length table
fn build_code_length_table(lengths: &[u8; 19]) -> Result<[u16; 128]> {
    let mut table = [0u16; 128];

    // Count codes of each length
    let mut count = [0u16; 8];
    for &len in lengths.iter() {
        if len > 0 && len <= 7 {
            count[len as usize] += 1;
        }
    }

    // Compute first code for each length
    let mut code = 0u32;
    let mut first_code = [0u32; 8];
    for len in 1..=7 {
        code = (code + count[len - 1] as u32) << 1;
        first_code[len] = code;
    }

    // Fill table
    let mut next_code = first_code;
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len = len as usize;
        let codeword = next_code[len];
        next_code[len] += 1;

        // Reverse bits
        let mut reversed = 0u32;
        let mut c = codeword;
        for _ in 0..len {
            reversed = (reversed << 1) | (c & 1);
            c >>= 1;
        }

        // Fill table entries
        let stride = 1usize << len;
        let mut idx = reversed as usize;
        while idx < 128 {
            table[idx] = ((symbol as u16) << 8) | (len as u16);
            idx += stride;
        }
    }

    Ok(table)
}

/// Main Huffman decode loop - libdeflate style
/// 
/// Key optimizations:
/// 1. Signed comparison for literal check: (entry as i32) < 0
/// 2. saved_bitbuf pattern for extra bit extraction
/// 3. Multi-literal unrolling in tight loop
fn decode_huffman(
    bits: &mut LibdeflateBits,
    output: &mut [u8],
    mut out_pos: usize,
    litlen_table: &LitLenTable,
    dist_table: &DistTable,
) -> Result<usize> {
    // Fastloop: only run when we have margin in output buffer
    const FASTLOOP_MARGIN: usize = 274; // max match length + 16

    'fastloop: while out_pos + FASTLOOP_MARGIN <= output.len() {
        bits.refill_branchless();

        // Save bitbuf for extracting extra bits later
        let saved_bitbuf = bits.peek_bits();

        // Look up entry (with subtable resolution)
        let mut entry = litlen_table.lookup(saved_bitbuf);
        if entry.is_subtable_ptr() {
            entry = litlen_table.lookup_subtable(entry, saved_bitbuf);
        }

        // Fast path: literal (bit 31 set)
        // Use signed comparison: (entry as i32) < 0
        if (entry.raw() as i32) < 0 {
            // LITERAL - tight loop with unsafe writes for max speed
            let out_ptr = output.as_mut_ptr();
            
            unsafe {
                *out_ptr.add(out_pos) = entry.literal_value();
            }
            out_pos += 1;
            bits.consume_entry(entry.raw());

            // Unrolled literal decode - up to 4 more without refill
            // Each literal uses max 15 bits, so 56+ bits = 3+ literals safe
            if bits.available() >= 45 {
                // Try 3 more literals in a row
                let s2 = bits.peek_bits();
                let mut e2 = litlen_table.lookup(s2);
                if e2.is_subtable_ptr() { e2 = litlen_table.lookup_subtable(e2, s2); }
                
                if (e2.raw() as i32) < 0 {
                    unsafe { *out_ptr.add(out_pos) = e2.literal_value(); }
                    out_pos += 1;
                    bits.consume_entry(e2.raw());
                    
                    let s3 = bits.peek_bits();
                    let mut e3 = litlen_table.lookup(s3);
                    if e3.is_subtable_ptr() { e3 = litlen_table.lookup_subtable(e3, s3); }
                    
                    if (e3.raw() as i32) < 0 {
                        unsafe { *out_ptr.add(out_pos) = e3.literal_value(); }
                        out_pos += 1;
                        bits.consume_entry(e3.raw());
                        
                        let s4 = bits.peek_bits();
                        let mut e4 = litlen_table.lookup(s4);
                        if e4.is_subtable_ptr() { e4 = litlen_table.lookup_subtable(e4, s4); }
                        
                        if (e4.raw() as i32) < 0 {
                            unsafe { *out_ptr.add(out_pos) = e4.literal_value(); }
                            out_pos += 1;
                            bits.consume_entry(e4.raw());
                        }
                    }
                }
            } else {
                // Slower path: check bits each time
                while bits.available() >= 15 {
                    let saved2 = bits.peek_bits();
                    let mut e2 = litlen_table.lookup(saved2);
                    if e2.is_subtable_ptr() {
                        e2 = litlen_table.lookup_subtable(e2, saved2);
                    }
                    
                    if (e2.raw() as i32) < 0 {
                        unsafe { *out_ptr.add(out_pos) = e2.literal_value(); }
                        out_pos += 1;
                        bits.consume_entry(e2.raw());
                    } else {
                        break;
                    }
                }
            }
            continue 'fastloop;
        }

        // Check for exceptional cases (EOB)
        if entry.is_exceptional() {
            if entry.is_end_of_block() {
                bits.consume_entry(entry.raw());
                return Ok(out_pos);
            }
            return Err(Error::new(ErrorKind::InvalidData, "Unresolved subtable"));
        }

        // LENGTH CODE - decode length and distance together for better pipelining
        bits.consume_entry(entry.raw());
        let length = entry.decode_length(saved_bitbuf);

        // Refill and preload distance entry
        bits.refill_branchless();
        let dist_saved = bits.peek_bits();
        let mut dist_entry = dist_table.lookup(dist_saved);
        
        // Prefetch the match source location (hide memory latency)
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Estimate source position (distance is typically small)
            let likely_src = output.as_ptr().add(out_pos.saturating_sub(32));
            std::arch::x86_64::_mm_prefetch(likely_src as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        
        if dist_entry.is_subtable_ptr() {
            dist_entry = dist_table.lookup_subtable(dist_entry, dist_saved);
        }

        if dist_entry.is_exceptional() {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance code"));
        }

        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_saved);

        // Validate (branch unlikely to be taken)
        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        // Copy match with optimized routine
        out_pos = copy_match(output, out_pos, distance, length);
    }

    // Generic loop for near end of output
    loop {
        if !bits.has_bits(15) {
            bits.refill_branchless();
        }

        let saved_bitbuf = bits.peek_bits();
        let entry = litlen_table.resolve(saved_bitbuf);

        if (entry.raw() as i32) < 0 {
            // Literal
            if out_pos >= output.len() {
                return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
            }
            output[out_pos] = entry.literal_value();
            out_pos += 1;
            bits.consume_entry(entry.raw());
            continue;
        }

        if entry.is_end_of_block() {
            bits.consume_entry(entry.raw());
            return Ok(out_pos);
        }

        if entry.is_exceptional() {
            return Err(Error::new(ErrorKind::InvalidData, "Unresolved subtable in generic loop"));
        }

        // Length
        bits.consume_entry(entry.raw());
        let length = entry.decode_length(saved_bitbuf);

        // Distance
        if !bits.has_bits(15) {
            bits.refill_branchless();
        }
        let dist_saved = bits.peek_bits();
        let dist_entry = dist_table.resolve(dist_saved);

        if dist_entry.is_exceptional() {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid distance code"));
        }

        bits.consume_entry(dist_entry.raw());
        let distance = dist_entry.decode_distance(dist_saved);

        if distance == 0 || distance as usize > out_pos {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid distance {} at pos {}", distance, out_pos),
            ));
        }

        if out_pos + length as usize > output.len() {
            return Err(Error::new(ErrorKind::WriteZero, "Output buffer full"));
        }

        out_pos = copy_match(output, out_pos, distance, length);
    }
}

/// Public entry point for decompression
pub fn inflate_libdeflate(deflate_data: &[u8], output: &mut [u8]) -> Result<usize> {
    decode_libdeflate(deflate_data, output)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    /// Test simple literal decompression
    #[test]
    fn test_libdeflate_decode_literals() {
        let original = b"Hello, World!";

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_libdeflate(&compressed, &mut output).expect("Failed to decode");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    /// Test RLE pattern (distance=1)
    #[test]
    fn test_libdeflate_decode_rle() {
        let original = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_libdeflate(&compressed, &mut output).expect("Failed to decode");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    /// Test matches with various distances
    #[test]
    fn test_libdeflate_decode_matches() {
        let original = b"abcabcabcabcabcabc"; // distance=3 matches

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_libdeflate(&compressed, &mut output).expect("Failed to decode");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    /// Test larger data
    #[test]
    fn test_libdeflate_decode_large() {
        let original = b"This is a test of the libdeflate-compatible decoder. ".repeat(1000);

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = vec![0u8; original.len() + 100];
        let size = inflate_libdeflate(&compressed, &mut output).expect("Failed to decode");

        assert_eq!(size, original.len());
        assert_eq!(&output[..size], original.as_slice());
    }

    /// Compare with libdeflate
    #[test]
    fn test_compare_with_libdeflate() {
        let original = b"Testing comparison with actual libdeflate output.".repeat(100);

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Our decode
        let mut our_output = vec![0u8; original.len() + 100];
        let our_size = inflate_libdeflate(&compressed, &mut our_output).expect("Our decode failed");

        // libdeflate decode
        let mut lib_output = vec![0u8; original.len() + 100];
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(&compressed, &mut lib_output)
            .expect("libdeflate failed");

        assert_eq!(our_size, lib_size, "Size mismatch");
        assert_eq!(
            &our_output[..our_size],
            &lib_output[..lib_size],
            "Content mismatch"
        );
    }

    /// Benchmark comparison
    #[test]
    fn bench_libdeflate_decode() {
        let original = b"Benchmark data for testing performance. ".repeat(10000);

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = vec![0u8; original.len() + 100];

        // Warmup
        for _ in 0..5 {
            let _ = inflate_libdeflate(&compressed, &mut output);
        }

        // Benchmark our decode
        let start = std::time::Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = inflate_libdeflate(&compressed, &mut output);
        }
        let our_time = start.elapsed();

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = libdeflater::Decompressor::new()
                .deflate_decompress(&compressed, &mut output);
        }
        let lib_time = start.elapsed();

        let bytes_per_iter = original.len();
        let our_throughput = (bytes_per_iter * iterations) as f64 / our_time.as_secs_f64() / 1e6;
        let lib_throughput = (bytes_per_iter * iterations) as f64 / lib_time.as_secs_f64() / 1e6;

        eprintln!("\n=== Libdeflate-compatible Decode Benchmark ===");
        eprintln!("Data size: {} bytes", bytes_per_iter);
        eprintln!("Our throughput:       {:>8.1} MB/s", our_throughput);
        eprintln!("libdeflate throughput: {:>8.1} MB/s", lib_throughput);
        eprintln!("Ratio: {:.1}%", 100.0 * our_throughput / lib_throughput);
        eprintln!("================================================\n");
    }

    /// Benchmark on silesia (real-world data)
    #[test]
    fn bench_silesia() {
        let gzip_data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[SKIP] No silesia benchmark file");
                return;
            }
        };

        // Extract deflate data
        let deflate_start = 10
            + if (gzip_data[3] & 0x08) != 0 {
                gzip_data[10..].iter().position(|&b| b == 0).unwrap_or(0) + 1
            } else {
                0
            };
        let deflate_end = gzip_data.len() - 8;
        let deflate_data = &gzip_data[deflate_start..deflate_end];

        // Get expected size from ISIZE
        let isize_bytes = &gzip_data[gzip_data.len() - 4..];
        let isize =
            u32::from_le_bytes([isize_bytes[0], isize_bytes[1], isize_bytes[2], isize_bytes[3]])
                as usize;

        let mut output = vec![0u8; isize + 1000];

        // Warmup + verify correctness
        let our_size = inflate_libdeflate(deflate_data, &mut output)
            .expect("Our silesia decode failed");
        
        let mut lib_output = vec![0u8; isize + 1000];
        let lib_size = libdeflater::Decompressor::new()
            .deflate_decompress(deflate_data, &mut lib_output)
            .expect("libdeflate silesia decode failed");

        assert_eq!(our_size, lib_size, "Silesia size mismatch");
        
        // Check first 10KB for correctness
        for i in 0..10000.min(our_size) {
            if output[i] != lib_output[i] {
                panic!("Silesia mismatch at byte {}: got {} expected {}", 
                    i, output[i], lib_output[i]);
            }
        }

        // Benchmark
        let iterations = 5;
        
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = inflate_libdeflate(deflate_data, &mut output);
        }
        let our_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = libdeflater::Decompressor::new()
                .deflate_decompress(deflate_data, &mut lib_output);
        }
        let lib_time = start.elapsed();

        let our_throughput = (isize * iterations) as f64 / our_time.as_secs_f64() / 1e6;
        let lib_throughput = (isize * iterations) as f64 / lib_time.as_secs_f64() / 1e6;

        eprintln!("\n=== SILESIA Benchmark ===");
        eprintln!("Data size: {} MB", isize / 1_000_000);
        eprintln!("Our throughput:       {:>8.1} MB/s", our_throughput);
        eprintln!("libdeflate throughput: {:>8.1} MB/s", lib_throughput);
        eprintln!("Ratio: {:.1}%", 100.0 * our_throughput / lib_throughput);
        eprintln!("=========================\n");
    }

    /// Profile the decode loop to find bottlenecks
    #[test]
    fn profile_decode_hotspots() {
        let gz = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => return,
        };

        let start = 10 + if (gz[3] & 0x08) != 0 {
            gz[10..].iter().position(|&b| b == 0).unwrap_or(0) + 1
        } else { 0 };
        let end = gz.len() - 8;
        let deflate = &gz[start..end];
        let isize = u32::from_le_bytes([gz[gz.len()-4], gz[gz.len()-3], gz[gz.len()-2], gz[gz.len()-1]]) as usize;

        let mut out = vec![0u8; isize + 1000];

        // Run decode and measure wall clock
        let iters = 10;
        let start_t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = inflate_libdeflate(deflate, &mut out);
        }
        let elapsed = start_t.elapsed();
        
        let throughput = (isize * iters) as f64 / elapsed.as_secs_f64() / 1e6;
        let ns_per_byte = elapsed.as_nanos() as f64 / (isize * iters) as f64;
        
        eprintln!("\n=== DECODE PROFILING ===");
        eprintln!("Total: {} MB in {:.2}s", (isize * iters) / 1_000_000, elapsed.as_secs_f64());
        eprintln!("Throughput: {:.1} MB/s", throughput);
        eprintln!("Time per byte: {:.2} ns", ns_per_byte);
        eprintln!("========================\n");
        
        // For deeper profiling, run: 
        // cargo flamegraph --bin gzippy -- -d benchmark_data/silesia-gzip.tar.gz -c > /dev/null
    }
}
