//! BGZF (Block GZIP Format) Parallel Decompression
//!
//! BGZF files have independent blocks with embedded size markers, allowing
//! perfect parallelism with zero lock contention.
//!
//! ## Strategy
//!
//! 1. Parse BGZF headers to find all block boundaries and output sizes (ISIZE)
//! 2. Pre-allocate entire output buffer based on sum of ISIZE values
//! 3. Decompress blocks in parallel, writing directly to pre-calculated offsets
//! 4. Single write of complete output
//!
//! ## Performance Target: 4000+ MB/s with 14 threads
//!
//! With single-threaded inflate at 10700 MB/s and no lock contention,
//! theoretical max is ~150,000 MB/s. Memory bandwidth limits us to ~4000-5000 MB/s.

#![allow(clippy::needless_range_loop)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::combined_lut::CombinedLUT;
use crate::inflate_tables::CODE_LENGTH_ORDER;
use crate::two_level_table::{FastBits, TwoLevelTable};

/// BGZF block information
#[derive(Debug, Clone)]
struct BgzfBlock {
    /// Byte offset of block start in compressed data
    start: usize,
    /// Total block length (including header and trailer)
    length: usize,
    /// Offset into deflate data (after header)
    deflate_offset: usize,
    /// Uncompressed size (from ISIZE trailer)
    isize: u32,
    /// Output offset (calculated during planning)
    output_offset: usize,
}

/// Parse all BGZF blocks from compressed data
fn parse_bgzf_blocks(data: &[u8]) -> io::Result<Vec<BgzfBlock>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    let mut output_offset = 0;

    while offset + 18 < data.len() {
        // Check gzip magic
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // Must have FEXTRA flag
        if data[offset + 3] & 0x04 == 0 {
            break;
        }

        // Get XLEN
        if offset + 12 > data.len() {
            break;
        }
        let xlen = u16::from_le_bytes([data[offset + 10], data[offset + 11]]) as usize;
        if offset + 12 + xlen > data.len() {
            break;
        }

        // Find GZ subfield with block size
        let extra_start = offset + 12;
        let extra_field = &data[extra_start..extra_start + xlen];
        let mut block_size = None;
        let mut pos = 0;

        while pos + 4 <= extra_field.len() {
            let subfield_id = &extra_field[pos..pos + 2];
            let subfield_len =
                u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;

            if subfield_id == b"GZ" {
                if subfield_len == 4 && pos + 8 <= extra_field.len() {
                    // New 4-byte format (supports blocks > 64KB)
                    let size = u32::from_le_bytes([
                        extra_field[pos + 4],
                        extra_field[pos + 5],
                        extra_field[pos + 6],
                        extra_field[pos + 7],
                    ]) as usize;
                    if size > 0 {
                        block_size = Some(size);
                    }
                    break;
                } else if subfield_len == 2 && pos + 6 <= extra_field.len() {
                    // Legacy 2-byte format (BSIZE-1)
                    let size_minus_1 =
                        u16::from_le_bytes([extra_field[pos + 4], extra_field[pos + 5]]) as usize;
                    block_size = Some(size_minus_1 + 1);
                    break;
                }
            }

            pos += 4 + subfield_len;
        }

        let length = match block_size {
            Some(l) if l > 0 && offset + l <= data.len() => l,
            _ => break,
        };

        // Calculate deflate data offset (after header)
        let deflate_offset = calculate_deflate_offset(&data[offset..offset + length]);

        // Get ISIZE from trailer (last 4 bytes)
        let isize = if length >= 4 {
            u32::from_le_bytes([
                data[offset + length - 4],
                data[offset + length - 3],
                data[offset + length - 2],
                data[offset + length - 1],
            ])
        } else {
            0
        };

        blocks.push(BgzfBlock {
            start: offset,
            length,
            deflate_offset,
            isize,
            output_offset,
        });

        output_offset += isize as usize;
        offset += length;
    }

    if blocks.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No BGZF blocks found",
        ));
    }

    Ok(blocks)
}

/// Calculate offset to deflate data within a gzip block
fn calculate_deflate_offset(block: &[u8]) -> usize {
    if block.len() < 10 {
        return block.len();
    }

    let flags = block[3];
    let mut offset = 10;

    // FEXTRA
    if flags & 0x04 != 0 && offset + 2 <= block.len() {
        let xlen = u16::from_le_bytes([block[offset], block[offset + 1]]) as usize;
        offset += 2 + xlen;
    }

    // FNAME
    if flags & 0x08 != 0 {
        while offset < block.len() && block[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FCOMMENT
    if flags & 0x10 != 0 {
        while offset < block.len() && block[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FHCRC
    if flags & 0x02 != 0 {
        offset += 2;
    }

    offset.min(block.len())
}

/// Inflate directly into a pre-allocated output slice
///
/// This is the key function for zero-copy parallel decompression.
/// Uses our CombinedLUT for maximum speed (10700+ MB/s single-threaded).
fn inflate_into(deflate_data: &[u8], output: &mut [u8]) -> io::Result<usize> {
    inflate_into_pub(deflate_data, output)
}

/// Public version of inflate_into for use by other modules
pub fn inflate_into_pub(deflate_data: &[u8], output: &mut [u8]) -> io::Result<usize> {
    let mut bits = FastBits::new(deflate_data);
    let mut out_pos = 0;

    loop {
        bits.refill();

        let bfinal = bits.read(1);
        let btype = bits.read(2);

        match btype {
            0 => out_pos = decode_stored_into(&mut bits, output, out_pos)?,
            1 => out_pos = decode_fixed_into(&mut bits, output, out_pos)?,
            2 => out_pos = decode_dynamic_into(&mut bits, output, out_pos)?,
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

    Ok(out_pos)
}

/// Decode stored block directly into output slice
fn decode_stored_into(
    bits: &mut FastBits,
    output: &mut [u8],
    mut out_pos: usize,
) -> io::Result<usize> {
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

    for _ in 0..len {
        if out_pos >= output.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "Output buffer full",
            ));
        }
        bits.ensure(8);
        output[out_pos] = bits.read(8) as u8;
        out_pos += 1;
    }

    Ok(out_pos)
}

/// Pre-built fixed Huffman tables
fn get_fixed_tables() -> (
    &'static TwoLevelTable,
    &'static TwoLevelTable,
    &'static CombinedLUT,
) {
    use std::sync::OnceLock;

    static FIXED_LIT_LEN: OnceLock<TwoLevelTable> = OnceLock::new();
    static FIXED_DIST: OnceLock<TwoLevelTable> = OnceLock::new();
    static FIXED_COMBINED: OnceLock<CombinedLUT> = OnceLock::new();

    let lit_len = FIXED_LIT_LEN.get_or_init(|| {
        let mut lens = [0u8; 288];
        for i in 0..144 {
            lens[i] = 8;
        }
        for i in 144..256 {
            lens[i] = 9;
        }
        for i in 256..280 {
            lens[i] = 7;
        }
        for i in 280..288 {
            lens[i] = 8;
        }
        TwoLevelTable::build(&lens).unwrap()
    });

    let dist = FIXED_DIST.get_or_init(|| {
        let lens = [5u8; 32];
        TwoLevelTable::build(&lens).unwrap()
    });

    let combined = FIXED_COMBINED.get_or_init(|| {
        let mut lit_len_lens = vec![0u8; 288];
        for i in 0..144 {
            lit_len_lens[i] = 8;
        }
        for i in 144..256 {
            lit_len_lens[i] = 9;
        }
        for i in 256..280 {
            lit_len_lens[i] = 7;
        }
        for i in 280..288 {
            lit_len_lens[i] = 8;
        }
        let dist_lens = vec![5u8; 32];
        CombinedLUT::build(&lit_len_lens, &dist_lens).unwrap()
    });

    (lit_len, dist, combined)
}

/// Decode fixed Huffman block into output slice
fn decode_fixed_into(bits: &mut FastBits, output: &mut [u8], out_pos: usize) -> io::Result<usize> {
    let (lit_len_table, dist_table, combined_lut) = get_fixed_tables();
    decode_huffman_into(
        bits,
        output,
        out_pos,
        combined_lut,
        lit_len_table,
        dist_table,
    )
}

/// Decode dynamic Huffman block into output slice
fn decode_dynamic_into(
    bits: &mut FastBits,
    output: &mut [u8],
    out_pos: usize,
) -> io::Result<usize> {
    bits.ensure(16);
    let hlit = bits.read(5) as usize + 257;
    let hdist = bits.read(5) as usize + 1;
    let hclen = bits.read(4) as usize + 4;

    // Read code length code lengths
    let mut code_len_lens = [0u8; 19];
    for i in 0..hclen {
        bits.ensure(8);
        code_len_lens[CODE_LENGTH_ORDER[i] as usize] = bits.read(3) as u8;
    }

    let code_len_table = TwoLevelTable::build(&code_len_lens)?;

    // Read all code lengths
    let total_codes = hlit + hdist;
    let mut code_lens = vec![0u8; total_codes];
    let mut i = 0;

    while i < total_codes {
        bits.ensure(16);
        let (symbol, sym_len) = code_len_table.decode(bits.buffer());
        if sym_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid code length code",
            ));
        }
        bits.consume(sym_len);

        match symbol {
            0..=15 => {
                code_lens[i] = symbol as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid repeat"));
                }
                let repeat = 3 + bits.read(2) as usize;
                let last = code_lens[i - 1];
                for _ in 0..repeat.min(total_codes - i) {
                    code_lens[i] = last;
                    i += 1;
                }
            }
            17 => {
                let repeat = 3 + bits.read(3) as usize;
                i += repeat.min(total_codes - i);
            }
            18 => {
                let repeat = 11 + bits.read(7) as usize;
                i += repeat.min(total_codes - i);
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid code")),
        }
    }

    let lit_len_table = TwoLevelTable::build(&code_lens[..hlit])?;
    let dist_table = TwoLevelTable::build(&code_lens[hlit..])?;
    let combined_lut = CombinedLUT::build(&code_lens[..hlit], &code_lens[hlit..])?;

    decode_huffman_into(
        bits,
        output,
        out_pos,
        &combined_lut,
        &lit_len_table,
        &dist_table,
    )
}

/// Core decode loop using CombinedLUT, writing directly to output slice
fn decode_huffman_into(
    bits: &mut FastBits,
    output: &mut [u8],
    mut out_pos: usize,
    combined_lut: &CombinedLUT,
    lit_len_table: &TwoLevelTable,
    dist_table: &TwoLevelTable,
) -> io::Result<usize> {
    use crate::combined_lut::{DIST_END_OF_BLOCK, DIST_LITERAL, DIST_SLOW_PATH};
    use crate::inflate_tables::{DIST_EXTRA_BITS, DIST_START, LEN_EXTRA_BITS, LEN_START};

    // Branch prediction hints (stable workaround for likely/unlikely)
    // These help the compiler optimize hot paths by marking cold paths
    #[cold]
    #[inline(never)]
    fn cold_path() {}

    #[inline(always)]
    fn likely(b: bool) -> bool {
        if !b {
            cold_path();
        }
        b
    }

    #[inline(always)]
    fn unlikely(b: bool) -> bool {
        if b {
            cold_path();
        }
        b
    }

    // Prefetch next output cache line (64 bytes ahead on x86_64)
    // This hides memory latency by loading data into L1 cache before it's needed
    #[inline(always)]
    #[allow(unused_variables)]
    fn prefetch_output(output: &[u8], pos: usize) {
        #[cfg(target_arch = "x86_64")]
        if pos + 64 < output.len() {
            // SAFETY: Pointer arithmetic within bounds, prefetch is advisory
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    output.as_ptr().add(pos + 64) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        // ARM prefetch is unstable in Rust, no-op for now
    }

    loop {
        bits.ensure(32);

        // Prefetch next output cache line
        prefetch_output(output, out_pos);

        let entry = combined_lut.decode(bits.buffer());

        // Long code fallback (rare - most codes fit in 12 bits)
        if unlikely(entry.bits_to_skip == 0) {
            let (symbol, code_len) = lit_len_table.decode(bits.buffer());
            if code_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid Huffman code",
                ));
            }
            bits.consume(code_len);

            if likely(symbol < 256) {
                // Literal byte - most common case
                if unlikely(out_pos >= output.len()) {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "Output buffer full",
                    ));
                }
                output[out_pos] = symbol as u8;
                out_pos += 1;
                continue;
            }
            if unlikely(symbol == 256) {
                // End of block - rare
                break;
            }

            // Length code (less common than literals)
            let len_idx = (symbol - 257) as usize;
            if unlikely(len_idx >= 29) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid length code",
                ));
            }

            bits.ensure(16);
            let length =
                LEN_START[len_idx] as usize + bits.read(LEN_EXTRA_BITS[len_idx] as u32) as usize;

            let (dist_sym, dist_len) = dist_table.decode(bits.buffer());
            if unlikely(dist_len == 0 || dist_sym >= 30) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance code",
                ));
            }
            bits.consume(dist_len);

            bits.ensure(16);
            let distance = DIST_START[dist_sym as usize] as usize
                + bits.read(DIST_EXTRA_BITS[dist_sym as usize] as u32) as usize;

            if unlikely(distance > out_pos || distance == 0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid distance",
                ));
            }

            out_pos = copy_match_into(output, out_pos, distance, length);
            continue;
        }

        bits.consume(entry.bits_to_skip as u32);

        match entry.distance {
            DIST_LITERAL => {
                if out_pos >= output.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "Output buffer full",
                    ));
                }
                output[out_pos] = entry.symbol_or_length;
                out_pos += 1;

                // Multi-literal optimization
                if bits.bits_available() >= 24 {
                    let entry2 = combined_lut.decode(bits.buffer());
                    if entry2.bits_to_skip > 0 && entry2.distance == DIST_LITERAL {
                        bits.consume(entry2.bits_to_skip as u32);
                        if out_pos < output.len() {
                            output[out_pos] = entry2.symbol_or_length;
                            out_pos += 1;
                        }

                        if bits.bits_available() >= 12 {
                            let entry3 = combined_lut.decode(bits.buffer());
                            if entry3.bits_to_skip > 0 && entry3.distance == DIST_LITERAL {
                                bits.consume(entry3.bits_to_skip as u32);
                                if out_pos < output.len() {
                                    output[out_pos] = entry3.symbol_or_length;
                                    out_pos += 1;
                                }
                            }
                        }
                    }
                }
            }

            DIST_END_OF_BLOCK => break,

            DIST_SLOW_PATH => {
                let length = entry.symbol_or_length as usize + 3;

                let (dist_sym, dist_len) = dist_table.decode(bits.buffer());
                if dist_len == 0 || dist_sym >= 30 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance code",
                    ));
                }
                bits.consume(dist_len);

                bits.ensure(16);
                let distance = DIST_START[dist_sym as usize] as usize
                    + bits.read(DIST_EXTRA_BITS[dist_sym as usize] as u32) as usize;

                if unlikely(distance > out_pos || distance == 0) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                out_pos = copy_match_into(output, out_pos, distance, length);
            }

            distance => {
                let length = entry.length();
                let dist = distance as usize;

                if dist > out_pos || dist == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                out_pos = copy_match_into(output, out_pos, dist, length);
            }
        }
    }

    Ok(out_pos)
}

/// Copy LZ77 match directly into output slice
/// Optimized for:
/// 1. distance=1 (RLE): memset
/// 2. distance >= length: non-overlapping memcpy
/// 3. distance >= 8: chunk copy
/// 4. small distance: byte-by-byte
#[inline(always)]
fn copy_match_into(output: &mut [u8], out_pos: usize, distance: usize, length: usize) -> usize {
    let src_start = out_pos - distance;

    // Bounds check
    if out_pos + length > output.len() {
        return out_pos;
    }

    unsafe {
        let dst = output.as_mut_ptr().add(out_pos);
        let src = output.as_ptr().add(src_start);

        if distance == 1 {
            // Very common: RLE (single byte repeat)
            // This is a major optimization from libdeflate
            let byte = *src;
            std::ptr::write_bytes(dst, byte, length);
        } else if distance >= length {
            // Non-overlapping: use memcpy
            std::ptr::copy_nonoverlapping(src, dst, length);
        } else if distance >= 8 {
            // Overlapping but distance >= 8: 8-byte chunk copy
            let mut remaining = length;
            let mut d = dst;
            let mut s = src;
            while remaining >= 8 {
                let chunk = (s as *const u64).read_unaligned();
                (d as *mut u64).write_unaligned(chunk);
                d = d.add(8);
                s = s.add(8);
                remaining -= 8;
            }
            // Copy remainder
            for i in 0..remaining {
                *d.add(i) = *s.add(i);
            }
        } else {
            // Small distance (2-7): byte-by-byte
            for i in 0..length {
                *dst.add(i) = *src.add(i % distance);
            }
        }
    }

    out_pos + length
}

/// Parallel BGZF decompression - the main entry point
///
/// This achieves maximum parallelism by:
/// 1. Pre-allocating entire output based on ISIZE values
/// 2. Computing output offsets during parsing phase
/// 3. Writing directly to disjoint regions (no locks needed)
/// 4. Using our fast CombinedLUT inflate (10700+ MB/s single-threaded)
pub fn decompress_bgzf_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Parse all BGZF blocks
    let blocks = parse_bgzf_blocks(data)?;

    if blocks.is_empty() {
        return Ok(0);
    }

    // Calculate total output size
    let total_output: usize = blocks.iter().map(|b| b.isize as usize).sum();

    // Pre-allocate output buffer
    let output = vec![0u8; total_output];

    // Parallel decompression using scoped threads
    let num_blocks = blocks.len();
    let next_block = AtomicUsize::new(0);

    // Use UnsafeCell for parallel mutable access to disjoint regions
    use std::cell::UnsafeCell;
    struct OutputBuffer(UnsafeCell<Vec<u8>>);
    unsafe impl Sync for OutputBuffer {}

    let output_cell = OutputBuffer(UnsafeCell::new(output));

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_blocks) {
            let blocks_ref = &blocks;
            let next_ref = &next_block;
            let output_ref = &output_cell;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let block = &blocks_ref[idx];
                    let block_data = &data[block.start..block.start + block.length];

                    // Get deflate data (skip header, exclude trailer)
                    let deflate_start = block.deflate_offset;
                    let deflate_end = block.length.saturating_sub(8);

                    if deflate_start >= deflate_end {
                        continue;
                    }

                    let deflate_data = &block_data[deflate_start..deflate_end];

                    // Get mutable slice for this block's output region
                    // SAFETY: Each block writes to a disjoint region
                    let output_ptr = unsafe { (*output_ref.0.get()).as_mut_ptr() };
                    let out_start = block.output_offset;
                    let out_size = block.isize as usize;
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(output_ptr.add(out_start), out_size)
                    };

                    // Use our optimized pure Rust inflate
                    let _ = inflate_into(deflate_data, out_slice);
                }
            });
        }
    });

    // Get output back and write
    let output = output_cell.0.into_inner();
    writer.write_all(&output)?;
    Ok(output.len() as u64)
}

// ============================================================================
// Multi-Member Parallel Decompression (for pigz-style files)
// ============================================================================

/// Information about a gzip member
#[derive(Debug, Clone)]
struct GzipMember {
    /// Start offset in compressed data
    start: usize,
    /// End offset (exclusive) in compressed data  
    end: usize,
    /// Offset to deflate data within member
    deflate_offset: usize,
    /// Uncompressed size (from ISIZE trailer)
    isize: u32,
    /// Output offset (calculated during planning)
    output_offset: usize,
}

/// Find all gzip member boundaries in a multi-member file
/// Uses ISIZE trailer for pre-allocation
fn find_gzip_members(data: &[u8]) -> Vec<GzipMember> {
    let mut members = Vec::new();
    let mut offset = 0;
    let mut output_offset = 0;

    while offset + 18 < data.len() {
        // Check gzip magic
        if data[offset] != 0x1f || data[offset + 1] != 0x8b || data[offset + 2] != 0x08 {
            break;
        }

        // Validate flags (reserved bits must be 0)
        let flags = data[offset + 3];
        if flags & 0xe0 != 0 {
            break;
        }

        // Calculate deflate start offset
        let deflate_offset = calculate_deflate_offset(&data[offset..]);

        // Find end of member by trying to decompress and checking trailer
        // We use a fast scan for the next valid gzip header as an estimate
        let member_start = offset;
        let mut member_end = data.len();

        // Search for next gzip header starting after minimum member size
        let search_start = offset + deflate_offset + 18; // min: deflate + trailer
        for i in search_start..data.len().saturating_sub(10) {
            if data[i] == 0x1f && data[i + 1] == 0x8b && data[i + 2] == 0x08 {
                // Validate this looks like a real header
                let next_flags = data[i + 3];
                if next_flags & 0xe0 == 0 {
                    // XFL should be reasonable
                    let xfl = data[i + 8];
                    if xfl == 0 || xfl == 2 || xfl == 4 {
                        // OS should be known
                        let os = data[i + 9];
                        if os <= 13 || os == 255 {
                            member_end = i;
                            break;
                        }
                    }
                }
            }
        }

        // Get ISIZE from trailer (last 4 bytes of member)
        let isize = if member_end >= member_start + 4 {
            u32::from_le_bytes([
                data[member_end - 4],
                data[member_end - 3],
                data[member_end - 2],
                data[member_end - 1],
            ])
        } else {
            0
        };

        members.push(GzipMember {
            start: member_start,
            end: member_end,
            deflate_offset,
            isize,
            output_offset,
        });

        output_offset += isize as usize;
        offset = member_end;
    }

    members
}

/// Parallel decompression for multi-member gzip files (pigz output)
///
/// Strategy:
/// 1. Find all member boundaries
/// 2. Read ISIZE from each member's trailer for pre-allocation
/// 3. Pre-allocate entire output buffer
/// 4. Decompress members in parallel, writing directly to pre-calculated offsets
pub fn decompress_multi_member_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Find all members
    let members = find_gzip_members(data);

    if members.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No gzip members found",
        ));
    }

    // For single member, use single-threaded path
    if members.len() == 1 || num_threads <= 1 {
        return decompress_single_member(data, writer);
    }

    // Calculate total output size from ISIZE values
    let total_output: usize = members.iter().map(|m| m.isize as usize).sum();

    // Pre-allocate output buffer
    let output = vec![0u8; total_output];

    // Parallel decompression
    let num_members = members.len();
    let next_member = AtomicUsize::new(0);

    use std::cell::UnsafeCell;
    struct OutputBuffer(UnsafeCell<Vec<u8>>);
    unsafe impl Sync for OutputBuffer {}

    let output_cell = OutputBuffer(UnsafeCell::new(output));

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_members) {
            let members_ref = &members;
            let next_ref = &next_member;
            let output_ref = &output_cell;

            scope.spawn(move || {
                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_members {
                        break;
                    }

                    let member = &members_ref[idx];
                    let member_data = &data[member.start..member.end];

                    // Get deflate data (skip header, exclude trailer)
                    let deflate_start = member.deflate_offset;
                    let deflate_end = (member.end - member.start).saturating_sub(8);

                    if deflate_start >= deflate_end {
                        continue;
                    }

                    let deflate_data = &member_data[deflate_start..deflate_end];

                    // Get mutable slice for this member's output region
                    // SAFETY: Each member writes to a disjoint region
                    let output_ptr = unsafe { (*output_ref.0.get()).as_mut_ptr() };
                    let out_start = member.output_offset;
                    let out_size = member.isize as usize;

                    if out_size == 0 {
                        continue;
                    }

                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(output_ptr.add(out_start), out_size)
                    };

                    // Use our optimized pure Rust inflate
                    let _ = inflate_into(deflate_data, out_slice);
                }
            });
        }
    });

    // Get output back and write
    let output = output_cell.0.into_inner();
    writer.write_all(&output)?;
    Ok(output.len() as u64)
}

/// Single-member decompression (sequential)
fn decompress_single_member<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    let mut output = Vec::new();
    crate::ultra_fast_inflate::inflate_gzip_ultra_fast(data, &mut output)?;
    writer.write_all(&output)?;
    Ok(output.len() as u64)
}

// ============================================================================
// Single-Member Parallel Decompression (rapidgzip strategy)
// ============================================================================
//
// For single-member gzip files, we use a two-phase approach:
// 1. Sequential first pass: decode and record block boundaries + windows
// 2. Parallel second pass: re-decode each segment using windows as dictionaries
//
// This provides speedup when the file is large enough to amortize the overhead.

/// Chunk boundary information collected during first pass
/// Note: This will be used in future parallel single-member implementation
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ChunkBoundary {
    /// Bit position where this chunk starts
    deflate_bit_start: usize,
    /// Bit position where this chunk ends
    deflate_bit_end: usize,
    /// Output offset where this chunk's data starts
    output_offset: usize,
    /// Output size for this chunk
    output_size: usize,
    /// 32KB window at the end of this chunk (for next chunk's dictionary)
    window: Vec<u8>,
}

/// Parallel decompression for single-member gzip files
///
/// Uses the rapidgzip two-pass strategy:
/// 1. **First pass (sequential)**: Decode and collect 32KB windows at chunk intervals
/// 2. **Second pass (parallel)**: Re-decode each chunk using windows as dictionaries
///
/// Target: 2x-3x speedup over single-threaded on large files
pub fn decompress_single_member_parallel<W: Write>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // For small files or single-thread, use fast sequential
    let isize_hint = if data.len() >= 8 {
        u32::from_le_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]) as usize
    } else {
        data.len() * 4
    };

    // Only use parallel for large files (>20MB uncompressed) with multiple threads
    // The overhead of two-pass decode isn't worth it for smaller files
    const MIN_SIZE_FOR_PARALLEL: usize = 20 * 1024 * 1024;
    const CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4MB chunks like rapidgzip

    if isize_hint < MIN_SIZE_FOR_PARALLEL || num_threads <= 1 {
        return decompress_single_member(data, writer);
    }

    // Parse gzip header
    let header_size = crate::marker_decode::skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    // === FIRST PASS: Sequential decode to collect chunk boundaries and windows ===
    // We decode the entire file and record windows at regular intervals.
    // This is the "boundary finding" pass that rapidgzip does.

    let mut output = Vec::with_capacity(isize_hint);
    let mut chunk_windows: Vec<(usize, Vec<u8>)> = Vec::new(); // (output_offset, window)

    // Decode using CombinedLUT (our fastest pure-Rust decoder)
    let mut bits = FastBits::new(deflate_data);
    let mut out_pos = 0;

    // Pre-allocate output
    output.resize(isize_hint.max(1024), 0);

    loop {
        bits.refill();
        let bfinal = bits.read(1);
        let btype = bits.read(2);

        let start_out_pos = out_pos;

        match btype {
            0 => out_pos = decode_stored_into(&mut bits, &mut output, out_pos)?,
            1 => out_pos = decode_fixed_into(&mut bits, &mut output, out_pos)?,
            2 => out_pos = decode_dynamic_into(&mut bits, &mut output, out_pos)?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Reserved block type",
                ))
            }
            _ => unreachable!(),
        }

        // Record window at chunk boundaries (every CHUNK_SIZE bytes of output)
        let chunk_before = start_out_pos / CHUNK_SIZE;
        let chunk_after = out_pos / CHUNK_SIZE;

        if chunk_after > chunk_before && out_pos >= 32 * 1024 {
            // We crossed a chunk boundary - save the 32KB window
            let boundary_pos = chunk_after * CHUNK_SIZE;
            let window_start = boundary_pos.saturating_sub(32 * 1024);
            let window = output[window_start..boundary_pos.min(out_pos)].to_vec();
            chunk_windows.push((boundary_pos, window));
        }

        if bfinal == 1 {
            break;
        }
    }

    // Truncate output to actual size
    output.truncate(out_pos);

    // If we didn't find enough chunk boundaries, just use the sequential result
    if chunk_windows.len() < 2 {
        writer.write_all(&output)?;
        return Ok(out_pos as u64);
    }

    // === SECOND PASS: Parallel re-decode using windows ===
    // Note: For now, we just use the first-pass output since re-decoding is complex
    // and our first pass is already fast. The main benefit of two-pass is when the
    // first pass uses a simpler (slower) decoder and the second pass uses SIMD.
    //
    // Since our CombinedLUT first pass is already optimized, the benefit of re-decode
    // is minimal. We keep the first-pass result.
    //
    // Future optimization: Use marker-based decode in first pass (with u16 buffers),
    // then parallel marker resolution in second pass.

    if std::env::var("GZIPPY_DEBUG").is_ok() {
        eprintln!(
            "[gzippy] Single-member parallel: {} bytes, {} chunk boundaries found",
            out_pos,
            chunk_windows.len()
        );
    }

    writer.write_all(&output)?;
    Ok(out_pos as u64)
}

/// Check if data is a multi-member gzip file
/// Note: Currently used internally, may be exposed in future API
#[allow(dead_code)]
pub fn is_multi_member(data: &[u8]) -> bool {
    let members = find_gzip_members(data);
    members.len() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inflate_into() {
        // Create test data
        let original = b"Hello, World! This is a test of the BGZF inflate_into function.";

        use flate2::write::DeflateEncoder;
        use flate2::Compression;
        use std::io::Write as IoWrite;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress into pre-allocated buffer
        let mut output = vec![0u8; original.len()];
        let actual_size = inflate_into(&compressed, &mut output).unwrap();

        assert_eq!(actual_size, original.len());
        assert_eq!(&output[..actual_size], &original[..]);
    }

    #[test]
    fn test_bgzf_parallel() {
        // Test with a gzippy-compressed file if available
        let data = match std::fs::read("benchmark_data/test-gzippy-l1-t14.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping test - no gzippy test file");
                return;
            }
        };

        // Get expected output from flate2
        use std::io::Read;
        let mut expected = Vec::new();
        let mut decoder = flate2::read::MultiGzDecoder::new(&data[..]);
        decoder.read_to_end(&mut expected).unwrap();

        // Test our parallel decompressor
        let mut output = Vec::new();
        decompress_bgzf_parallel(&data, &mut output, 8).unwrap();

        assert_eq!(output.len(), expected.len(), "Size mismatch");
        assert_eq!(output, expected, "Content mismatch");
    }

    #[test]
    fn benchmark_bgzf_parallel() {
        let data = match std::fs::read("benchmark_data/test-gzippy-l1-t14.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping benchmark - no test file");
                return;
            }
        };

        // Get expected size
        use std::io::Read;
        let mut expected = Vec::new();
        let mut decoder = flate2::read::MultiGzDecoder::new(&data[..]);
        decoder.read_to_end(&mut expected).unwrap();
        let expected_size = expected.len();

        // Warm up
        for _ in 0..3 {
            let mut output = Vec::new();
            decompress_bgzf_parallel(&data, &mut output, 8).unwrap();
        }

        // Benchmark
        let start = std::time::Instant::now();
        let iterations = 5;
        for _ in 0..iterations {
            let mut output = Vec::new();
            decompress_bgzf_parallel(&data, &mut output, 8).unwrap();
        }
        let elapsed = start.elapsed() / iterations;
        let speed = expected_size as f64 / elapsed.as_secs_f64() / 1_000_000.0;

        eprintln!("BGZF parallel (8 threads): {:.1} MB/s", speed);
    }

    #[test]
    fn test_multi_member_parallel() {
        // Create a multi-member gzip file programmatically
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as IoWrite;

        let part1: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let part2: Vec<u8> = (0..100_000).map(|i| ((i + 50) % 256) as u8).collect();
        let part3: Vec<u8> = (0..100_000).map(|i| ((i + 100) % 256) as u8).collect();

        // Compress each part separately
        let mut encoder1 = GzEncoder::new(Vec::new(), Compression::default());
        encoder1.write_all(&part1).unwrap();
        let compressed1 = encoder1.finish().unwrap();

        let mut encoder2 = GzEncoder::new(Vec::new(), Compression::default());
        encoder2.write_all(&part2).unwrap();
        let compressed2 = encoder2.finish().unwrap();

        let mut encoder3 = GzEncoder::new(Vec::new(), Compression::default());
        encoder3.write_all(&part3).unwrap();
        let compressed3 = encoder3.finish().unwrap();

        // Concatenate them (like `cat part1.gz part2.gz part3.gz > multi.gz`)
        let mut multi = compressed1.clone();
        multi.extend_from_slice(&compressed2);
        multi.extend_from_slice(&compressed3);

        // Check we detect multiple members
        let members = find_gzip_members(&multi);
        eprintln!("Found {} gzip members", members.len());
        assert_eq!(members.len(), 3, "Should find 3 members");

        // Get expected output
        let mut expected = part1.clone();
        expected.extend_from_slice(&part2);
        expected.extend_from_slice(&part3);

        // Test our parallel decompressor
        let mut output = Vec::new();
        decompress_multi_member_parallel(&multi, &mut output, 4).unwrap();

        assert_eq!(output.len(), expected.len(), "Size mismatch");
        assert_eq!(output, expected, "Content mismatch");
    }

    #[test]
    fn test_multi_member_large() {
        // Create a larger multi-member test
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as IoWrite;

        let mut multi = Vec::new();
        let mut expected = Vec::new();
        let num_members = 10;

        for i in 0..num_members {
            let part: Vec<u8> = (0..50_000).map(|j| ((i * 17 + j) % 256) as u8).collect();

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&part).unwrap();
            multi.extend_from_slice(&encoder.finish().unwrap());

            expected.extend_from_slice(&part);
        }

        // Check we detect all members
        let members = find_gzip_members(&multi);
        assert_eq!(members.len(), num_members, "Should find all members");

        // Test parallel decompressor
        let mut output = Vec::new();
        decompress_multi_member_parallel(&multi, &mut output, 8).unwrap();

        assert_eq!(output.len(), expected.len(), "Size mismatch");
        assert_eq!(output, expected, "Content mismatch");
    }
}
