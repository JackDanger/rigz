//! rapidgzip-Style Parallel Deflate Decoder
//!
//! This implements the core rapidgzip algorithm for parallel decompression:
//!
//! 1. **Speculative decoding** - Start from guessed positions, track unresolved refs
//! 2. **Bit-aligned probing** - Try all 8 bit offsets to find valid block starts  
//! 3. **Window propagation** - Forward 32KB windows through chunk chain
//! 4. **Result stitching** - Merge outputs with resolved back-references
//!
//! # Key Insight
//!
//! Deflate back-references can span up to 32KB. When we start decoding mid-stream:
//! - Some refs point to data we have (within the chunk)
//! - Some refs point to data we don't have (needs window from previous chunk)
//!
//! We decode everything we can and mark unresolved refs. Once we have the
//! window from the previous chunk, we resolve them.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::needless_range_loop)]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Window size (32KB)
const WINDOW_SIZE: usize = 32 * 1024;

/// Minimum chunk size for parallel processing
const MIN_CHUNK_SIZE: usize = 256 * 1024;

/// Maximum distance to search for block start
const MAX_BLOCK_SEARCH: usize = 32 * 1024;

/// An unresolved back-reference
#[derive(Clone, Debug)]
struct UnresolvedRef {
    /// Position in output where this ref starts
    output_pos: usize,
    /// Distance back (may be beyond available window)
    distance: usize,
    /// Length of the match
    length: usize,
    /// How many bytes into the window we need
    window_offset: usize,
}

/// Result of speculative chunk decoding
#[derive(Clone)]
struct ChunkResult {
    /// Chunk index
    index: usize,
    /// Starting bit offset that worked (0-7), None if failed
    valid_bit_offset: Option<u8>,
    /// Decoded output (may have holes for unresolved refs)
    output: Vec<u8>,
    /// Unresolved back-references
    unresolved: Vec<UnresolvedRef>,
    /// Final 32KB window from this chunk
    final_window: Vec<u8>,
    /// Whether decode succeeded
    success: bool,
    /// How many bytes were written before first unresolved ref
    bytes_before_unresolved: usize,
    /// Ending bit position (for validation)
    end_bit_pos: usize,
}

/// Bit reader for deflate streams
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn at_offset(data: &'a [u8], byte_offset: usize, bit_offset: u8) -> Self {
        Self {
            data,
            byte_pos: byte_offset,
            bit_pos: bit_offset,
        }
    }

    #[inline]
    fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    #[inline]
    fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    #[inline]
    fn read_bit(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    #[inline]
    fn read_bits(&mut self, count: u8) -> io::Result<u32> {
        let mut value = 0u32;
        for i in 0..count {
            value |= (self.read_bit()? as u32) << i;
        }
        Ok(value)
    }

    #[inline]
    fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }

    #[inline]
    fn read_byte(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }
        let b = self.data[self.byte_pos];
        self.byte_pos += 1;
        Ok(b)
    }

    #[inline]
    fn read_u16_le(&mut self) -> io::Result<u16> {
        let lo = self.read_byte()? as u16;
        let hi = self.read_byte()? as u16;
        Ok(lo | (hi << 8))
    }
}

/// Huffman table for decoding
struct HuffmanTable {
    /// Lookup table: code -> (symbol, length)
    table: Vec<(u16, u8)>,
    max_bits: u8,
}

impl HuffmanTable {
    fn from_lengths(lengths: &[u8]) -> io::Result<Self> {
        let max_bits = *lengths.iter().max().unwrap_or(&0);
        if max_bits == 0 || max_bits > 15 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid lengths",
            ));
        }

        // Count codes per length
        let mut bl_count = [0u32; 16];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Generate next_code array
        let mut code = 0u32;
        let mut next_code = [0u32; 16];
        for bits in 1..=max_bits as usize {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes
        let mut codes = vec![0u32; lengths.len()];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[sym] = next_code[len as usize];
                next_code[len as usize] += 1;
            }
        }

        // Build lookup table
        let table_size = 1 << max_bits;
        let mut table = vec![(0u16, 0u8); table_size];

        for (sym, (&code, &len)) in codes.iter().zip(lengths.iter()).enumerate() {
            if len > 0 {
                let reversed = reverse_bits(code, len);
                let fill = 1 << (max_bits - len);
                for i in 0..fill {
                    let idx = (reversed | (i << len)) as usize;
                    if idx < table.len() {
                        table[idx] = (sym as u16, len);
                    }
                }
            }
        }

        Ok(Self { table, max_bits })
    }

    #[inline]
    fn decode(&self, reader: &mut BitReader) -> io::Result<u16> {
        if reader.byte_pos + 2 >= reader.data.len() {
            return self.decode_slow(reader);
        }

        // Fast path: peek max_bits
        let mut bits = (reader.data[reader.byte_pos] >> reader.bit_pos) as u32;
        let mut available = 8 - reader.bit_pos;

        if available < self.max_bits && reader.byte_pos + 1 < reader.data.len() {
            bits |= (reader.data[reader.byte_pos + 1] as u32) << available;
            available += 8;
        }
        if available < self.max_bits && reader.byte_pos + 2 < reader.data.len() {
            bits |= (reader.data[reader.byte_pos + 2] as u32) << available;
        }

        let idx = (bits & ((1 << self.max_bits) - 1)) as usize;
        let (sym, len) = self.table[idx];

        if len == 0 {
            return self.decode_slow(reader);
        }

        // Consume bits
        reader.bit_pos += len;
        while reader.bit_pos >= 8 {
            reader.bit_pos -= 8;
            reader.byte_pos += 1;
        }

        Ok(sym)
    }

    fn decode_slow(&self, reader: &mut BitReader) -> io::Result<u16> {
        let mut code = 0u32;
        for len in 1..=self.max_bits {
            code = (code << 1) | reader.read_bit()? as u32;
            let reversed = reverse_bits(code, len);
            if reversed < self.table.len() as u32 {
                let (sym, code_len) = self.table[reversed as usize];
                if code_len == len {
                    return Ok(sym);
                }
            }
        }
        Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid code"))
    }
}

#[inline]
fn reverse_bits(value: u32, bits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Fixed Huffman tables
fn fixed_litlen_table() -> HuffmanTable {
    let mut lengths = vec![0u8; 288];
    for i in 0..144 {
        lengths[i] = 8;
    }
    for i in 144..256 {
        lengths[i] = 9;
    }
    for i in 256..280 {
        lengths[i] = 7;
    }
    for i in 280..288 {
        lengths[i] = 8;
    }
    HuffmanTable::from_lengths(&lengths).unwrap()
}

fn fixed_dist_table() -> HuffmanTable {
    HuffmanTable::from_lengths(&[5u8; 32]).unwrap()
}

/// Length/distance tables
static LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
static LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
static DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
static DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
static CODELEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Speculative decoder that tracks unresolved references
struct SpeculativeDecoder<'a> {
    reader: BitReader<'a>,
    output: Vec<u8>,
    window: Vec<u8>,
    window_pos: usize,
    total_output: usize,
    unresolved: Vec<UnresolvedRef>,
    /// Available window size (starts at 0 for speculative decode)
    available_window: usize,
}

impl<'a> SpeculativeDecoder<'a> {
    fn new(data: &'a [u8], bit_offset: u8) -> Self {
        Self {
            reader: BitReader::at_offset(data, 0, bit_offset),
            output: Vec::new(),
            window: vec![0u8; WINDOW_SIZE],
            window_pos: 0,
            total_output: 0,
            unresolved: Vec::new(),
            available_window: 0, // No window available initially
        }
    }

    fn with_window(data: &'a [u8], bit_offset: u8, window: &[u8]) -> Self {
        let mut w = vec![0u8; WINDOW_SIZE];
        let len = window.len().min(WINDOW_SIZE);
        w[..len].copy_from_slice(&window[..len]);

        Self {
            reader: BitReader::at_offset(data, 0, bit_offset),
            output: Vec::new(),
            window: w,
            window_pos: len % WINDOW_SIZE,
            total_output: 0,
            unresolved: Vec::new(),
            available_window: len, // Full window available
        }
    }

    fn decode(&mut self) -> io::Result<()> {
        loop {
            let bfinal = self.reader.read_bit()?;
            let btype = self.reader.read_bits(2)? as u8;

            match btype {
                0 => self.decode_stored()?,
                1 => self.decode_fixed()?,
                2 => self.decode_dynamic()?,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid block type",
                    ))
                }
            }

            if bfinal == 1 {
                break;
            }
        }
        Ok(())
    }

    fn decode_stored(&mut self) -> io::Result<()> {
        self.reader.align_to_byte();
        let len = self.reader.read_u16_le()?;
        let nlen = self.reader.read_u16_le()?;

        if len != !nlen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LEN/NLEN mismatch",
            ));
        }

        for _ in 0..len {
            let byte = self.reader.read_byte()?;
            self.output_byte(byte);
        }
        Ok(())
    }

    fn decode_fixed(&mut self) -> io::Result<()> {
        let litlen = fixed_litlen_table();
        let dist = fixed_dist_table();
        self.decode_huffman(&litlen, &dist)
    }

    fn decode_dynamic(&mut self) -> io::Result<()> {
        let hlit = self.reader.read_bits(5)? as usize + 257;
        let hdist = self.reader.read_bits(5)? as usize + 1;
        let hclen = self.reader.read_bits(4)? as usize + 4;

        let mut codelen_lengths = [0u8; 19];
        for i in 0..hclen {
            codelen_lengths[CODELEN_ORDER[i]] = self.reader.read_bits(3)? as u8;
        }

        let codelen_table = HuffmanTable::from_lengths(&codelen_lengths)?;
        let mut all_lengths = vec![0u8; hlit + hdist];
        let mut i = 0;

        while i < all_lengths.len() {
            let sym = codelen_table.decode(&mut self.reader)?;
            match sym {
                0..=15 => {
                    all_lengths[i] = sym as u8;
                    i += 1;
                }
                16 => {
                    if i == 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid"));
                    }
                    let count = self.reader.read_bits(2)? as usize + 3;
                    let prev = all_lengths[i - 1];
                    for _ in 0..count {
                        if i >= all_lengths.len() {
                            break;
                        }
                        all_lengths[i] = prev;
                        i += 1;
                    }
                }
                17 => {
                    i += self.reader.read_bits(3)? as usize + 3;
                }
                18 => {
                    i += self.reader.read_bits(7)? as usize + 11;
                }
                _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid")),
            }
        }

        let litlen = HuffmanTable::from_lengths(&all_lengths[..hlit])?;
        let dist = if hdist > 0 {
            HuffmanTable::from_lengths(&all_lengths[hlit..])?
        } else {
            fixed_dist_table()
        };

        self.decode_huffman(&litlen, &dist)
    }

    fn decode_huffman(&mut self, litlen: &HuffmanTable, dist: &HuffmanTable) -> io::Result<()> {
        loop {
            let sym = litlen.decode(&mut self.reader)?;

            if sym < 256 {
                self.output_byte(sym as u8);
            } else if sym == 256 {
                break;
            } else {
                let len_idx = (sym - 257) as usize;
                if len_idx >= LENGTH_BASE.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid length"));
                }

                let extra = LENGTH_EXTRA[len_idx];
                let length = LENGTH_BASE[len_idx] as usize
                    + if extra > 0 {
                        self.reader.read_bits(extra)? as usize
                    } else {
                        0
                    };

                let dist_sym = dist.decode(&mut self.reader)? as usize;
                if dist_sym >= DIST_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance",
                    ));
                }

                let dist_extra = DIST_EXTRA[dist_sym];
                let distance = DIST_BASE[dist_sym] as usize
                    + if dist_extra > 0 {
                        self.reader.read_bits(dist_extra)? as usize
                    } else {
                        0
                    };

                self.copy_match(distance, length)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn output_byte(&mut self, byte: u8) {
        self.output.push(byte);
        self.window[self.window_pos] = byte;
        self.window_pos = (self.window_pos + 1) & (WINDOW_SIZE - 1);
        self.total_output += 1;
    }

    fn copy_match(&mut self, distance: usize, length: usize) -> io::Result<()> {
        // Check if this reference can be resolved
        let available = self
            .total_output
            .min(WINDOW_SIZE)
            .min(self.available_window + self.total_output);

        if distance > available {
            // Unresolved reference - record it and output placeholder zeros
            let window_offset = distance - available;
            self.unresolved.push(UnresolvedRef {
                output_pos: self.output.len(),
                distance,
                length,
                window_offset,
            });

            // Output zeros as placeholders
            for _ in 0..length {
                self.output_byte(0);
            }
            return Ok(());
        }

        // Resolve from window
        let mut src = (self.window_pos + WINDOW_SIZE - distance) & (WINDOW_SIZE - 1);

        for _ in 0..length {
            let byte = self.window[src];
            self.output_byte(byte);
            src = (src + 1) & (WINDOW_SIZE - 1);
        }

        Ok(())
    }

    fn get_final_window(&self) -> Vec<u8> {
        // Return last 32KB of output
        if self.output.len() >= WINDOW_SIZE {
            self.output[self.output.len() - WINDOW_SIZE..].to_vec()
        } else {
            self.output.clone()
        }
    }

    fn resolve_refs(&mut self, previous_window: &[u8]) {
        for uref in &self.unresolved {
            // Calculate where in the previous window this data is
            let window_len = previous_window.len();
            if uref.window_offset > window_len {
                continue; // Can't resolve
            }

            let start_in_window = window_len - uref.window_offset;

            for i in 0..uref.length {
                let src_idx = (start_in_window + i) % window_len;
                if src_idx < previous_window.len() && uref.output_pos + i < self.output.len() {
                    self.output[uref.output_pos + i] = previous_window[src_idx];
                }
            }
        }
    }
}

/// Skip gzip header, return offset to deflate data
fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 || data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not gzip"));
    }

    let flags = data[3];
    let mut offset = 10;

    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + xlen;
    }
    if flags & 0x08 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }
    if flags & 0x10 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }
    if flags & 0x02 != 0 {
        offset += 2;
    }

    if offset > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
    }
    Ok(offset)
}

/// Main parallel decompressor using rapidgzip algorithm
pub struct RapidgzipDecoder {
    num_threads: usize,
}

impl RapidgzipDecoder {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    pub fn decompress<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let header_size = skip_gzip_header(data)?;
        let deflate_data = &data[header_size..data.len().saturating_sub(8)];

        if deflate_data.len() < MIN_CHUNK_SIZE * 2 || self.num_threads <= 1 {
            return self.decompress_sequential(deflate_data, writer);
        }

        // Calculate chunk boundaries
        let chunk_size = deflate_data.len() / self.num_threads;
        let chunk_size = chunk_size.clamp(MIN_CHUNK_SIZE, 4 * 1024 * 1024);
        let num_chunks = deflate_data.len().div_ceil(chunk_size);

        // Phase 1: Parallel speculative decode
        let results = self.parallel_speculative_decode(deflate_data, num_chunks, chunk_size);

        // Check if we got valid results
        let valid_first = results.first().map(|r| r.success).unwrap_or(false);
        if !valid_first {
            return self.decompress_sequential(deflate_data, writer);
        }

        // Phase 2: Window propagation and stitching
        let output = self.stitch_with_window_propagation(&results, deflate_data)?;

        writer.write_all(&output)?;
        writer.flush()?;
        Ok(output.len() as u64)
    }

    fn parallel_speculative_decode(
        &self,
        data: &[u8],
        num_chunks: usize,
        chunk_size: usize,
    ) -> Vec<ChunkResult> {
        let results: Vec<Mutex<Option<ChunkResult>>> =
            (0..num_chunks).map(|_| Mutex::new(None)).collect();
        let next_chunk = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            for _ in 0..self.num_threads.min(num_chunks) {
                let results_ref = &results;
                let next_ref = &next_chunk;

                scope.spawn(move || {
                    loop {
                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_chunks {
                            break;
                        }

                        let start = idx * chunk_size;
                        if start >= data.len() {
                            break;
                        }

                        let chunk_data = &data[start..];

                        // First chunk: known to start at bit 0
                        if idx == 0 {
                            let result = decode_chunk_speculative(chunk_data, idx, 0, true);
                            *results_ref[idx].lock().unwrap() = Some(result);
                            continue;
                        }

                        // Other chunks: find valid block start
                        let result = if let Some((block_offset, bit_offset)) =
                            find_block_start(data, start, MAX_BLOCK_SEARCH)
                        {
                            // Found a potential block start, try to decode from there
                            let adjusted_data = &data[block_offset..];
                            if let Some(r) = try_decode_at_bit(adjusted_data, idx, bit_offset) {
                                r
                            } else {
                                // Try all bit offsets at the found position
                                let mut best: Option<ChunkResult> = None;
                                for bo in 0..8 {
                                    if let Some(r) = try_decode_at_bit(adjusted_data, idx, bo) {
                                        if r.success {
                                            best = Some(r);
                                            break;
                                        }
                                    }
                                }
                                best.unwrap_or(ChunkResult {
                                    index: idx,
                                    valid_bit_offset: None,
                                    output: Vec::new(),
                                    unresolved: Vec::new(),
                                    final_window: Vec::new(),
                                    success: false,
                                    bytes_before_unresolved: 0,
                                    end_bit_pos: 0,
                                })
                            }
                        } else {
                            // No block start found, try all bit offsets at chunk start
                            let mut best: Option<ChunkResult> = None;
                            for bit_offset in 0..8 {
                                if let Some(r) = try_decode_at_bit(chunk_data, idx, bit_offset) {
                                    if r.success {
                                        best = Some(r);
                                        break;
                                    }
                                }
                            }
                            best.unwrap_or(ChunkResult {
                                index: idx,
                                valid_bit_offset: None,
                                output: Vec::new(),
                                unresolved: Vec::new(),
                                final_window: Vec::new(),
                                success: false,
                                bytes_before_unresolved: 0,
                                end_bit_pos: 0,
                            })
                        };

                        *results_ref[idx].lock().unwrap() = Some(result);
                    }
                });
            }
        });

        results
            .into_iter()
            .map(|m| {
                m.into_inner().unwrap().unwrap_or_else(|| ChunkResult {
                    index: 0,
                    valid_bit_offset: None,
                    output: Vec::new(),
                    unresolved: Vec::new(),
                    final_window: Vec::new(),
                    success: false,
                    bytes_before_unresolved: 0,
                    end_bit_pos: 0,
                })
            })
            .collect()
    }

    fn stitch_with_window_propagation(
        &self,
        results: &[ChunkResult],
        data: &[u8],
    ) -> io::Result<Vec<u8>> {
        // Simple case: all chunks succeeded with no unresolved refs
        let all_clean = results.iter().all(|r| r.success && r.unresolved.is_empty());

        if all_clean {
            let total_size: usize = results.iter().map(|r| r.output.len()).sum();
            let mut output = Vec::with_capacity(total_size);
            for result in results {
                output.extend_from_slice(&result.output);
            }
            return Ok(output);
        }

        // Complex case: need window propagation
        let mut output = Vec::new();
        let mut prev_window: Vec<u8> = Vec::new();

        for result in results {
            if !result.success {
                // Fall back to sequential from here
                output.clear();
                self.decompress_sequential(data, &mut output)?;
                return Ok(output);
            }

            if result.unresolved.is_empty() {
                output.extend_from_slice(&result.output);
            } else {
                // Re-decode this chunk with the window from previous
                let chunk_start = result.index * (data.len() / results.len());
                let chunk_data = &data[chunk_start..];
                let bit_offset = result.valid_bit_offset.unwrap_or(0);

                let mut decoder =
                    SpeculativeDecoder::with_window(chunk_data, bit_offset, &prev_window);
                decoder.decode()?;
                output.extend_from_slice(&decoder.output);
            }

            // Update window for next chunk
            if output.len() >= WINDOW_SIZE {
                prev_window = output[output.len() - WINDOW_SIZE..].to_vec();
            } else {
                prev_window = output.clone();
            }
        }

        Ok(output)
    }

    fn decompress_sequential<W: Write>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let mut decoder = SpeculativeDecoder::with_window(data, 0, &[]);
        decoder.available_window = WINDOW_SIZE; // Full window available
        decoder.decode()?;
        writer.write_all(&decoder.output)?;
        writer.flush()?;
        Ok(decoder.output.len() as u64)
    }
}

/// Decode a chunk speculatively
fn decode_chunk_speculative(
    data: &[u8],
    index: usize,
    bit_offset: u8,
    has_window: bool,
) -> ChunkResult {
    // Try ISA-L first if we have the feature enabled
    #[cfg(feature = "isal")]
    {
        if let Some(result) = decode_chunk_isal(data, index, bit_offset, has_window) {
            return result;
        }
    }

    // Fallback to custom decoder
    let mut decoder = if has_window {
        let mut d = SpeculativeDecoder::new(data, bit_offset);
        d.available_window = WINDOW_SIZE; // First chunk has full virtual window
        d
    } else {
        SpeculativeDecoder::new(data, bit_offset)
    };

    match decoder.decode() {
        Ok(()) => {
            let final_window = decoder.get_final_window();
            let end_pos = decoder.reader.bit_position();
            ChunkResult {
                index,
                valid_bit_offset: Some(bit_offset),
                output: decoder.output,
                unresolved: decoder.unresolved,
                final_window,
                success: true,
                bytes_before_unresolved: 0,
                end_bit_pos: end_pos,
            }
        }
        Err(_) => ChunkResult {
            index,
            valid_bit_offset: None,
            output: Vec::new(),
            unresolved: Vec::new(),
            final_window: Vec::new(),
            success: false,
            bytes_before_unresolved: 0,
            end_bit_pos: 0,
        },
    }
}

/// Try to decode a chunk using ISA-L for maximum speed
#[cfg(feature = "isal")]
fn decode_chunk_isal(
    data: &[u8],
    index: usize,
    bit_offset: u8,
    has_window: bool,
) -> Option<ChunkResult> {
    use crate::isal::IsalInflater;

    // ISA-L works best when starting at byte boundaries with raw deflate
    // For non-zero bit offsets, we need to adjust the data
    if bit_offset != 0 {
        // For non-byte-aligned starts, use our custom decoder
        return None;
    }

    // Try to decompress with ISA-L
    let mut inflater = IsalInflater::new().ok()?;

    // Estimate output size (typically 3-10x compression ratio)
    let estimated_output = data.len() * 5;
    let mut output = vec![0u8; estimated_output.max(64 * 1024)];
    let mut total_out = 0;

    // If this is the first chunk, we have a virtual window
    // Otherwise, we'll need a previous chunk's window (handled elsewhere)
    if !has_window && index > 0 {
        // Can't use ISA-L without window for mid-stream decompression
        return None;
    }

    // Try decompression
    loop {
        let result = inflater.decompress(data, &mut output[total_out..]);
        match result {
            Ok(n) => {
                total_out += n;
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WriteZero => {
                // Need more output space
                output.resize(output.len() * 2, 0);
            }
            Err(_) => {
                // ISA-L failed, fall back to custom decoder
                return None;
            }
        }
    }

    output.truncate(total_out);

    // Get final window (last 32KB)
    let final_window = if output.len() >= WINDOW_SIZE {
        output[output.len() - WINDOW_SIZE..].to_vec()
    } else {
        output.clone()
    };

    Some(ChunkResult {
        index,
        valid_bit_offset: Some(bit_offset),
        output,
        unresolved: Vec::new(), // ISA-L handles all references internally
        final_window,
        success: true,
        bytes_before_unresolved: 0,
        end_bit_pos: data.len() * 8,
    })
}

/// Try to decode at a specific bit offset
fn try_decode_at_bit(data: &[u8], index: usize, bit_offset: u8) -> Option<ChunkResult> {
    let mut decoder = SpeculativeDecoder::new(data, bit_offset);

    match decoder.decode() {
        Ok(()) => {
            let final_window = decoder.get_final_window();
            let end_pos = decoder.reader.bit_position();
            Some(ChunkResult {
                index,
                valid_bit_offset: Some(bit_offset),
                output: decoder.output,
                unresolved: decoder.unresolved,
                final_window,
                success: true,
                bytes_before_unresolved: 0,
                end_bit_pos: end_pos,
            })
        }
        Err(_) => None,
    }
}

/// Find a valid block start near a target position
/// Returns (byte_offset, bit_offset) if found
fn find_block_start(data: &[u8], target: usize, search_range: usize) -> Option<(usize, u8)> {
    use crate::block_finder::BlockFinder;

    let start_bit = target.saturating_sub(search_range / 2) * 8;
    let end_bit = ((target + search_range / 2).min(data.len().saturating_sub(5))) * 8;

    // Use the sophisticated block finder with LUT and precode validation
    let finder = BlockFinder::new(data);
    let blocks = finder.find_blocks(start_bit, end_bit);

    if let Some(block) = blocks.first() {
        let byte_offset = block.bit_offset / 8;
        let bit_offset = (block.bit_offset % 8) as u8;
        return Some((byte_offset, bit_offset));
    }

    // Fallback: Try stored blocks (BTYPE=00) with LEN/NLEN pattern
    let start = target.saturating_sub(search_range / 2);
    let end = (target + search_range / 2).min(data.len().saturating_sub(5));

    for offset in start..end {
        if offset + 4 >= data.len() {
            break;
        }

        let len = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let nlen = u16::from_le_bytes([data[offset + 2], data[offset + 3]]);

        if len == !nlen && len > 0 && len < 65535 {
            return Some((offset.saturating_sub(1), 0));
        }
    }

    None
}

/// Try to validate that a position is a valid block start
fn try_validate_block_start(data: &[u8], byte_offset: usize, bit_offset: u8) -> bool {
    if byte_offset + 20 >= data.len() {
        return false;
    }

    // Read BFINAL and BTYPE
    let bit_pos = byte_offset * 8 + bit_offset as usize;

    // Quick heuristic: try to decode a few symbols and see if it fails
    let mut decoder = SpeculativeDecoder::new(&data[byte_offset..], bit_offset);

    // Try to decode just one block header
    match decoder.reader.read_bit() {
        Ok(_bfinal) => {}
        Err(_) => return false,
    }

    match decoder.reader.read_bits(2) {
        Ok(btype) => {
            // BTYPE=3 is reserved/invalid
            if btype == 3 {
                return false;
            }

            // For stored blocks, validate LEN/NLEN
            if btype == 0 {
                decoder.reader.align_to_byte();
                if let (Ok(len), Ok(nlen)) =
                    (decoder.reader.read_u16_le(), decoder.reader.read_u16_le())
                {
                    return len == !nlen;
                }
                return false;
            }

            // For Huffman blocks, just assume valid for now
            true
        }
        Err(_) => false,
    }
}

/// Public API
pub fn decompress_rapidgzip<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    RapidgzipDecoder::new(num_threads).decompress(data, writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    #[test]
    fn test_sequential() {
        let original = b"Hello, World!";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_rapidgzip(&compressed, &mut output, 1).unwrap();
        assert_slices_eq!(&output, original);
    }

    #[test]
    fn test_parallel() {
        let original: Vec<u8> = (0..500_000).map(|i| (i % 256) as u8).collect();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_rapidgzip(&compressed, &mut output, 4).unwrap();
        assert_slices_eq!(output, original);
    }
}
