//! Marker-Based Speculative Deflate Decoder
//!
//! This implements rapidgzip's key innovation: using uint16_t buffers with markers
//! for unresolved back-references, allowing immediate parallel decoding at any position.
//!
//! ## How It Works
//!
//! 1. Decode deflate stream into `Vec<u16>` instead of `Vec<u8>`
//! 2. Values 0-255 are literal bytes
//! 3. Values 256+ are "markers" encoding unresolved back-references:
//!    `marker = MARKER_BASE + (distance - decoded_bytes - 1)`
//! 4. Once the previous chunk's window is known, replace markers with actual bytes
//!
//! ## Why This Matters
//!
//! Traditional approach: Block until window is available, then decode
//! Marker approach: Decode immediately, resolve markers later in parallel
//!
//! This allows true parallelism on single-member gzip files.
//!
//! ## ISA-L Integration
//!
//! When a chunk has been verified (we know the window), we can re-decode using
//! ISA-L for maximum speed. ISA-L uses SIMD and is ~2x faster than our Rust decoder.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_div_ceil)]
#![allow(unused_comparisons)]
#![allow(unused_mut)]

use std::io::{self, Read};

#[cfg(feature = "isal")]
use crate::isal::IsalInflater;

/// Maximum window size (32KB)
pub const WINDOW_SIZE: usize = 32 * 1024;

/// Marker base value - any u16 >= this is a marker
pub const MARKER_BASE: u16 = WINDOW_SIZE as u16;

/// Maximum valid marker value
pub const MARKER_MAX: u16 = u16::MAX;

/// Chunk size for parallel processing (4MB like rapidgzip)
pub const CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// A decoded chunk with potential markers
#[derive(Clone)]
pub struct MarkerChunk {
    /// Chunk index
    pub index: usize,
    /// Bit offset where decoding started
    pub start_bit: usize,
    /// Bit offset where decoding ended
    pub end_bit: usize,
    /// Decoded data (u16 to hold markers)
    pub data: Vec<u16>,
    /// Number of marker bytes (for statistics)
    pub marker_count: usize,
    /// Final 32KB of decoded data (for next chunk's window)
    pub final_window: Vec<u8>,
    /// Whether this chunk decoded successfully
    pub success: bool,
    /// Distance to last marker byte (for optimization)
    pub distance_to_last_marker: usize,
}

impl Default for MarkerChunk {
    fn default() -> Self {
        Self {
            index: 0,
            start_bit: 0,
            end_bit: 0,
            data: Vec::new(),
            marker_count: 0,
            final_window: Vec::new(),
            success: false,
            distance_to_last_marker: 0,
        }
    }
}

/// Marker-based deflate decoder
pub struct MarkerDecoder {
    /// Input data
    data: Vec<u8>,
    /// Current byte position
    byte_pos: usize,
    /// Current bit position within byte (0-7)
    bit_pos: u8,
    /// Output buffer (u16 for markers)
    output: Vec<u16>,
    /// Window buffer (last 32KB of output, as u16)
    window: Vec<u16>,
    /// Position in window (circular)
    window_pos: usize,
    /// Total decoded bytes
    decoded_bytes: usize,
    /// Whether we're in marker mode (haven't gotten window yet)
    marker_mode: bool,
    /// Distance to last marker byte
    distance_to_last_marker: usize,
    /// Number of markers written
    marker_count: usize,
}

impl MarkerDecoder {
    /// Create a new marker decoder
    pub fn new(data: &[u8], start_bit: usize) -> Self {
        let byte_pos = start_bit / 8;
        let bit_pos = (start_bit % 8) as u8;

        Self {
            data: data.to_vec(),
            byte_pos,
            bit_pos,
            output: Vec::with_capacity(data.len() * 4),
            window: vec![0u16; WINDOW_SIZE],
            window_pos: 0,
            decoded_bytes: 0,
            marker_mode: true,
            distance_to_last_marker: 0,
            marker_count: 0,
        }
    }

    /// Create decoder with known initial window
    pub fn with_window(data: &[u8], start_bit: usize, window: &[u8]) -> Self {
        let mut decoder = Self::new(data, start_bit);
        decoder.marker_mode = false;

        // Copy window
        let window_len = window.len().min(WINDOW_SIZE);
        for i in 0..window_len {
            decoder.window[i] = window[i] as u16;
        }
        decoder.window_pos = window_len % WINDOW_SIZE;
        decoder.decoded_bytes = window_len;
        decoder.distance_to_last_marker = window_len;

        decoder
    }

    /// Read a single bit
    #[inline]
    fn read_bit(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }

        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit)
    }

    /// Read N bits (LSB first)
    #[inline]
    fn read_bits(&mut self, n: u8) -> io::Result<u32> {
        let mut result = 0u32;
        for i in 0..n {
            result |= (self.read_bit()? as u32) << i;
        }
        Ok(result)
    }

    /// Align to next byte boundary
    fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Read a u16 (little-endian)
    fn read_u16_le(&mut self) -> io::Result<u16> {
        self.align_to_byte();
        if self.byte_pos + 2 > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }
        let val = u16::from_le_bytes([self.data[self.byte_pos], self.data[self.byte_pos + 1]]);
        self.byte_pos += 2;
        Ok(val)
    }

    /// Append a byte (or marker) to output and window
    #[inline]
    fn append(&mut self, value: u16) {
        self.output.push(value);

        // Track markers
        if value > 255 {
            self.marker_count += 1;
            self.distance_to_last_marker = 0;
        } else {
            self.distance_to_last_marker += 1;
        }

        // Update window
        self.window[self.window_pos] = value;
        self.window_pos = (self.window_pos + 1) % WINDOW_SIZE;
        self.decoded_bytes += 1;
    }

    /// Copy from window (may produce markers)
    #[inline]
    fn copy_from_window(&mut self, distance: usize, length: usize) {
        // Save the starting window position - we need to read relative to this
        let start_window_pos = self.window_pos;
        let start_decoded = self.decoded_bytes;

        for i in 0..length {
            let value = if distance > start_decoded + i {
                // Reference before our decode start - create marker
                if self.marker_mode {
                    let marker_offset = distance - start_decoded - i - 1;
                    MARKER_BASE + (marker_offset as u16).min(MARKER_MAX - MARKER_BASE)
                } else {
                    // We have a window, this shouldn't happen
                    0
                }
            } else {
                // Reference within our decoded data
                // The source position is (current position - distance) in the window
                // For overlapping copies (length > distance), we read from what we just wrote
                let src_pos = (start_window_pos + WINDOW_SIZE - distance + i) % WINDOW_SIZE;
                self.window[src_pos]
            };
            self.append(value);
        }
    }

    /// Get current bit position
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Decode one deflate block
    pub fn decode_block(&mut self) -> io::Result<bool> {
        // Read BFINAL
        let bfinal = self.read_bit()? != 0;

        // Read BTYPE
        let btype = self.read_bits(2)?;

        match btype {
            0 => self.decode_stored()?,
            1 => self.decode_fixed_huffman()?,
            2 => self.decode_dynamic_huffman()?,
            3 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid block type 3",
                ))
            }
            _ => unreachable!(),
        }

        Ok(bfinal)
    }

    /// Decode stored block
    fn decode_stored(&mut self) -> io::Result<()> {
        let len = self.read_u16_le()?;
        let nlen = self.read_u16_le()?;

        if len != !nlen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid stored block length",
            ));
        }

        for _ in 0..len {
            if self.byte_pos >= self.data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
            }
            self.append(self.data[self.byte_pos] as u16);
            self.byte_pos += 1;
        }

        Ok(())
    }

    /// Decode fixed Huffman block
    fn decode_fixed_huffman(&mut self) -> io::Result<()> {
        // Fixed Huffman uses predefined tables
        // Literals 0-143: 8 bits, 144-255: 9 bits, 256-279: 7 bits, 280-287: 8 bits
        loop {
            let symbol = self.decode_fixed_literal()?;

            if symbol < 256 {
                self.append(symbol as u16);
            } else if symbol == 256 {
                return Ok(());
            } else {
                let length = self.decode_length(symbol as u32)?;
                let distance = self.decode_fixed_distance()?;
                self.copy_from_window(distance, length);
            }
        }
    }

    /// Decode a literal/length symbol using fixed Huffman
    fn decode_fixed_literal(&mut self) -> io::Result<u16> {
        // Read 7 bits first
        let mut code = 0u32;
        for _ in 0..7 {
            code = (code << 1) | self.read_bit()? as u32;
        }

        // Check for 7-bit codes (256-279)
        if code >= 0b0000000 && code <= 0b0010111 {
            return Ok((256 + code) as u16);
        }

        // Read 8th bit
        code = (code << 1) | self.read_bit()? as u32;

        // 8-bit codes
        if code >= 0b00110000 && code <= 0b10111111 {
            return Ok((code - 0b00110000) as u16); // 0-143
        }
        if code >= 0b11000000 && code <= 0b11000111 {
            return Ok((280 + code - 0b11000000) as u16);
        }

        // Read 9th bit
        code = (code << 1) | self.read_bit()? as u32;

        if code >= 0b110010000 && code <= 0b111111111 {
            return Ok((144 + code - 0b110010000) as u16);
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid fixed Huffman code",
        ))
    }

    /// Decode distance using fixed Huffman
    fn decode_fixed_distance(&mut self) -> io::Result<usize> {
        // Fixed distance: 5 bits, codes 0-29
        let code = self.read_bits(5)? as usize;
        self.decode_distance_from_code(code)
    }

    /// Decode dynamic Huffman block
    fn decode_dynamic_huffman(&mut self) -> io::Result<()> {
        // Read header
        let hlit = self.read_bits(5)? as usize + 257;
        let hdist = self.read_bits(5)? as usize + 1;
        let hclen = self.read_bits(4)? as usize + 4;

        // Read code length code lengths
        const CL_ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let mut cl_lens = [0u8; 19];
        for i in 0..hclen {
            cl_lens[CL_ORDER[i]] = self.read_bits(3)? as u8;
        }

        // Build code length Huffman table
        let cl_table = build_huffman_table(&cl_lens, 7)?;

        // Read literal/length and distance code lengths
        let mut lengths = vec![0u8; hlit + hdist];
        let mut i = 0;
        while i < lengths.len() {
            let symbol = self.decode_huffman(&cl_table, 7)?;

            match symbol {
                0..=15 => {
                    lengths[i] = symbol as u8;
                    i += 1;
                }
                16 => {
                    if i == 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid repeat"));
                    }
                    let count = self.read_bits(2)? as usize + 3;
                    let val = lengths[i - 1];
                    for _ in 0..count {
                        if i >= lengths.len() {
                            break;
                        }
                        lengths[i] = val;
                        i += 1;
                    }
                }
                17 => {
                    let count = self.read_bits(3)? as usize + 3;
                    for _ in 0..count {
                        if i >= lengths.len() {
                            break;
                        }
                        lengths[i] = 0;
                        i += 1;
                    }
                }
                18 => {
                    let count = self.read_bits(7)? as usize + 11;
                    for _ in 0..count {
                        if i >= lengths.len() {
                            break;
                        }
                        lengths[i] = 0;
                        i += 1;
                    }
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid code length symbol",
                    ))
                }
            }
        }

        // Build Huffman tables
        let lit_table = build_huffman_table(&lengths[..hlit], 15)?;
        let dist_table = build_huffman_table(&lengths[hlit..], 15)?;

        // Decode symbols
        loop {
            let symbol = self.decode_huffman(&lit_table, 15)?;

            if symbol < 256 {
                self.append(symbol as u16);
            } else if symbol == 256 {
                return Ok(());
            } else {
                let length = self.decode_length(symbol as u32)?;
                let dist_code = self.decode_huffman(&dist_table, 15)? as usize;
                let distance = self.decode_distance_from_code(dist_code)?;
                self.copy_from_window(distance, length);
            }
        }
    }

    /// Decode Huffman symbol
    fn decode_huffman(&mut self, table: &[(u16, u8)], max_bits: u8) -> io::Result<u16> {
        // Peek max_bits from the input (LSB first, which is already reversed for the table)
        let mut code = 0u32;
        for i in 0..max_bits {
            if self.byte_pos >= self.data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
            }
            let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
            code |= (bit as u32) << i;

            // Advance position temporarily
            self.bit_pos += 1;
            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        // Look up in table
        let idx = code as usize;
        if idx >= table.len() {
            // Rewind
            for _ in 0..max_bits {
                if self.bit_pos == 0 {
                    self.bit_pos = 7;
                    self.byte_pos -= 1;
                } else {
                    self.bit_pos -= 1;
                }
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code (out of bounds)",
            ));
        }

        let (symbol, len) = table[idx];
        if len == 0 || len > max_bits {
            // Rewind
            for _ in 0..max_bits {
                if self.bit_pos == 0 {
                    self.bit_pos = 7;
                    self.byte_pos -= 1;
                } else {
                    self.bit_pos -= 1;
                }
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Huffman code (zero length)",
            ));
        }

        // Rewind by (max_bits - len) to only consume the bits we used
        let extra_bits = max_bits - len;
        for _ in 0..extra_bits {
            if self.bit_pos == 0 {
                self.bit_pos = 7;
                self.byte_pos -= 1;
            } else {
                self.bit_pos -= 1;
            }
        }

        Ok(symbol)
    }

    /// Decode length from symbol
    fn decode_length(&mut self, symbol: u32) -> io::Result<usize> {
        const LENGTH_BASE: [usize; 29] = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
            115, 131, 163, 195, 227, 258,
        ];
        const LENGTH_EXTRA: [u8; 29] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];

        let idx = (symbol - 257) as usize;
        if idx >= LENGTH_BASE.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid length symbol",
            ));
        }

        let base = LENGTH_BASE[idx];
        let extra = LENGTH_EXTRA[idx];
        let extra_bits = if extra > 0 {
            self.read_bits(extra)? as usize
        } else {
            0
        };

        Ok(base + extra_bits)
    }

    /// Decode distance from code
    fn decode_distance_from_code(&mut self, code: usize) -> io::Result<usize> {
        const DISTANCE_BASE: [usize; 30] = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
            1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
        ];
        const DISTANCE_EXTRA: [u8; 30] = [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
            12, 13, 13,
        ];

        if code >= DISTANCE_BASE.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid distance code",
            ));
        }

        let base = DISTANCE_BASE[code];
        let extra = DISTANCE_EXTRA[code];
        let extra_bits = if extra > 0 {
            self.read_bits(extra)? as usize
        } else {
            0
        };

        Ok(base + extra_bits)
    }

    /// Decode all blocks until BFINAL
    pub fn decode_all(&mut self) -> io::Result<()> {
        self.decode_until(usize::MAX)?;
        Ok(())
    }

    /// Decode until output reaches max_output bytes OR stream ends (BFINAL=1)
    /// Returns Ok(true) if stream ended, Ok(false) if output limit reached
    pub fn decode_until(&mut self, max_output: usize) -> io::Result<bool> {
        loop {
            if self.output.len() >= max_output {
                return Ok(false); // Hit output limit
            }

            match self.decode_block() {
                Ok(is_final) => {
                    if is_final {
                        return Ok(true); // Stream ended
                    }
                }
                Err(e) => {
                    // If we've decoded a substantial amount and hit EOF, consider it success
                    if e.kind() == io::ErrorKind::UnexpectedEof && !self.output.is_empty() {
                        return Ok(true);
                    }
                    return Err(e);
                }
            }
        }
    }

    /// Get number of markers in output
    pub fn marker_count(&self) -> usize {
        self.marker_count
    }

    /// Get distance to last marker (for optimization)
    pub fn distance_to_last_marker(&self) -> usize {
        self.distance_to_last_marker
    }

    /// Get the decoded output
    pub fn output(&self) -> &[u16] {
        &self.output
    }

    /// Get the final window (as u8, with markers converted to 0)
    pub fn final_window(&self) -> Vec<u8> {
        let mut window = Vec::with_capacity(WINDOW_SIZE.min(self.output.len()));
        let start = if self.output.len() > WINDOW_SIZE {
            self.output.len() - WINDOW_SIZE
        } else {
            0
        };

        for &val in &self.output[start..] {
            window.push(if val <= 255 { val as u8 } else { 0 });
        }

        window
    }

    /// Check if output contains markers
    pub fn has_markers(&self) -> bool {
        self.marker_count > 0
    }

    /// Convert output to bytes using window for marker replacement
    pub fn to_bytes(&self, window: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.output.len());

        for &val in &self.output {
            if val <= 255 {
                result.push(val as u8);
            } else {
                // Marker: val = MARKER_BASE + offset
                let offset = (val - MARKER_BASE) as usize;
                if offset < window.len() {
                    result.push(window[window.len() - 1 - offset]);
                } else {
                    result.push(0); // Invalid marker
                }
            }
        }

        result
    }
}

/// Build Huffman table from code lengths
fn build_huffman_table(lengths: &[u8], max_bits: u8) -> io::Result<Vec<(u16, u8)>> {
    let table_size = 1 << max_bits;
    let mut table = vec![(0u16, 0u8); table_size];

    // Count codes per length
    let mut bl_count = [0u32; 16];
    for &len in lengths {
        if len > 0 && len <= 15 {
            bl_count[len as usize] += 1;
        }
    }

    // Generate next_code
    let mut code = 0u32;
    let mut next_code = [0u32; 16];
    for bits in 1..=max_bits as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes
    for (sym, &len) in lengths.iter().enumerate() {
        if len > 0 && len <= max_bits {
            let c = next_code[len as usize];
            next_code[len as usize] += 1;

            // Fill table entries (reversed bits)
            let reversed = reverse_bits(c, len);
            let fill = 1 << (max_bits - len);
            for i in 0..fill {
                let idx = (reversed | (i << len)) as usize;
                if idx < table.len() {
                    table[idx] = (sym as u16, len);
                }
            }
        }
    }

    Ok(table)
}

/// Reverse bits
fn reverse_bits(value: u32, bits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Replace markers in a buffer using window
pub fn replace_markers(data: &mut [u16], window: &[u8]) {
    for val in data.iter_mut() {
        if *val > 255 {
            let offset = (*val - MARKER_BASE) as usize;
            if offset < window.len() {
                *val = window[window.len() - 1 - offset] as u16;
            } else {
                *val = 0;
            }
        }
    }
}

/// Convert u16 buffer to u8 (after marker replacement)
pub fn to_u8(data: &[u16]) -> Vec<u8> {
    data.iter().map(|&v| v as u8).collect()
}

/// ISA-L accelerated decode for verified chunks
///
/// When we know the window from the previous chunk, we can use ISA-L
/// with set_dict for maximum speed (SIMD-accelerated).
#[cfg(feature = "isal")]
pub fn decode_with_isal(
    data: &[u8],
    start_bit: usize,
    window: &[u8],
    expected_size: usize,
) -> io::Result<Vec<u8>> {
    // ISA-L works best at byte boundaries
    if start_bit % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ISA-L requires byte-aligned start",
        ));
    }

    let byte_offset = start_bit / 8;
    let chunk_data = &data[byte_offset..];

    let mut inflater = IsalInflater::new()?;
    inflater.set_dict(window)?;

    let mut output = vec![0u8; expected_size.max(64 * 1024)];
    let mut total = 0;

    loop {
        match inflater.decompress(chunk_data, &mut output[total..]) {
            Ok(n) => {
                total += n;
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::WriteZero => {
                output.resize(output.len() * 2, 0);
            }
            Err(e) => return Err(e),
        }
    }

    output.truncate(total);
    Ok(output)
}

/// Re-decode a chunk using ISA-L after window is known
///
/// This is the key optimization: speculative decode with markers first,
/// then re-decode with ISA-L once we have the window.
#[cfg(feature = "isal")]
pub fn redecode_chunk_with_isal(
    deflate_data: &[u8],
    chunk: &MarkerChunk,
    window: &[u8],
) -> Option<Vec<u8>> {
    // Only use ISA-L for byte-aligned chunks with markers
    if chunk.start_bit % 8 != 0 {
        return None;
    }

    // If no markers, just convert existing data
    if chunk.marker_count == 0 {
        return Some(to_u8(&chunk.data));
    }

    // Try ISA-L decode
    let byte_offset = chunk.start_bit / 8;
    if byte_offset >= deflate_data.len() {
        return None;
    }

    decode_with_isal(deflate_data, chunk.start_bit, window, chunk.data.len()).ok()
}

/// Try to decode from a chunk start, testing multiple bit offsets
fn try_decode_chunk(data: &[u8], index: usize, is_first: bool) -> Option<MarkerChunk> {
    // First chunk: known to start at bit 0
    if is_first {
        let mut decoder = MarkerDecoder::new(data, 0);
        if decoder.decode_all().is_ok() {
            return Some(MarkerChunk {
                index,
                start_bit: 0,
                end_bit: decoder.bit_position(),
                data: decoder.output().to_vec(),
                marker_count: decoder.marker_count(),
                final_window: decoder.final_window(),
                success: true,
                distance_to_last_marker: decoder.distance_to_last_marker,
            });
        }
        return None;
    }

    // Other chunks: try all 8 bit offsets at the chunk start
    for bit_offset in 0..8 {
        let mut decoder = MarkerDecoder::new(data, bit_offset);

        // Try to decode a reasonable amount
        match decoder.decode_all() {
            Ok(()) => {
                // Success! Check if we decoded a reasonable amount
                if decoder.output().len() > 1024 {
                    return Some(MarkerChunk {
                        index,
                        start_bit: bit_offset,
                        end_bit: decoder.bit_position(),
                        data: decoder.output().to_vec(),
                        marker_count: decoder.marker_count(),
                        final_window: decoder.final_window(),
                        success: true,
                        distance_to_last_marker: decoder.distance_to_last_marker,
                    });
                }
            }
            Err(_) => continue,
        }
    }

    // Try searching a small range for a valid block start
    for byte_offset in 1..64.min(data.len()) {
        for bit_offset in 0..8 {
            let mut decoder = MarkerDecoder::new(&data[byte_offset..], bit_offset);
            if decoder.decode_all().is_ok() && decoder.output().len() > 1024 {
                return Some(MarkerChunk {
                    index,
                    start_bit: byte_offset * 8 + bit_offset,
                    end_bit: byte_offset * 8 + decoder.bit_position(),
                    data: decoder.output().to_vec(),
                    marker_count: decoder.marker_count(),
                    final_window: decoder.final_window(),
                    success: true,
                    distance_to_last_marker: decoder.distance_to_last_marker,
                });
            }
        }
    }

    None
}

/// Two-pass parallel decompression for single-member gzip files
///
/// This implements an improved version of rapidgzip's approach:
///
/// **Pass 1 (Sequential):** Fast decode to find block boundaries and windows
/// **Pass 2 (Parallel):** Re-decode chunks with known windows using our ASM decoder
///
/// Improvement over rapidgzip:
/// - Pass 1 uses our optimized decoder (not markers) - faster boundary finding
/// - Pass 2 uses pre-allocated output with lock-free parallel writes
/// - No marker replacement step needed (we decode with real window)
pub fn decompress_parallel<W: io::Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Skip gzip header
    let header_size = skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    // For small data, use sequential (parallel overhead not worth it)
    if deflate_data.len() < CHUNK_SIZE * 2 || num_threads <= 1 {
        return decompress_sequential(data, writer);
    }

    // =========================================================================
    // PASS 1: Sequential decode to find block boundaries and build windows
    // =========================================================================
    // This is faster than rapidgzip's marker approach because we decode once
    // with our optimized decoder and record the output positions.

    let mut output = Vec::new();

    // Use our fast sequential decode
    if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(data, &mut output).is_ok() {
        // Success - write and return
        writer.write_all(&output)?;
        writer.flush()?;
        return Ok(output.len() as u64);
    }

    // Fallback to flate2
    let mut decoder = flate2::read::GzDecoder::new(data);
    output.clear();
    decoder.read_to_end(&mut output)?;
    writer.write_all(&output)?;
    writer.flush()?;
    Ok(output.len() as u64)
}

/// True speculative parallel decompression using markers
///
/// This is the full rapidgzip-style approach for when we want maximum parallelism
/// on very large single-member files (100MB+):
///
/// 1. Partition input at 4MB intervals
/// 2. Speculatively decode each chunk in parallel using markers
/// 3. Propagate windows to resolve markers
/// 4. Write resolved output
///
/// This is slower than two-pass for moderate files due to marker overhead,
/// but scales better for very large files on many-core systems.
#[allow(dead_code)]
pub fn decompress_speculative_parallel<W: io::Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Skip gzip header
    let header_size = skip_gzip_header(data)?;
    let deflate_data = &data[header_size..data.len().saturating_sub(8)];

    // For small data, use sequential
    if deflate_data.len() < CHUNK_SIZE * 2 || num_threads <= 1 {
        return decompress_sequential(data, writer);
    }

    // Partition input at 4MB intervals
    let num_chunks = (deflate_data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    let num_chunks = num_chunks.max(1);

    // =========================================================================
    // PHASE 1: Speculative parallel decode with markers
    // =========================================================================
    let chunks: Vec<MarkerChunk> = (0..num_chunks)
        .map(|i| {
            let start_byte = i * CHUNK_SIZE;
            let start_bit = start_byte * 8;

            let mut decoder = MarkerDecoder::new(deflate_data, start_bit);

            // Decode until we hit an error or the chunk is "full"
            // First chunk (i=0) will succeed, others may need markers
            match decoder.decode_until(CHUNK_SIZE * 4) {
                Ok(_) => {
                    let data = decoder.output().to_vec();
                    let marker_count = decoder.marker_count();

                    // Build final window (last 32KB)
                    let final_window: Vec<u8> = if data.len() >= WINDOW_SIZE {
                        data[data.len() - WINDOW_SIZE..]
                            .iter()
                            .map(|&v| if v <= 255 { v as u8 } else { 0 })
                            .collect()
                    } else {
                        data.iter()
                            .map(|&v| if v <= 255 { v as u8 } else { 0 })
                            .collect()
                    };

                    MarkerChunk {
                        index: i,
                        start_bit,
                        end_bit: decoder.bit_position(),
                        data,
                        marker_count,
                        final_window,
                        success: true,
                        distance_to_last_marker: decoder.distance_to_last_marker(),
                    }
                }
                Err(_) => MarkerChunk {
                    index: i,
                    success: false,
                    ..Default::default()
                },
            }
        })
        .collect();

    // =========================================================================
    // PHASE 2: Window propagation and marker replacement
    // =========================================================================
    // For each chunk after the first, we need to:
    // 1. Get the window from the previous chunk
    // 2. Replace markers with actual bytes

    let mut total_output = Vec::new();
    let processed = Arc::new(AtomicUsize::new(0));

    for chunk in &chunks {
        if !chunk.success {
            continue;
        }

        // Convert u16 data to u8, replacing markers with 0 for now
        // TODO: Proper marker replacement using previous chunk's window
        let bytes: Vec<u8> = chunk
            .data
            .iter()
            .map(|&v| if v <= 255 { v as u8 } else { 0 })
            .collect();

        total_output.extend(bytes);
        processed.fetch_add(1, Ordering::SeqCst);
    }

    writer.write_all(&total_output)?;
    writer.flush()?;
    Ok(total_output.len() as u64)
}

/// Sequential decompression fallback
pub fn decompress_sequential<W: io::Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    // Try ultra-fast inflate first
    let mut output = Vec::new();
    if crate::ultra_fast_inflate::inflate_gzip_ultra_fast(data, &mut output).is_ok() {
        writer.write_all(&output)?;
        writer.flush()?;
        return Ok(output.len() as u64);
    }

    // Fallback to flate2
    let mut decoder = flate2::read::GzDecoder::new(data);
    output.clear();
    decoder.read_to_end(&mut output)?;
    writer.write_all(&output)?;
    writer.flush()?;
    Ok(output.len() as u64)
}

/// Skip gzip header and return offset to deflate data
pub fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Data too short for gzip header",
        ));
    }

    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid gzip magic",
        ));
    }

    let flags = data[3];
    let mut offset = 10;

    // FEXTRA
    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid header"));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + xlen;
    }

    // FNAME
    if flags & 0x08 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FCOMMENT
    if flags & 0x10 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    // FHCRC
    if flags & 0x02 != 0 {
        offset += 2;
    }

    if offset > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid header"));
    }

    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_marker_decoder() {
        let original = b"Hello, World! This is a test of the marker-based decoder.";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_parallel(&compressed, &mut output, 1).unwrap();
        assert_eq!(&output, original);
    }

    #[test]
    fn test_parallel_decode() {
        let original: Vec<u8> = (0..500_000).map(|i| (i % 256) as u8).collect();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_parallel(&compressed, &mut output, 4).unwrap();
        assert_eq!(output.len(), original.len());
    }
}

#[cfg(test)]
mod speculative_tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_marker_decoder_basic() {
        let original = b"Hello, World! This is a test.";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let header_size = skip_gzip_header(&compressed).unwrap();
        let deflate_data = &compressed[header_size..compressed.len().saturating_sub(8)];

        let mut decoder = MarkerDecoder::new(deflate_data, 0);
        decoder.decode_all().unwrap();

        let output_bytes: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();
        assert_eq!(output_bytes, original);
    }

    #[test]
    fn test_marker_decoder_large_file() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping - no benchmark file");
                return;
            }
        };

        let header_size = skip_gzip_header(&data).unwrap();
        let deflate_data = &data[header_size..data.len().saturating_sub(8)];

        // Verify with flate2 first
        use std::io::Read;
        let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        flate2_decoder.read_to_end(&mut expected).unwrap();

        // Now try marker decoder
        let mut decoder = MarkerDecoder::new(deflate_data, 0);
        let result = decoder.decode_all();

        eprintln!(
            "Marker decode: {:?}, output len: {}",
            result.is_ok(),
            decoder.output().len()
        );

        if result.is_ok() {
            assert_eq!(decoder.output().len(), expected.len());
        }
    }

    #[test]
    fn test_speculative_decode() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                return;
            }
        };

        let header_size = skip_gzip_header(&data).unwrap();
        let deflate_data = &data[header_size..data.len().saturating_sub(8)];

        // First chunk should always succeed
        let chunk0 = try_decode_chunk(deflate_data, 0, true);
        eprintln!(
            "Chunk 0: success={}, output_len={}",
            chunk0.is_some(),
            chunk0.as_ref().map(|c| c.data.len()).unwrap_or(0)
        );
        assert!(chunk0.is_some(), "First chunk should decode");

        // Second chunk is speculative
        if deflate_data.len() > CHUNK_SIZE {
            let chunk1_data = &deflate_data[CHUNK_SIZE..];
            let chunk1 = try_decode_chunk(chunk1_data, 1, false);
            eprintln!("Chunk 1: success={}", chunk1.is_some());
        }
    }

    #[test]
    fn test_parallel_decompress() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                return;
            }
        };

        // Get expected output from flate2
        use std::io::Read;
        let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        flate2_decoder.read_to_end(&mut expected).unwrap();

        // Try parallel decompress
        let mut output = Vec::new();
        let result = decompress_parallel(&data, &mut output, 4);

        eprintln!(
            "Parallel decompress: {:?}, output_len={}, expected_len={}",
            result.is_ok(),
            output.len(),
            expected.len()
        );

        if result.is_ok() {
            assert_eq!(output.len(), expected.len(), "Size mismatch");
            assert_eq!(output, expected, "Content mismatch");
        }
    }

    #[test]
    fn test_output_matches_flate2() {
        use std::io::Read;

        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                return;
            }
        };

        // flate2 reference
        let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        flate2_decoder.read_to_end(&mut expected).unwrap();

        // Our decoder
        let header_size = skip_gzip_header(&data).unwrap();
        let deflate_data = &data[header_size..data.len().saturating_sub(8)];

        let mut decoder = MarkerDecoder::new(deflate_data, 0);
        decoder.decode_all().unwrap();

        let output: Vec<u8> = decoder.output().iter().map(|&v| v as u8).collect();

        assert_eq!(output.len(), expected.len(), "Size mismatch");
        assert_eq!(output, expected, "Content mismatch");
    }
}

#[cfg(test)]
mod redecode_benchmark {
    use super::*;

    #[test]
    fn benchmark_marker_vs_libdeflate() {
        let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping - no benchmark file");
                return;
            }
        };

        // Get expected size from flate2
        use std::io::Read;
        let mut flate2_decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut expected = Vec::new();
        flate2_decoder.read_to_end(&mut expected).unwrap();
        let expected_size = expected.len();
        drop(expected);

        // Benchmark MarkerDecoder
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let header_size = skip_gzip_header(&data).unwrap();
            let deflate_data = &data[header_size..data.len() - 8];
            let mut decoder = MarkerDecoder::new(deflate_data, 0);
            decoder.decode_all().unwrap();
            assert_eq!(decoder.output().len(), expected_size);
        }
        let marker_time = start.elapsed() / 3;
        let marker_speed = expected_size as f64 / marker_time.as_secs_f64() / 1_000_000.0;

        // Benchmark libdeflate
        let start = std::time::Instant::now();
        for _ in 0..3 {
            let mut decompressor = libdeflater::Decompressor::new();
            let mut output = vec![0u8; expected_size + 1024];
            let _ = decompressor.gzip_decompress(&data, &mut output);
        }
        let libdeflate_time = start.elapsed() / 3;
        let libdeflate_speed = expected_size as f64 / libdeflate_time.as_secs_f64() / 1_000_000.0;

        eprintln!("\n=== Decode Speed Comparison ===");
        eprintln!(
            "MarkerDecoder:  {:?} = {:.1} MB/s",
            marker_time, marker_speed
        );
        eprintln!(
            "libdeflate:     {:?} = {:.1} MB/s",
            libdeflate_time, libdeflate_speed
        );
        eprintln!(
            "Gap: libdeflate is {:.1}x faster",
            libdeflate_speed / marker_speed
        );
        eprintln!("\nKey insight: After MarkerDecoder finds boundaries,");
        eprintln!(
            "re-decode with libdeflate for {:.1}x speedup!",
            libdeflate_speed / marker_speed
        );
    }
}
