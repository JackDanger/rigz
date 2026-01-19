//! Custom Deflate Decoder with Block Boundary Tracking
//!
//! This is a full implementation of the DEFLATE algorithm (RFC 1951) that
//! tracks block boundaries during decompression. This enables parallel
//! decompression by:
//!
//! 1. Recording where each deflate block starts (bit offset)
//! 2. Recording the sliding window state at each boundary
//! 3. Allowing decompression to resume from any recorded boundary
//!
//! # Architecture
//!
//! The decoder consists of:
//! - `BitReader`: Bit-level stream reading
//! - `HuffmanDecoder`: Fast Huffman decoding with lookup tables
//! - `DeflateDecoder`: Main decoder with block tracking
//! - `BlockBoundary`: Recorded boundary with window state

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::len_zero)]

use std::io::{self, Write};

/// Maximum back-reference distance (32KB)
pub const WINDOW_SIZE: usize = 32 * 1024;

/// Maximum match length
pub const MAX_MATCH_LENGTH: usize = 258;

/// End of block symbol
const END_OF_BLOCK: u16 = 256;

/// Number of literal/length codes
const NUM_LITLEN_CODES: usize = 286;

/// Number of distance codes
const NUM_DIST_CODES: usize = 30;

/// Bit reader for deflate streams
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0-7, bit position within current byte
    bits_consumed: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            bits_consumed: 0,
        }
    }

    /// Create reader at a specific bit offset
    pub fn at_bit_offset(data: &'a [u8], bit_offset: usize) -> Self {
        Self {
            data,
            byte_pos: bit_offset / 8,
            bit_pos: (bit_offset % 8) as u8,
            bits_consumed: bit_offset as u64,
        }
    }

    /// Current bit position in stream
    #[inline]
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Bytes remaining in stream
    #[inline]
    pub fn bytes_remaining(&self) -> usize {
        self.data.len().saturating_sub(self.byte_pos)
    }

    /// Check if at end of data
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Read a single bit (LSB first per deflate spec)
    #[inline]
    pub fn read_bit(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of stream",
            ));
        }

        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        self.bits_consumed += 1;

        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit)
    }

    /// Read multiple bits (up to 25), LSB first
    #[inline]
    pub fn read_bits(&mut self, count: u8) -> io::Result<u32> {
        debug_assert!(count <= 25);

        // Fast path: can read from current position
        if count <= 16 && self.byte_pos + 2 < self.data.len() {
            let remaining_in_byte = 8 - self.bit_pos;
            if count <= remaining_in_byte {
                // All bits in current byte
                let mask = (1u32 << count) - 1;
                let value = ((self.data[self.byte_pos] >> self.bit_pos) as u32) & mask;
                self.bit_pos += count;
                self.bits_consumed += count as u64;
                if self.bit_pos >= 8 {
                    self.bit_pos -= 8;
                    self.byte_pos += 1;
                }
                return Ok(value);
            }
        }

        // Slow path: read bit by bit
        let mut value = 0u32;
        for i in 0..count {
            let bit = self.read_bit()?;
            value |= (bit as u32) << i;
        }
        Ok(value)
    }

    /// Read bits in reverse order (for Huffman codes)
    #[inline]
    pub fn read_bits_reversed(&mut self, count: u8) -> io::Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit()? as u32;
        }
        Ok(value)
    }

    /// Align to next byte boundary
    #[inline]
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bits_consumed += (8 - self.bit_pos) as u64;
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }

    /// Read a byte (must be byte-aligned)
    #[inline]
    pub fn read_byte(&mut self) -> io::Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "End of stream",
            ));
        }
        let byte = self.data[self.byte_pos];
        self.byte_pos += 1;
        self.bits_consumed += 8;
        Ok(byte)
    }

    /// Read a 16-bit little-endian value (byte-aligned)
    #[inline]
    pub fn read_u16_le(&mut self) -> io::Result<u16> {
        let lo = self.read_byte()? as u16;
        let hi = self.read_byte()? as u16;
        Ok(lo | (hi << 8))
    }
}

/// Huffman decoder with fast lookup table
pub struct HuffmanTable {
    /// Lookup table: indexed by first N bits, contains (symbol, code_length)
    lookup: Vec<(u16, u8)>,
    /// Maximum code length
    max_len: u8,
    /// Lookup table bits (typically 9)
    lookup_bits: u8,
}

impl HuffmanTable {
    /// Build a Huffman table from code lengths
    pub fn from_lengths(lengths: &[u8]) -> io::Result<Self> {
        let max_len = *lengths.iter().max().unwrap_or(&0);
        if max_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty code lengths",
            ));
        }
        if max_len > 15 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Code length > 15",
            ));
        }

        // Count codes of each length
        let mut bl_count = [0u32; 16];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Check for valid Huffman tree
        let mut code = 0u32;
        let mut next_code = [0u32; 16];
        for bits in 1..=max_len as usize {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes to symbols
        let mut codes = vec![0u32; lengths.len()];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = next_code[len as usize];
                next_code[len as usize] += 1;
            }
        }

        // Build lookup table
        let lookup_bits = max_len.min(9);
        let lookup_size = 1 << lookup_bits;
        let mut lookup = vec![(0u16, 0u8); lookup_size];

        for (symbol, (&code, &len)) in codes.iter().zip(lengths.iter()).enumerate() {
            if len > 0 && len <= lookup_bits {
                // Reverse the code for LSB-first reading
                let reversed = reverse_bits(code, len);
                let fill_count = 1 << (lookup_bits - len);
                for i in 0..fill_count {
                    let idx = (reversed | (i << len)) as usize;
                    if idx < lookup.len() {
                        lookup[idx] = (symbol as u16, len);
                    }
                }
            }
        }

        Ok(Self {
            lookup,
            max_len,
            lookup_bits,
        })
    }

    /// Decode a symbol from the bit stream
    #[inline]
    pub fn decode(&self, reader: &mut BitReader) -> io::Result<u16> {
        // Fast path: use lookup table
        if reader.bytes_remaining() >= 2 {
            let peek = peek_bits(reader, self.lookup_bits);
            let (symbol, len) = self.lookup[peek as usize];
            if len > 0 {
                // Consume the bits
                reader.read_bits(len)?;
                return Ok(symbol);
            }
        }

        // Slow path: decode bit by bit
        self.decode_slow(reader)
    }

    fn decode_slow(&self, reader: &mut BitReader) -> io::Result<u16> {
        let mut code = 0u32;
        for len in 1..=self.max_len {
            code = (code << 1) | reader.read_bit()? as u32;

            // Check if this code matches
            for (symbol, entry) in self.lookup.iter().enumerate() {
                if entry.1 == len {
                    let expected = reverse_bits(code, len);
                    if self.lookup.get(expected as usize).map(|e| e.0) == Some(symbol as u16) {
                        return Ok(symbol as u16);
                    }
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid Huffman code",
        ))
    }
}

/// Reverse bits in a value
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

/// Peek at bits without consuming them
#[inline]
fn peek_bits(reader: &BitReader, count: u8) -> u32 {
    if reader.byte_pos >= reader.data.len() {
        return 0;
    }

    let mut value = (reader.data[reader.byte_pos] >> reader.bit_pos) as u32;
    let mut bits_available = 8 - reader.bit_pos;

    if bits_available < count && reader.byte_pos + 1 < reader.data.len() {
        value |= (reader.data[reader.byte_pos + 1] as u32) << bits_available;
        bits_available += 8;
    }
    if bits_available < count && reader.byte_pos + 2 < reader.data.len() {
        value |= (reader.data[reader.byte_pos + 2] as u32) << bits_available;
    }

    value & ((1 << count) - 1)
}

/// Fixed Huffman tables (BTYPE=01)
pub fn fixed_litlen_table() -> HuffmanTable {
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

pub fn fixed_dist_table() -> HuffmanTable {
    let lengths = vec![5u8; 32];
    HuffmanTable::from_lengths(&lengths).unwrap()
}

/// Extra bits for length codes
static LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Base lengths for length codes
static LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits for distance codes
static DISTANCE_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Base distances for distance codes
static DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Code length alphabet order for dynamic Huffman
static CODELEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Recorded block boundary
#[derive(Clone)]
pub struct BlockBoundary {
    /// Bit offset where block starts in compressed stream
    pub bit_offset: usize,
    /// Byte offset in decompressed output
    pub output_offset: usize,
    /// Block type (0=stored, 1=fixed, 2=dynamic)
    pub block_type: u8,
    /// Is this the final block?
    pub is_final: bool,
    /// Last 32KB of output (window for LZ77 back-references)
    pub window: Vec<u8>,
    /// Window position (for circular buffer)
    pub window_pos: usize,
}

/// Sliding window for LZ77 decoding
pub struct SlidingWindow {
    buffer: Vec<u8>,
    pos: usize,
    total_output: usize,
}

impl SlidingWindow {
    pub fn new() -> Self {
        Self {
            buffer: vec![0u8; WINDOW_SIZE],
            pos: 0,
            total_output: 0,
        }
    }

    /// Initialize from a previous window state
    pub fn from_window(window: &[u8], window_pos: usize) -> Self {
        let mut buffer = vec![0u8; WINDOW_SIZE];
        let len = window.len().min(WINDOW_SIZE);
        buffer[..len].copy_from_slice(&window[..len]);
        Self {
            buffer,
            pos: window_pos % WINDOW_SIZE,
            total_output: len,
        }
    }

    /// Output a literal byte
    #[inline]
    pub fn output_byte<W: Write>(&mut self, byte: u8, writer: &mut W) -> io::Result<()> {
        self.buffer[self.pos] = byte;
        self.pos = (self.pos + 1) & (WINDOW_SIZE - 1);
        self.total_output += 1;
        writer.write_all(&[byte])
    }

    /// Copy from back-reference
    #[inline]
    pub fn copy_match<W: Write>(
        &mut self,
        distance: usize,
        length: usize,
        writer: &mut W,
    ) -> io::Result<()> {
        if distance > self.total_output.min(WINDOW_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid distance {} (max {})",
                    distance,
                    self.total_output.min(WINDOW_SIZE)
                ),
            ));
        }

        // Calculate source position in circular buffer
        let mut src = (self.pos + WINDOW_SIZE - distance) & (WINDOW_SIZE - 1);

        // Copy bytes (handle overlapping case)
        let mut output_buf = [0u8; MAX_MATCH_LENGTH];
        for i in 0..length {
            let byte = self.buffer[src];
            output_buf[i] = byte;
            self.buffer[self.pos] = byte;
            self.pos = (self.pos + 1) & (WINDOW_SIZE - 1);
            src = (src + 1) & (WINDOW_SIZE - 1);
        }

        self.total_output += length;
        writer.write_all(&output_buf[..length])
    }

    /// Get current window state for checkpoint
    pub fn get_window(&self) -> (Vec<u8>, usize) {
        (self.buffer.clone(), self.pos)
    }

    /// Total bytes output so far
    pub fn total_output(&self) -> usize {
        self.total_output
    }

    /// Output a byte without writing (for range decode)
    #[inline]
    pub fn output_byte_internal(&mut self, byte: u8) {
        self.buffer[self.pos] = byte;
        self.pos = (self.pos + 1) & (WINDOW_SIZE - 1);
        self.total_output += 1;
    }

    /// Copy match with range filtering
    #[inline]
    pub fn copy_match_range<W: Write>(
        &mut self,
        distance: usize,
        length: usize,
        writer: &mut W,
        start_output: usize,
        end_output: usize,
    ) -> io::Result<usize> {
        if distance > self.total_output.min(WINDOW_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid distance {} (max {})",
                    distance,
                    self.total_output.min(WINDOW_SIZE)
                ),
            ));
        }

        let mut src = (self.pos + WINDOW_SIZE - distance) & (WINDOW_SIZE - 1);
        let mut bytes_written = 0;

        for _ in 0..length {
            let byte = self.buffer[src];
            let pos = self.total_output;

            self.buffer[self.pos] = byte;
            self.pos = (self.pos + 1) & (WINDOW_SIZE - 1);
            self.total_output += 1;
            src = (src + 1) & (WINDOW_SIZE - 1);

            if pos >= start_output && pos < end_output {
                writer.write_all(&[byte])?;
                bytes_written += 1;
            }
        }

        Ok(bytes_written)
    }
}

impl Default for SlidingWindow {
    fn default() -> Self {
        Self::new()
    }
}

/// Main deflate decoder with block boundary tracking
pub struct DeflateDecoder<'a> {
    reader: BitReader<'a>,
    window: SlidingWindow,
    boundaries: Vec<BlockBoundary>,
    min_boundary_spacing: usize,
    last_boundary_output: usize,
}

impl<'a> DeflateDecoder<'a> {
    /// Create a new decoder
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
            window: SlidingWindow::new(),
            boundaries: Vec::new(),
            min_boundary_spacing: 256 * 1024, // Record boundary every 256KB
            last_boundary_output: 0,
        }
    }

    /// Create decoder starting at a specific bit offset with a window
    pub fn with_window(
        data: &'a [u8],
        bit_offset: usize,
        window: &[u8],
        window_pos: usize,
    ) -> Self {
        Self {
            reader: BitReader::at_bit_offset(data, bit_offset),
            window: SlidingWindow::from_window(window, window_pos),
            boundaries: Vec::new(),
            min_boundary_spacing: 256 * 1024,
            last_boundary_output: 0,
        }
    }

    /// Set boundary recording spacing
    pub fn set_boundary_spacing(&mut self, spacing: usize) {
        self.min_boundary_spacing = spacing;
    }

    /// Decode the deflate stream
    pub fn decode<W: Write>(&mut self, writer: &mut W) -> io::Result<usize> {
        // Record initial boundary
        self.record_boundary(0, false);

        loop {
            let bfinal = self.reader.read_bit()?;
            let btype = self.reader.read_bits(2)? as u8;

            let block_start_bit = self.reader.bit_position() - 3;
            let block_start_output = self.window.total_output();

            // Record boundary at block start
            self.record_boundary(btype, bfinal == 1);

            match btype {
                0 => self.decode_stored_block(writer)?,
                1 => self.decode_fixed_block(writer)?,
                2 => self.decode_dynamic_block(writer)?,
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

        Ok(self.window.total_output())
    }

    /// Record a block boundary
    fn record_boundary(&mut self, block_type: u8, is_final: bool) {
        let output_offset = self.window.total_output();

        // Only record if we've output enough since last boundary
        if output_offset >= self.last_boundary_output + self.min_boundary_spacing
            || self.boundaries.is_empty()
        {
            let (window, window_pos) = self.window.get_window();

            self.boundaries.push(BlockBoundary {
                bit_offset: self.reader.bit_position(),
                output_offset,
                block_type,
                is_final,
                window,
                window_pos,
            });

            self.last_boundary_output = output_offset;
        }
    }

    /// Decode a stored block (BTYPE=00)
    fn decode_stored_block<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
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
            self.window.output_byte(byte, writer)?;
        }

        Ok(())
    }

    /// Decode a block with fixed Huffman codes (BTYPE=01)
    fn decode_fixed_block<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        let litlen_table = fixed_litlen_table();
        let dist_table = fixed_dist_table();
        self.decode_huffman_block(writer, &litlen_table, &dist_table)
    }

    /// Decode a block with dynamic Huffman codes (BTYPE=10)
    fn decode_dynamic_block<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        // Read header
        let hlit = self.reader.read_bits(5)? as usize + 257;
        let hdist = self.reader.read_bits(5)? as usize + 1;
        let hclen = self.reader.read_bits(4)? as usize + 4;

        // Read code length code lengths
        let mut codelen_lengths = [0u8; 19];
        for i in 0..hclen {
            codelen_lengths[CODELEN_ORDER[i]] = self.reader.read_bits(3)? as u8;
        }

        // Build code length table
        let codelen_table = HuffmanTable::from_lengths(&codelen_lengths)?;

        // Read literal/length and distance code lengths
        let mut all_lengths = vec![0u8; hlit + hdist];
        let mut i = 0;

        while i < all_lengths.len() {
            let symbol = codelen_table.decode(&mut self.reader)?;

            match symbol {
                0..=15 => {
                    all_lengths[i] = symbol as u8;
                    i += 1;
                }
                16 => {
                    if i == 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid repeat"));
                    }
                    let count = self.reader.read_bits(2)? as usize + 3;
                    let prev = all_lengths[i - 1];
                    for _ in 0..count {
                        if i >= all_lengths.len() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Repeat overflow",
                            ));
                        }
                        all_lengths[i] = prev;
                        i += 1;
                    }
                }
                17 => {
                    let count = self.reader.read_bits(3)? as usize + 3;
                    i += count;
                }
                18 => {
                    let count = self.reader.read_bits(7)? as usize + 11;
                    i += count;
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid code length symbol",
                    ))
                }
            }
        }

        // Build tables
        let litlen_table = HuffmanTable::from_lengths(&all_lengths[..hlit])?;
        let dist_table = if hdist > 0 {
            HuffmanTable::from_lengths(&all_lengths[hlit..])?
        } else {
            fixed_dist_table()
        };

        self.decode_huffman_block(writer, &litlen_table, &dist_table)
    }

    /// Decode a Huffman-coded block
    fn decode_huffman_block<W: Write>(
        &mut self,
        writer: &mut W,
        litlen_table: &HuffmanTable,
        dist_table: &HuffmanTable,
    ) -> io::Result<()> {
        loop {
            let symbol = litlen_table.decode(&mut self.reader)?;

            if symbol < 256 {
                // Literal
                self.window.output_byte(symbol as u8, writer)?;
            } else if symbol == END_OF_BLOCK {
                // End of block
                break;
            } else {
                // Length/distance pair
                let length_code = (symbol - 257) as usize;
                if length_code >= LENGTH_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid length code",
                    ));
                }

                let extra_bits = LENGTH_EXTRA_BITS[length_code];
                let length = LENGTH_BASE[length_code] as usize
                    + if extra_bits > 0 {
                        self.reader.read_bits(extra_bits)? as usize
                    } else {
                        0
                    };

                let dist_code = dist_table.decode(&mut self.reader)? as usize;
                if dist_code >= DISTANCE_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance code",
                    ));
                }

                let dist_extra = DISTANCE_EXTRA_BITS[dist_code];
                let distance = DISTANCE_BASE[dist_code] as usize
                    + if dist_extra > 0 {
                        self.reader.read_bits(dist_extra)? as usize
                    } else {
                        0
                    };

                self.window.copy_match(distance, length, writer)?;
            }

            // Check if we should record a boundary
            let output = self.window.total_output();
            if output >= self.last_boundary_output + self.min_boundary_spacing {
                self.record_boundary(2, false);
            }
        }

        Ok(())
    }

    /// Get recorded boundaries
    pub fn boundaries(&self) -> &[BlockBoundary] {
        &self.boundaries
    }

    /// Current bit position
    pub fn bit_position(&self) -> usize {
        self.reader.bit_position()
    }

    /// Total output bytes
    pub fn total_output(&self) -> usize {
        self.window.total_output()
    }

    /// Get current window state
    pub fn get_window(&self) -> (Vec<u8>, usize) {
        self.window.get_window()
    }

    /// Decode a range of output bytes (for parallel decompression)
    ///
    /// This decodes the stream but only outputs bytes in the range
    /// [start_output, end_output). The window must be properly initialized.
    pub fn decode_range<W: Write>(
        &mut self,
        writer: &mut W,
        start_output: usize,
        end_output: usize,
    ) -> io::Result<usize> {
        let mut bytes_written = 0usize;

        loop {
            let bfinal = self.reader.read_bit()?;
            let btype = self.reader.read_bits(2)? as u8;

            match btype {
                0 => {
                    bytes_written +=
                        self.decode_stored_block_range(writer, start_output, end_output)?;
                }
                1 => {
                    bytes_written +=
                        self.decode_fixed_block_range(writer, start_output, end_output)?;
                }
                2 => {
                    bytes_written +=
                        self.decode_dynamic_block_range(writer, start_output, end_output)?;
                }
                3 => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Reserved block type",
                    ))
                }
                _ => unreachable!(),
            }

            // Stop if we've reached the end of our target range
            if self.window.total_output() >= end_output {
                break;
            }

            if bfinal == 1 {
                break;
            }
        }

        Ok(bytes_written)
    }

    /// Decode stored block with range filtering
    fn decode_stored_block_range<W: Write>(
        &mut self,
        writer: &mut W,
        start_output: usize,
        end_output: usize,
    ) -> io::Result<usize> {
        self.reader.align_to_byte();

        let len = self.reader.read_u16_le()?;
        let nlen = self.reader.read_u16_le()?;

        if len != !nlen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LEN/NLEN mismatch",
            ));
        }

        let mut bytes_written = 0;
        for _ in 0..len {
            let byte = self.reader.read_byte()?;
            let pos = self.window.total_output();
            self.window.output_byte_internal(byte);

            if pos >= start_output && pos < end_output {
                writer.write_all(&[byte])?;
                bytes_written += 1;
            }
        }

        Ok(bytes_written)
    }

    /// Decode fixed Huffman block with range filtering
    fn decode_fixed_block_range<W: Write>(
        &mut self,
        writer: &mut W,
        start_output: usize,
        end_output: usize,
    ) -> io::Result<usize> {
        let litlen_table = fixed_litlen_table();
        let dist_table = fixed_dist_table();
        self.decode_huffman_block_range(
            writer,
            &litlen_table,
            &dist_table,
            start_output,
            end_output,
        )
    }

    /// Decode dynamic Huffman block with range filtering
    fn decode_dynamic_block_range<W: Write>(
        &mut self,
        writer: &mut W,
        start_output: usize,
        end_output: usize,
    ) -> io::Result<usize> {
        // Read header
        let hlit = self.reader.read_bits(5)? as usize + 257;
        let hdist = self.reader.read_bits(5)? as usize + 1;
        let hclen = self.reader.read_bits(4)? as usize + 4;

        // Read code length code lengths
        let mut codelen_lengths = [0u8; 19];
        for i in 0..hclen {
            codelen_lengths[CODELEN_ORDER[i]] = self.reader.read_bits(3)? as u8;
        }

        // Build code length table
        let codelen_table = HuffmanTable::from_lengths(&codelen_lengths)?;

        // Read literal/length and distance code lengths
        let mut all_lengths = vec![0u8; hlit + hdist];
        let mut i = 0;

        while i < all_lengths.len() {
            let symbol = codelen_table.decode(&mut self.reader)?;

            match symbol {
                0..=15 => {
                    all_lengths[i] = symbol as u8;
                    i += 1;
                }
                16 => {
                    if i == 0 {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid repeat"));
                    }
                    let count = self.reader.read_bits(2)? as usize + 3;
                    let prev = all_lengths[i - 1];
                    for _ in 0..count {
                        if i >= all_lengths.len() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Repeat overflow",
                            ));
                        }
                        all_lengths[i] = prev;
                        i += 1;
                    }
                }
                17 => {
                    let count = self.reader.read_bits(3)? as usize + 3;
                    i += count;
                }
                18 => {
                    let count = self.reader.read_bits(7)? as usize + 11;
                    i += count;
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid code length symbol",
                    ))
                }
            }
        }

        // Build tables
        let litlen_table = HuffmanTable::from_lengths(&all_lengths[..hlit])?;
        let dist_table = if hdist > 0 {
            HuffmanTable::from_lengths(&all_lengths[hlit..])?
        } else {
            fixed_dist_table()
        };

        self.decode_huffman_block_range(
            writer,
            &litlen_table,
            &dist_table,
            start_output,
            end_output,
        )
    }

    /// Decode Huffman block with range filtering
    fn decode_huffman_block_range<W: Write>(
        &mut self,
        writer: &mut W,
        litlen_table: &HuffmanTable,
        dist_table: &HuffmanTable,
        start_output: usize,
        end_output: usize,
    ) -> io::Result<usize> {
        let mut bytes_written = 0;

        loop {
            let symbol = litlen_table.decode(&mut self.reader)?;

            if symbol < 256 {
                // Literal
                let byte = symbol as u8;
                let pos = self.window.total_output();
                self.window.output_byte_internal(byte);

                if pos >= start_output && pos < end_output {
                    writer.write_all(&[byte])?;
                    bytes_written += 1;
                }
            } else if symbol == END_OF_BLOCK {
                break;
            } else {
                // Length/distance pair
                let length_code = (symbol - 257) as usize;
                if length_code >= LENGTH_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid length code",
                    ));
                }

                let extra_bits = LENGTH_EXTRA_BITS[length_code];
                let length = LENGTH_BASE[length_code] as usize
                    + if extra_bits > 0 {
                        self.reader.read_bits(extra_bits)? as usize
                    } else {
                        0
                    };

                let dist_code = dist_table.decode(&mut self.reader)? as usize;
                if dist_code >= DISTANCE_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid distance code",
                    ));
                }

                let dist_extra = DISTANCE_EXTRA_BITS[dist_code];
                let distance = DISTANCE_BASE[dist_code] as usize
                    + if dist_extra > 0 {
                        self.reader.read_bits(dist_extra)? as usize
                    } else {
                        0
                    };

                let start_pos = self.window.total_output();
                bytes_written += self.window.copy_match_range(
                    distance,
                    length,
                    writer,
                    start_output,
                    end_output,
                )?;
            }

            // Stop if we've reached end of target range
            if self.window.total_output() >= end_output {
                break;
            }
        }

        Ok(bytes_written)
    }
}

/// Skip gzip header and return offset to deflate data
pub fn skip_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Too short"));
    }

    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not gzip"));
    }

    let flags = data[3];
    let mut offset = 10;

    // FEXTRA
    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Truncated header",
        ));
    }

    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
    }

    #[test]
    fn test_fixed_tables() {
        let litlen = fixed_litlen_table();
        let dist = fixed_dist_table();

        // Tables should be valid
        assert!(litlen.max_len > 0);
        assert!(dist.max_len > 0);
    }

    #[test]
    fn test_decode_simple() {
        // Create test data
        let original = b"Hello, World! Hello, World!";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Skip gzip header
        let header_size = skip_gzip_header(&compressed).unwrap();
        let deflate_data = &compressed[header_size..compressed.len() - 8];

        // Decode
        let mut decoder = DeflateDecoder::new(deflate_data);
        let mut output = Vec::new();
        decoder.decode(&mut output).unwrap();

        assert_eq!(&output, original);
    }

    #[test]
    fn test_decode_with_boundaries() {
        // Create larger test data
        let original: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let header_size = skip_gzip_header(&compressed).unwrap();
        let deflate_data = &compressed[header_size..compressed.len() - 8];

        let mut decoder = DeflateDecoder::new(deflate_data);
        decoder.set_boundary_spacing(10_000);
        let mut output = Vec::new();
        decoder.decode(&mut output).unwrap();

        assert_eq!(output, original);
        assert!(decoder.boundaries().len() >= 1);
    }
}
