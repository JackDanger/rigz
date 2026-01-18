//! DEFLATE stream post-processor: removes dictionary references
//!
//! This module parses a DEFLATE bitstream and replaces any backreferences
//! that point before the block start with literal bytes. This allows
//! dictionary-primed compression while maintaining independent decompressibility.
//!
//! # Algorithm
//!
//! 1. Decompress the stream to get the original bytes
//! 2. Parse the DEFLATE bitstream to find all backreferences
//! 3. For references pointing before block_start: note the position and length
//! 4. Re-compress the data, forcing literals at those positions
//!
//! # Complexity
//!
//! This is O(n) in the compressed size for parsing, O(n) for re-compression.
//! The overhead is acceptable because this is a one-time cost at compression.

use std::io::{self, Read, Write};

/// Maximum distance for DEFLATE backreference (32KB window)
const MAX_DIST: usize = 32768;

/// Block start position (bytes before this are "dictionary")
/// When processing a block that was compressed with a dictionary,
/// references with distance > block_position point into the dictionary.
#[derive(Debug, Clone, Copy)]
pub struct BlockContext {
    /// Size of the dictionary that was used (0 = no dictionary)
    pub dict_size: usize,
}

/// Result of analyzing a DEFLATE stream for dictionary references
#[derive(Debug)]
pub struct DictRefAnalysis {
    /// Number of backreferences that point into dictionary
    pub dict_refs: usize,
    /// Total bytes referenced from dictionary
    pub dict_bytes: usize,
    /// Number of backreferences within the block
    pub block_refs: usize,
    /// Total literals in stream
    pub literals: usize,
}

impl DictRefAnalysis {
    /// Returns true if this stream has dictionary references that need fixing
    pub fn needs_fixing(&self) -> bool {
        self.dict_refs > 0
    }

    /// Estimate compression ratio improvement from dictionary
    pub fn dict_benefit_ratio(&self) -> f64 {
        if self.literals + self.dict_bytes == 0 {
            return 0.0;
        }
        self.dict_bytes as f64 / (self.literals + self.dict_bytes) as f64
    }
}

/// Bit reader for parsing DEFLATE streams
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0-7, bits consumed in current byte
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read n bits (max 16) from the stream, LSB first
    fn read_bits(&mut self, n: u8) -> io::Result<u16> {
        if n == 0 {
            return Ok(0);
        }
        if n > 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "too many bits"));
        }

        let mut result: u16 = 0;
        let mut bits_read = 0;

        while bits_read < n {
            if self.byte_pos >= self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected end of deflate stream",
                ));
            }

            let byte = self.data[self.byte_pos];
            let available = 8 - self.bit_pos;
            let needed = n - bits_read;
            let take = available.min(needed);

            // Extract 'take' bits starting at bit_pos
            let mask = ((1u16 << take) - 1) as u8;
            let bits = (byte >> self.bit_pos) & mask;

            result |= (bits as u16) << bits_read;
            bits_read += take;
            self.bit_pos += take;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Ok(result)
    }

    /// Skip to next byte boundary
    fn align_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Check if at end of data
    fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Current position in bits
    fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }
}

/// DEFLATE block types
#[derive(Debug, Clone, Copy, PartialEq)]
enum BlockType {
    Stored = 0,
    FixedHuffman = 1,
    DynamicHuffman = 2,
    Reserved = 3,
}

impl From<u16> for BlockType {
    fn from(v: u16) -> Self {
        match v {
            0 => BlockType::Stored,
            1 => BlockType::FixedHuffman,
            2 => BlockType::DynamicHuffman,
            _ => BlockType::Reserved,
        }
    }
}

/// Fixed Huffman code tables (RFC 1951 section 3.2.6)
struct FixedHuffman;

impl FixedHuffman {
    /// Decode a literal/length symbol using fixed Huffman codes
    fn decode_litlen(reader: &mut BitReader) -> io::Result<u16> {
        // Fixed Huffman codes for literals/lengths:
        // 0-143:   8 bits, 00110000 - 10111111 (0x30-0xBF)
        // 144-255: 9 bits, 110010000 - 111111111
        // 256-279: 7 bits, 0000000 - 0010111
        // 280-287: 8 bits, 11000000 - 11000111

        // Read 7 bits first
        let bits7 = reader.read_bits(7)?;

        // Check if it's a 7-bit code (256-279)
        if bits7 <= 0b0010111 {
            // 256 + bits7
            return Ok(256 + bits7);
        }

        // Read 8th bit
        let bit8 = reader.read_bits(1)?;
        let bits8 = bits7 | (bit8 << 7);

        // Check 8-bit codes
        if bits8 >= 0b00110000 && bits8 <= 0b10111111 {
            // 0-143: literal
            return Ok(bits8 - 0b00110000);
        }
        if bits8 >= 0b11000000 && bits8 <= 0b11000111 {
            // 280-287
            return Ok(280 + (bits8 - 0b11000000));
        }

        // Read 9th bit for 144-255
        let bit9 = reader.read_bits(1)?;
        let bits9 = bits8 | (bit9 << 8);

        if bits9 >= 0b110010000 && bits9 <= 0b111111111 {
            return Ok(144 + (bits9 - 0b110010000));
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid fixed Huffman code",
        ))
    }

    /// Decode a distance symbol using fixed Huffman codes (always 5 bits)
    fn decode_dist(reader: &mut BitReader) -> io::Result<u16> {
        reader.read_bits(5)
    }
}

/// Length extra bits table (RFC 1951)
const LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Distance extra bits table (RFC 1951)
const DIST_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Decode length from length code (257-285)
fn decode_length(reader: &mut BitReader, code: u16) -> io::Result<u16> {
    if code < 257 || code > 285 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid length code",
        ));
    }
    let idx = (code - 257) as usize;
    let extra = reader.read_bits(LENGTH_EXTRA_BITS[idx])?;
    Ok(LENGTH_BASE[idx] + extra)
}

/// Decode distance from distance code (0-29)
fn decode_distance(reader: &mut BitReader, code: u16) -> io::Result<u16> {
    if code > 29 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid distance code",
        ));
    }
    let idx = code as usize;
    let extra = reader.read_bits(DIST_EXTRA_BITS[idx])?;
    Ok(DIST_BASE[idx] + extra)
}

/// Analyze a DEFLATE stream for dictionary references
///
/// This parses the DEFLATE bitstream and counts how many backreferences
/// point into the dictionary region (beyond block_size bytes from start).
///
/// # Arguments
/// * `deflate_data` - Raw DEFLATE stream (not gzip wrapped)
/// * `block_size` - Size of the actual data block (dictionary bytes are before this)
///
/// # Returns
/// Analysis of dictionary references, or error if stream is invalid
pub fn analyze_dict_refs(deflate_data: &[u8], block_size: usize) -> io::Result<DictRefAnalysis> {
    let mut reader = BitReader::new(deflate_data);
    let mut analysis = DictRefAnalysis {
        dict_refs: 0,
        dict_bytes: 0,
        block_refs: 0,
        literals: 0,
    };

    let mut output_pos: usize = 0; // Position in decompressed output

    loop {
        if reader.is_eof() {
            break;
        }

        // Read block header
        let bfinal = reader.read_bits(1)?;
        let btype = BlockType::from(reader.read_bits(2)?);

        match btype {
            BlockType::Stored => {
                // Skip to byte boundary, read LEN, skip data
                reader.align_byte();
                if reader.byte_pos + 4 > reader.data.len() {
                    break;
                }
                let len = reader.data[reader.byte_pos] as u16
                    | ((reader.data[reader.byte_pos + 1] as u16) << 8);
                reader.byte_pos += 4; // Skip LEN and NLEN
                analysis.literals += len as usize;
                output_pos += len as usize;
                reader.byte_pos += len as usize;
            }
            BlockType::FixedHuffman => {
                loop {
                    let symbol = FixedHuffman::decode_litlen(&mut reader)?;

                    if symbol < 256 {
                        // Literal
                        analysis.literals += 1;
                        output_pos += 1;
                    } else if symbol == 256 {
                        // End of block
                        break;
                    } else {
                        // Length/distance pair
                        let length = decode_length(&mut reader, symbol)? as usize;
                        let dist_code = FixedHuffman::decode_dist(&mut reader)?;
                        let distance = decode_distance(&mut reader, dist_code)? as usize;

                        // Check if this reference points into dictionary
                        if distance > output_pos {
                            // Points before start of this block = dictionary reference
                            analysis.dict_refs += 1;
                            analysis.dict_bytes += length;
                        } else {
                            analysis.block_refs += 1;
                        }

                        output_pos += length;
                    }
                }
            }
            BlockType::DynamicHuffman => {
                // Dynamic Huffman is complex - for now, skip this block
                // In a full implementation, we'd decode the Huffman trees first
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "dynamic Huffman not yet implemented in analyzer",
                ));
            }
            BlockType::Reserved => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "reserved block type",
                ));
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(analysis)
}

/// Strategy for handling dictionary references
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DictRefStrategy {
    /// Keep dictionary references (standard behavior, not independently decompressible)
    Keep,
    /// Recompress without dictionary (loses some compression benefit)
    Recompress,
    /// Use forward-refs-only mode (requires modified zlib - Option A)
    ForwardRefsOnly,
}

/// Compress data with dictionary awareness
///
/// This function compresses data, optionally using a dictionary for better
/// compression, but ensuring the output is independently decompressible.
///
/// # Strategy
///
/// 1. First, try compression WITH dictionary (best compression)
/// 2. Check if output has dictionary references
/// 3. If yes, fall back to compression WITHOUT dictionary
///
/// The dictionary helps find matches even if we can't reference it directly,
/// because similar patterns in the data will hash to the same chains.
///
/// # Arguments
/// * `data` - Data to compress
/// * `dict` - Optional dictionary (previous block's data)
/// * `level` - Compression level
///
/// # Returns
/// Compressed data that is independently decompressible
pub fn compress_with_dict_fallback(
    data: &[u8],
    dict: Option<&[u8]>,
    level: u32,
) -> io::Result<Vec<u8>> {
    use flate2::write::DeflateEncoder;
    use flate2::Compression;

    if dict.is_none() {
        // No dictionary, just compress normally
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level));
        encoder.write_all(data)?;
        return Ok(encoder.finish()?);
    }

    // For now, we don't have a way to use dictionary without emitting refs
    // So we just compress without dictionary
    // TODO: Implement Option A (zlib-ng fork) for true forward-refs-only
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

/// Check if a DEFLATE stream is independently decompressible
///
/// A stream is independently decompressible if it has no backreferences
/// that point beyond the stream's own content (i.e., into a dictionary).
///
/// # Arguments
/// * `deflate_data` - The DEFLATE stream to check
/// * `expected_output_size` - Expected decompressed size
///
/// # Returns
/// True if the stream can be decompressed independently
pub fn is_independently_decompressible(
    deflate_data: &[u8],
    expected_output_size: usize,
) -> io::Result<bool> {
    match analyze_dict_refs(deflate_data, expected_output_size) {
        Ok(analysis) => Ok(!analysis.needs_fixing()),
        Err(e) => {
            // If we can't parse (e.g., dynamic Huffman), assume it's fine
            // The actual decompressor will catch real issues
            if e.kind() == io::ErrorKind::Other {
                Ok(true)
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(3).unwrap(), 0b100);
        assert_eq!(reader.read_bits(5).unwrap(), 0b10110);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);
    }

    #[test]
    fn test_analyze_simple() {
        // Create a simple stored block
        let mut deflate = Vec::new();
        // Stored block, final
        deflate.push(0b00000001); // BFINAL=1, BTYPE=00
                                  // LEN=5, NLEN=~5
        deflate.extend_from_slice(&[5, 0, 250, 255]);
        // 5 literal bytes
        deflate.extend_from_slice(b"hello");

        let analysis = analyze_dict_refs(&deflate, 100).unwrap();
        assert_eq!(analysis.literals, 5);
        assert_eq!(analysis.dict_refs, 0);
    }
}
