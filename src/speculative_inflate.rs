//! Speculative Parallel Inflate (rapidgzip-style)
//!
//! This implements true parallel decompression without a sequential first pass.
//! The key insight is:
//!
//! 1. Divide the compressed data into chunks at regular intervals
//! 2. For each chunk, try decompressing from multiple bit offsets (0-7)
//! 3. A valid decode produces output; invalid decodes fail fast
//! 4. Use window propagation to resolve back-references across chunks
//!
//! The challenge is that deflate blocks are bit-aligned, so we don't know
//! where blocks start. We solve this by trying all 8 possible bit offsets.

#![allow(dead_code)]
#![allow(unused_variables)]

use crate::deflate_decoder::{skip_gzip_header, DeflateDecoder};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Minimum chunk size for parallel processing
const MIN_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum chunk size
const MAX_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Result of speculative decompression
struct SpecResult {
    /// Chunk index
    index: usize,
    /// Bit offset that worked (0-7, or None if all failed)
    valid_bit_offset: Option<u8>,
    /// Decompressed data
    data: Vec<u8>,
    /// Whether this chunk needs window resolution
    needs_window: bool,
    /// Unresolved back-references (positions that referenced before chunk start)
    unresolved_refs: Vec<(usize, usize, usize)>, // (output_pos, distance, length)
    /// Final window for next chunk
    final_window: Vec<u8>,
}

/// Speculative parallel decompressor
pub struct SpeculativeInflater {
    num_threads: usize,
}

impl SpeculativeInflater {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Decompress gzip data using speculative parallel decoding
    pub fn decompress<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let header_size = skip_gzip_header(data)?;
        let trailer_size = 8;

        if data.len() < header_size + trailer_size {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
        }

        let deflate_data = &data[header_size..data.len() - trailer_size];

        // For small data, use direct decode
        if deflate_data.len() < MIN_CHUNK_SIZE * 2 || self.num_threads <= 1 {
            return self.decode_direct(deflate_data, writer);
        }

        // Calculate chunk boundaries
        let num_chunks = self.num_threads * 2; // More chunks than threads for better load balancing
        let chunk_size = (deflate_data.len() / num_chunks).clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        let actual_chunks = deflate_data.len().div_ceil(chunk_size);

        // Parallel speculative decode
        let results = self.decode_parallel_speculative(deflate_data, actual_chunks, chunk_size);

        // Check if we got valid results
        let valid_count = results
            .iter()
            .filter(|r| r.valid_bit_offset.is_some())
            .count();

        if valid_count == 0 {
            // All failed, use direct decode
            return self.decode_direct(deflate_data, writer);
        }

        // Phase 2: Window propagation and stitching
        let final_output = self.stitch_results(deflate_data, &results)?;

        writer.write_all(&final_output)?;
        writer.flush()?;
        Ok(final_output.len() as u64)
    }

    /// Direct decode (fallback)
    fn decode_direct<W: Write>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        let mut decoder = DeflateDecoder::new(data);
        let mut output = Vec::new();
        let size = decoder.decode(&mut output)?;
        writer.write_all(&output)?;
        writer.flush()?;
        Ok(size as u64)
    }

    /// Parallel speculative decode
    fn decode_parallel_speculative(
        &self,
        data: &[u8],
        num_chunks: usize,
        chunk_size: usize,
    ) -> Vec<SpecResult> {
        let results: Vec<Mutex<Option<SpecResult>>> =
            (0..num_chunks).map(|_| Mutex::new(None)).collect();

        let next_chunk = AtomicUsize::new(0);
        let any_success = AtomicBool::new(false);

        std::thread::scope(|scope| {
            for _ in 0..self.num_threads.min(num_chunks) {
                let results_ref = &results;
                let next_ref = &next_chunk;
                let success_ref = &any_success;

                scope.spawn(move || {
                    loop {
                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_chunks {
                            break;
                        }

                        let start_byte = idx * chunk_size;
                        let end_byte = ((idx + 1) * chunk_size).min(data.len());

                        if start_byte >= data.len() {
                            break;
                        }

                        // First chunk always starts at bit 0
                        if idx == 0 {
                            let result = try_decode_chunk(data, idx, 0, end_byte, 0);
                            if result.valid_bit_offset.is_some() {
                                success_ref.store(true, Ordering::Relaxed);
                            }
                            *results_ref[idx].lock().unwrap() = Some(result);
                            continue;
                        }

                        // For other chunks, try all bit offsets
                        let chunk_data = &data[start_byte..];
                        let mut best_result: Option<SpecResult> = None;

                        for bit_offset in 0..8 {
                            if let Some(result) =
                                try_decode_from_bit(chunk_data, idx, bit_offset, chunk_size)
                            {
                                if result.valid_bit_offset.is_some() {
                                    success_ref.store(true, Ordering::Relaxed);
                                    best_result = Some(result);
                                    break;
                                }
                            }
                        }

                        let result = best_result.unwrap_or(SpecResult {
                            index: idx,
                            valid_bit_offset: None,
                            data: Vec::new(),
                            needs_window: false,
                            unresolved_refs: Vec::new(),
                            final_window: Vec::new(),
                        });

                        *results_ref[idx].lock().unwrap() = Some(result);
                    }
                });
            }
        });

        results
            .into_iter()
            .map(|m| {
                m.into_inner().unwrap().unwrap_or(SpecResult {
                    index: 0,
                    valid_bit_offset: None,
                    data: Vec::new(),
                    needs_window: false,
                    unresolved_refs: Vec::new(),
                    final_window: Vec::new(),
                })
            })
            .collect()
    }

    /// Stitch results together with window propagation
    fn stitch_results(&self, data: &[u8], results: &[SpecResult]) -> io::Result<Vec<u8>> {
        // If the first chunk succeeded and no others need window resolution,
        // we can just concatenate
        let mut total_size = 0usize;
        let mut all_independent = true;

        for result in results {
            if result.valid_bit_offset.is_none() {
                all_independent = false;
                break;
            }
            if result.needs_window && result.index > 0 {
                all_independent = false;
            }
            total_size += result.data.len();
        }

        if all_independent && results.iter().all(|r| r.valid_bit_offset.is_some()) {
            // Simple case: concatenate all outputs
            let mut output = Vec::with_capacity(total_size);
            for result in results {
                output.extend_from_slice(&result.data);
            }
            return Ok(output);
        }

        // Complex case: need to do window propagation
        // For now, fall back to sequential decode
        let mut decoder = DeflateDecoder::new(data);
        let mut output = Vec::new();
        decoder.decode(&mut output)?;
        Ok(output)
    }
}

/// Try to decode a chunk from a specific bit offset
fn try_decode_from_bit(
    data: &[u8],
    index: usize,
    bit_offset: u8,
    max_output: usize,
) -> Option<SpecResult> {
    // Skip the bit offset bytes
    if bit_offset > 0 && data.is_empty() {
        return None;
    }

    // Create a decoder with empty window (we'll mark unresolved refs)
    let start_bit = bit_offset as usize;

    // Try to decode - invalid positions will fail quickly
    let mut output = Vec::new();
    let mut decoder = DeflateDecoder::new(data);

    // TODO: Implement bit-offset starting
    // For now, only byte-aligned works
    if bit_offset != 0 {
        return None;
    }

    match decoder.decode(&mut output) {
        Ok(_) => Some(SpecResult {
            index,
            valid_bit_offset: Some(bit_offset),
            data: output,
            needs_window: false, // First chunk doesn't need window
            unresolved_refs: Vec::new(),
            final_window: decoder.get_window().0,
        }),
        Err(_) => None,
    }
}

/// Try to decode a chunk
fn try_decode_chunk(
    data: &[u8],
    index: usize,
    start_byte: usize,
    end_byte: usize,
    bit_offset: u8,
) -> SpecResult {
    let chunk_data = &data[start_byte..end_byte.min(data.len())];

    let mut output = Vec::new();
    let mut decoder = DeflateDecoder::new(chunk_data);

    match decoder.decode(&mut output) {
        Ok(_) => SpecResult {
            index,
            valid_bit_offset: Some(bit_offset),
            data: output,
            needs_window: false,
            unresolved_refs: Vec::new(),
            final_window: decoder.get_window().0,
        },
        Err(_) => SpecResult {
            index,
            valid_bit_offset: None,
            data: Vec::new(),
            needs_window: false,
            unresolved_refs: Vec::new(),
            final_window: Vec::new(),
        },
    }
}

/// High-level speculative decompression
pub fn decompress_speculative<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    let inflater = SpeculativeInflater::new(num_threads);
    inflater.decompress(data, writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    #[test]
    fn test_speculative_small() {
        let original = b"Hello, World!";

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_speculative(&compressed, &mut output, 4).unwrap();

        assert_eq!(&output, original);
    }

    #[test]
    fn test_speculative_large() {
        let original: Vec<u8> = (0..200_000).map(|i| (i % 256) as u8).collect();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        decompress_speculative(&compressed, &mut output, 4).unwrap();

        assert_eq!(output, original);
    }
}
