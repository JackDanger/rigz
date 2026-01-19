#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

//! Ultra-Fast Parallel Decompression Engine
//!
//! This is the main decompression engine that achieves rapidgzip-level
//! performance by combining:
//!
//! 1. ISA-L for raw decompression speed
//! 2. Block-level parallelism for multi-member files
//! 3. Prefetching and cache optimization
//! 4. Zero-copy operations where possible
//!
//! # Strategy by File Type
//!
//! - BGZF (gzippy): Perfect parallelism using embedded markers
//! - Multi-member (pigz): Parallel per-member with ISA-L
//! - Single-member: Streaming with large buffers and prefetch

use crate::isal::{decompress_parallel as isal_parallel, IsalInflater};
use std::cell::RefCell;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Minimum file size for parallel decompression
const PARALLEL_THRESHOLD: usize = 1024 * 1024; // 1MB

/// Block size for prefetching
const PREFETCH_SIZE: usize = 256 * 1024; // 256KB

// Thread-local ISA-L inflater
thread_local! {
    static INFLATER: RefCell<IsalInflater> = RefCell::new(IsalInflater::new().unwrap());
}

/// Main ultra-fast decompressor
pub struct UltraDecompressor {
    num_threads: usize,
}

impl UltraDecompressor {
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self { num_threads }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Decompress gzip data using the fastest available method
    pub fn decompress<W: Write + Send>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        // Check for BGZF markers (gzippy output)
        if has_bgzf_markers(data) {
            return self.decompress_bgzf(data, writer);
        }

        // Check for multi-member (pigz output)
        let members = find_members_fast(data);
        if members.len() > 1 && data.len() >= PARALLEL_THRESHOLD {
            return self.decompress_members(data, &members, writer);
        }

        // Single member: try rapidgzip-style parallel decompression
        if data.len() >= PARALLEL_THRESHOLD && self.num_threads > 1 {
            if let Ok(bytes) = self.decompress_single_parallel(data, writer) {
                return Ok(bytes);
            }
            // Fall through to sequential if parallel fails
        }

        // Small file or parallel failed: optimized sequential
        self.decompress_sequential(data, writer)
    }

    /// Decompress BGZF (gzippy output) - uses embedded block markers
    fn decompress_bgzf<W: Write>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        // Parse BGZF blocks using the GZ markers
        let blocks = parse_bgzf_blocks(data);

        if blocks.is_empty() || self.num_threads <= 1 {
            return self.decompress_sequential(data, writer);
        }

        let num_blocks = blocks.len();
        let outputs: Vec<Mutex<Option<Vec<u8>>>> =
            (0..num_blocks).map(|_| Mutex::new(None)).collect();

        let next_block = AtomicUsize::new(0);
        let error_flag = AtomicBool::new(false);

        std::thread::scope(|scope| {
            // Worker threads
            for _ in 0..self.num_threads.min(num_blocks) {
                let outputs_ref = &outputs;
                let next_ref = &next_block;
                let blocks_ref = &blocks;
                let error_ref = &error_flag;

                scope.spawn(move || {
                    let mut prefetch_idx = 0;

                    loop {
                        if error_ref.load(Ordering::Relaxed) {
                            break;
                        }

                        let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_blocks {
                            break;
                        }

                        // Prefetch next blocks
                        for i in 1..=4 {
                            let future_idx = idx + i;
                            if future_idx < num_blocks {
                                let (start, len) = blocks_ref[future_idx];
                                prefetch_data(&data[start..start + len.min(64)]);
                            }
                        }

                        let (start, len) = blocks_ref[idx];
                        let block_data = &data[start..start + len];

                        // Decompress this block
                        let result = INFLATER.with(|inf| {
                            let mut inflater = inf.borrow_mut();
                            inflater.reset().ok();

                            // Get ISIZE hint
                            let isize_hint = if len >= 8 {
                                let trailer = &block_data[len - 4..];
                                u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]])
                                    as usize
                            } else {
                                len * 4
                            };

                            inflater.decompress_all(block_data, isize_hint.max(1024))
                        });

                        match result {
                            Ok(decompressed) => {
                                *outputs_ref[idx].lock().unwrap() = Some(decompressed);
                            }
                            Err(_) => {
                                // Try flate2 fallback
                                let mut decoder = flate2::read::GzDecoder::new(block_data);
                                let mut buf = Vec::new();
                                if decoder.read_to_end(&mut buf).is_ok() {
                                    *outputs_ref[idx].lock().unwrap() = Some(buf);
                                }
                            }
                        }
                    }
                });
            }
        });

        // Write outputs in order
        let mut total = 0u64;
        for output_mutex in outputs {
            if let Some(output) = output_mutex.into_inner().unwrap() {
                writer.write_all(&output)?;
                total += output.len() as u64;
            }
        }

        writer.flush()?;
        Ok(total)
    }

    /// Decompress multi-member gzip (pigz output)
    fn decompress_members<W: Write + Send>(
        &self,
        data: &[u8],
        members: &[(usize, usize)],
        writer: &mut W,
    ) -> io::Result<u64> {
        isal_parallel(data, writer, self.num_threads)
    }

    /// Parallel decompression for single-member gzip using ultra-fast engine
    fn decompress_single_parallel<W: Write + Send>(
        &self,
        data: &[u8],
        writer: &mut W,
    ) -> io::Result<u64> {
        // Try the ultra-fast LUT-based parallel decompress first
        match crate::ultra_inflate::decompress_ultra_fast(data, writer, self.num_threads) {
            Ok(bytes) => Ok(bytes),
            Err(_) => {
                // Fallback to rapidgzip decoder
                crate::rapidgzip_decoder::decompress_rapidgzip(data, writer, self.num_threads)
            }
        }
    }

    /// Optimized sequential decompression
    fn decompress_sequential<W: Write>(&self, data: &[u8], writer: &mut W) -> io::Result<u64> {
        INFLATER.with(|inf| {
            let mut inflater = inf.borrow_mut();
            inflater.reset().ok();

            // Get ISIZE hint
            let isize_hint = if data.len() >= 8 {
                let trailer = &data[data.len() - 4..];
                u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
            } else {
                data.len() * 4
            };

            let output = inflater.decompress_all(data, isize_hint.max(1024))?;
            writer.write_all(&output)?;
            writer.flush()?;

            Ok(output.len() as u64)
        })
    }
}

impl Default for UltraDecompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Check for BGZF-style "GZ" markers
fn has_bgzf_markers(data: &[u8]) -> bool {
    if data.len() < 16 {
        return false;
    }

    // Check FEXTRA flag
    if data[3] & 0x04 == 0 {
        return false;
    }

    // Parse extra field
    let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
    if 12 + xlen > data.len() {
        return false;
    }

    // Look for "GZ" subfield ID
    let mut pos = 12;
    while pos + 4 <= 12 + xlen {
        let id = [data[pos], data[pos + 1]];
        if id == [b'G', b'Z'] || id == [b'B', b'C'] {
            return true;
        }
        let len = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
        pos += 4 + len;
    }

    false
}

/// Parse BGZF blocks using embedded size markers
fn parse_bgzf_blocks(data: &[u8]) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut offset = 0;

    while offset + 18 < data.len() {
        // Check gzip magic
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // Check FEXTRA
        if data[offset + 3] & 0x04 == 0 {
            break;
        }

        // Parse extra field for block size
        let xlen = u16::from_le_bytes([data[offset + 10], data[offset + 11]]) as usize;
        if offset + 12 + xlen > data.len() {
            break;
        }

        // Find block size in extra field
        let mut block_size = None;
        let mut pos = offset + 12;
        while pos + 4 <= offset + 12 + xlen {
            let id = [data[pos], data[pos + 1]];
            let len = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;

            if (id == [b'G', b'Z'] || id == [b'B', b'C']) && len == 2 {
                // BGZF block size (includes header)
                let bsize = u16::from_le_bytes([data[pos + 4], data[pos + 5]]) as usize;
                block_size = Some(bsize + 1);
                break;
            }

            pos += 4 + len;
        }

        match block_size {
            Some(size) if size > 0 && offset + size <= data.len() => {
                blocks.push((offset, size));
                offset += size;
            }
            _ => break,
        }
    }

    blocks
}

/// Fast member detection (only looks at headers, not content)
fn find_members_fast(data: &[u8]) -> Vec<(usize, usize)> {
    let mut members = Vec::new();

    // Sequential scan for gzip magic
    // We use the decompressor to find true member boundaries
    let mut offset = 0;

    while offset < data.len() {
        if offset + 10 > data.len() {
            break;
        }

        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // This is a valid member start, find the end
        // For now, scan for next member or EOF
        let start = offset;
        let mut end = data.len();

        // Skip past header
        let header_len = parse_gzip_header(&data[offset..]).unwrap_or(10);
        let search_start = offset + header_len + 18; // minimum: header + 8 deflate + 8 trailer

        // Look for next member
        for i in search_start..data.len().saturating_sub(10) {
            if data[i] == 0x1f
                && data[i + 1] == 0x8b
                && data[i + 2] == 0x08
                && parse_gzip_header(&data[i..]).is_some()
            {
                end = i;
                break;
            }
        }

        members.push((start, end));
        offset = end;
    }

    members
}

/// Parse gzip header, return length
fn parse_gzip_header(data: &[u8]) -> Option<usize> {
    if data.len() < 10 {
        return None;
    }

    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }

    let flags = data[3];
    let mut offset = 10;

    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return None;
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
        return None;
    }

    Some(offset)
}

/// Prefetch data into CPU cache
#[inline]
fn prefetch_data(data: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        for chunk in data.chunks(64) {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    chunk.as_ptr() as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        for chunk in data.chunks(128) {
            let _ = unsafe { std::ptr::read_volatile(chunk.as_ptr()) };
        }
    }
}

/// Public function for use in decompression.rs
pub fn decompress_ultra<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    let decompressor = UltraDecompressor::with_threads(num_threads);
    decompressor.decompress(data, writer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultra_decompressor() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let original = b"Test data for ultra decompressor!";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decompressor = UltraDecompressor::new();
        let mut output = Vec::new();
        decompressor.decompress(&compressed, &mut output).unwrap();

        assert_eq!(&output, original);
    }

    #[test]
    fn test_bgzf_detection() {
        // Normal gzip (no BGZF)
        let normal = [0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff];
        assert!(!has_bgzf_markers(&normal));
    }

    #[test]
    fn test_member_detection() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        // Single member
        let original = b"Single member test";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let members = find_members_fast(&compressed);
        assert_eq!(members.len(), 1);
    }
}
