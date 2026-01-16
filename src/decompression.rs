//! Ultra-fast decompression using libdeflate + zlib-ng
//!
//! Strategy:
//! - Single-member gzip: libdeflate (30-50% faster than zlib)
//! - Multi-member gzip: zlib-ng via flate2 (reliable member boundary handling)
//! - Stdin streaming: flate2 MultiGzDecoder
//!
//! Key optimizations:
//! - ISIZE trailer hint for accurate buffer pre-allocation
//! - SIMD-accelerated header detection via memchr
//! - Cache-line aligned buffers (64 bytes on x86, 128 on Apple Silicon)
//!
//! Key insight: Deflate streams can contain bytes that look like gzip headers
//! (0x1f 0x8b 0x08), so we can't reliably detect member boundaries by scanning.
//! Instead, we use flate2's GzDecoder which properly parses each member.

use std::fs::File;
use std::io::{self, stdin, stdout, BufReader, BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::cli::RigzArgs;
use crate::error::{RigzError, RigzResult};
use crate::format::CompressionFormat;
use crate::utils::strip_compression_extension;

/// Output buffer size for streaming (256KB for better throughput)
const STREAM_BUFFER_SIZE: usize = 256 * 1024;

/// Cache line size for buffer alignment
#[cfg(target_os = "macos")]
const CACHE_LINE_SIZE: usize = 128; // Apple Silicon uses 128-byte cache lines

#[cfg(not(target_os = "macos"))]
const CACHE_LINE_SIZE: usize = 64; // x86 and most ARM use 64-byte cache lines

/// Allocate a buffer aligned to cache line boundaries
#[inline]
fn alloc_aligned_buffer(size: usize) -> Vec<u8> {
    // Round up size to cache line boundary for better memory access patterns
    let aligned_size = (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
    vec![0u8; aligned_size]
}

use std::cell::RefCell;

// Thread-local decompressor to avoid repeated initialization overhead
thread_local! {
    static DECOMPRESSOR: RefCell<libdeflater::Decompressor> =
        RefCell::new(libdeflater::Decompressor::new());
}

pub fn decompress_file(filename: &str, args: &RigzArgs) -> RigzResult<i32> {
    if filename == "-" {
        return decompress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(RigzError::FileNotFound(filename.to_string()));
    }

    if input_path.is_dir() {
        return Err(RigzError::invalid_argument(format!(
            "{} is a directory",
            filename
        )));
    }

    let output_path = if args.stdout {
        None
    } else {
        Some(get_output_filename(input_path, args))
    };

    if let Some(ref output_path) = output_path {
        if output_path.exists() && !args.force {
            return Err(RigzError::invalid_argument(format!(
                "Output file {} already exists",
                output_path.display()
            )));
        }
    }

    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();
    let mmap = unsafe { Mmap::map(&input_file)? };

    let format = detect_compression_format_from_path(input_path)?;

    let result = if args.stdout {
        let stdout = stdout();
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
        decompress_mmap_libdeflate(&mmap, &mut writer, format)
    } else {
        let output_path = output_path.clone().unwrap();
        let output_file = File::create(&output_path)?;
        let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, output_file);
        decompress_mmap_libdeflate(&mmap, &mut writer, format)
    };

    match result {
        Ok(output_size) => {
            if args.verbosity > 0 && !args.quiet {
                print_decompression_stats(file_size, output_size, input_path);
            }
            if !args.keep && !args.stdout {
                std::fs::remove_file(input_path)?;
            }
            Ok(0)
        }
        Err(e) => {
            if !args.stdout {
                let cleanup_path = get_output_filename(input_path, args);
                if cleanup_path.exists() {
                    let _ = std::fs::remove_file(&cleanup_path);
                }
            }
            Err(e)
        }
    }
}

pub fn decompress_stdin(_args: &RigzArgs) -> RigzResult<i32> {
    use flate2::read::MultiGzDecoder;

    let stdin = stdin();
    let input = BufReader::with_capacity(STREAM_BUFFER_SIZE, stdin.lock());
    let stdout = stdout();
    let mut output = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());

    let mut decoder = MultiGzDecoder::new(input);
    io::copy(&mut decoder, &mut output)?;
    output.flush()?;

    Ok(0)
}

/// Decompress using libdeflate (fastest for in-memory data)
fn decompress_mmap_libdeflate<W: Write>(
    mmap: &Mmap,
    writer: &mut W,
    format: CompressionFormat,
) -> RigzResult<u64> {
    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            decompress_gzip_libdeflate(&mmap[..], writer)
        }
        CompressionFormat::Zlib => decompress_zlib_libdeflate(&mmap[..], writer),
    }
}

/// Quick check if data contains multiple gzip members
/// Uses SIMD-accelerated search via memchr (10-50x faster than byte-by-byte)
/// Only scans first 256KB to detect parallel-compressed files
#[inline]
fn is_multi_member_quick(data: &[u8]) -> bool {
    use memchr::memmem;

    const SCAN_LIMIT: usize = 256 * 1024;
    const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b, 0x08];

    let scan_end = data.len().min(SCAN_LIMIT);

    // Skip past the first gzip header (minimum 10 bytes)
    // and look for another gzip magic sequence
    if scan_end <= 10 {
        return false;
    }

    // memmem uses SIMD (AVX2/NEON) internally for fast searching
    memmem::find(&data[10..scan_end], GZIP_MAGIC).is_some()
}

/// Decompress gzip - chooses optimal strategy based on content
///
/// - Single member: libdeflate (fastest, 30-50% faster than zlib)
/// - BGZF-style (rigz output): parallel libdeflate using embedded block sizes
/// - Other multi-member: sequential zlib-ng
fn decompress_gzip_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(0);
    }

    // Fast path: check if this is likely multi-member (from parallel compression)
    // Only scan first 256KB - if no second header found, use direct single-member path
    if !is_multi_member_quick(data) {
        return decompress_single_member_libdeflate(data, writer);
    }

    // Check for BGZF-style markers (rigz output with embedded block sizes)
    // These allow parallel decompression without scanning for boundaries
    if has_bgzf_markers(data) {
        return decompress_bgzf_parallel(data, writer);
    }

    // Multi-member file without markers: use zlib-ng sequential
    decompress_multi_member_zlibng(data, writer)
}

/// Check if data has BGZF-style "RZ" markers in the first gzip header
#[inline]
fn has_bgzf_markers(data: &[u8]) -> bool {
    // Minimum header with FEXTRA: 10 base + 2 XLEN + 4 subfield header
    if data.len() < 16 {
        return false;
    }

    // Check FEXTRA flag (bit 2 of flags byte at offset 3)
    if data[3] & 0x04 == 0 {
        return false;
    }

    // Get XLEN (2 bytes at offset 10, little-endian)
    let xlen = u16::from_le_bytes([data[10], data[11]]) as usize;
    if xlen < 6 || data.len() < 12 + xlen {
        return false;
    }

    // Look for "RZ" subfield ID
    let extra_field = &data[12..12 + xlen];
    let mut pos = 0;
    while pos + 4 <= extra_field.len() {
        let subfield_id = &extra_field[pos..pos + 2];
        let subfield_len = u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;

        if subfield_id == crate::parallel_compress::RIGZ_SUBFIELD_ID.as_slice() {
            return true;
        }

        pos += 4 + subfield_len;
    }

    false
}

/// Parse BGZF block boundaries from "RZ" markers
/// Returns vector of (start_offset, block_size) tuples
fn parse_bgzf_blocks(data: &[u8]) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        // Check for gzip magic
        if data.len() - offset < 18 {
            break;
        }
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // Check FEXTRA flag
        if data[offset + 3] & 0x04 == 0 {
            break; // No FEXTRA, can't parse block size
        }

        // Get XLEN
        if data.len() - offset < 12 {
            break;
        }
        let xlen = u16::from_le_bytes([data[offset + 10], data[offset + 11]]) as usize;
        if data.len() - offset < 12 + xlen {
            break;
        }

        // Find "RZ" subfield
        let extra_start = offset + 12;
        let extra_field = &data[extra_start..extra_start + xlen];
        let mut block_size = None;
        let mut pos = 0;

        while pos + 4 <= extra_field.len() {
            let subfield_id = &extra_field[pos..pos + 2];
            let subfield_len =
                u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;

            if subfield_id == crate::parallel_compress::RIGZ_SUBFIELD_ID.as_slice()
                && subfield_len >= 2
                && pos + 4 + 2 <= extra_field.len()
            {
                // Block size is stored as (size - 1)
                let size_minus_1 =
                    u16::from_le_bytes([extra_field[pos + 4], extra_field[pos + 5]]);
                block_size = Some((size_minus_1 as usize) + 1);
                break;
            }

            pos += 4 + subfield_len;
        }

        match block_size {
            Some(size) if size > 0 && offset + size <= data.len() => {
                blocks.push((offset, size));
                offset += size;
            }
            _ => break, // Invalid or missing block size
        }
    }

    blocks
}

/// Parallel decompression for BGZF-style files (rigz output)
/// Uses embedded block size markers to find boundaries without inflating
fn decompress_bgzf_parallel<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use libdeflater::{DecompressionError, Decompressor};
    use rayon::prelude::*;

    let blocks = parse_bgzf_blocks(data);

    // Fall back to sequential if we couldn't parse blocks
    if blocks.is_empty() {
        return decompress_multi_member_zlibng(data, writer);
    }

    // For few blocks, sequential is faster (avoids rayon overhead)
    if blocks.len() < 4 {
        return decompress_multi_member_sequential(data, writer);
    }

    // Decompress blocks in parallel
    let results: Vec<Result<Vec<u8>, String>> = blocks
        .par_iter()
        .map(|&(start, len)| {
            let block_data = &data[start..start + len];
            let mut decompressor = Decompressor::new();

            // Read ISIZE from trailer for buffer sizing
            let isize_hint = if len >= 8 {
                let trailer = &block_data[len - 4..];
                u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
            } else {
                0
            };

            let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
                isize_hint + 1024
            } else {
                len.saturating_mul(4).max(64 * 1024)
            };

            let mut output = vec![0u8; initial_size];

            loop {
                match decompressor.gzip_decompress(block_data, &mut output) {
                    Ok(size) => {
                        output.truncate(size);
                        return Ok(output);
                    }
                    Err(DecompressionError::InsufficientSpace) => {
                        let new_size = output.len().saturating_mul(2);
                        output.resize(new_size, 0);
                        continue;
                    }
                    Err(e) => return Err(format!("decompression error: {:?}", e)),
                }
            }
        })
        .collect();

    // Collect results and write in order
    let mut total_bytes = 0u64;
    for result in results {
        match result {
            Ok(decompressed) => {
                writer.write_all(&decompressed)?;
                total_bytes += decompressed.len() as u64;
            }
            Err(e) => return Err(RigzError::invalid_argument(e)),
        }
    }

    writer.flush()?;
    Ok(total_bytes)
}

/// Read the ISIZE field from gzip trailer (last 4 bytes) for buffer sizing
/// Returns uncompressed size mod 2^32 (per RFC 1952)
#[inline]
fn read_gzip_isize(data: &[u8]) -> Option<u32> {
    if data.len() < 18 {
        // Minimum gzip: 10 header + 8 trailer
        return None;
    }
    let isize_bytes = &data[data.len() - 4..];
    Some(u32::from_le_bytes([
        isize_bytes[0],
        isize_bytes[1],
        isize_bytes[2],
        isize_bytes[3],
    ]))
}

/// Decompress single-member gzip using libdeflate (fastest path)
/// Uses thread-local decompressor to avoid initialization overhead
fn decompress_single_member_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use libdeflater::DecompressionError;

    // Use ISIZE from trailer for accurate buffer sizing (avoids resize loop)
    // Add small margin for safety, handle files >4GB (ISIZE wraps at 2^32)
    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        // Trust ISIZE for files under 1GB, add 1KB margin
        isize_hint + 1024
    } else {
        // Fallback: estimate 4x compression ratio
        data.len().saturating_mul(4).max(64 * 1024)
    };

    // Use cache-aligned buffer for better memory access
    let mut output_buf = alloc_aligned_buffer(initial_size);

    // Reuse thread-local decompressor to avoid repeated initialization
    DECOMPRESSOR.with(|decomp| {
        let mut decompressor = decomp.borrow_mut();

        loop {
            match decompressor.gzip_decompress(data, &mut output_buf) {
                Ok(decompressed_size) => {
                    writer.write_all(&output_buf[..decompressed_size])?;
                    writer.flush()?;
                    return Ok(decompressed_size as u64);
                }
                Err(DecompressionError::InsufficientSpace) => {
                    // Grow buffer and retry (rare with ISIZE hint)
                    let new_size = output_buf.len().saturating_mul(2);
                    output_buf.resize(new_size, 0);
                    continue;
                }
                Err(_) => {
                    return Err(RigzError::invalid_argument(
                        "gzip decompression failed".to_string(),
                    ));
                }
            }
        }
    })
}

/// Decompress multi-member gzip using sequential zlib-ng
///
/// Note: We previously had a parallel decompression path, but it required
/// finding member boundaries first (which means decompressing once to find
/// boundaries, then again in parallel). This 2x overhead made it slower
/// for files with many small members (like rigz's 128KB chunks).
///
/// The sequential MultiGzDecoder is fast enough and doesn't have this overhead.
fn decompress_multi_member_zlibng<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    decompress_multi_member_sequential(data, writer)
}

/// Sequential multi-member decompression using flate2
fn decompress_multi_member_sequential<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use flate2::bufread::MultiGzDecoder;
    use std::io::Read;

    let mut total_bytes = 0u64;
    let mut decoder = MultiGzDecoder::new(data);

    // Use cache-aligned buffer for better memory access patterns
    let mut buf = alloc_aligned_buffer(STREAM_BUFFER_SIZE);

    loop {
        match decoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                writer.write_all(&buf[..n])?;
                total_bytes += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(RigzError::Io(e)),
        }
    }

    writer.flush()?;
    Ok(total_bytes)
}

/// Find gzip member boundaries by inflating each member
/// Returns vector of (start_offset, length) for each member
///
/// Note: Currently unused - the parallel decompression path was disabled
/// because the boundary-finding overhead (requires full decompression)
/// made it slower than sequential for files with many small members.
#[allow(dead_code)]
fn find_member_boundaries(data: &[u8]) -> Vec<(usize, usize)> {
    use flate2::bufread::GzDecoder;
    use std::io::Read;

    let mut boundaries = Vec::new();
    let mut offset = 0;
    let mut buf = [0u8; 8192];

    while offset < data.len() && data.len() - offset >= 10 {
        // Check for gzip magic
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        let start = offset;
        let remaining = &data[offset..];
        let mut decoder = GzDecoder::new(remaining);

        // Consume the entire member
        loop {
            match decoder.read(&mut buf) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(_) => break,
            }
        }

        // Get how many bytes were consumed
        let consumed = remaining.len() - decoder.into_inner().len();
        if consumed == 0 {
            break;
        }

        boundaries.push((start, consumed));
        offset += consumed;
    }

    boundaries
}

/// Parallel multi-member decompression using rayon + libdeflate
///
/// Note: Currently unused - requires finding member boundaries first,
/// which means decompressing once to find boundaries, then again in parallel.
/// This 2x overhead made it slower for files with many small members.
#[allow(dead_code)]
fn decompress_multi_member_parallel<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use libdeflater::{DecompressionError, Decompressor};
    use rayon::prelude::*;
    use std::io::IoSlice;

    // Find member boundaries (this is sequential but fast)
    let boundaries = find_member_boundaries(data);

    // Need at least 2 members to benefit from parallelism
    if boundaries.len() < 2 {
        return decompress_multi_member_sequential(data, writer);
    }

    // Decompress members in parallel
    let results: Vec<Result<Vec<u8>, String>> = boundaries
        .par_iter()
        .map(|&(start, len)| {
            let member_data = &data[start..start + len];
            let mut decompressor = Decompressor::new();

            // Estimate output size from ISIZE trailer (last 4 bytes of member)
            let isize_hint = if len >= 8 {
                let trailer = &member_data[len - 4..];
                u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
            } else {
                0
            };

            let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
                isize_hint + 1024
            } else {
                len.saturating_mul(4).max(64 * 1024)
            };

            let mut output = vec![0u8; initial_size];

            loop {
                match decompressor.gzip_decompress(member_data, &mut output) {
                    Ok(size) => {
                        output.truncate(size);
                        return Ok(output);
                    }
                    Err(DecompressionError::InsufficientSpace) => {
                        let new_size = output.len().saturating_mul(2);
                        output.resize(new_size, 0);
                        continue;
                    }
                    Err(e) => return Err(format!("decompression error: {:?}", e)),
                }
            }
        })
        .collect();

    // Collect successful results and check for errors
    let mut decompressed_blocks: Vec<Vec<u8>> = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(block) => decompressed_blocks.push(block),
            Err(e) => return Err(RigzError::invalid_argument(e)),
        }
    }

    // Use vectorized I/O to write all blocks efficiently
    let total_bytes: u64 = decompressed_blocks.iter().map(|b| b.len() as u64).sum();

    // For many blocks, use write_vectored to reduce syscalls
    if decompressed_blocks.len() > 4 {
        const MAX_IOVECS: usize = 64;
        for chunk in decompressed_blocks.chunks(MAX_IOVECS) {
            let slices: Vec<IoSlice<'_>> = chunk.iter().map(|b| IoSlice::new(b)).collect();
            let mut remaining = &slices[..];

            while !remaining.is_empty() {
                let written = writer.write_vectored(remaining)?;
                if written == 0 {
                    return Err(RigzError::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write decompressed data",
                    )));
                }

                // Advance past written data
                let mut bytes_left = written;
                let mut consumed = 0;
                for slice in remaining.iter() {
                    if bytes_left >= slice.len() {
                        bytes_left -= slice.len();
                        consumed += 1;
                    } else {
                        break;
                    }
                }
                remaining = &remaining[consumed..];

                // Handle partial writes
                if bytes_left > 0 && !remaining.is_empty() {
                    writer.write_all(&remaining[0][bytes_left..])?;
                    for slice in &remaining[1..] {
                        writer.write_all(slice)?;
                    }
                    break;
                }
            }
        }
    } else {
        for block in &decompressed_blocks {
            writer.write_all(block)?;
        }
    }

    writer.flush()?;
    Ok(total_bytes)
}

/// Decompress zlib using libdeflate
fn decompress_zlib_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> RigzResult<u64> {
    use libdeflater::{DecompressionError, Decompressor};

    let mut decompressor = Decompressor::new();
    let mut output_buf = vec![0u8; data.len().saturating_mul(4).max(64 * 1024)];

    loop {
        match decompressor.zlib_decompress(data, &mut output_buf) {
            Ok(decompressed_size) => {
                writer.write_all(&output_buf[..decompressed_size])?;
                writer.flush()?;
                return Ok(decompressed_size as u64);
            }
            Err(DecompressionError::InsufficientSpace) => {
                let new_size = output_buf.len().saturating_mul(2);
                output_buf.resize(new_size, 0);
                continue;
            }
            Err(_) => {
                return Err(RigzError::invalid_argument(
                    "zlib decompression failed".to_string(),
                ));
            }
        }
    }
}

fn detect_compression_format_from_path(path: &Path) -> RigzResult<CompressionFormat> {
    if let Some(format) = crate::utils::detect_format_from_file(path) {
        Ok(format)
    } else {
        Ok(CompressionFormat::Gzip)
    }
}

fn get_output_filename(input_path: &Path, args: &RigzArgs) -> std::path::PathBuf {
    if args.stdout {
        return input_path.to_path_buf();
    }
    let mut output_path = strip_compression_extension(input_path);
    if output_path == input_path {
        output_path = input_path.to_path_buf();
        let current_name = output_path.file_name().unwrap().to_str().unwrap();
        output_path.set_file_name(format!("{}.out", current_name));
    }
    output_path
}

fn print_decompression_stats(input_size: u64, output_size: u64, path: &Path) {
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("<unknown>");

    let ratio = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        1.0
    };

    let (in_size, in_unit) = format_size(input_size);
    let (out_size, out_unit) = format_size(output_size);

    eprintln!(
        "{}: {:.1}{} → {:.1}{} ({:.1}x expansion)",
        filename,
        in_size,
        in_unit,
        out_size,
        out_unit,
        1.0 / ratio
    );
}

fn format_size(bytes: u64) -> (f64, &'static str) {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        (bytes as f64 / GB as f64, "GB")
    } else if bytes >= MB {
        (bytes as f64 / MB as f64, "MB")
    } else if bytes >= KB {
        (bytes as f64 / KB as f64, "KB")
    } else {
        (bytes as f64, "B")
    }
}
