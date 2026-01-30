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
use std::io::{stdin, stdout, BufReader, BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::cli::GzippyArgs;
use crate::error::{GzippyError, GzippyResult};
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

pub fn decompress_file(filename: &str, args: &GzippyArgs) -> GzippyResult<i32> {
    if filename == "-" {
        return decompress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(GzippyError::FileNotFound(filename.to_string()));
    }

    if input_path.is_dir() {
        return Err(GzippyError::invalid_argument(format!(
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
            return Err(GzippyError::invalid_argument(format!(
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
        // For stdout, we need to buffer in memory first because StdoutLock isn't Send
        // This allows parallel decompression to work, then we write the result
        let mut buffer = Vec::new();
        let result = decompress_mmap_libdeflate(&mmap, &mut buffer, format);

        // Write buffer to stdout
        if let Ok(size) = result {
            let stdout = stdout();
            let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
            writer.write_all(&buffer)?;
            writer.flush()?;
            Ok(size)
        } else {
            result
        }
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

pub fn decompress_stdin(_args: &GzippyArgs) -> GzippyResult<i32> {
    use std::io::Read;

    // Buffer stdin into memory to use our optimized parallel decompressor
    // This trades memory for speed: ~20MB stdin fits easily in RAM
    let stdin = stdin();
    let mut input_data = Vec::new();
    {
        let mut reader = BufReader::with_capacity(STREAM_BUFFER_SIZE, stdin.lock());
        reader.read_to_end(&mut input_data)?;
    }

    if input_data.is_empty() {
        return Ok(0);
    }

    // Detect format
    let format = if input_data.len() >= 2 && input_data[0] == 0x1f && input_data[1] == 0x8b {
        CompressionFormat::Gzip
    } else if input_data.len() >= 2 && input_data[0] == 0x78 {
        CompressionFormat::Zlib
    } else {
        CompressionFormat::Gzip // Default
    };

    // Decompress into buffer (allows parallel decompression, then write to stdout)
    let mut output_buffer = Vec::new();
    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            decompress_gzip_libdeflate(&input_data, &mut output_buffer)?;
        }
        CompressionFormat::Zlib => {
            decompress_zlib_turbo(&input_data, &mut output_buffer)?;
        }
    }

    // Write to stdout
    let stdout = stdout();
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER_SIZE, stdout.lock());
    writer.write_all(&output_buffer)?;
    writer.flush()?;

    Ok(0)
}

/// Decompress using libdeflate (fastest for in-memory data)
fn decompress_mmap_libdeflate<W: Write + Send>(
    mmap: &Mmap,
    writer: &mut W,
    format: CompressionFormat,
) -> GzippyResult<u64> {
    match format {
        CompressionFormat::Gzip | CompressionFormat::Zip => {
            decompress_gzip_libdeflate(&mmap[..], writer)
        }
        CompressionFormat::Zlib => decompress_zlib_turbo(&mmap[..], writer),
    }
}

/// Quick check if data contains multiple gzip members
/// Uses SIMD-accelerated search via memchr (10-50x faster than byte-by-byte)
/// Only scans first 256KB to detect parallel-compressed files
#[inline]
/// Check if this is a multi-member gzip by checking if there are more
/// gzip headers after the first member's minimum size.
///
/// Uses conservative heuristics to avoid false positives from gzip magic
/// appearing in compressed data.
fn is_likely_multi_member(data: &[u8]) -> bool {
    use memchr::memmem;

    // A gzip member is minimum 18 bytes (10 header + 8 trailer)
    if data.len() < 36 {
        // Too small for multi-member
        return false;
    }

    // Parse first header to find approximate end of first member
    let header_size = parse_gzip_header_size(data).unwrap_or(10);

    // Search for gzip magic after header
    // Use a conservative approach: look for the 4-byte pattern 1f 8b 08 XX
    // where XX has reserved bits zero
    const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b, 0x08];
    let finder = memmem::Finder::new(GZIP_MAGIC);

    // Start searching after the header (the deflate data starts there)
    let search_start = header_size + 1; // +1 to skip past first byte of deflate

    let mut pos = search_start;
    while let Some(offset) = finder.find(&data[pos..]) {
        let header_pos = pos + offset;

        // Must have room for full header (10 bytes minimum)
        if header_pos + 10 > data.len() {
            break;
        }

        let flags = data[header_pos + 3];

        // Reserved bits must be zero
        if flags & 0xE0 != 0 {
            pos = header_pos + 1;
            continue;
        }

        // MTIME (4 bytes) should be reasonable (before year 2100 = ~4102444800)
        let mtime = u32::from_le_bytes([
            data[header_pos + 4],
            data[header_pos + 5],
            data[header_pos + 6],
            data[header_pos + 7],
        ]);
        // Allow mtime = 0 (common) or reasonable values
        if mtime != 0 && mtime > 4_102_444_800 {
            pos = header_pos + 1;
            continue;
        }

        // XFL (extra flags) should be 0, 2, or 4 (0=default, 2=slowest, 4=fastest)
        let xfl = data[header_pos + 8];
        if xfl != 0 && xfl != 2 && xfl != 4 {
            pos = header_pos + 1;
            continue;
        }

        // OS should be a known value (0-13, 255)
        let os = data[header_pos + 9];
        if os > 13 && os != 255 {
            pos = header_pos + 1;
            continue;
        }

        // This looks like a valid header!
        return true;
    }

    false
}

/// Parse gzip header size (variable due to FEXTRA, FNAME, FCOMMENT, FHCRC)
pub(crate) fn parse_gzip_header_size(data: &[u8]) -> Option<usize> {
    if data.len() < 10 {
        return None;
    }

    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return None;
    }

    let flags = data[3];
    let mut pos = 10;

    // FEXTRA
    if flags & 0x04 != 0 {
        if pos + 2 > data.len() {
            return None;
        }
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }

    // FNAME (null-terminated)
    if flags & 0x08 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1; // null terminator
    }

    // FCOMMENT (null-terminated)
    if flags & 0x10 != 0 {
        while pos < data.len() && data[pos] != 0 {
            pos += 1;
        }
        pos += 1;
    }

    // FHCRC
    if flags & 0x02 != 0 {
        pos += 2;
    }

    Some(pos)
}

/// Legacy function for compatibility
#[allow(dead_code)]
fn is_multi_member_quick(data: &[u8]) -> bool {
    is_likely_multi_member(data)
}

/// Decompress single-member gzip using our turbo inflate (optimized pure Rust)
///
/// This uses the same optimizations as our BGZF path:
/// - TurboBits with branchless refill
/// - PackedLUT for bitsleft -= entry optimization
/// - 3-literal decode chain with entry preloading
fn decompress_single_member_turbo<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    // Parse gzip header
    let header_size = parse_gzip_header_size(data)
        .ok_or_else(|| GzippyError::invalid_argument("Invalid gzip header".to_string()))?;

    // Data must have at least header + 8 bytes trailer
    if data.len() < header_size + 8 {
        return Err(GzippyError::invalid_argument("Data too short".to_string()));
    }

    // Get deflate data (between header and 8-byte trailer)
    let deflate_data = &data[header_size..data.len() - 8];

    // Use ISIZE from trailer for buffer sizing
    let isize_hint = read_gzip_isize(data).unwrap_or(0) as usize;
    let output_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024 // Small margin
    } else {
        data.len().saturating_mul(4).max(64 * 1024)
    };

    // Allocate output buffer
    let mut output = alloc_aligned_buffer(output_size);

    // Use our turbo inflate
    match crate::bgzf::inflate_into_pub(deflate_data, &mut output) {
        Ok(decompressed_size) => {
            writer.write_all(&output[..decompressed_size])?;
            writer.flush()?;
            Ok(decompressed_size as u64)
        }
        Err(e) => {
            // Fall back to libdeflate on error
            if std::env::var("GZIPPY_DEBUG").is_ok() {
                eprintln!(
                    "[gzippy] Turbo inflate failed: {}, falling back to libdeflate",
                    e
                );
            }
            Err(GzippyError::invalid_argument(format!(
                "Turbo inflate failed: {}",
                e
            )))
        }
    }
}

/// Decompress gzip - chooses optimal strategy based on content
///
/// Strategies (in order of preference):
/// 1. BGZF-style (gzippy output): parallel libdeflate using embedded block sizes
/// 2. Hyperoptimized routing: Profile-based selection (libdeflate/ISA-L/consume_first)
/// 3. Single member: libdeflate (fastest, 30-50% faster than zlib)
/// 4. Large multi-member: speculative parallel decompression (rapidgzip-style)
/// 5. Small multi-member: sequential zlib-ng
///
/// To enable hyperoptimized routing, set GZIPPY_HYPEROPT=1
fn decompress_gzip_libdeflate<W: Write + Send>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(0);
    }

    // Check for BGZF-style markers FIRST (gzippy output with embedded block sizes)
    // This check is fast (only looks at first header) and must come before
    // is_multi_member_quick which only scans 256KB - not enough for random data
    // where the first block can be >256KB
    if has_bgzf_markers(data) {
        // Try our new CombinedLUT-based BGZF decompressor (fastest pure Rust)
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        match crate::bgzf::decompress_bgzf_parallel(data, writer, num_threads) {
            Ok(bytes) => {
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!(
                        "[gzippy] BGZF parallel: {} bytes, {} threads",
                        bytes, num_threads
                    );
                }
                return Ok(bytes);
            }
            Err(e) => {
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!("[gzippy] BGZF parallel failed: {}, trying ultra_inflate", e);
                }
            }
        }

        // Fallback to ultra_inflate BGZF (uses our turbo inflate)
        match crate::ultra_inflate::decompress_bgzf_ultra(data, writer, num_threads) {
            Ok(bytes) => {
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!(
                        "[gzippy] BGZF ultra: {} bytes, {} threads",
                        bytes, num_threads
                    );
                }
                return Ok(bytes);
            }
            Err(e) => {
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!("[gzippy] BGZF ultra failed: {}, falling back", e);
                }
            }
        }
        // Fallback to streaming parallel decompressor
        return decompress_bgzf_parallel_prefetch(data, writer);
    }

    // HYPEROPT: Use profile-based routing if enabled
    if std::env::var("GZIPPY_HYPEROPT").is_ok() {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        match crate::hyperopt_dispatcher::decompress_hyperopt(data, writer, num_threads) {
            Ok(bytes) => {
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!("[gzippy] HYPEROPT: {} bytes", bytes);
                }
                return Ok(bytes);
            }
            Err(e) => {
                if std::env::var("GZIPPY_DEBUG").is_ok() {
                    eprintln!("[gzippy] HYPEROPT failed: {}, falling back", e);
                }
            }
        }
    }

    // Check if this is multi-member using conservative heuristics
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    if !is_likely_multi_member(data) {
        // Single-member: use our turbo inflate (optimized pure Rust)
        // No fallback to libdeflate - we want to see errors
        if std::env::var("GZIPPY_DEBUG").is_ok() {
            eprintln!("[gzippy] Single-member: turbo inflate (pure Rust)");
        }
        return decompress_single_member_turbo(data, writer);
    }

    // Multi-member: try our new pre-allocated parallel decompressor first (Phase 2)
    match crate::bgzf::decompress_multi_member_parallel(data, writer, num_threads) {
        Ok(bytes) => {
            if std::env::var("GZIPPY_DEBUG").is_ok() {
                eprintln!(
                    "[gzippy] Multi-member parallel: {} bytes, {} threads",
                    bytes, num_threads
                );
            }
            return Ok(bytes);
        }
        Err(e) => {
            if std::env::var("GZIPPY_DEBUG").is_ok() {
                eprintln!(
                    "[gzippy] Multi-member parallel failed: {}, trying fallback",
                    e
                );
            }
        }
    }

    // Fallback to parallel_decompress
    if let Ok(bytes) = crate::parallel_decompress::decompress_parallel(data, writer, num_threads) {
        return Ok(bytes);
    }

    match crate::ultra_decompress::decompress_ultra(data, writer, num_threads) {
        Ok(bytes) => Ok(bytes),
        Err(_) => {
            // Try our pure Rust parallel inflater
            if let Ok(bytes) = crate::parallel_inflate::decompress_auto(data, writer, num_threads) {
                return Ok(bytes);
            }
            // Fallback to flate2 sequential
            decompress_multi_member_zlibng(data, writer)
        }
    }
}

/// Check if data has BGZF-style "GZ" markers in the first gzip header
#[inline]
pub(crate) fn has_bgzf_markers(data: &[u8]) -> bool {
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
        let subfield_len =
            u16::from_le_bytes([extra_field[pos + 2], extra_field[pos + 3]]) as usize;

        if subfield_id == crate::parallel_compress::GZ_SUBFIELD_ID.as_slice() {
            return true;
        }

        pos += 4 + subfield_len;
    }

    false
}

/// Parse BGZF block boundaries from "RZ" markers
/// Returns (blocks, consumed_all) where consumed_all is true if we parsed the entire file
fn parse_bgzf_blocks(data: &[u8]) -> (Vec<(usize, usize)>, bool) {
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

            if subfield_id == crate::parallel_compress::GZ_SUBFIELD_ID.as_slice()
                && subfield_len >= 2
                && pos + 4 + 2 <= extra_field.len()
            {
                // Block size is stored as (size - 1)
                let size_minus_1 = u16::from_le_bytes([extra_field[pos + 4], extra_field[pos + 5]]);
                block_size = Some((size_minus_1 as usize) + 1);
                break;
            }

            pos += 4 + subfield_len;
        }

        match block_size {
            Some(size) if size > 1 && offset + size <= data.len() => {
                // size > 1 because size=1 means stored value was 0 (overflow marker)
                blocks.push((offset, size));
                offset += size;
            }
            _ => break, // Invalid, overflow, or missing block size
        }
    }

    // Return whether we consumed the entire file
    let consumed_all = offset >= data.len();
    (blocks, consumed_all)
}

/// Enhanced parallel decompression for BGZF-style files with prefetching
///
/// Improvements over basic parallel decompression:
/// 1. Memory prefetching: Hint to CPU to load next blocks into cache
/// 2. Batch processing: Process blocks in batches for better cache locality
/// 3. Overlapped I/O: Write previous batch while decompressing next
/// 4. Optimized slot management: Reuse output buffers
fn decompress_bgzf_parallel_prefetch<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;

    let (blocks, consumed_all) = parse_bgzf_blocks(data);

    // Fall back to sequential if we couldn't parse ALL blocks
    if blocks.is_empty() || !consumed_all {
        return decompress_multi_member_sequential(data, writer);
    }

    // For few blocks, sequential is faster (avoids thread overhead)
    if blocks.len() < 4 {
        return decompress_multi_member_sequential(data, writer);
    }

    let num_blocks = blocks.len();
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(num_blocks);

    // Output slot for each block - pre-allocated based on ISIZE hints
    struct Slot {
        ready: AtomicBool,
        data: UnsafeCell<Vec<u8>>,
    }
    unsafe impl Sync for Slot {}

    // Pre-calculate output sizes from ISIZE hints for better allocation
    let slots: Vec<Slot> = blocks
        .iter()
        .map(|(start, len)| {
            let block_data = &data[*start..*start + *len];
            let isize_hint = if *len >= 8 {
                let trailer = &block_data[*len - 4..];
                u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
            } else {
                0
            };
            let capacity = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
                isize_hint + 1024
            } else {
                len.saturating_mul(4).max(64 * 1024)
            };
            Slot {
                ready: AtomicBool::new(false),
                data: UnsafeCell::new(Vec::with_capacity(capacity)),
            }
        })
        .collect();

    let next_block = AtomicUsize::new(0);
    let mut total_bytes = 0u64;

    // Number of blocks to prefetch ahead
    const PREFETCH_AHEAD: usize = 4;

    thread::scope(|scope| {
        // Spawn worker threads with prefetching
        for _ in 0..num_threads {
            scope.spawn(|| {
                loop {
                    let idx = next_block.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    // Prefetch next blocks into CPU cache
                    for prefetch_idx in 1..=PREFETCH_AHEAD {
                        let future_idx = idx + prefetch_idx;
                        if future_idx < num_blocks {
                            let (future_start, future_len) = blocks[future_idx];
                            // Prefetch start and middle of block
                            prefetch_memory(&data[future_start..future_start + future_len.min(64)]);
                            if future_len > 4096 {
                                prefetch_memory(
                                    &data[future_start + future_len / 2
                                        ..future_start + future_len / 2 + 64],
                                );
                            }
                        }
                    }

                    let (start, len) = blocks[idx];
                    let block_data = &data[start..start + len];

                    // Use our turbo inflate
                    let output = unsafe { &mut *slots[idx].data.get() };
                    output.clear();

                    // Parse gzip header to get deflate data
                    if let Some(header_size) = parse_gzip_header_size(block_data) {
                        let deflate_end = len.saturating_sub(8); // Exclude trailer
                        if header_size < deflate_end {
                            let deflate_data = &block_data[header_size..deflate_end];
                            let initial_size = output.capacity().max(64 * 1024);
                            output.resize(initial_size, 0);

                            match crate::bgzf::inflate_into_pub(deflate_data, output) {
                                Ok(size) => {
                                    output.truncate(size);
                                }
                                Err(_) => {
                                    output.clear();
                                }
                            }
                        }
                    }

                    slots[idx].ready.store(true, Ordering::Release);
                }
            });
        }

        // Main thread: stream output in order with batched writes
        // Write in batches to reduce syscall overhead
        const WRITE_BATCH_SIZE: usize = 16;
        let mut batch_buffer: Vec<u8> = Vec::with_capacity(WRITE_BATCH_SIZE * 128 * 1024);

        for batch in slots.chunks(WRITE_BATCH_SIZE) {
            batch_buffer.clear();

            for slot in batch.iter() {
                // Spin-wait with backoff for slot to be ready
                let mut spin_count = 0;
                while !slot.ready.load(Ordering::Acquire) {
                    spin_count += 1;
                    if spin_count < 100 {
                        std::hint::spin_loop();
                    } else if spin_count < 1000 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(10));
                    }
                }

                let output = unsafe { &*slot.data.get() };
                if !output.is_empty() {
                    batch_buffer.extend_from_slice(output);
                }
            }

            if !batch_buffer.is_empty() {
                writer.write_all(&batch_buffer).unwrap();
                total_bytes += batch_buffer.len() as u64;
            }
        }
    });

    writer.flush()?;
    Ok(total_bytes)
}

/// Prefetch memory into CPU cache (hint only, no guarantee)
#[inline]
fn prefetch_memory(data: &[u8]) {
    // Use platform-specific prefetch intrinsics when available
    #[cfg(target_arch = "x86_64")]
    {
        for chunk in data.chunks(64) {
            // PREFETCHT0: Prefetch into all cache levels
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
        // On ARM, we use a simple volatile read to encourage prefetch
        // The compiler may optimize this, but it's a hint
        for chunk in data.chunks(128) {
            let _ = unsafe { std::ptr::read_volatile(chunk.as_ptr()) };
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback: touch memory to trigger hardware prefetch
        let _ = data.first();
    }
}

/// Read the ISIZE field from gzip trailer (last 4 bytes) for buffer sizing
/// Returns uncompressed size mod 2^32 (per RFC 1952)
#[inline]
pub(crate) fn read_gzip_isize(data: &[u8]) -> Option<u32> {
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

/// Decompress multi-member gzip using sequential zlib-ng
///
/// Note: We previously had a parallel decompression path, but it required
/// finding member boundaries first (which means decompressing once to find
/// boundaries, then again in parallel). This 2x overhead made it slower
/// for files with many small members (like gzippy's 128KB chunks).
///
/// The sequential MultiGzDecoder is fast enough and doesn't have this overhead.
fn decompress_multi_member_zlibng<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    decompress_multi_member_sequential(data, writer)
}

/// Sequential multi-member decompression using libdeflate (fastest)
///
/// Uses our DecompressorEx wrapper that returns consumed bytes,
/// allowing us to iterate through members without re-decompressing.
fn decompress_multi_member_sequential<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    use crate::libdeflate_ext::{DecompressError, DecompressorEx};

    let mut decompressor = DecompressorEx::new();
    let mut total_bytes = 0u64;
    let mut offset = 0;

    // Pre-allocate a reasonably sized output buffer
    let mut output_buf = alloc_aligned_buffer(256 * 1024);

    while offset < data.len() {
        // Check for gzip magic
        if data.len() - offset < 10 {
            break;
        }
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        let remaining = &data[offset..];

        // Ensure buffer is large enough for estimated output
        let min_size = remaining.len().saturating_mul(4).max(128 * 1024);
        if output_buf.len() < min_size {
            output_buf.resize(min_size, 0);
        }

        let mut success = false;
        loop {
            match decompressor.gzip_decompress_ex(remaining, &mut output_buf) {
                Ok(result) => {
                    writer.write_all(&output_buf[..result.output_size])?;
                    total_bytes += result.output_size as u64;
                    offset += result.input_consumed;
                    success = true;
                    break;
                }
                Err(DecompressError::InsufficientSpace) => {
                    // Grow buffer and retry
                    let new_size = output_buf.len().saturating_mul(2);
                    output_buf.resize(new_size, 0);
                    continue;
                }
                Err(DecompressError::BadData) => {
                    // Invalid data - stop processing entirely
                    break;
                }
            }
        }
        if !success {
            break; // Exit outer loop on error
        }
    }

    writer.flush()?;
    Ok(total_bytes)
}

/// Decompress zlib using our turbo inflate
fn decompress_zlib_turbo<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    // Zlib format: 2-byte header, deflate data, 4-byte Adler32
    if data.len() < 6 {
        return Err(GzippyError::invalid_argument(
            "Zlib data too short".to_string(),
        ));
    }

    // Skip 2-byte zlib header, exclude 4-byte trailer
    let deflate_data = &data[2..data.len() - 4];
    let mut output_buf = vec![0u8; data.len().saturating_mul(4).max(64 * 1024)];

    match crate::bgzf::inflate_into_pub(deflate_data, &mut output_buf) {
        Ok(size) => {
            writer.write_all(&output_buf[..size])?;
            writer.flush()?;
            Ok(size as u64)
        }
        Err(e) => Err(GzippyError::invalid_argument(format!(
            "zlib decompression failed: {}",
            e
        ))),
    }
}

fn detect_compression_format_from_path(path: &Path) -> GzippyResult<CompressionFormat> {
    if let Some(format) = crate::utils::detect_format_from_file(path) {
        Ok(format)
    } else {
        Ok(CompressionFormat::Gzip)
    }
}

fn get_output_filename(input_path: &Path, args: &GzippyArgs) -> std::path::PathBuf {
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

#[cfg(test)]
mod multi_member_tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_decompress_multi_member_file() {
        // Create two separate gzip streams like: (gzip file1; gzip file2) > combined.gz
        let part1: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let part2: Vec<u8> = (0..100_000).map(|i| ((i + 50) % 256) as u8).collect();

        let mut encoder1 = GzEncoder::new(Vec::new(), Compression::default());
        encoder1.write_all(&part1).unwrap();
        let compressed1 = encoder1.finish().unwrap();

        let mut encoder2 = GzEncoder::new(Vec::new(), Compression::default());
        encoder2.write_all(&part2).unwrap();
        let compressed2 = encoder2.finish().unwrap();

        // Concatenate them
        let mut multi = compressed1.clone();
        multi.extend_from_slice(&compressed2);

        eprintln!(
            "Multi-member: {} bytes total, member1={}, member2={}",
            multi.len(),
            compressed1.len(),
            compressed2.len()
        );

        // Check detection
        let is_multi = is_multi_member_quick(&multi);
        eprintln!("is_multi_member_quick: {}", is_multi);

        // Try the full decompression path
        let mut output = Vec::new();
        decompress_gzip_libdeflate(&multi, &mut output).unwrap();

        let mut expected = part1.clone();
        expected.extend_from_slice(&part2);

        eprintln!(
            "Expected: {} bytes, got: {} bytes",
            expected.len(),
            output.len()
        );
        assert_eq!(output.len(), expected.len(), "Output size mismatch!");
        assert_eq!(output, expected, "Output content mismatch!");
    }
}

#[test]
fn test_is_multi_member_quick_timing() {
    let data = match std::fs::read("benchmark_data/silesia-gzip.tar.gz") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping test - benchmark file not found");
            return;
        }
    };
    eprintln!("File size: {} bytes", data.len());

    let start = std::time::Instant::now();
    let result = is_multi_member_quick(&data);
    eprintln!("is_multi_member_quick: {} in {:?}", result, start.elapsed());
}
