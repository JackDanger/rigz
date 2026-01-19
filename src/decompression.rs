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

use std::cell::RefCell;

// Thread-local decompressor and buffer to avoid repeated allocation
thread_local! {
    static DECOMPRESSOR: RefCell<libdeflater::Decompressor> =
        RefCell::new(libdeflater::Decompressor::new());
    static DECOMPRESS_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(1024 * 1024));
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
fn decompress_mmap_libdeflate<W: Write + Send>(
    mmap: &Mmap,
    writer: &mut W,
    format: CompressionFormat,
) -> GzippyResult<u64> {
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
/// Strategies (in order of preference):
/// 1. BGZF-style (gzippy output): parallel libdeflate using embedded block sizes
/// 2. Single member: libdeflate (fastest, 30-50% faster than zlib)
/// 3. Large multi-member: speculative parallel decompression (rapidgzip-style)
/// 4. Small multi-member: sequential zlib-ng
fn decompress_gzip_libdeflate<W: Write + Send>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return Ok(0);
    }

    // Check for BGZF-style markers FIRST (gzippy output with embedded block sizes)
    // This check is fast (only looks at first header) and must come before
    // is_multi_member_quick which only scans 256KB - not enough for random data
    // where the first block can be >256KB
    if has_bgzf_markers(data) {
        return decompress_bgzf_parallel_prefetch(data, writer);
    }

    // Fast path: check if this is likely multi-member (from parallel compression)
    // Only scan first 256KB - if no second header found, use direct single-member path
    if !is_multi_member_quick(data) {
        return decompress_single_member_libdeflate(data, writer);
    }

    // Use the ultra-fast parallel decompressor
    // This uses ISA-L and block-level parallelism for maximum speed
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    match crate::ultra_decompress::decompress_ultra(data, writer, num_threads) {
        Ok(bytes) => Ok(bytes),
        Err(_) => {
            // Fallback to flate2 sequential
            decompress_multi_member_zlibng(data, writer)
        }
    }
}

/// Check if data has BGZF-style "GZ" markers in the first gzip header
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

                    // Use thread-local decompressor
                    DECOMPRESSOR.with(|decomp_cell| {
                        let mut decompressor = decomp_cell.borrow_mut();
                        let output = unsafe { &mut *slots[idx].data.get() };

                        output.clear();
                        // Capacity is pre-allocated based on ISIZE hint
                        let initial_size = output.capacity().max(64 * 1024);
                        output.resize(initial_size, 0);

                        loop {
                            match decompressor.gzip_decompress(block_data, output) {
                                Ok(size) => {
                                    output.truncate(size);
                                    break;
                                }
                                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                                    let new_size = output.len().saturating_mul(2);
                                    output.resize(new_size, 0);
                                    continue;
                                }
                                Err(_) => {
                                    output.clear();
                                    break;
                                }
                            }
                        }
                    });

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

/// Parallel decompression for BGZF-style files (gzippy output) - basic version
/// Uses embedded block size markers to find boundaries without inflating
///
/// Uses our custom scheduler for zero-overhead parallelism with streaming output.
#[allow(dead_code)]
fn decompress_bgzf_parallel<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
    use libdeflater::DecompressionError;
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

    // Output slot for each block
    struct Slot {
        ready: AtomicBool,
        data: UnsafeCell<Vec<u8>>,
    }
    unsafe impl Sync for Slot {}

    let slots: Vec<Slot> = (0..num_blocks)
        .map(|_| Slot {
            ready: AtomicBool::new(false),
            data: UnsafeCell::new(Vec::with_capacity(128 * 1024)),
        })
        .collect();

    let next_block = AtomicUsize::new(0);
    let mut total_bytes = 0u64;

    thread::scope(|scope| {
        // Spawn worker threads
        for _ in 0..num_threads {
            scope.spawn(|| {
                loop {
                    let idx = next_block.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_blocks {
                        break;
                    }

                    let (start, len) = blocks[idx];
                    let block_data = &data[start..start + len];

                    // Read ISIZE from trailer for buffer sizing
                    let isize_hint = if len >= 8 {
                        let trailer = &block_data[len - 4..];
                        u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]])
                            as usize
                    } else {
                        0
                    };

                    let initial_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
                        isize_hint + 1024
                    } else {
                        len.saturating_mul(4).max(64 * 1024)
                    };

                    // Use thread-local decompressor
                    DECOMPRESSOR.with(|decomp_cell| {
                        let mut decompressor = decomp_cell.borrow_mut();
                        let output = unsafe { &mut *slots[idx].data.get() };

                        output.clear();
                        if output.capacity() < initial_size {
                            output.reserve(initial_size - output.capacity());
                        }
                        output.resize(initial_size, 0);

                        loop {
                            match decompressor.gzip_decompress(block_data, output) {
                                Ok(size) => {
                                    output.truncate(size);
                                    break;
                                }
                                Err(DecompressionError::InsufficientSpace) => {
                                    let new_size = output.len().saturating_mul(2);
                                    output.resize(new_size, 0);
                                    continue;
                                }
                                Err(_) => {
                                    output.clear();
                                    break;
                                }
                            }
                        }
                    });

                    slots[idx].ready.store(true, Ordering::Release);
                }
            });
        }

        // Main thread: stream output in order
        for slot in slots.iter() {
            while !slot.ready.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let output = unsafe { &*slot.data.get() };
            if !output.is_empty() {
                writer.write_all(output).unwrap();
                total_bytes += output.len() as u64;
            }
        }
    });

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
fn decompress_single_member_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
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
                    return Err(GzippyError::invalid_argument(
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

/// Decompress zlib using libdeflate
fn decompress_zlib_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> GzippyResult<u64> {
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
                return Err(GzippyError::invalid_argument(
                    "zlib decompression failed".to_string(),
                ));
            }
        }
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
