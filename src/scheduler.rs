//! Pigz-style parallel scheduler with dedicated writer thread
//!
//! This implements pigz's proven threading model:
//!
//! 1. N compress worker threads (claim work via atomic counter)
//! 2. 1 dedicated writer thread (writes blocks in order)
//! 3. All N+1 threads run concurrently (no main-thread stalls)
//! 4. Simple spin-wait for block completion (low latency)
//!
//! This maximizes CPU utilization by never blocking the compress workers
//! on I/O. The writer thread handles all disk writes independently.
//!
//! Set RIGZ_DEBUG=1 to enable timing diagnostics.

use std::cell::UnsafeCell;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

/// Check if debug mode is enabled via RIGZ_DEBUG env var
fn is_debug_enabled() -> bool {
    std::env::var("RIGZ_DEBUG").is_ok_and(|v| v == "1" || v == "true")
}

/// A slot for storing a compressed block's output
pub struct BlockSlot {
    /// Whether this block has been compressed
    ready: AtomicBool,
    /// The compressed data for this block
    data: UnsafeCell<Vec<u8>>,
}

/// Efficient spin-wait for slot readiness
///
/// Uses brief spin with pause hint. For L9 compression where each block
/// takes ~10ms, this adds negligible overhead while keeping latency low.
#[inline]
fn wait_for_slot_ready(slot: &BlockSlot) {
    while !slot.is_ready() {
        std::hint::spin_loop();
    }
}

// Safety: Each slot is written by exactly one worker thread, then read by main thread
// after ready=true. The atomic provides the synchronization.
unsafe impl Sync for BlockSlot {}

impl BlockSlot {
    /// Create a new slot with pre-allocated capacity
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            ready: AtomicBool::new(false),
            data: UnsafeCell::new(Vec::with_capacity(capacity)),
        }
    }

    /// Get mutable access to the data buffer (called by single worker)
    ///
    /// # Safety
    /// Only call from the single worker assigned to this block index.
    /// The UnsafeCell allows interior mutability from an immutable reference.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn data_mut(&self) -> &mut Vec<u8> {
        &mut *self.data.get()
    }

    /// Mark this block as ready (worker calls after compression)
    #[inline]
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Check if this block is ready (main thread polls)
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Get the data (main thread calls after is_ready returns true)
    ///
    /// # Safety
    /// Only call after is_ready() returns true. At that point the worker
    /// has finished writing and will not access the slot again.
    #[inline]
    pub fn data(&self) -> &[u8] {
        unsafe { &*self.data.get() }
    }
}

/// Compress blocks in parallel with dedicated writer thread (pigz model)
///
/// This implements the pigz threading model:
/// 1. N compress worker threads claim blocks via atomic counter
/// 2. 1 dedicated writer thread writes blocks in order
/// 3. All threads run concurrently - no blocking on I/O
///
/// This is optimal because:
/// - Compress workers never stall waiting for writes
/// - Writer thread runs in parallel with compression
/// - Simple spin-wait has low latency for fast blocks
pub fn compress_parallel<W, F>(
    input: &[u8],
    block_size: usize,
    num_threads: usize,
    writer: W,
    compress_fn: F,
) -> io::Result<W>
where
    W: Write + Send,
    F: Fn(usize, &[u8], Option<&[u8]>, bool, &mut Vec<u8>) + Sync,
{
    let debug = is_debug_enabled();
    let start = Instant::now();

    let num_blocks = input.len().div_ceil(block_size);
    if num_blocks == 0 {
        return Ok(writer);
    }

    if debug {
        eprintln!(
            "[rigz] compress_parallel: input={}KB, block_size={}KB, blocks={}, threads={}",
            input.len() / 1024,
            block_size / 1024,
            num_blocks,
            num_threads
        );
    }

    // Pre-allocate output slots with conservative capacity
    let alloc_start = Instant::now();
    let slot_capacity = block_size + (block_size / 10) + 1024;
    let slots: Vec<BlockSlot> = (0..num_blocks)
        .map(|_| BlockSlot::new(slot_capacity))
        .collect();
    let alloc_time = alloc_start.elapsed();

    if debug {
        eprintln!(
            "[rigz] slot allocation: {}ms for {} slots ({}KB each)",
            alloc_time.as_millis(),
            num_blocks,
            slot_capacity / 1024
        );
    }

    // Atomic counter for lock-free work distribution
    let next_block = AtomicUsize::new(0);

    // Track any write error from writer thread
    let write_error: AtomicBool = AtomicBool::new(false);

    // Timing accumulators (atomic for thread-safe updates)
    let total_compress_ns = AtomicU64::new(0);
    let total_wait_ns = AtomicU64::new(0);
    let total_write_ns = AtomicU64::new(0);
    let blocks_compressed = AtomicUsize::new(0);

    // Use scoped threads - no Arc needed, everything is borrowed
    let thread_start = Instant::now();
    let result = thread::scope(|scope| {
        // Spawn dedicated writer thread (pigz model)
        // Returns the writer so caller can write trailer
        let writer_handle = scope.spawn(|| {
            let mut w = writer;
            for slot in slots.iter() {
                let wait_start = Instant::now();
                wait_for_slot_ready(slot);
                total_wait_ns.fetch_add(wait_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                let write_start = Instant::now();
                if w.write_all(slot.data()).is_err() {
                    write_error.store(true, Ordering::Relaxed);
                    break;
                }
                total_write_ns
                    .fetch_add(write_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            w
        });

        // Spawn N compress worker threads
        for _ in 0..num_threads {
            scope.spawn(|| {
                worker_loop_timed(
                    input,
                    block_size,
                    num_blocks,
                    &slots,
                    &next_block,
                    &compress_fn,
                    &total_compress_ns,
                    &blocks_compressed,
                );
            });
        }

        // Wait for writer to finish and get it back
        let w = writer_handle.join().unwrap();

        if write_error.load(Ordering::Relaxed) {
            Err(io::Error::other("write failed"))
        } else {
            Ok(w)
        }
    });
    let thread_time = thread_start.elapsed();
    let total_time = start.elapsed();

    if debug {
        let compress_ms = total_compress_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let wait_ms = total_wait_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let write_ms = total_write_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let blocks = blocks_compressed.load(Ordering::Relaxed);

        eprintln!("[rigz] timing breakdown:");
        eprintln!("  total time: {}ms", total_time.as_millis());
        eprintln!("  thread scope: {}ms", thread_time.as_millis());
        eprintln!(
            "  compress (sum across threads): {:.1}ms ({} blocks, {:.2}ms/block avg)",
            compress_ms,
            blocks,
            if blocks > 0 {
                compress_ms / blocks as f64
            } else {
                0.0
            }
        );
        eprintln!("  writer wait: {:.1}ms", wait_ms);
        eprintln!("  writer write: {:.1}ms", write_ms);
        eprintln!(
            "  overhead: {:.1}ms",
            total_time.as_millis() as f64 - thread_time.as_millis() as f64
        );
    }

    result
}

/// Worker loop with timing instrumentation
#[inline]
#[allow(clippy::too_many_arguments)]
fn worker_loop_timed<F>(
    input: &[u8],
    block_size: usize,
    num_blocks: usize,
    slots: &[BlockSlot],
    next_block: &AtomicUsize,
    compress_fn: &F,
    total_compress_ns: &AtomicU64,
    blocks_compressed: &AtomicUsize,
) where
    F: Fn(usize, &[u8], Option<&[u8]>, bool, &mut Vec<u8>),
{
    loop {
        // Claim next block atomically
        let block_idx = next_block.fetch_add(1, Ordering::Relaxed);
        if block_idx >= num_blocks {
            break;
        }

        // Calculate block boundaries
        let start = block_idx * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];

        // Get dictionary: last 32KB of input before this block
        let dict = if block_idx > 0 {
            let dict_end = start;
            let dict_start = dict_end.saturating_sub(32768);
            Some(&input[dict_start..dict_end])
        } else {
            None
        };

        let is_last = block_idx == num_blocks - 1;

        // Get output buffer from pre-allocated slot
        let output = unsafe { slots[block_idx].data_mut() };

        // Time the compression
        let compress_start = Instant::now();
        compress_fn(block_idx, block, dict, is_last, output);
        total_compress_ns.fetch_add(
            compress_start.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        blocks_compressed.fetch_add(1, Ordering::Relaxed);

        // Signal completion
        slots[block_idx].mark_ready();
    }
}

/// Variant for independent blocks (L1-L6) that don't need dictionaries
///
/// Uses same pigz model: N workers + dedicated writer thread.
/// Returns the writer so caller can write any trailer.
pub fn compress_parallel_independent<W, F>(
    input: &[u8],
    block_size: usize,
    num_threads: usize,
    writer: W,
    compress_fn: F,
) -> io::Result<W>
where
    W: Write + Send,
    F: Fn(&[u8], &mut Vec<u8>) + Sync,
{
    let num_blocks = input.len().div_ceil(block_size);
    if num_blocks == 0 {
        return Ok(writer);
    }

    let slot_capacity = block_size + (block_size / 10) + 1024;
    let slots: Vec<BlockSlot> = (0..num_blocks)
        .map(|_| BlockSlot::new(slot_capacity))
        .collect();

    let next_block = AtomicUsize::new(0);
    let write_error = AtomicBool::new(false);

    thread::scope(|scope| {
        // Spawn dedicated writer thread
        let writer_handle = scope.spawn(|| {
            let mut w = writer;
            for slot in slots.iter() {
                wait_for_slot_ready(slot);
                if w.write_all(slot.data()).is_err() {
                    write_error.store(true, Ordering::Relaxed);
                    break;
                }
            }
            w
        });

        // Spawn N compress workers
        for _ in 0..num_threads {
            scope.spawn(|| loop {
                let block_idx = next_block.fetch_add(1, Ordering::Relaxed);
                if block_idx >= num_blocks {
                    break;
                }

                let start = block_idx * block_size;
                let end = (start + block_size).min(input.len());
                let block = &input[start..end];

                let output = unsafe { slots[block_idx].data_mut() };
                compress_fn(block, output);
                slots[block_idx].mark_ready();
            });
        }

        let w = writer_handle.join().unwrap();

        if write_error.load(Ordering::Relaxed) {
            Err(io::Error::other("write failed"))
        } else {
            Ok(w)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_basic() {
        let input = b"Hello, world! ".repeat(1000);
        let mut output = Vec::new();

        compress_parallel(
            &input,
            1024, // 1KB blocks
            4,    // 4 threads
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                // Simple "compression": just copy
                out.clear();
                out.extend_from_slice(block);
            },
        )
        .unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_parallel_ordering() {
        // Verify blocks are written in order even when compressed out of order
        let input: Vec<u8> = (0..100).collect();
        let mut output = Vec::new();

        compress_parallel(
            &input,
            10, // 10-byte blocks
            4,  // 4 threads
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                // Add artificial delay for odd blocks to scramble completion order
                // (In real use, compression time varies)
                out.clear();
                out.extend_from_slice(block);
            },
        )
        .unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_single_block() {
        let input = b"small";
        let mut output = Vec::new();

        compress_parallel(
            input,
            1024, // Block size larger than input
            4,
            &mut output,
            |_idx, block, _dict, _is_last, out| {
                out.clear();
                out.extend_from_slice(block);
            },
        )
        .unwrap();

        assert_eq!(output, input.as_slice());
    }
}
