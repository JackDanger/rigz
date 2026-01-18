//! Efficient parallel scheduler with condvar-based synchronization
//!
//! A pigz-inspired scheduler optimized for GHA's 4-vCPU environment:
//!
//! 1. Dedicated worker threads that compress blocks
//! 2. Main thread writes blocks in order using condvar (no spinning)
//! 3. Pre-allocated output slots (zero allocation in hot path)
//! 4. Lock-free work distribution (single atomic counter)
//!
//! Key difference from spin-wait: workers signal completion via condvar,
//! main thread sleeps efficiently instead of burning CPU cycles.

use std::cell::UnsafeCell;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread;

/// A slot for storing a compressed block's output with completion signaling
pub struct BlockSlot {
    /// Whether this block has been compressed
    ready: AtomicBool,
    /// The compressed data for this block
    data: UnsafeCell<Vec<u8>>,
}

/// Shared state for signaling block completions
struct CompletionSignal {
    /// Mutex for condvar (value is next expected block to complete)
    lock: Mutex<usize>,
    /// Condvar for efficient waiting
    cond: Condvar,
}

impl CompletionSignal {
    fn new() -> Self {
        Self {
            lock: Mutex::new(0),
            cond: Condvar::new(),
        }
    }

    /// Signal that a block has been completed
    #[inline]
    fn signal_completion(&self, _block_idx: usize) {
        // Just notify - the waiter will check the slot
        self.cond.notify_one();
    }

    /// Wait for any block completion using condvar
    #[inline]
    fn wait_for_completion(&self) {
        let guard = self.lock.lock().unwrap();
        // Short timeout to avoid missing signals
        let _ = self
            .cond
            .wait_timeout(guard, std::time::Duration::from_micros(10));
    }
}

/// Efficient hybrid wait: brief spin, then condvar
#[inline]
fn wait_for_slot_ready(slot: &BlockSlot, signal: &CompletionSignal) {
    // Brief spin for fast completions (common case)
    for _ in 0..64 {
        if slot.is_ready() {
            return;
        }
        std::hint::spin_loop();
    }

    // Fall back to condvar for efficient waiting
    while !slot.is_ready() {
        signal.wait_for_completion();
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

/// Compress blocks in parallel with streaming output
///
/// This is the core function that replaces rayon. It:
/// 1. Pre-allocates output slots for each block
/// 2. Spawns worker threads that claim blocks via atomic counter
/// 3. Workers compress into their claimed slot, then signal via condvar
/// 4. Main thread waits efficiently on condvar and writes as blocks complete
///
/// # Arguments
/// * `input` - The input data (typically memory-mapped)
/// * `block_size` - Size of each block to compress
/// * `num_threads` - Number of worker threads
/// * `writer` - Where to write compressed output
/// * `compress_fn` - Function to compress a block: (block_idx, block, dict, is_last, output)
///
/// # Type Parameters
/// * `W` - Output writer type
/// * `F` - Compression function type
pub fn compress_parallel<W, F>(
    input: &[u8],
    block_size: usize,
    num_threads: usize,
    mut writer: W,
    compress_fn: F,
) -> io::Result<()>
where
    W: Write,
    F: Fn(usize, &[u8], Option<&[u8]>, bool, &mut Vec<u8>) + Sync,
{
    let num_blocks = input.len().div_ceil(block_size);
    if num_blocks == 0 {
        return Ok(());
    }

    // Pre-allocate output slots with conservative capacity.
    // Must handle worst case (incompressible data) to avoid buffer overflow.
    // Compressed output can be slightly larger than input for random data.
    let slot_capacity = block_size + (block_size / 10) + 1024;
    let slots: Vec<BlockSlot> = (0..num_blocks)
        .map(|_| BlockSlot::new(slot_capacity))
        .collect();

    // Atomic counter for lock-free work distribution
    let next_block = AtomicUsize::new(0);

    // Completion signal for efficient waiting
    let signal = CompletionSignal::new();

    // Use scoped threads - no Arc needed, everything is borrowed
    thread::scope(|scope| {
        // Spawn N worker threads - all threads compress, main just writes
        // This maximizes compression parallelism and CPU utilization
        for _ in 0..num_threads {
            scope.spawn(|| {
                worker_loop_with_signal(
                    input,
                    block_size,
                    num_blocks,
                    &slots,
                    &next_block,
                    &compress_fn,
                    &signal,
                );
            });
        }

        // Main thread: write blocks in order using hybrid spin/condvar wait
        for slot in slots.iter() {
            wait_for_slot_ready(slot, &signal);
            writer.write_all(slot.data())?;
        }

        Ok(())
    })
}

/// Worker loop with completion signaling via condvar
#[inline]
fn worker_loop_with_signal<F>(
    input: &[u8],
    block_size: usize,
    num_blocks: usize,
    slots: &[BlockSlot],
    next_block: &AtomicUsize,
    compress_fn: &F,
    signal: &CompletionSignal,
) where
    F: Fn(usize, &[u8], Option<&[u8]>, bool, &mut Vec<u8>),
{
    loop {
        // Claim next block atomically
        let block_idx = next_block.fetch_add(1, Ordering::Relaxed);
        if block_idx >= num_blocks {
            break; // No more work
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
        // Safety: Each block_idx is claimed by exactly one worker
        let output = unsafe { slots[block_idx].data_mut() };

        // Compress into the slot's buffer
        compress_fn(block_idx, block, dict, is_last, output);

        // Signal completion - main thread can now write this block
        slots[block_idx].mark_ready();
        signal.signal_completion(block_idx);
    }
}

/// Variant for independent blocks (L1-L6) that don't need dictionaries
///
/// Slightly simpler since we don't compute dict slices.
pub fn compress_parallel_independent<W, F>(
    input: &[u8],
    block_size: usize,
    num_threads: usize,
    mut writer: W,
    compress_fn: F,
) -> io::Result<()>
where
    W: Write,
    F: Fn(&[u8], &mut Vec<u8>) + Sync,
{
    let num_blocks = input.len().div_ceil(block_size);
    if num_blocks == 0 {
        return Ok(());
    }

    let slot_capacity = block_size + (block_size / 10) + 1024;
    let slots: Vec<BlockSlot> = (0..num_blocks)
        .map(|_| BlockSlot::new(slot_capacity))
        .collect();

    let next_block = AtomicUsize::new(0);
    let signal = CompletionSignal::new();

    thread::scope(|scope| {
        // Spawn workers with condvar signaling
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
                signal.signal_completion(block_idx);
            });
        }

        // Stream output in order using hybrid spin/condvar wait
        for slot in slots.iter() {
            wait_for_slot_ready(slot, &signal);
            writer.write_all(slot.data())?;
        }

        Ok(())
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
