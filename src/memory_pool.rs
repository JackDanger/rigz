use std::sync::{Arc, Mutex, Condvar};
use std::collections::VecDeque;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::hardware_analysis::SystemProfile;

/// High-performance memory pool for zero-copy operations
pub struct ZeroCopyMemoryPool {
    pools: Vec<BufferPool>,
    system_profile: SystemProfile,
    total_allocated: AtomicUsize,
    peak_usage: AtomicUsize,
}

#[derive(Debug)]
struct BufferPool {
    buffer_size: usize,
    available_buffers: Arc<Mutex<VecDeque<AlignedBuffer>>>,
    buffer_count: AtomicUsize,
    max_buffers: usize,
    allocation_condvar: Arc<Condvar>,
}

#[derive(Debug)]
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    size: usize,
    alignment: usize,
    pool_index: usize,
}

#[derive(Debug)]
pub struct ManagedBuffer {
    buffer: AlignedBuffer,
    pool: Arc<BufferPool>,
    returned: bool,
}

impl ZeroCopyMemoryPool {
    pub fn new(system_profile: SystemProfile) -> Self {
        let pools = Self::create_size_pools(&system_profile);
        
        Self {
            pools,
            system_profile,
            total_allocated: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
        }
    }
    
    fn create_size_pools(system_profile: &SystemProfile) -> Vec<BufferPool> {
        let cache_line_size = system_profile.cpu.cache_line_size;
        let available_memory = system_profile.memory.available_memory;
        
        // Create pools for different buffer sizes
        let pool_configs = [
            (4 * 1024, 256),           // 4KB buffers, up to 256 (1MB total)
            (64 * 1024, 128),          // 64KB buffers, up to 128 (8MB total)
            (256 * 1024, 64),          // 256KB buffers, up to 64 (16MB total)
            (1024 * 1024, 32),         // 1MB buffers, up to 32 (32MB total)
            (4 * 1024 * 1024, 16),     // 4MB buffers, up to 16 (64MB total)
            (16 * 1024 * 1024, 8),     // 16MB buffers, up to 8 (128MB total)
        ];
        
        pool_configs.into_iter()
            .enumerate()
            .map(|(index, (size, max_count))| {
                // Adjust max count based on available memory
                let adjusted_max = if available_memory < 2 * 1024 * 1024 * 1024 {
                    max_count / 2 // Reduce for systems with < 2GB available
                } else {
                    max_count
                };
                
                BufferPool::new(index, size, adjusted_max, cache_line_size)
            })
            .collect()
    }
    
    pub fn get_buffer(&self, requested_size: usize) -> Option<ManagedBuffer> {
        // Find the smallest pool that can accommodate the request
        let pool_index = self.pools.iter()
            .position(|pool| pool.buffer_size >= requested_size)?;
        
        let pool = &self.pools[pool_index];
        let buffer = pool.get_buffer()?;
        
        // Update memory tracking
        self.total_allocated.fetch_add(buffer.size, Ordering::Relaxed);
        let current_usage = self.total_allocated.load(Ordering::Relaxed);
        self.peak_usage.fetch_max(current_usage, Ordering::Relaxed);
        
        Some(ManagedBuffer {
            buffer,
            pool: Arc::new(pool.clone()),
            returned: false,
        })
    }
    
    pub fn get_aligned_buffer(&self, size: usize, alignment: usize) -> Option<ManagedBuffer> {
        // For custom alignment requirements, allocate directly
        if alignment > self.system_profile.cpu.cache_line_size {
            return Self::allocate_custom_buffer(size, alignment);
        }
        
        self.get_buffer(size)
    }
    
    fn allocate_custom_buffer(size: usize, alignment: usize) -> Option<ManagedBuffer> {
        let layout = Layout::from_size_align(size, alignment).ok()?;
        
        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return None;
            }
            
            let buffer = AlignedBuffer {
                ptr: NonNull::new_unchecked(ptr),
                size,
                alignment,
                pool_index: usize::MAX, // Special marker for custom buffers
            };
            
            // Create a dummy pool for cleanup
            let dummy_pool = BufferPool::dummy();
            
            Some(ManagedBuffer {
                buffer,
                pool: Arc::new(dummy_pool),
                returned: false,
            })
        }
    }
    
    pub fn prefault_buffers(&self, count: usize) {
        // Pre-allocate and touch memory pages to avoid page faults during compression
        for pool in &self.pools {
            pool.prefault_buffers(count.min(pool.max_buffers / 4));
        }
    }
    
    pub fn memory_stats(&self) -> MemoryStats {
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let peak_usage = self.peak_usage.load(Ordering::Relaxed);
        
        let pool_stats: Vec<PoolStats> = self.pools.iter()
            .enumerate()
            .map(|(index, pool)| PoolStats {
                pool_index: index,
                buffer_size: pool.buffer_size,
                allocated_count: pool.buffer_count.load(Ordering::Relaxed),
                max_buffers: pool.max_buffers,
                total_bytes: pool.buffer_count.load(Ordering::Relaxed) * pool.buffer_size,
            })
            .collect();
        
        MemoryStats {
            total_allocated,
            peak_usage,
            pool_stats,
        }
    }
    
    pub fn compact(&self) {
        // Free unused buffers to reduce memory usage
        for pool in &self.pools {
            pool.compact();
        }
    }
}

impl BufferPool {
    fn new(_index: usize, buffer_size: usize, max_buffers: usize, _alignment: usize) -> Self {
        Self {
            buffer_size,
            available_buffers: Arc::new(Mutex::new(VecDeque::new())),
            buffer_count: AtomicUsize::new(0),
            max_buffers,
            allocation_condvar: Arc::new(Condvar::new()),
        }
    }
    
    fn dummy() -> Self {
        Self {
            buffer_size: 0,
            available_buffers: Arc::new(Mutex::new(VecDeque::new())),
            buffer_count: AtomicUsize::new(0),
            max_buffers: 0,
            allocation_condvar: Arc::new(Condvar::new()),
        }
    }
    
    fn get_buffer(&self) -> Option<AlignedBuffer> {
        // Try to get from available buffers first
        {
            let mut available = self.available_buffers.lock().ok()?;
            if let Some(buffer) = available.pop_front() {
                return Some(buffer);
            }
        }
        
        // Allocate new buffer if under limit
        let current_count = self.buffer_count.load(Ordering::Relaxed);
        if current_count < self.max_buffers {
            if let Some(buffer) = self.allocate_new_buffer() {
                self.buffer_count.fetch_add(1, Ordering::Relaxed);
                return Some(buffer);
            }
        }
        
        // Wait for a buffer to become available (with timeout)
        let available = self.available_buffers.lock().ok()?;
        let (mut available, timeout_result) = self.allocation_condvar
            .wait_timeout(available, std::time::Duration::from_millis(100))
            .ok()?;
        
        if !timeout_result.timed_out() {
            available.pop_front()
        } else {
            None
        }
    }
    
    fn allocate_new_buffer(&self) -> Option<AlignedBuffer> {
        let alignment = 64; // Cache line alignment
        let layout = Layout::from_size_align(self.buffer_size, alignment).ok()?;
        
        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return None;
            }
            
            // Touch all pages to prefault them
            let pages = self.buffer_size / 4096 + 1;
            for i in 0..pages {
                let offset = (i * 4096).min(self.buffer_size - 1);
                ptr.add(offset).write_volatile(0);
            }
            
            Some(AlignedBuffer {
                ptr: NonNull::new_unchecked(ptr),
                size: self.buffer_size,
                alignment,
                pool_index: 0, // Will be set by caller
            })
        }
    }
    
    fn return_buffer(&self, buffer: AlignedBuffer) {
        if let Ok(mut available) = self.available_buffers.lock() {
            available.push_back(buffer);
            self.allocation_condvar.notify_one();
        }
    }
    
    fn prefault_buffers(&self, count: usize) {
        let mut allocated = Vec::new();
        
        // Allocate buffers
        for _ in 0..count {
            if let Some(buffer) = self.get_buffer() {
                // Touch the memory to ensure it's paged in
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(buffer.ptr.as_ptr(), buffer.size);
                    slice.iter_mut().step_by(4096).for_each(|byte| *byte = 0);
                }
                allocated.push(buffer);
            } else {
                break;
            }
        }
        
        // Return buffers to pool
        if let Ok(mut available) = self.available_buffers.lock() {
            for buffer in allocated {
                available.push_back(buffer);
            }
        }
    }
    
    fn compact(&self) {
        // Remove excess buffers, keeping only a reasonable number
        let target_size = self.max_buffers / 4;
        
        if let Ok(mut available) = self.available_buffers.lock() {
            while available.len() > target_size {
                if let Some(buffer) = available.pop_back() {
                    unsafe {
                        let layout = Layout::from_size_align_unchecked(buffer.size, buffer.alignment);
                        dealloc(buffer.ptr.as_ptr(), layout);
                    }
                    self.buffer_count.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
    }
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            buffer_size: self.buffer_size,
            available_buffers: self.available_buffers.clone(),
            buffer_count: AtomicUsize::new(self.buffer_count.load(Ordering::Relaxed)),
            max_buffers: self.max_buffers,
            allocation_condvar: self.allocation_condvar.clone(),
        }
    }
}

impl AlignedBuffer {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
    
    pub fn size(&self) -> usize {
        self.size
    }
    
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    pub fn is_aligned(&self, alignment: usize) -> bool {
        self.ptr.as_ptr() as usize % alignment == 0
    }
}

impl ManagedBuffer {
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }
    
    pub fn size(&self) -> usize {
        self.buffer.size()
    }
    
    pub fn resize(&mut self, new_size: usize) -> Result<(), String> {
        if new_size > self.buffer.size {
            return Err(format!("Cannot resize buffer to {} bytes, maximum is {}", new_size, self.buffer.size));
        }
        // Note: This doesn't actually resize the allocation, just changes the usable size
        // Real implementation would need more sophisticated handling
        Ok(())
    }
    
    pub fn zero_fill(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.buffer.ptr.as_ptr(), 0, self.buffer.size);
        }
    }
    
    pub fn copy_from_slice(&mut self, src: &[u8]) -> Result<(), String> {
        if src.len() > self.buffer.size {
            return Err(format!("Source slice too large: {} > {}", src.len(), self.buffer.size));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffer.ptr.as_ptr(), src.len());
        }
        
        Ok(())
    }
}

impl Drop for ManagedBuffer {
    fn drop(&mut self) {
        if !self.returned {
            if self.buffer.pool_index == usize::MAX {
                // Custom allocated buffer - deallocate directly
                unsafe {
                    let layout = Layout::from_size_align_unchecked(self.buffer.size, self.buffer.alignment);
                    dealloc(self.buffer.ptr.as_ptr(), layout);
                }
            } else {
                // Return to pool
                self.pool.return_buffer(std::mem::replace(&mut self.buffer, AlignedBuffer {
                    ptr: NonNull::dangling(),
                    size: 0,
                    alignment: 1,
                    pool_index: 0,
                }));
            }
            self.returned = true;
        }
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}
unsafe impl Send for ManagedBuffer {}
unsafe impl Sync for ManagedBuffer {}

#[derive(Debug)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub pool_stats: Vec<PoolStats>,
}

#[derive(Debug)]
pub struct PoolStats {
    pub pool_index: usize,
    pub buffer_size: usize,
    pub allocated_count: usize,
    pub max_buffers: usize,
    pub total_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_basic() {
        let system_profile = SystemProfile::detect();
        let pool = ZeroCopyMemoryPool::new(system_profile);
        
        // Test getting a buffer
        let buffer = pool.get_buffer(64 * 1024).expect("Failed to get buffer");
        assert!(buffer.size() >= 64 * 1024);
        assert!(buffer.as_slice().len() >= 64 * 1024);
        
        // Test buffer alignment
        assert!(buffer.buffer.is_aligned(64));
    }
    
    #[test]
    fn test_buffer_reuse() {
        let system_profile = SystemProfile::detect();
        let pool = ZeroCopyMemoryPool::new(system_profile);
        
        let initial_stats = pool.memory_stats();
        
        // Get and immediately drop a buffer
        {
            let _buffer = pool.get_buffer(64 * 1024).expect("Failed to get buffer");
        }
        
        // Get another buffer - should reuse the first one
        let _buffer2 = pool.get_buffer(64 * 1024).expect("Failed to get second buffer");
        
        let final_stats = pool.memory_stats();
        
        // Should not have allocated more memory for the second buffer
        assert_eq!(initial_stats.total_allocated, final_stats.total_allocated);
    }
    
    #[test]
    fn test_zero_copy_operations() {
        let system_profile = SystemProfile::detect();
        let pool = ZeroCopyMemoryPool::new(system_profile);
        
        let mut buffer = pool.get_buffer(1024).expect("Failed to get buffer");
        
        // Test zero fill
        buffer.zero_fill();
        assert!(buffer.as_slice().iter().all(|&b| b == 0));
        
        // Test copy from slice
        let test_data = b"Hello, zero-copy world!";
        buffer.copy_from_slice(test_data).expect("Failed to copy data");
        
        assert_eq!(&buffer.as_slice()[..test_data.len()], test_data);
    }
    
    #[test]
    fn test_custom_alignment() {
        let system_profile = SystemProfile::detect();
        let pool = ZeroCopyMemoryPool::new(system_profile);
        
        let buffer = pool.get_aligned_buffer(1024, 256).expect("Failed to get aligned buffer");
        assert!(buffer.buffer.is_aligned(256));
    }
    
    #[test]
    fn test_memory_stats() {
        let system_profile = SystemProfile::detect();
        let pool = ZeroCopyMemoryPool::new(system_profile);
        
        let _buffer1 = pool.get_buffer(4 * 1024);
        let _buffer2 = pool.get_buffer(64 * 1024);
        
        let stats = pool.memory_stats();
        assert!(stats.total_allocated > 0);
        assert!(stats.pool_stats.len() > 0);
        
        println!("Memory stats: {:#?}", stats);
    }
}