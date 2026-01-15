use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use crossbeam_deque::{Injector, Stealer, Worker};
// use rayon::prelude::*; // Will use when needed

use crate::hardware_analysis::{SystemProfile, CompressionStrategy};
use crate::profiler::AdvancedProfiler;

#[derive(Clone)]
pub struct WorkItem<T> 
where T: Clone {
    pub data: T,
    pub priority: u8,
    pub estimated_work: usize, // In CPU cycles or time units
    pub affinity_hint: Option<usize>, // Preferred CPU core
}

pub struct WorkStealingPool<T: Send + Clone + 'static> {
    workers: Vec<Worker<WorkItem<T>>>,
    stealers: Vec<Stealer<WorkItem<T>>>,
    injector: Arc<Injector<WorkItem<T>>>,
    threads: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<ThreadPoolStats>,
    profiler: Option<Arc<AdvancedProfiler>>,
}

#[derive(Default)]
pub struct ThreadPoolStats {
    pub tasks_completed: AtomicUsize,
    pub tasks_stolen: AtomicUsize,
    pub total_work_time: AtomicUsize, // In microseconds
    pub total_idle_time: AtomicUsize,
    pub cache_misses: AtomicUsize,
}

impl<T: Send + Clone + 'static> WorkStealingPool<T> {
    pub fn new<F>(
        num_threads: usize,
        processor: F,
        _system_profile: &SystemProfile,
    ) -> Self 
    where
        F: Fn(T) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static + Clone,
    {
        let mut workers = Vec::with_capacity(num_threads);
        let mut stealers = Vec::with_capacity(num_threads);
        
        for _ in 0..num_threads {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }
        
        let injector = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(ThreadPoolStats::default());
        
        let mut threads = Vec::with_capacity(num_threads);
        
        // Create optimized threads with CPU affinity
        for (i, worker) in workers.into_iter().enumerate() {
            let stealers_clone = stealers.clone();
            let injector_clone = injector.clone();
            let shutdown_clone = shutdown.clone();
            let stats_clone = stats.clone();
            let processor_clone = processor.clone();
            
            let thread_handle = thread::Builder::new()
                .name(format!("rigz-worker-{}", i))
                .spawn(move || {
                    // Set CPU affinity if possible - don't need system_profile data here
                    set_thread_affinity(i);
                    
                    // Main work loop
                    run_worker_loop(
                        worker,
                        stealers_clone,
                        injector_clone,
                        shutdown_clone,
                        stats_clone,
                        processor_clone,
                        i,
                    );
                })
                .expect("Failed to create worker thread");
            
            threads.push(thread_handle);
        }
        
        Self {
            workers: Vec::new(), // Moved to threads
            stealers,
            injector,
            threads,
            shutdown,
            stats,
            profiler: None,
        }
    }
    
    pub fn with_profiler(mut self, profiler: Arc<AdvancedProfiler>) -> Self {
        self.profiler = Some(profiler);
        self
    }
    
    pub fn submit(&self, work_item: WorkItem<T>) {
        // Try to place in least loaded worker first
        let target_worker = self.find_least_loaded_worker();
        
        if let Some(stealer) = self.stealers.get(target_worker) {
            // Try direct injection to preferred worker
            if stealer.is_empty() {
                self.injector.push(work_item);
                return;
            }
        }
        
        // Fallback to global injector
        self.injector.push(work_item);
    }
    
    pub fn submit_batch(&self, work_items: Vec<WorkItem<T>>) {
        // Batch submission for better cache locality
        for chunk in work_items.chunks(64) { // Process in cache-friendly chunks
            for item in chunk {
                self.injector.push(item.clone());
            }
        }
    }
    
    pub fn wait_for_completion(&self) {
        // Wait until all work is done
        while !self.is_idle() {
            thread::sleep(Duration::from_millis(1));
        }
    }
    
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::Release);
        
        for handle in self.threads {
            handle.join().expect("Worker thread panicked");
        }
    }
    
    pub fn stats(&self) -> ThreadPoolStatsSummary {
        ThreadPoolStatsSummary {
            tasks_completed: self.stats.tasks_completed.load(Ordering::Relaxed),
            tasks_stolen: self.stats.tasks_stolen.load(Ordering::Relaxed),
            average_work_time: self.stats.total_work_time.load(Ordering::Relaxed) as f64 
                / self.stats.tasks_completed.load(Ordering::Relaxed).max(1) as f64,
            average_idle_time: self.stats.total_idle_time.load(Ordering::Relaxed) as f64 
                / self.threads.len() as f64,
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
        }
    }
    
    fn find_least_loaded_worker(&self) -> usize {
        // Simple heuristic: find worker with least items
        self.stealers
            .iter()
            .enumerate()
            .min_by_key(|(_, stealer)| stealer.len())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    fn is_idle(&self) -> bool {
        self.injector.is_empty() && 
        self.stealers.iter().all(|s| s.is_empty())
    }
}

#[derive(Debug)]
pub struct ThreadPoolStatsSummary {
    pub tasks_completed: usize,
    pub tasks_stolen: usize,
    pub average_work_time: f64,
    pub average_idle_time: f64,
    pub cache_misses: usize,
}

fn run_worker_loop<T, F>(
    worker: Worker<WorkItem<T>>,
    stealers: Vec<Stealer<WorkItem<T>>>,
    injector: Arc<Injector<WorkItem<T>>>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<ThreadPoolStats>,
    processor: F,
    worker_id: usize,
) 
where
    T: Send + Clone,
    F: Fn(T) -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
{
    let mut idle_start: Option<Instant> = None;
    let mut consecutive_steals = 0;
    
    while !shutdown.load(Ordering::Acquire) {
        let _work_start = Instant::now();
        
        // Try to get work from local queue first
        if let Some(work_item) = worker.pop() {
            if let Some(idle_time) = idle_start.take() {
                let idle_duration = idle_time.elapsed();
                stats.total_idle_time.fetch_add(
                    idle_duration.as_micros() as usize, 
                    Ordering::Relaxed
                );
            }
            
            // Process the work item
            let work_time = Instant::now();
            if let Err(_) = processor(work_item.data) {
                // Handle error (could log or retry)
            }
            
            let work_duration = work_time.elapsed();
            stats.total_work_time.fetch_add(
                work_duration.as_micros() as usize,
                Ordering::Relaxed
            );
            stats.tasks_completed.fetch_add(1, Ordering::Relaxed);
            
            consecutive_steals = 0;
            continue;
        }
        
        // Try to steal from global injector
        if let Some(work_item) = injector.steal().success() {
            if let Some(idle_time) = idle_start.take() {
                let idle_duration = idle_time.elapsed();
                stats.total_idle_time.fetch_add(
                    idle_duration.as_micros() as usize,
                    Ordering::Relaxed
                );
            }
            
            let work_time = Instant::now();
            if let Err(_) = processor(work_item.data) {
                // Handle error
            }
            
            let work_duration = work_time.elapsed();
            stats.total_work_time.fetch_add(
                work_duration.as_micros() as usize,
                Ordering::Relaxed
            );
            stats.tasks_completed.fetch_add(1, Ordering::Relaxed);
            stats.tasks_stolen.fetch_add(1, Ordering::Relaxed);
            
            consecutive_steals = 0;
            continue;
        }
        
        // Try to steal from other workers
        let mut found_work = false;
        for (i, stealer) in stealers.iter().enumerate() {
            if i == worker_id { continue; } // Don't steal from ourselves
            
            if let Some(work_item) = stealer.steal().success() {
                if let Some(idle_time) = idle_start.take() {
                    let idle_duration = idle_time.elapsed();
                    stats.total_idle_time.fetch_add(
                        idle_duration.as_micros() as usize,
                        Ordering::Relaxed
                    );
                }
                
                let work_time = Instant::now();
                if let Err(_) = processor(work_item.data) {
                    // Handle error
                }
                
                let work_duration = work_time.elapsed();
                stats.total_work_time.fetch_add(
                    work_duration.as_micros() as usize,
                    Ordering::Relaxed
                );
                stats.tasks_completed.fetch_add(1, Ordering::Relaxed);
                stats.tasks_stolen.fetch_add(1, Ordering::Relaxed);
                
                consecutive_steals += 1;
                found_work = true;
                break;
            }
        }
        
        if !found_work {
            // No work found, enter idle state
            if idle_start.is_none() {
                idle_start = Some(Instant::now());
            }
            
            // Adaptive backoff strategy
            let backoff_duration = calculate_backoff_duration(consecutive_steals);
            if backoff_duration > Duration::from_micros(1) {
                thread::sleep(backoff_duration);
            } else {
                // Very short spin
                for _ in 0..10 {
                    std::hint::spin_loop();
                }
            }
        }
    }
}

fn calculate_backoff_duration(consecutive_fails: usize) -> Duration {
    match consecutive_fails {
        0..=10 => Duration::from_nanos(0), // Immediate retry
        11..=50 => Duration::from_micros(1), // Very short sleep
        51..=200 => Duration::from_micros(10),
        201..=1000 => Duration::from_micros(100),
        _ => Duration::from_millis(1), // Longer sleep for persistent idle
    }
}

fn set_thread_affinity(_thread_id: usize) {
    // Try to set CPU affinity for better cache locality
    #[cfg(target_os = "linux")]
    {
        use core_affinity::{get_core_ids, set_for_current};
        
        if let Some(core_ids) = get_core_ids() {
            if !core_ids.is_empty() {
                let core_id = core_ids[thread_id % core_ids.len()];
                set_for_current(core_id);
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS doesn't allow setting CPU affinity directly
        // We can try to set thread priority instead
        use libc::qos_class_t;
        
        unsafe {
            let _ = libc::pthread_set_qos_class_self_np(
                qos_class_t::QOS_CLASS_USER_INITIATED,
                0
            );
        }
    }
}

// Specialized compression thread pool
pub struct CompressionThreadPool {
    pool: WorkStealingPool<CompressionTask>,
    strategy: CompressionStrategy,
}

#[derive(Clone)]
pub struct CompressionTask {
    pub chunk_id: usize,
    pub data: Vec<u8>,
    pub compression_level: u8,
    pub algorithm: CompressionAlgorithm,
}

#[derive(Clone, Debug)]
pub enum CompressionAlgorithm {
    Gzip,
    Deflate,
    Zstd,
    Lz4,
}

impl CompressionThreadPool {
    pub fn new(system_profile: &SystemProfile, compression_level: u8, file_size: usize) -> Self {
        let strategy = system_profile.recommend_compression_strategy(file_size, compression_level);
        
        let processor = move |task: CompressionTask| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            // Process compression task
            process_compression_task(task)
        };
        
        let pool = WorkStealingPool::new(
            strategy.thread_count,
            processor,
            system_profile,
        );
        
        Self { pool, strategy }
    }
    
    pub fn compress_chunk(&self, chunk_id: usize, data: Vec<u8>, compression_level: u8) {
        let estimated_work = estimate_compression_work(&data, compression_level);
        
        let task = CompressionTask {
            chunk_id,
            data,
            compression_level,
            algorithm: select_optimal_algorithm(compression_level, &self.strategy),
        };
        
        let work_item = WorkItem {
            data: task,
            priority: calculate_priority(compression_level),
            estimated_work,
            affinity_hint: Some(chunk_id % self.strategy.thread_count),
        };
        
        self.pool.submit(work_item);
    }
    
    pub fn wait_completion(&self) {
        self.pool.wait_for_completion();
    }
    
    pub fn shutdown(self) {
        self.pool.shutdown();
    }
    
    pub fn stats(&self) -> ThreadPoolStatsSummary {
        self.pool.stats()
    }
}

fn process_compression_task(task: CompressionTask) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Implement actual compression logic here
    match task.algorithm {
        CompressionAlgorithm::Gzip => {
            // Use gzip compression
            compress_with_gzip(&task.data, task.compression_level)?;
        },
        CompressionAlgorithm::Deflate => {
            // Use deflate compression
            compress_with_deflate(&task.data, task.compression_level)?;
        },
        CompressionAlgorithm::Zstd => {
            // Use zstd compression
            compress_with_zstd(&task.data, task.compression_level)?;
        },
        CompressionAlgorithm::Lz4 => {
            // Use lz4 compression
            compress_with_lz4(&task.data)?;
        },
    }
    
    Ok(())
}

fn select_optimal_algorithm(compression_level: u8, _strategy: &CompressionStrategy) -> CompressionAlgorithm {
    match compression_level {
        1..=3 => CompressionAlgorithm::Lz4, // Fast compression
        4..=6 => CompressionAlgorithm::Gzip, // Balanced
        7..=9 => CompressionAlgorithm::Zstd, // High compression
        _ => CompressionAlgorithm::Gzip, // Default
    }
}

fn calculate_priority(compression_level: u8) -> u8 {
    // Higher compression levels get higher priority
    compression_level
}

fn estimate_compression_work(data: &[u8], compression_level: u8) -> usize {
    // Estimate work based on data size and compression level
    data.len() * (compression_level as usize + 1)
}

// Placeholder compression functions
fn compress_with_gzip(_data: &[u8], _level: u8) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // Implement gzip compression
    Ok(Vec::new())
}

fn compress_with_deflate(_data: &[u8], _level: u8) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // Implement deflate compression
    Ok(Vec::new())
}

fn compress_with_zstd(_data: &[u8], _level: u8) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // Implement zstd compression
    Ok(Vec::new())
}

fn compress_with_lz4(_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // Implement lz4 compression
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_work_stealing_pool() {
        let system_profile = SystemProfile::detect();
        
        let processor = |data: usize| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            std::thread::sleep(Duration::from_millis(data as u64));
            Ok(())
        };
        
        let pool = WorkStealingPool::new(4, processor, &system_profile);
        
        // Submit some work
        for i in 0..10 {
            let work_item = WorkItem {
                data: i % 5 + 1, // 1-5 ms of work
                priority: 1,
                estimated_work: i * 1000,
                affinity_hint: Some(i % 4),
            };
            pool.submit(work_item);
        }
        
        pool.wait_for_completion();
        
        let stats = pool.stats();
        assert_eq!(stats.tasks_completed, 10);
        
        pool.shutdown();
    }
    
    #[test]
    fn test_compression_thread_pool() {
        let system_profile = SystemProfile::detect();
        let pool = CompressionThreadPool::new(&system_profile, 6, 1024 * 1024);
        
        // Submit compression tasks
        for i in 0..5 {
            let data = vec![i as u8; 1024];
            pool.compress_chunk(i, data, 6);
        }
        
        pool.wait_completion();
        
        let stats = pool.stats();
        assert_eq!(stats.tasks_completed, 5);
        
        pool.shutdown();
    }
}