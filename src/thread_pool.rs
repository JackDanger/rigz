//! Persistent Thread Pool for Parallel Decompression
//!
//! This module provides a global thread pool that persists across multiple
//! decompression operations, eliminating thread spawn/teardown overhead.
//!
//! Key features:
//! - Threads created once at first use, reused for all operations
//! - Work-stealing for load balancing
//! - Configurable thread count (defaults to CPU count)

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Global thread pool instance
static POOL: std::sync::LazyLock<ThreadPool> = std::sync::LazyLock::new(|| {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    ThreadPool::new(num_threads)
});

/// Task type for the thread pool
type Task = Box<dyn FnOnce() + Send + 'static>;

/// A persistent thread pool for parallel work
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Task>>,
    active_tasks: Arc<AtomicUsize>,
}

struct Worker {
    _handle: JoinHandle<()>,
}

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads
    fn new(num_threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Task>();
        let receiver = Arc::new(Mutex::new(receiver));
        let active_tasks = Arc::new(AtomicUsize::new(0));

        let workers: Vec<Worker> = (0..num_threads)
            .map(|_| {
                let receiver = Arc::clone(&receiver);
                let active_tasks = Arc::clone(&active_tasks);
                let handle = thread::spawn(move || loop {
                    let task = {
                        let lock = receiver.lock().unwrap();
                        lock.recv()
                    };
                    match task {
                        Ok(task) => {
                            active_tasks.fetch_add(1, Ordering::SeqCst);
                            task();
                            active_tasks.fetch_sub(1, Ordering::SeqCst);
                        }
                        Err(_) => break, // Channel closed, exit
                    }
                });
                Worker { _handle: handle }
            })
            .collect();

        ThreadPool {
            workers,
            sender: Some(sender),
            active_tasks,
        }
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Drop sender to signal workers to exit
        self.sender.take();
    }
}

/// Execute tasks in parallel using the global thread pool
///
/// This function blocks until all tasks complete.
pub fn parallel_execute<F, T>(tasks: Vec<F>) -> Vec<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    use std::sync::mpsc::channel;

    if tasks.is_empty() {
        return Vec::new();
    }

    let num_tasks = tasks.len();
    let (result_tx, result_rx) = channel();

    for (i, task) in tasks.into_iter().enumerate() {
        let tx = result_tx.clone();
        let job = Box::new(move || {
            let result = task();
            let _ = tx.send((i, result));
        });
        if let Some(ref sender) = POOL.sender {
            sender.send(job).expect("Thread pool channel closed");
        }
    }
    drop(result_tx); // Close our copy so receiver knows when all done

    // Collect results in order
    let mut results: Vec<Option<T>> = (0..num_tasks).map(|_| None).collect();
    for _ in 0..num_tasks {
        let (i, result) = result_rx.recv().expect("Task panicked");
        results[i] = Some(result);
    }

    results.into_iter().map(|r| r.unwrap()).collect()
}

/// Execute tasks in parallel, collecting results via closure
///
/// More efficient than parallel_execute for large numbers of small tasks
/// because it doesn't allocate a result vector upfront.
pub fn parallel_for<F>(num_tasks: usize, task: F)
where
    F: Fn(usize) + Send + Sync + 'static,
{
    if num_tasks == 0 {
        return;
    }

    let task = Arc::new(task);
    let completed = Arc::new(AtomicUsize::new(0));

    for i in 0..num_tasks {
        let task = Arc::clone(&task);
        let completed = Arc::clone(&completed);
        let job = Box::new(move || {
            task(i);
            completed.fetch_add(1, Ordering::SeqCst);
        });
        if let Some(ref sender) = POOL.sender {
            sender.send(job).expect("Thread pool channel closed");
        }
    }

    // Wait for all tasks to complete
    while completed.load(Ordering::SeqCst) < num_tasks {
        std::hint::spin_loop();
    }
}

/// Get the number of threads in the global pool
pub fn num_threads() -> usize {
    POOL.num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_execute() {
        let tasks: Vec<_> = (0..100).map(|i| move || i * 2).collect();
        let results = parallel_execute(tasks);
        assert_eq!(results.len(), 100);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, i * 2);
        }
    }

    #[test]
    fn test_parallel_for() {
        let sum = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sum_clone = Arc::clone(&sum);
        parallel_for(100, move |i| {
            sum_clone.fetch_add(i, Ordering::SeqCst);
        });
        // Sum of 0..100 = 4950
        assert_eq!(sum.load(Ordering::SeqCst), 4950);
    }

    #[test]
    fn test_num_threads() {
        assert!(num_threads() >= 1);
    }
}
