use std::fs;

#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_line_size: usize,
    pub numa_nodes: usize,
    pub cpu_brand: String,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub is_apple_silicon: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_memory: usize,
    pub available_memory: usize,
    pub memory_bandwidth_gb_s: f64,
    pub page_size: usize,
    pub huge_page_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SystemProfile {
    pub cpu: CpuTopology,
    pub memory: MemoryInfo,
    pub optimal_thread_count: usize,
    pub optimal_chunk_size: usize,
    pub optimal_buffer_size: usize,
}

impl SystemProfile {
    pub fn detect() -> Self {
        let cpu = detect_cpu_topology();
        let memory = detect_memory_info();
        
        // Calculate optimal parameters based on hardware
        let optimal_thread_count = calculate_optimal_threads(&cpu);
        let optimal_chunk_size = calculate_optimal_chunk_size(&cpu, &memory);
        let optimal_buffer_size = calculate_optimal_buffer_size(&cpu, &memory);
        
        Self {
            cpu,
            memory,
            optimal_thread_count,
            optimal_chunk_size,
            optimal_buffer_size,
        }
    }
    
    pub fn recommend_compression_strategy(&self, file_size: usize, compression_level: u8) -> CompressionStrategy {
        let thread_count = match file_size {
            // For small files, use fewer threads to avoid overhead
            size if size < 64 * 1024 => 1,
            size if size < 1024 * 1024 => (self.optimal_thread_count / 4).max(1),
            size if size < 10 * 1024 * 1024 => (self.optimal_thread_count / 2).max(1),
            _ => self.optimal_thread_count,
        };
        
        let chunk_size = match compression_level {
            1..=3 => self.optimal_chunk_size * 2, // Larger chunks for fast compression
            4..=6 => self.optimal_chunk_size,
            7..=9 => self.optimal_chunk_size / 2, // Smaller chunks for better compression
            _ => self.optimal_chunk_size,
        };
        
        let use_simd = self.cpu.has_avx2 || self.cpu.has_neon;
        let use_prefetch = file_size > 1024 * 1024; // Only for larger files
        
        CompressionStrategy {
            thread_count,
            chunk_size,
            buffer_size: self.optimal_buffer_size,
            use_simd,
            use_prefetch,
            use_huge_pages: self.memory.huge_page_size.is_some() && file_size > 100 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressionStrategy {
    pub thread_count: usize,
    pub chunk_size: usize,
    pub buffer_size: usize,
    pub use_simd: bool,
    pub use_prefetch: bool,
    pub use_huge_pages: bool,
}

fn detect_cpu_topology() -> CpuTopology {
    let logical_cores = num_cpus::get();
    let physical_cores = num_cpus::get_physical();
    
    // Detect CPU brand and features
    let cpu_brand = detect_cpu_brand();
    let is_apple_silicon = cpu_brand.contains("Apple") || std::env::consts::ARCH == "aarch64";
    
    // Cache sizes (defaults, can be improved with platform-specific detection)
    let (l1_cache_size, l2_cache_size, l3_cache_size) = if is_apple_silicon {
        // Apple Silicon typical cache sizes
        (128 * 1024, 4 * 1024 * 1024, 0) // M1/M2 don't have L3
    } else {
        // Intel/AMD typical cache sizes
        (32 * 1024, 256 * 1024, 8 * 1024 * 1024)
    };
    
    CpuTopology {
        physical_cores,
        logical_cores,
        l1_cache_size,
        l2_cache_size,
        l3_cache_size,
        cache_line_size: 64, // Standard cache line size
        numa_nodes: 1, // Simplified for now
        cpu_brand,
        has_avx2: {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            { is_x86_feature_detected!("avx2") && !is_apple_silicon }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            { false }
        },
        has_avx512: {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            { is_x86_feature_detected!("avx512f") && !is_apple_silicon }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            { false }
        },
        has_neon: is_apple_silicon || cfg!(target_arch = "aarch64"),
        is_apple_silicon,
    }
}

fn detect_memory_info() -> MemoryInfo {
    // Get system memory info
    let total_memory = get_total_memory();
    let available_memory = get_available_memory();
    
    // Estimate memory bandwidth based on system type
    let memory_bandwidth_gb_s = estimate_memory_bandwidth();
    
    MemoryInfo {
        total_memory,
        available_memory,
        memory_bandwidth_gb_s,
        page_size: 4096, // Standard page size
        huge_page_size: detect_huge_page_size(),
    }
}

fn calculate_optimal_threads(cpu: &CpuTopology) -> usize {
    // For compression workloads, physical cores are usually optimal
    // But consider hyperthreading benefits for I/O bound portions
    if cpu.is_apple_silicon {
        // Apple Silicon efficiency cores strategy
        cpu.physical_cores
    } else {
        // Intel/AMD strategy: use hyperthreading but not beyond physical cores * 1.5
        (cpu.logical_cores).min(cpu.physical_cores * 3 / 2)
    }
}

fn calculate_optimal_chunk_size(cpu: &CpuTopology, _memory: &MemoryInfo) -> usize {
    // Balance between cache efficiency and parallelism
    let base_chunk = if cpu.l3_cache_size > 0 {
        cpu.l3_cache_size / cpu.physical_cores / 4 // Quarter of L3 per core
    } else {
        cpu.l2_cache_size / 2 // Half of L2 for Apple Silicon
    };
    
    // Ensure chunk size is reasonable (64KB - 4MB range)
    base_chunk.max(64 * 1024).min(4 * 1024 * 1024)
}

fn calculate_optimal_buffer_size(cpu: &CpuTopology, memory: &MemoryInfo) -> usize {
    // Buffer size should accommodate multiple chunks and be memory-bandwidth aware
    let base_buffer = (memory.available_memory / 64).min(128 * 1024 * 1024); // Max 128MB
    
    // Align to cache line size for optimal memory access
    align_to_cache_line(base_buffer, cpu.cache_line_size)
}

fn align_to_cache_line(size: usize, cache_line_size: usize) -> usize {
    (size + cache_line_size - 1) & !(cache_line_size - 1)
}

fn detect_cpu_brand() -> String {
    // Try to read CPU brand from various sources
    if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if line.starts_with("model name") {
                if let Some(brand) = line.split(':').nth(1) {
                    return brand.trim().to_string();
                }
            }
        }
    }
    
    // Fallback for macOS
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").arg("-n").arg("machdep.cpu.brand_string").output() {
            if let Ok(brand) = String::from_utf8(output.stdout) {
                return brand.trim().to_string();
            }
        }
    }
    
    "Unknown CPU".to_string()
}

fn get_total_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output() {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(mem_bytes) = mem_str.trim().parse::<usize>() {
                    return mem_bytes;
                }
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    8 * 1024 * 1024 * 1024 // Fallback: 8GB
}

fn get_available_memory() -> usize {
    // Conservative estimate: 50% of total memory
    get_total_memory() / 2
}

fn estimate_memory_bandwidth() -> f64 {
    // Rough estimates based on common hardware
    #[cfg(target_arch = "aarch64")]
    {
        200.0 // Apple Silicon typical bandwidth
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        100.0 // Intel/AMD typical bandwidth
    }
}

fn detect_huge_page_size() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("Hugepagesize:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_system_detection() {
        let profile = SystemProfile::detect();
        println!("Detected system profile: {:#?}", profile);
        
        assert!(profile.cpu.logical_cores > 0);
        assert!(profile.cpu.physical_cores > 0);
        assert!(profile.optimal_thread_count > 0);
        assert!(profile.optimal_chunk_size > 0);
    }
    
    #[test]
    fn test_compression_strategy() {
        let profile = SystemProfile::detect();
        
        // Test different file sizes and compression levels
        let small_file = profile.recommend_compression_strategy(1024, 6);
        let large_file = profile.recommend_compression_strategy(100 * 1024 * 1024, 6);
        
        println!("Small file strategy: {:#?}", small_file);
        println!("Large file strategy: {:#?}", large_file);
        
        // Large files should use more threads
        assert!(large_file.thread_count >= small_file.thread_count);
    }
}