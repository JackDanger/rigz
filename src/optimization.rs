use std::io::Read;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Text,
    Binary,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionBackend {
    Gzp,
    Flate2,
}

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub thread_count: usize,
    pub buffer_size: usize,
    pub backend: CompressionBackend,
    #[allow(dead_code)]
    pub content_type: ContentType,
    #[allow(dead_code)]
    pub use_numa_pinning: bool,
    pub compression_level: u8,
}

impl OptimizationConfig {
    pub fn new(
        requested_threads: usize,
        file_size: u64,
        compression_level: u8,
        content_type: ContentType,
    ) -> Self {
        let thread_count = optimal_thread_count(requested_threads, file_size, compression_level);
        let buffer_size = optimal_buffer_size(file_size, content_type);
        let backend = choose_compression_backend(compression_level, content_type, file_size, thread_count);
        let use_numa_pinning = should_use_numa_pinning(thread_count, file_size);

        OptimizationConfig {
            thread_count,
            buffer_size,
            backend,
            content_type,
            use_numa_pinning,
            compression_level,
        }
    }
}

/// Detect content type by analyzing the first chunk of data
pub fn detect_content_type<R: Read>(reader: &mut R) -> std::io::Result<ContentType> {
    let mut sample = vec![0u8; 8192]; // 8KB sample
    let bytes_read = reader.read(&mut sample)?;

    if bytes_read == 0 {
        return Ok(ContentType::Binary);
    }

    sample.truncate(bytes_read);
    Ok(analyze_content_type(&sample))
}

pub fn analyze_content_type(sample: &[u8]) -> ContentType {
    if sample.is_empty() {
        return ContentType::Binary;
    }

    let mut text_chars = 0;
    let mut control_chars = 0;

    for &byte in sample {
        match byte {
            // ASCII printable + common whitespace
            0x20..=0x7E | 0x09 | 0x0A | 0x0D => text_chars += 1,
            // Control characters that might be in text files
            0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F => control_chars += 1,
            // High bytes - could be UTF-8 or binary
            0x80..=0xFF => {
                // Simple UTF-8 heuristic - check for valid sequences
                if is_likely_utf8_byte(byte) {
                    text_chars += 1;
                }
                // Binary chars don't affect classification, only text/control ratio
            }
            _ => {} // Binary - doesn't affect classification
        }
    }

    let total = sample.len();
    let text_ratio = text_chars as f64 / total as f64;
    let control_ratio = control_chars as f64 / total as f64;

    // Classify based on character distribution
    if text_ratio > 0.8 && control_ratio < 0.1 {
        ContentType::Text
    } else if text_ratio < 0.3 && is_random_like(sample) {
        ContentType::Random
    } else {
        ContentType::Binary
    }
}

fn is_likely_utf8_byte(byte: u8) -> bool {
    // Simple heuristic for UTF-8 continuation bytes and common patterns
    matches!(byte, 0x80..=0xBF | 0xC0..=0xDF | 0xE0..=0xEF | 0xF0..=0xF7)
}

fn is_random_like(sample: &[u8]) -> bool {
    if sample.len() < 256 {
        return false; // Need sufficient data for analysis
    }

    // Check if data looks random (even distribution of bytes)
    let mut counts = [0u32; 256];
    for &byte in sample {
        counts[byte as usize] += 1;
    }

    // Calculate variance in byte distribution
    let mean = sample.len() as f64 / 256.0;
    let variance: f64 = counts
        .iter()
        .map(|&count| {
            let diff = count as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / 256.0;

    // Check for entropy indicators
    let mut non_zero_buckets = 0;
    let mut max_count = 0;
    for &count in &counts {
        if count > 0 {
            non_zero_buckets += 1;
        }
        max_count = max_count.max(count);
    }

    // Random data should have:
    // 1. Low variance (even distribution)
    // 2. High number of unique bytes
    // 3. No single byte dominates
    let variance_threshold = mean * 1.5; // Tighter threshold
    let diversity_ratio = non_zero_buckets as f64 / 256.0;
    let dominance_ratio = max_count as f64 / sample.len() as f64;

    variance < variance_threshold && diversity_ratio > 0.8 && dominance_ratio < 0.1
}

/// Optimize thread count based on workload characteristics
fn optimal_thread_count(requested: usize, file_size: u64, compression_level: u8) -> usize {
    // Base thread count - respect what user requested, capped at CPU count
    let max_threads = requested.min(num_cpus::get());

    // Key insight from pigz: thread overhead only matters for very small files
    // For 1MB+, parallel compression is always beneficial
    
    match compression_level {
        1 => {
            // Level 1: Maximum throughput, use all threads regardless of file size
            // pigz uses all threads at L1 - we must too
            max_threads
        }
        2..=5 => {
            // Fast compression - use all threads for files > 100KB
            if file_size <= 102_400 {
                (max_threads / 2).max(1)
            } else {
                max_threads
            }
        }
        6 => {
            // Level 6: Use all requested threads
            // The "2-thread problem" was specific to gzp which we no longer use
            max_threads
        }
        7..=9 => {
            // High compression: Use all threads - compression is CPU-bound
            // pigz uses all threads at high levels, we should too
            max_threads
        }
        _ => max_threads,
    }
}

/// Optimize buffer size based on file characteristics
fn optimal_buffer_size(file_size: u64, content_type: ContentType) -> usize {
    let base_size = match file_size {
        0..=102_400 => 32_768,             // 32KB for small files
        102_401..=1_048_576 => 65_536,     // 64KB for medium files
        1_048_577..=10_485_760 => 131_072, // 128KB for large files
        _ => 524_288,                      // 512KB for very large files (increased for level 1 speed)
    };

    // Adjust based on content type
    match content_type {
        ContentType::Text => base_size, // Text compresses well with standard buffers
        ContentType::Binary => base_size * 2, // Binary needs larger buffers for patterns
        ContentType::Random => base_size / 2, // Random data won't compress much anyway
    }
}

/// Choose optimal compression backend
fn choose_compression_backend(
    _compression_level: u8,
    _content_type: ContentType,
    file_size: u64,
    thread_count: usize,
) -> CompressionBackend {
    // For very small files (<64KB), flate2 has lower startup overhead
    if file_size <= 65_536 {
        return CompressionBackend::Flate2;
    }

    // Multi-threaded: Use Gzp for parallel compression
    if thread_count > 1 {
        return CompressionBackend::Gzp;
    }

    // Single-threaded: Always use flate2 (zlib-ng) for correct compression ratios
    // gzp at level 1 produces poor compression, flate2 matches gzip output
    CompressionBackend::Flate2
}

/// Determine if NUMA pinning would be beneficial
fn should_use_numa_pinning(thread_count: usize, file_size: u64) -> bool {
    // Only beneficial for high thread counts on large files
    thread_count >= 4 && file_size >= 10_485_760 // >= 10MB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_detection() {
        // Text content - high ratio of printable ASCII
        let text_sample = b"Hello, world! This is a text file with normal content.";
        assert_eq!(analyze_content_type(text_sample), ContentType::Text);

        // Binary content - null bytes
        let binary_sample = vec![0u8; 100];
        assert_eq!(analyze_content_type(&binary_sample), ContentType::Binary);
    }

    #[test]
    fn test_thread_count_respects_request() {
        // Thread count should respect the request (capped at CPU count)
        let result = optimal_thread_count(4, 10_000_000, 6);
        assert!(result >= 1 && result <= 4);
    }

    #[test]
    fn test_buffer_sizing_scales_with_file_size() {
        // Larger files should get larger buffers
        let small = optimal_buffer_size(1024, ContentType::Text);
        let large = optimal_buffer_size(100_000_000, ContentType::Text);
        assert!(large >= small);
    }
}
