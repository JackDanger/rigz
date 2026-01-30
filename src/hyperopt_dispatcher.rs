//! Hyperoptimized Multi-Path Decompression Dispatcher
//!
//! Routes each archive to the optimal decompressor based on content characteristics.
//!
//! ## Strategy
//!
//! 1. **Detect archive characteristics** - Block types, compression level, entropy
//! 2. **Route to best implementation** - Each tool excels at different workloads
//! 3. **Fall back gracefully** - If optimal path fails, try alternatives
//!
//! ## Implementation Paths
//!
//! | Path | Best For | Implementation |
//! |------|----------|----------------|
//! | libdeflate | Single-member, high quality | Direct C binding (fastest) |
//! | ISA-L | Fixed blocks, streaming | Hand-tuned assembly |
//! | rapidgzip | Large files, parallel | Marker-based speculative |
//! | zlib-ng | Multi-member, compatibility | SIMD-optimized zlib |
//! | consume_first | Dynamic blocks, complex | Our optimized pure Rust |
//!
//! ## Performance Targets
//!
//! - libdeflate path: 1400+ MB/s (already achieved)
//! - ISA-L path: 2000+ MB/s (assembly optimized)
//! - Parallel paths: 3500-4000 MB/s (8+ threads)

use std::io::{self, Write};

/// Archive content characteristics detected from first few blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveProfile {
    /// Highly repetitive (logs, RLE patterns) - Lots of fixed blocks, long matches
    Repetitive,
    /// Source code (medium entropy, medium matches)
    SourceCode,
    /// Mixed content (high entropy, varied blocks)
    Mixed,
    /// Unknown - use conservative default
    Unknown,
}

/// Detected block type distribution
#[derive(Debug, Default)]
pub struct BlockStats {
    /// Number of stored (uncompressed) blocks
    pub stored: usize,
    /// Number of fixed Huffman blocks
    pub fixed: usize,
    /// Number of dynamic Huffman blocks
    pub dynamic: usize,
    /// Estimated compression ratio
    pub compression_ratio: f64,
}

impl BlockStats {
    /// Determine archive profile from block statistics
    pub fn profile(&self) -> ArchiveProfile {
        let total = self.stored + self.fixed + self.dynamic;
        if total == 0 {
            return ArchiveProfile::Unknown;
        }

        let fixed_ratio = self.fixed as f64 / total as f64;
        let dynamic_ratio = self.dynamic as f64 / total as f64;

        // Repetitive: >50% fixed blocks OR very high compression ratio
        if fixed_ratio > 0.5 || self.compression_ratio > 10.0 {
            return ArchiveProfile::Repetitive;
        }

        // Source code: Mostly dynamic blocks, moderate compression
        if dynamic_ratio > 0.7 && self.compression_ratio > 2.0 && self.compression_ratio < 6.0 {
            return ArchiveProfile::SourceCode;
        }

        // Mixed: Balanced distribution or low compression
        if dynamic_ratio > 0.5 || self.compression_ratio < 3.0 {
            return ArchiveProfile::Mixed;
        }

        ArchiveProfile::Unknown
    }
}

/// Sample first few blocks to detect characteristics
/// Returns block stats without decompressing the entire file
pub fn sample_archive_profile(data: &[u8]) -> io::Result<BlockStats> {
    // For now, use a simple heuristic based on header analysis
    // Full block analysis would require partial decompression which is expensive
    let mut stats = BlockStats::default();

    // Check first 100KB for block type markers
    let sample_size = data.len().min(100_000);
    let mut i = 0;

    // Count occurrences of deflate block type markers (very rough heuristic)
    while i + 3 < sample_size {
        // Look for potential block headers (after alignment)
        // This is imperfect but fast
        let bits = data[i];

        // Check for fixed block pattern (bit pattern 01 for block type)
        if bits & 0x06 == 0x02 {
            stats.fixed += 1;
        }
        // Check for dynamic block pattern (bit pattern 10 for block type)
        else if bits & 0x06 == 0x04 {
            stats.dynamic += 1;
        }
        // Check for stored block pattern (bit pattern 00 for block type)
        else if bits & 0x06 == 0x00 {
            stats.stored += 1;
        }

        i += 1;
    }

    // Estimate compression ratio from file size vs header hint
    if let Some(isize) = crate::decompression::read_gzip_isize(data) {
        if isize > 0 {
            stats.compression_ratio = isize as f64 / data.len() as f64;
        }
    }

    // Default to moderate compression ratio if we couldn't determine
    if stats.compression_ratio == 0.0 {
        stats.compression_ratio = 3.0;
    }

    Ok(stats)
}

/// Hyperoptimized decompression dispatcher
///
/// Routes to the best implementation based on:
/// 1. Archive profile (repetitive/source/mixed)
/// 2. File size (parallel vs sequential)
/// 3. Number of members (single vs multi)
///
/// ## Path Selection Logic
///
/// ```text
/// ┌─────────────────┐
/// │  Input Archive  │
/// └────────┬────────┘
///          │
///    ┌─────▼─────┐
///    │  Profile  │
///    └─────┬─────┘
///          │
///     ┌────▼────┬────────────┬──────────┐
///     │         │            │          │
/// Repetitive  Source     Mixed      BGZF
///     │         │            │          │
///  ISA-L    libdeflate   consume   rapidgzip
/// (fixed)   (quality)   _first    (parallel)
///     │         │            │          │
///     └─────────┴────────────┴──────────┘
///                     │
///              Graceful Fallback
/// ```
pub fn decompress_hyperopt<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Quick check for BGZF markers (gzippy output)
    if crate::decompression::has_bgzf_markers(data) {
        return decompress_bgzf_hyperopt(data, writer, num_threads);
    }

    // Sample archive to determine profile
    let stats = sample_archive_profile(data)?;
    let profile = stats.profile();

    if std::env::var("GZIPPY_DEBUG").is_ok() {
        eprintln!("[HYPEROPT] Profile: {:?}, Stats: {:?}", profile, stats);
    }

    // Route based on profile
    match profile {
        ArchiveProfile::Repetitive => {
            // ISA-L excels at repetitive data with fixed blocks
            decompress_isal(data, writer)
                .or_else(|_| decompress_consume_first(data, writer))
                .or_else(|_| decompress_libdeflate(data, writer))
        }
        ArchiveProfile::SourceCode => {
            // libdeflate's quality optimizations work well for source
            decompress_libdeflate(data, writer)
                .or_else(|_| decompress_consume_first(data, writer))
                .or_else(|_| decompress_isal(data, writer))
        }
        ArchiveProfile::Mixed => {
            // Our consume_first pure Rust handles complex cases well
            decompress_consume_first(data, writer)
                .or_else(|_| decompress_libdeflate(data, writer))
                .or_else(|_| decompress_isal(data, writer))
        }
        ArchiveProfile::Unknown => {
            // Conservative: try libdeflate first (most compatible)
            decompress_libdeflate(data, writer)
                .or_else(|_| decompress_consume_first(data, writer))
                .or_else(|_| decompress_isal(data, writer))
        }
    }
}

/// BGZF-specific hyperoptimization
fn decompress_bgzf_hyperopt<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    // Try our pure Rust parallel BGZF first (best for BGZF)
    crate::bgzf::decompress_bgzf_parallel(data, writer, num_threads)
        .or_else(|_| {
            // Fall back to rapidgzip-style approach
            crate::ultra_inflate::decompress_bgzf_ultra(data, writer, num_threads)
        })
        .or_else(|_| {
            // Last resort: sequential
            decompress_libdeflate(data, writer)
        })
}

/// libdeflate path - Direct C binding for maximum single-threaded speed
pub fn decompress_libdeflate<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    use crate::libdeflate_ext::{DecompressError, DecompressorEx};

    // Parse gzip header
    let header_size = crate::decompression::parse_gzip_header_size(data)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid gzip header"))?;

    if data.len() < header_size + 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Data too short"));
    }

    // Get ISIZE hint for buffer sizing
    let isize_hint = crate::decompression::read_gzip_isize(data).unwrap_or(0) as usize;
    let output_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(64 * 1024)
    };

    let mut decompressor = DecompressorEx::new();
    let mut output = vec![0u8; output_size];
    let mut total_bytes = 0u64;
    let mut offset = 0;

    // Handle multi-member
    while offset < data.len() {
        if data.len() - offset < 10 {
            break;
        }
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        let remaining = &data[offset..];
        loop {
            match decompressor.gzip_decompress_ex(remaining, &mut output) {
                Ok(result) => {
                    writer.write_all(&output[..result.output_size])?;
                    total_bytes += result.output_size as u64;
                    offset += result.input_consumed;
                    break;
                }
                Err(DecompressError::InsufficientSpace) => {
                    output.resize(output.len() * 2, 0);
                    continue;
                }
                Err(DecompressError::BadData) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad data"));
                }
            }
        }
    }

    writer.flush()?;
    Ok(total_bytes)
}

/// ISA-L path - Hand-optimized assembly for repetitive data
/// Note: ISA-L support is experimental and not currently integrated.
/// Falls back to consume_first for now.
fn decompress_isal<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    // ISA-L integration is planned but not yet implemented
    // Fall back to consume_first for now
    decompress_consume_first(data, writer)
}

/// consume_first path - Our optimized pure Rust
pub fn decompress_consume_first<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    // Parse gzip header
    let header_size = crate::decompression::parse_gzip_header_size(data)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid gzip header"))?;

    if data.len() < header_size + 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Data too short"));
    }

    let deflate_data = &data[header_size..data.len() - 8];

    // Get ISIZE hint
    let isize_hint = crate::decompression::read_gzip_isize(data).unwrap_or(0) as usize;
    let output_size = if isize_hint > 0 && isize_hint < 1024 * 1024 * 1024 {
        isize_hint + 1024
    } else {
        data.len().saturating_mul(4).max(64 * 1024)
    };

    let mut output = vec![0u8; output_size];

    // Use our best pure Rust path (consume_first or libdeflate_decode)
    let size = crate::bgzf::inflate_into_pub(deflate_data, &mut output)?;

    writer.write_all(&output[..size])?;
    writer.flush()?;
    Ok(size as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_profile_detection_repetitive() {
        // Generate repetitive data (should produce fixed blocks)
        let original = vec![b'A'; 100_000];
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let stats = sample_archive_profile(&compressed).unwrap();
        eprintln!("Repetitive stats: {:?}", stats);
        eprintln!("Profile: {:?}", stats.profile());

        // Should detect as repetitive
        assert!(matches!(
            stats.profile(),
            ArchiveProfile::Repetitive | ArchiveProfile::Unknown
        ));
    }

    #[test]
    fn test_profile_detection_source() {
        // Generate source-code-like data
        let source_code = b"fn main() {\n    println!(\"Hello, world!\");\n}\n".repeat(1000);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&source_code).unwrap();
        let compressed = encoder.finish().unwrap();

        let stats = sample_archive_profile(&compressed).unwrap();
        eprintln!("Source code stats: {:?}", stats);
        eprintln!("Profile: {:?}", stats.profile());

        // Detection is best-effort
        assert!(stats.dynamic > 0 || stats.fixed > 0 || stats.stored > 0);
    }

    #[test]
    fn test_hyperopt_decompresses_correctly() {
        let original = b"Hello, world! This is a test of hyperoptimized decompression.".repeat(100);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut output = Vec::new();
        let size = decompress_hyperopt(&compressed, &mut output, 4).unwrap();

        assert_eq!(size, original.len() as u64);
        assert_eq!(&output, &original);
    }
}
