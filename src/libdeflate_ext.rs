//! Direct libdeflate bindings for `_ex` functions
//!
//! This module wraps `libdeflate-sys` to expose `gzip_decompress_ex`,
//! which returns both output size and input bytes consumed. Essential
//! for iterating through multi-member gzip files at maximum speed.

use std::ptr::NonNull;

/// Result of a gzip decompression with extended info
#[derive(Debug)]
pub struct DecompressResult {
    /// Number of decompressed bytes written to output
    pub output_size: usize,
    /// Number of compressed bytes consumed from input
    pub input_consumed: usize,
}

/// Error from decompression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompressError {
    /// The compressed data is invalid
    BadData,
    /// The output buffer is too small
    InsufficientSpace,
}

impl std::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressError::BadData => write!(f, "invalid compressed data"),
            DecompressError::InsufficientSpace => write!(f, "output buffer too small"),
        }
    }
}

impl std::error::Error for DecompressError {}

/// A decompressor that can return consumed bytes count
pub struct DecompressorEx {
    ptr: NonNull<libdeflate_sys::libdeflate_decompressor>,
}

// Safety: libdeflate decompressors are thread-safe
unsafe impl Send for DecompressorEx {}

impl Default for DecompressorEx {
    fn default() -> Self {
        Self::new()
    }
}

impl DecompressorEx {
    /// Create a new decompressor
    pub fn new() -> Self {
        let ptr = unsafe { libdeflate_sys::libdeflate_alloc_decompressor() };
        let ptr = NonNull::new(ptr).expect("libdeflate_alloc_decompressor returned NULL");
        Self { ptr }
    }

    /// Decompress gzip data, returning both output size and input consumed
    ///
    /// This is essential for iterating through multi-member gzip files,
    /// as it tells us where each member ends.
    pub fn gzip_decompress_ex(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<DecompressResult, DecompressError> {
        let mut actual_in = 0usize;
        let mut actual_out = 0usize;

        let result = unsafe {
            libdeflate_sys::libdeflate_gzip_decompress_ex(
                self.ptr.as_ptr(),
                input.as_ptr() as *const std::ffi::c_void,
                input.len(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                output.len(),
                &mut actual_in,
                &mut actual_out,
            )
        };

        match result {
            libdeflate_sys::libdeflate_result_LIBDEFLATE_SUCCESS => Ok(DecompressResult {
                output_size: actual_out,
                input_consumed: actual_in,
            }),
            libdeflate_sys::libdeflate_result_LIBDEFLATE_BAD_DATA => Err(DecompressError::BadData),
            libdeflate_sys::libdeflate_result_LIBDEFLATE_INSUFFICIENT_SPACE => {
                Err(DecompressError::InsufficientSpace)
            }
            _ => panic!("libdeflate_gzip_decompress_ex returned unknown result"),
        }
    }
}

impl Drop for DecompressorEx {
    fn drop(&mut self) {
        unsafe {
            libdeflate_sys::libdeflate_free_decompressor(self.ptr.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn test_gzip_decompress_ex_single_member() {
        // Create a simple gzip-compressed buffer
        let original = b"Hello, world! This is a test of the decompressor.";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        // Decompress with our _ex function
        let mut decompressor = DecompressorEx::new();
        let mut output = vec![0u8; original.len() + 1024];

        let result = decompressor
            .gzip_decompress_ex(&compressed, &mut output)
            .unwrap();

        assert_eq!(result.output_size, original.len());
        assert_eq!(result.input_consumed, compressed.len());
        assert_eq!(&output[..result.output_size], &original[..]);
    }

    #[test]
    fn test_gzip_decompress_ex_multi_member() {
        // Create two gzip members concatenated
        let part1 = b"First part of the data.";
        let part2 = b"Second part of the data.";

        let mut encoder1 = GzEncoder::new(Vec::new(), Compression::default());
        encoder1.write_all(part1).unwrap();
        let compressed1 = encoder1.finish().unwrap();

        let mut encoder2 = GzEncoder::new(Vec::new(), Compression::default());
        encoder2.write_all(part2).unwrap();
        let compressed2 = encoder2.finish().unwrap();

        // Concatenate the two gzip members
        let mut multi_member = compressed1.clone();
        multi_member.extend_from_slice(&compressed2);

        // Decompress first member
        let mut decompressor = DecompressorEx::new();
        let mut output = vec![0u8; 1024];

        let result1 = decompressor
            .gzip_decompress_ex(&multi_member, &mut output)
            .unwrap();

        assert_eq!(result1.output_size, part1.len());
        assert_eq!(result1.input_consumed, compressed1.len());
        assert_eq!(&output[..result1.output_size], &part1[..]);

        // Decompress second member (starting after first)
        let remaining = &multi_member[result1.input_consumed..];
        let result2 = decompressor
            .gzip_decompress_ex(remaining, &mut output)
            .unwrap();

        assert_eq!(result2.output_size, part2.len());
        assert_eq!(result2.input_consumed, compressed2.len());
        assert_eq!(&output[..result2.output_size], &part2[..]);
    }
}
