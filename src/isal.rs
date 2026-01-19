#![allow(dead_code)]
#![allow(clippy::io_other_error)]
#![allow(clippy::collapsible_if)]

//! ISA-L High-Performance Decompression
//!
//! This module provides direct bindings to Intel's ISA-L library for
//! the fastest possible gzip decompression. ISA-L uses hand-optimized
//! SIMD code that significantly outperforms other implementations.
//!
//! When ISA-L is not available, we fall back to libdeflate.

use std::io::{self, Write};

/// ISA-L inflate state structure
/// This matches the C struct layout from isa-l/include/igzip_lib.h
#[repr(C)]
pub struct InflateState {
    next_out: *mut u8,
    avail_out: u32,
    total_out: u32,
    next_in: *const u8,
    read_in: u64,
    avail_in: u32,
    read_in_length: u32,
    // ... rest of the struct (we only access the first fields)
    _pad: [u8; 1024], // Padding to ensure enough space
}

impl Default for InflateState {
    fn default() -> Self {
        Self {
            next_out: std::ptr::null_mut(),
            avail_out: 0,
            total_out: 0,
            next_in: std::ptr::null(),
            read_in: 0,
            avail_in: 0,
            read_in_length: 0,
            _pad: [0; 1024],
        }
    }
}

/// ISA-L return codes
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsalReturnCode {
    DecompOk = 0,
    EndInput = 1,
    OutOverflow = 2,
    NeedDict = 3,
    InvalidBlock = -1,
    InvalidSymbol = -2,
    InvalidLookback = -3,
    InvalidWrapper = -4,
    UnsupportedMethod = -5,
    IncorrectChecksum = -6,
}

// External ISA-L functions
#[cfg(feature = "isal")]
extern "C" {
    fn isal_inflate_init(state: *mut InflateState) -> i32;
    fn isal_inflate_reset(state: *mut InflateState) -> i32;
    fn isal_inflate(state: *mut InflateState) -> i32;
    fn isal_inflate_set_dict(state: *mut InflateState, dict: *const u8, dict_len: u32) -> i32;
}

/// Check if ISA-L is available
#[cfg(feature = "isal")]
pub fn is_available() -> bool {
    true
}

#[cfg(not(feature = "isal"))]
pub fn is_available() -> bool {
    false
}

/// High-performance ISA-L decompressor
#[cfg(feature = "isal")]
pub struct IsalInflater {
    state: Box<InflateState>,
}

#[cfg(feature = "isal")]
impl IsalInflater {
    /// Create a new ISA-L inflater
    pub fn new() -> io::Result<Self> {
        let mut state = Box::new(InflateState::default());

        let ret = unsafe { isal_inflate_init(state.as_mut()) };
        if ret != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("isal_inflate_init failed: {}", ret),
            ));
        }

        Ok(Self { state })
    }

    /// Reset the inflater for a new stream
    pub fn reset(&mut self) -> io::Result<()> {
        let ret = unsafe { isal_inflate_reset(self.state.as_mut()) };
        if ret != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("isal_inflate_reset failed: {}", ret),
            ));
        }
        Ok(())
    }

    /// Set dictionary/window for speculative decompression
    ///
    /// This is critical for parallel decompression: when starting mid-stream,
    /// back-references may point to data before our chunk. This sets that data.
    pub fn set_dict(&mut self, dict: &[u8]) -> io::Result<()> {
        let ret =
            unsafe { isal_inflate_set_dict(self.state.as_mut(), dict.as_ptr(), dict.len() as u32) };
        if ret != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("isal_inflate_set_dict failed: {}", ret),
            ));
        }
        Ok(())
    }

    /// Prime the inflater with bits for non-byte-aligned starts
    ///
    /// When starting decompression at a bit offset that isn't byte-aligned,
    /// we need to prime the bit buffer with the leading bits.
    pub fn prime(&mut self, bits: u64, num_bits: u32) {
        self.state.read_in = bits;
        self.state.read_in_length = num_bits;
    }

    /// Decompress gzip data
    /// Returns the number of bytes written to output
    pub fn decompress(&mut self, input: &[u8], output: &mut [u8]) -> io::Result<usize> {
        self.state.next_in = input.as_ptr();
        self.state.avail_in = input.len() as u32;
        self.state.next_out = output.as_mut_ptr();
        self.state.avail_out = output.len() as u32;

        let ret = unsafe { isal_inflate(self.state.as_mut()) };

        match ret {
            0 | 1 => {
                // Success
                let bytes_written = output.len() - self.state.avail_out as usize;
                Ok(bytes_written)
            }
            2 => Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "Output buffer too small",
            )),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("isal_inflate failed: {}", ret),
            )),
        }
    }

    /// Decompress a complete gzip stream
    pub fn decompress_all(&mut self, input: &[u8], initial_size: usize) -> io::Result<Vec<u8>> {
        let mut output = vec![0u8; initial_size];
        let mut total_written = 0;

        self.state.next_in = input.as_ptr();
        self.state.avail_in = input.len() as u32;

        loop {
            self.state.next_out = output[total_written..].as_mut_ptr();
            self.state.avail_out = (output.len() - total_written) as u32;

            let ret = unsafe { isal_inflate(self.state.as_mut()) };
            total_written = output.len() - self.state.avail_out as usize;

            match ret {
                0 | 1 => {
                    // Done
                    output.truncate(total_written);
                    return Ok(output);
                }
                2 => {
                    // Need more output space
                    output.resize(output.len() * 2, 0);
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("isal_inflate failed: {}", ret),
                    ));
                }
            }
        }
    }
}

#[cfg(feature = "isal")]
impl Default for IsalInflater {
    fn default() -> Self {
        Self::new().expect("Failed to create ISA-L inflater")
    }
}

// Fallback when ISA-L is not available
#[cfg(not(feature = "isal"))]
pub struct IsalInflater {
    inner: libdeflater::Decompressor,
}

#[cfg(not(feature = "isal"))]
impl IsalInflater {
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            inner: libdeflater::Decompressor::new(),
        })
    }

    pub fn reset(&mut self) -> io::Result<()> {
        self.inner = libdeflater::Decompressor::new();
        Ok(())
    }

    pub fn decompress(&mut self, input: &[u8], output: &mut [u8]) -> io::Result<usize> {
        match self.inner.gzip_decompress(input, output) {
            Ok(n) => Ok(n),
            Err(e) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Decompression failed: {:?}", e),
            )),
        }
    }

    pub fn decompress_all(&mut self, input: &[u8], initial_size: usize) -> io::Result<Vec<u8>> {
        let mut output = vec![0u8; initial_size];

        loop {
            match self.inner.gzip_decompress(input, &mut output) {
                Ok(n) => {
                    output.truncate(n);
                    return Ok(output);
                }
                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                    output.resize(output.len() * 2, 0);
                }
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Decompression failed: {:?}", e),
                    ));
                }
            }
        }
    }
}

#[cfg(not(feature = "isal"))]
impl Default for IsalInflater {
    fn default() -> Self {
        Self::new().expect("Failed to create fallback inflater")
    }
}

/// Parallel decompression using ISA-L
///
/// This function decompresses gzip data using multiple threads.
/// Each thread uses its own ISA-L inflater for maximum performance.
pub fn decompress_parallel<W: Write + Send>(
    data: &[u8],
    writer: &mut W,
    num_threads: usize,
) -> io::Result<u64> {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    // Parse to find gzip member boundaries
    let members = find_gzip_members(data);

    if members.len() < 2 {
        // Single member - use optimized sequential
        return decompress_sequential(data, writer);
    }

    // Parallel decompression of members
    let outputs: Vec<Mutex<Option<Vec<u8>>>> =
        (0..members.len()).map(|_| Mutex::new(None)).collect();

    let next_member = AtomicUsize::new(0);
    let num_members = members.len();

    std::thread::scope(|scope| {
        for _ in 0..num_threads.min(num_members) {
            let outputs_ref = &outputs;
            let next_ref = &next_member;
            let members_ref = &members;

            scope.spawn(move || {
                thread_local! {
                    static INFLATER: RefCell<IsalInflater> =
                        RefCell::new(IsalInflater::new().unwrap());
                }

                loop {
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_members {
                        break;
                    }

                    let (start, end) = members_ref[idx];
                    let member_data = &data[start..end];

                    // Estimate output size from ISIZE trailer
                    let isize_hint = if member_data.len() >= 8 {
                        let trailer = &member_data[member_data.len() - 4..];
                        u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]])
                            as usize
                    } else {
                        member_data.len() * 4
                    };

                    let result = INFLATER.with(|inflater| {
                        let mut inf = inflater.borrow_mut();
                        inf.reset().ok();
                        inf.decompress_all(member_data, isize_hint.max(1024))
                    });

                    match result {
                        Ok(decompressed) => {
                            *outputs_ref[idx].lock().unwrap() = Some(decompressed);
                        }
                        Err(_) => {
                            // Fallback to flate2
                            let mut decoder = flate2::read::GzDecoder::new(member_data);
                            let mut buf = Vec::new();
                            if std::io::Read::read_to_end(&mut decoder, &mut buf).is_ok() {
                                *outputs_ref[idx].lock().unwrap() = Some(buf);
                            }
                        }
                    }
                }
            });
        }
    });

    // Write outputs in order
    let mut total = 0u64;
    for output_mutex in outputs {
        if let Some(output) = output_mutex.into_inner().unwrap() {
            writer.write_all(&output)?;
            total += output.len() as u64;
        }
    }

    writer.flush()?;
    Ok(total)
}

/// Find gzip member boundaries
fn find_gzip_members(data: &[u8]) -> Vec<(usize, usize)> {
    let mut members = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        if offset + 10 > data.len() {
            break;
        }
        if data[offset] != 0x1f || data[offset + 1] != 0x8b {
            break;
        }

        // Parse header to find content start
        let header_size = match parse_gzip_header(&data[offset..]) {
            Ok(size) => size,
            Err(_) => break,
        };

        // Find end by decompressing (we need to know where the trailer is)
        // For now, use a heuristic: look for next member or end of data
        let mut end = data.len();
        for i in (offset + header_size + 18)..data.len().saturating_sub(10) {
            if data[i] == 0x1f && data[i + 1] == 0x8b && data[i + 2] == 0x08 {
                if parse_gzip_header(&data[i..]).is_ok() {
                    end = i;
                    break;
                }
            }
        }

        members.push((offset, end));
        offset = end;
    }

    members
}

/// Parse gzip header, return size
fn parse_gzip_header(data: &[u8]) -> io::Result<usize> {
    if data.len() < 10 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Too short"));
    }

    if data[0] != 0x1f || data[1] != 0x8b || data[2] != 0x08 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not gzip"));
    }

    let flags = data[3];
    let mut offset = 10;

    if flags & 0x04 != 0 {
        if offset + 2 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
        }
        let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + xlen;
    }

    if flags & 0x08 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    if flags & 0x10 != 0 {
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        offset += 1;
    }

    if flags & 0x02 != 0 {
        offset += 2;
    }

    if offset > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated"));
    }

    Ok(offset)
}

/// Sequential decompression using ISA-L
fn decompress_sequential<W: Write>(data: &[u8], writer: &mut W) -> io::Result<u64> {
    let mut inflater = IsalInflater::new()?;

    // Get ISIZE hint
    let isize_hint = if data.len() >= 8 {
        let trailer = &data[data.len() - 4..];
        u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]) as usize
    } else {
        data.len() * 4
    };

    let output = inflater.decompress_all(data, isize_hint.max(1024))?;
    writer.write_all(&output)?;
    writer.flush()?;

    Ok(output.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isal_available() {
        println!("ISA-L available: {}", is_available());
    }

    #[test]
    fn test_decompress() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let original = b"Hello, World! This is a test.";
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, original).unwrap();
        let compressed = encoder.finish().unwrap();

        let mut inflater = IsalInflater::new().unwrap();
        let decompressed = inflater.decompress_all(&compressed, 1024).unwrap();

        assert_eq!(&decompressed, original);
    }
}
