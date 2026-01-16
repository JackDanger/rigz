//! Decompression module
//!
//! Uses flate2's MultiGzDecoder to handle concatenated gzip members
//! (which rigz produces in parallel mode).
//!
//! Optimizations:
//! - Memory-mapped input for zero-copy reads
//! - Large output buffers to reduce syscall overhead

use std::fs::File;
use std::io::{self, stdin, stdout, BufReader, BufWriter, Cursor, Read, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::cli::RigzArgs;
use crate::error::{RigzError, RigzResult};
use crate::format::CompressionFormat;
use crate::utils::strip_compression_extension;

/// Large buffer size for I/O operations (1MB)
const BUFFER_SIZE: usize = 1024 * 1024;

/// Minimum file size to use mmap (smaller files don't benefit)
const MMAP_THRESHOLD: u64 = 64 * 1024;

pub fn decompress_file(filename: &str, args: &RigzArgs) -> RigzResult<i32> {
    if filename == "-" {
        return decompress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(RigzError::FileNotFound(filename.to_string()));
    }

    if input_path.is_dir() {
        return Err(RigzError::invalid_argument(format!(
            "{} is a directory",
            filename
        )));
    }

    // Determine output filename
    let output_path = if args.stdout {
        None
    } else {
        Some(get_output_filename(input_path, args))
    };

    // Check if output file exists and handle force flag
    if let Some(ref output_path) = output_path {
        if output_path.exists() && !args.force {
            return Err(RigzError::invalid_argument(format!(
                "Output file {} already exists",
                output_path.display()
            )));
        }
    }

    // Open input file
    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();

    // Determine compression format
    let format = detect_compression_format_from_path(input_path)?;

    // Use mmap for larger files - zero-copy from kernel page cache
    let use_mmap = file_size >= MMAP_THRESHOLD;

    let result = if use_mmap {
        // MMAP path - zero-copy input
        let mmap = unsafe { Mmap::map(&input_file)? };
        let reader = Cursor::new(&mmap[..]);
        
        if args.stdout {
            let stdout = stdout();
            let writer = BufWriter::with_capacity(BUFFER_SIZE, stdout.lock());
            decompress_stream(reader, writer, format)
        } else {
            let output_path = output_path.clone().unwrap();
            let output_file = File::create(&output_path)?;
            let writer = BufWriter::with_capacity(BUFFER_SIZE, output_file);
            decompress_stream(reader, writer, format)
        }
    } else {
        // Buffered I/O for small files
        let input_reader = BufReader::with_capacity(BUFFER_SIZE, input_file);
        
        if args.stdout {
            let stdout = stdout();
            let writer = BufWriter::with_capacity(BUFFER_SIZE, stdout.lock());
            decompress_stream(input_reader, writer, format)
        } else {
            let output_path = output_path.clone().unwrap();
            let output_file = File::create(&output_path)?;
            let writer = BufWriter::with_capacity(BUFFER_SIZE, output_file);
            decompress_stream(input_reader, writer, format)
        }
    };

    match result {
        Ok(output_size) => {
            if args.verbosity > 0 && !args.quiet {
                print_decompression_stats(file_size, output_size, input_path);
            }

            // Delete original file if not keeping it
            if !args.keep && !args.stdout {
                std::fs::remove_file(input_path)?;
            }

            Ok(0)
        }
        Err(e) => {
            // Clean up output file on error if we created one
            if !args.stdout {
                let cleanup_path = get_output_filename(input_path, args);
                if cleanup_path.exists() {
                    let _ = std::fs::remove_file(&cleanup_path);
                }
            }
            Err(e)
        }
    }
}

pub fn decompress_stdin(args: &RigzArgs) -> RigzResult<i32> {
    let stdin = stdin();
    let input = BufReader::with_capacity(BUFFER_SIZE, stdin.lock());
    
    let stdout = stdout();
    let output = BufWriter::with_capacity(BUFFER_SIZE, stdout.lock());

    let format = CompressionFormat::Gzip;
    let result = decompress_stream(input, output, format);

    match result {
        Ok(_) => Ok(0),
        Err(e) => Err(e),
    }
}

/// Core decompression using flate2
fn decompress_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    format: CompressionFormat,
) -> RigzResult<u64> {
    match format {
        CompressionFormat::Gzip => {
            // Use MultiGzDecoder to handle concatenated gzip members
            // (rigz produces multiple members in parallel mode, per RFC 1952)
            use flate2::read::MultiGzDecoder;
            let mut decoder = MultiGzDecoder::new(reader);
            
            let bytes_written = copy_with_buffer(&mut decoder, &mut writer)?;
            writer.flush()?;
            Ok(bytes_written)
        }
        CompressionFormat::Zlib => {
            use flate2::read::ZlibDecoder;
            let mut decoder = ZlibDecoder::new(reader);
            let bytes_written = copy_with_buffer(&mut decoder, &mut writer)?;
            writer.flush()?;
            Ok(bytes_written)
        }
        CompressionFormat::Zip => {
            use flate2::read::MultiGzDecoder;
            let mut decoder = MultiGzDecoder::new(reader);
            let bytes_written = copy_with_buffer(&mut decoder, &mut writer)?;
            writer.flush()?;
            Ok(bytes_written)
        }
    }
}

/// Copy with a larger buffer than io::copy's default 8KB
fn copy_with_buffer<R: Read, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<u64> {
    let mut buf = vec![0u8; BUFFER_SIZE];
    let mut total = 0u64;
    
    loop {
        let bytes_read = reader.read(&mut buf)?;
        if bytes_read == 0 {
            break;
        }
        writer.write_all(&buf[..bytes_read])?;
        total += bytes_read as u64;
    }
    
    Ok(total)
}

fn detect_compression_format_from_path(path: &Path) -> RigzResult<CompressionFormat> {
    if let Some(format) = crate::utils::detect_format_from_file(path) {
        Ok(format)
    } else {
        Ok(CompressionFormat::Gzip)
    }
}

fn get_output_filename(input_path: &Path, args: &RigzArgs) -> std::path::PathBuf {
    if args.stdout {
        return input_path.to_path_buf();
    }

    let mut output_path = strip_compression_extension(input_path);

    if output_path == input_path {
        output_path = input_path.to_path_buf();
        let current_name = output_path.file_name().unwrap().to_str().unwrap();
        output_path.set_file_name(format!("{}.out", current_name));
    }

    output_path
}

fn print_decompression_stats(input_size: u64, output_size: u64, path: &Path) {
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("<unknown>");
    
    let ratio = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        1.0
    };
    
    let (in_size, in_unit) = format_size(input_size);
    let (out_size, out_unit) = format_size(output_size);
    
    eprintln!(
        "{}: {:.1}{} → {:.1}{} ({:.1}x expansion)",
        filename, in_size, in_unit, out_size, out_unit, 1.0 / ratio
    );
}

fn format_size(bytes: u64) -> (f64, &'static str) {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    
    if bytes >= GB {
        (bytes as f64 / GB as f64, "GB")
    } else if bytes >= MB {
        (bytes as f64 / MB as f64, "MB")
    } else if bytes >= KB {
        (bytes as f64 / KB as f64, "KB")
    } else {
        (bytes as f64, "B")
    }
}
