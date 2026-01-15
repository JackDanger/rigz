//! File compression module
//!
//! Uses system zlib for identical output to gzip at ALL compression levels.

use std::fs::File;
use std::io::{self, stdin, stdout, BufWriter, Cursor, Read, Write};
use std::path::Path;

use crate::cli::RigzArgs;
use crate::error::{RigzError, RigzResult};
use crate::optimization::{detect_content_type, ContentType, OptimizationConfig};
use crate::simple_optimizations::SimpleOptimizer;

pub fn compress_file(filename: &str, args: &RigzArgs) -> RigzResult<i32> {
    if filename == "-" {
        return compress_stdin(args);
    }

    let input_path = Path::new(filename);
    if !input_path.exists() {
        return Err(RigzError::FileNotFound(filename.to_string()));
    }

    // Handle directory recursion
    if input_path.is_dir() {
        return if args.recursive {
            compress_directory(filename, args)
        } else {
            Err(RigzError::invalid_argument(format!(
                "{} is a directory",
                filename
            )))
        };
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

    // Open input file once and get metadata from the handle (fewer syscalls)
    let input_file = File::open(input_path)?;
    let file_size = input_file.metadata()?.len();

    // Skip content detection for single-threaded mode - no benefit and adds overhead
    // For multi-threaded with L4-L9, detect content type for optimization decisions
    let content_type = if args.processes <= 1 || args.compression_level <= 3 {
        // Fast path: skip content detection
        ContentType::Binary
    } else {
        // Multi-threaded with higher compression levels: detect for optimization
        let mut sample_file = File::open(input_path)?;
        detect_content_type(&mut sample_file).unwrap_or(ContentType::Binary)
    };

    // Create optimization configuration
    let opt_config = OptimizationConfig::new(
        args.processes as usize,
        file_size,
        args.compression_level,
        content_type,
    );

    if args.verbosity >= 2 {
        eprintln!(
            "rigz: optimizing for {:?} content, {} threads, {}KB buffer, {:?} backend",
            content_type,
            opt_config.thread_count,
            opt_config.buffer_size / 1024,
            opt_config.backend
        );
    }

    // Use advanced compression pipeline for better performance
    // For multi-threaded large file compression, use mmap for zero-copy access
    // Mmap has overhead that only pays off for files > 50MB
    let use_mmap = opt_config.thread_count > 1 && file_size > 50 * 1024 * 1024;
    
    let result = if use_mmap {
        // MMAP PATH: Zero-copy parallel compression for large files
        if args.verbosity >= 2 {
            eprintln!(
                "rigz: using mmap parallel backend with {} threads",
                opt_config.thread_count,
            );
        }
        let optimizer = SimpleOptimizer::new(opt_config.clone());
        if args.stdout {
            optimizer.compress_file(input_path, stdout()).map_err(|e| e.into())
        } else {
            let output_path = output_path.clone().unwrap();
            let output_file = BufWriter::new(File::create(&output_path)?);
            optimizer.compress_file(input_path, output_file).map_err(|e| e.into())
        }
    } else if args.stdout {
        compress_with_pipeline(input_file, stdout(), args, &opt_config)
    } else {
        let output_path = output_path.clone().unwrap();
        let output_file = BufWriter::new(File::create(&output_path)?);
        compress_with_pipeline(input_file, output_file, args, &opt_config)
    };

    match result {
        Ok(output_size) => {
            if args.verbosity > 0 && !args.quiet {
                print_compression_stats(file_size, output_size, input_path, args);
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

pub fn compress_stdin(args: &RigzArgs) -> RigzResult<i32> {
    let mut input = stdin();
    let output = stdout();

    // For stdin, we need to buffer some data to detect content type
    let mut buffer = Vec::new();
    let mut sample = vec![0u8; 8192];
    let bytes_read = input.read(&mut sample)?;

    if bytes_read > 0 {
        sample.truncate(bytes_read);
        buffer.extend_from_slice(&sample);

        // Read the rest of stdin
        input.read_to_end(&mut buffer)?;
    }

    let file_size = buffer.len() as u64;
    let content_type = if !sample.is_empty() {
        crate::optimization::analyze_content_type(&sample)
    } else {
        ContentType::Binary
    };

    // Create optimization configuration
    let opt_config = OptimizationConfig::new(
        args.processes as usize,
        file_size,
        args.compression_level,
        content_type,
    );

    let cursor = Cursor::new(buffer);
    let result = compress_with_pipeline(cursor, output, args, &opt_config);

    match result {
        Ok(_) => Ok(0),
        Err(e) => Err(e),
    }
}

fn compress_directory(dirname: &str, args: &RigzArgs) -> RigzResult<i32> {
    use walkdir::WalkDir;

    let mut exit_code = 0;

    for entry in WalkDir::new(dirname) {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let path_str = path.to_string_lossy();
            match compress_file(&path_str, args) {
                Ok(code) => {
                    if code != 0 {
                        exit_code = code;
                    }
                }
                Err(e) => {
                    eprintln!("rigz: {}: {}", path_str, e);
                    exit_code = 1;
                }
            }
        }
    }

    Ok(exit_code)
}

fn compress_with_pipeline<R: Read, W: Write>(
    mut reader: R,
    writer: W,
    args: &RigzArgs,
    opt_config: &OptimizationConfig,
) -> RigzResult<u64> {
    // FAST PATH: Single-threaded goes directly to flate2 with minimal overhead
    // This is critical for L1 performance where every microsecond matters
    if opt_config.thread_count == 1 {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        
        if args.verbosity >= 2 {
            eprintln!("rigz: using direct flate2 single-threaded path");
        }
        
        let compression = Compression::new(args.compression_level as u32);
        let mut encoder = GzEncoder::new(writer, compression);
        let bytes = io::copy(&mut reader, &mut encoder)?;
        encoder.finish()?;
        return Ok(bytes);
    }

    // MULTI-THREADED PATH: Use optimizer for parallel compression
    let optimizer = SimpleOptimizer::new(opt_config.clone());

    if args.verbosity >= 2 {
        eprintln!(
            "rigz: using parallel backend with {} threads",
            opt_config.thread_count,
        );
    }

    optimizer.compress(reader, writer).map_err(|e| e.into())
}

fn get_output_filename(input_path: &Path, args: &RigzArgs) -> std::path::PathBuf {
    let mut output_path = input_path.to_path_buf();

    // Remove existing compression extensions if forcing
    if args.force {
        let mut stem = input_path
            .file_stem()
            .unwrap_or(input_path.as_os_str())
            .to_str()
            .unwrap();
        if stem.ends_with(".tar") {
            stem = &stem[..stem.len() - 4];
        }
        output_path.set_file_name(stem);
    }

    // Add appropriate extension
    let current_extension = output_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("");
    let new_extension = if current_extension.is_empty() {
        args.suffix.trim_start_matches('.')
    } else {
        &format!("{}{}", current_extension, args.suffix)
    };

    output_path.set_extension(new_extension);
    output_path
}

fn print_compression_stats(input_size: u64, output_size: u64, path: &Path, args: &RigzArgs) {
    if args.verbosity >= 1 {
        let filename = path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("<unknown>");
        let compression_ratio = if input_size > 0 {
            (output_size as f64 / input_size as f64) * 100.0
        } else {
            0.0
        };

        let speedup_estimate = if input_size > 1_000_000 {
            match args.processes {
                1 => "1.0x",
                2 => "1.8x",
                4 => "3.2x",
                8 => "5.5x",
                _ => "?x",
            }
        } else {
            "1.0x"
        };

        eprintln!(
            "{}: {:.1}% compression, ~{} speedup",
            filename,
            100.0 - compression_ratio,
            speedup_estimate
        );
    }
}
