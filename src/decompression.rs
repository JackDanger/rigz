use std::fs::File;
use std::io::{self, stdin, stdout, BufReader, BufWriter, Read, Write};
use std::path::Path;

// Note: gzp primarily focuses on compression, so we'll use flate2 for decompression

use crate::cli::RigzArgs;
use crate::error::{RigzError, RigzResult};
use crate::format::CompressionFormat;
use crate::utils::strip_compression_extension;

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
    let input_reader = BufReader::new(input_file);

    // Determine compression format
    let format = detect_compression_format_from_path(input_path)?;

    // Perform decompression
    let result = if args.stdout {
        decompress_with_gzp(input_reader, stdout(), format, args)
    } else {
        let output_path = output_path.unwrap();
        let output_file = BufWriter::new(File::create(&output_path)?);
        decompress_with_gzp(input_reader, output_file, format, args)
    };

    match result {
        Ok(output_size) => {
            if args.verbosity > 0 && !args.quiet {
                print_decompression_stats(output_size, input_path, args);
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
    let input = stdin();
    let output = stdout();

    // For stdin, assume gzip format (most common)
    let format = CompressionFormat::Gzip;

    let result = decompress_with_gzp(input, output, format, args);

    match result {
        Ok(_) => Ok(0),
        Err(e) => Err(e),
    }
}

fn decompress_with_gzp<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    format: CompressionFormat,
    _args: &RigzArgs,
) -> RigzResult<u64> {
    match format {
        CompressionFormat::Gzip => {
            // Try to use gzp's parallel decompression if available
            // For now, use simple single-threaded decompression
            use flate2::read::GzDecoder;
            let mut decoder = GzDecoder::new(reader);
            let bytes_written = io::copy(&mut decoder, &mut writer)?;
            Ok(bytes_written)
        }
        CompressionFormat::Zlib => {
            // Use flate2 for zlib decompression
            use flate2::read::ZlibDecoder;
            let mut decoder = ZlibDecoder::new(reader);
            let bytes_written = io::copy(&mut decoder, &mut writer)?;
            Ok(bytes_written)
        }
        CompressionFormat::Zip => {
            // For ZIP, fallback to gzip decompression
            use flate2::read::GzDecoder;
            let mut decoder = GzDecoder::new(reader);
            let bytes_written = io::copy(&mut decoder, &mut writer)?;
            Ok(bytes_written)
        }
    }
}

fn detect_compression_format_from_path(path: &Path) -> RigzResult<CompressionFormat> {
    if let Some(format) = crate::utils::detect_format_from_file(path) {
        Ok(format)
    } else {
        // Default to gzip if we can't detect
        Ok(CompressionFormat::Gzip)
    }
}

fn get_output_filename(input_path: &Path, args: &RigzArgs) -> std::path::PathBuf {
    if args.stdout {
        // This shouldn't be called if stdout is true, but just in case
        return input_path.to_path_buf();
    }

    // Strip compression extension to get original filename
    let mut output_path = strip_compression_extension(input_path);

    // If stripping didn't change the path, just add .out
    if output_path == input_path {
        output_path = input_path.to_path_buf();
        let current_name = output_path.file_name().unwrap().to_str().unwrap();
        output_path.set_file_name(format!("{}.out", current_name));
    }

    output_path
}

fn print_decompression_stats(output_size: u64, path: &Path, args: &RigzArgs) {
    if args.verbosity >= 1 {
        let filename = path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("<unknown>");
        eprintln!("{}: decompressed {} bytes", filename, output_size);
    }
}
