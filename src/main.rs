//! gzippy - The fastest parallel gzip
//!
//! A drop-in replacement for gzip that uses multiple processors for compression.
//! Inspired by [pigz](https://zlib.net/pigz/) by Mark Adler.

use std::env;
use std::path::Path;
use std::process;

mod bgzf;
mod block_finder;
mod block_finder_lut;
mod cli;
mod combined_lut;
mod compression;
mod consume_first_table;
mod decompression;
mod error;
mod fast_inflate;
mod format;
mod hyper_parallel;
mod inflate_tables;
mod isal;
mod libdeflate_ext;
mod marker_decode;
mod multi_symbol;
mod optimization;
mod packed_lut;
mod parallel_compress;
mod parallel_decompress;
mod parallel_inflate;
mod pipelined_compress;
mod rapidgzip_decoder;
mod scheduler;
mod simd_copy;
mod simd_huffman;
mod simd_inflate;
mod simple_optimizations;
mod thread_pool;
mod turbo_inflate;
mod two_level_table;
mod ultra_decompress;
mod ultra_fast_inflate;
mod ultra_inflate;
mod utils;

use cli::GzippyArgs;
use error::GzippyError;

const VERSION: &str = concat!("gzippy ", env!("CARGO_PKG_VERSION"));

fn main() {
    let result = run();

    match result {
        Ok(exit_code) => process::exit(exit_code),
        Err(e) => {
            eprintln!("gzippy: {}", e);
            process::exit(1);
        }
    }
}

fn run() -> Result<i32, GzippyError> {
    let args = GzippyArgs::parse()?;

    if args.version {
        println!("{}", VERSION);
        return Ok(0);
    }

    if args.help {
        print_help();
        return Ok(0);
    }

    if args.license {
        print_license();
        return Ok(0);
    }

    // Support gunzip/ungzippy symlinks
    let program_path = env::args().next().unwrap_or_else(|| "gzippy".to_string());
    let program_name = Path::new(&program_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("gzippy");

    let decompress =
        args.decompress || program_name.contains("ungzippy") || program_name.contains("gunzip");

    let mut exit_code = 0;

    if args.files.is_empty() {
        // Process stdin
        if decompress {
            exit_code = decompression::decompress_stdin(&args)?;
        } else {
            exit_code = compression::compress_stdin(&args)?;
        }
    } else {
        // Process files
        for file in &args.files {
            let result = if decompress {
                decompression::decompress_file(file, &args)
            } else {
                compression::compress_file(file, &args)
            };

            match result {
                Ok(code) => {
                    if code != 0 {
                        exit_code = code;
                    }
                }
                Err(e) => {
                    eprintln!("gzippy: {}: {}", file, e);
                    exit_code = 1;
                }
            }
        }
    }

    Ok(exit_code)
}

fn print_help() {
    println!("Usage: gzippy [OPTION]... [FILE]...");
    println!();
    println!("Compress or decompress FILEs (by default, compress in place).");
    println!("Uses multiple processors for parallel compression.");
    println!();
    println!("Options:");
    println!("  -1..-9           Compression level (1=fast, 9=best, default=6)");
    println!("  --level N        Set compression level 1-12");
    println!("  --ultra          Ultra compression (level 11, near-zopfli)");
    println!("  --max            Maximum compression (level 12, closest to zopfli)");
    println!("  -c, --stdout     Write to stdout, keep original files");
    println!("  -d, --decompress Decompress");
    println!("  -f, --force      Force overwrite of output file");
    println!("  -k, --keep       Keep original file");
    println!("  -p, --processes  Number of threads (default: all CPUs)");
    println!("  -r, --recursive  Recurse into directories");
    println!("  -q, --quiet      Suppress output");
    println!("  -v, --verbose    Verbose output");
    println!("  -h, --help       Show this help");
    println!("  -V, --version    Show version");
    println!("  -L, --license    Show license");
    println!();
    println!("Compression levels:");
    println!("  1-6              Fast (libdeflate, parallel decompress)");
    println!("  7-9              Balanced (zlib-ng, gzip-compatible)");
    println!("  10-12            Ultra (libdeflate high, near-zopfli ratio)");
    println!();
    println!("Examples:");
    println!("  gzippy file.txt          Compress file.txt → file.txt.gz");
    println!("  gzippy -d file.txt.gz    Decompress file.txt.gz → file.txt");
    println!("  gzippy -p4 -9 file.txt   Compress with 4 threads, best compression");
    println!("  cat file | gzippy > out  Compress stdin to stdout");
}

fn print_license() {
    println!("gzippy - The fastest gzip");
    println!();
    println!("Inspired by pigz by Mark Adler, Copyright (C) 2007-2023");
    println!();
    println!("zlib License - see LICENSE file for details.");
}
