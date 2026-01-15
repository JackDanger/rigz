//! rigz - Rust parallel gzip replacement
//!
//! A drop-in replacement for gzip that uses multiple processors for compression.
//! Based on [pigz](https://zlib.net/pigz/) by Mark Adler.

use std::env;
use std::path::Path;
use std::process;

mod cli;
mod compression;
mod decompression;
mod error;
mod format;
mod optimization;
mod parallel_compress;
mod simple_optimizations;
mod utils;

use cli::RigzArgs;
use error::RigzError;

const VERSION: &str = concat!("rigz ", env!("CARGO_PKG_VERSION"));

fn main() {
    let result = run();

    match result {
        Ok(exit_code) => process::exit(exit_code),
        Err(e) => {
            eprintln!("rigz: {}", e);
            process::exit(1);
        }
    }
}

fn run() -> Result<i32, RigzError> {
    let args = RigzArgs::parse()?;

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

    // Support gunzip/unrigz symlinks
    let program_path = env::args().next().unwrap_or_else(|| "rigz".to_string());
    let program_name = Path::new(&program_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("rigz");

    let decompress =
        args.decompress || program_name.contains("unrigz") || program_name.contains("gunzip");

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
                    eprintln!("rigz: {}: {}", file, e);
                    exit_code = 1;
                }
            }
        }
    }

    Ok(exit_code)
}

fn print_help() {
    println!("Usage: rigz [OPTION]... [FILE]...");
    println!();
    println!("Compress or decompress FILEs (by default, compress in place).");
    println!("Uses multiple processors for parallel compression.");
    println!();
    println!("Options:");
    println!("  -1..-9           Compression level (1=fast, 9=best, default=6)");
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
    println!("Examples:");
    println!("  rigz file.txt          Compress file.txt → file.txt.gz");
    println!("  rigz -d file.txt.gz    Decompress file.txt.gz → file.txt");
    println!("  rigz -p4 -9 file.txt   Compress with 4 threads, best compression");
    println!("  cat file | rigz > out  Compress stdin to stdout");
}

fn print_license() {
    println!("rigz - Rust parallel gzip replacement");
    println!();
    println!("Based on pigz by Mark Adler, Copyright (C) 2007-2023");
    println!();
    println!("zlib License - see LICENSE file for details.");
}
