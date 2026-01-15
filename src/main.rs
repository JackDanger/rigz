use std::env;
use std::path::Path;
use std::process;

// Core modules - used in main compression flow
mod cli;
mod compression;
mod decompression;
mod error;
mod format;
mod optimization;
mod parallel_compress;
mod simple_optimizations;
mod utils;

// Experimental/future optimization modules (not yet integrated)
#[allow(dead_code)]
mod adaptive_compression;
#[allow(dead_code)]
mod advanced_algorithms;
#[allow(dead_code)]
mod hardware_analysis;
#[allow(dead_code)]
mod lockfree_threading;
#[allow(dead_code)]
mod memory_pool;
#[allow(dead_code)]
mod profiler;
#[allow(dead_code)]
mod simd_compression;

use cli::RigzArgs;
use error::RigzError;

const VERSION: &str = "rigz 0.1.0 (Rust port of pigz 2.8)";

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

    // Check if we should show version info
    if args.version {
        println!("{}", VERSION);
        if args.verbosity >= 2 {
            println!("Using flate2 crate");
        }
        return Ok(0);
    }

    // Check if we should show help
    if args.help {
        print_help();
        return Ok(0);
    }

    // Check if we should show license
    if args.license {
        print_license();
        return Ok(0);
    }

    // Determine operation mode based on program name
    let program_path = env::args().next().unwrap_or_else(|| "rigz".to_string());
    let program_name = Path::new(&program_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("rigz");

    let decompress =
        args.decompress || program_name.contains("unrigz") || program_name.contains("gunzip");

    // Process files
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
    println!("Usage: rigz [options] [files ...]");
    println!("  will compress files in place, adding the suffix '.gz'. If no files are");
    println!("  specified, stdin will be compressed to stdout. rigz does what gzip does,");
    println!("  but spreads the work over multiple processors and cores when compressing.");
    println!();
    println!("Options:");
    println!("  -0 to -9             Compression level");
    println!("  --fast, --best       Compression levels 1 and 9 respectively");
    println!("  -b, --blocksize mmm  Set compression block size to mmmK (default 128K)");
    println!("  -c, --stdout         Write all processed output to stdout (won't delete)");
    println!("  -d, --decompress     Decompress the compressed input");
    println!("  -f, --force          Force overwrite, compress .gz, links, and to terminal");
    println!("  -h, --help           Display a help screen and quit");
    println!("  -i, --independent    Compress blocks independently for damage recovery");
    println!("  -k, --keep           Do not delete original file after processing");
    println!("  -l, --list           List the contents of the compressed input");
    println!("  -L, --license        Display the rigz license and quit");
    println!("  -p, --processes n    Allow up to n compression threads (default is the");
    println!("                       number of online processors, or 8 if unknown)");
    println!("  -q, --quiet          Print no messages, even on error");
    println!("  -r, --recursive      Process the contents of all subdirectories");
    println!("  -S, --suffix .sss    Use suffix .sss instead of .gz (for compression)");
    println!("  -t, --test           Test the integrity of the compressed input");
    println!("  -v, --verbose        Provide more verbose output");
    println!("  -V, --version        Show the version of rigz");
    println!("  -z, --zlib           Compress to zlib (.zz) instead of gzip format");
    println!("  --                   All arguments after \"--\" are treated as files");
}

fn print_license() {
    println!("rigz - Rust port of pigz (parallel gzip replacement)");
    println!();
    println!("Based on pigz by Mark Adler, Copyright (C) 2007-2023");
    println!("Rust port implementation");
    println!();
    println!("This software is provided 'as-is', without any express or implied");
    println!("warranty. In no event will the author be held liable for any damages");
    println!("arising from the use of this software.");
    println!();
    println!("Permission is granted to anyone to use this software for any purpose,");
    println!("including commercial applications, and to alter it and redistribute it");
    println!("freely, subject to the following restrictions:");
    println!();
    println!("1. The origin of this software must not be misrepresented; you must not");
    println!("   claim that you wrote the original software. If you use this software");
    println!("   in a product, an acknowledgment in the product documentation would be");
    println!("   appreciated but is not required.");
    println!("2. Altered source versions must be plainly marked as such, and must not be");
    println!("   misrepresented as being the original software.");
    println!("3. This notice may not be removed or altered from any source distribution.");
}
