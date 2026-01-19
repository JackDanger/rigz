//! Build script for gzippy
//!
//! This script:
//! 1. Builds ISA-L from source (if available)
//! 2. Links the ISA-L library for high-performance decompression

use std::path::PathBuf;

fn main() {
    // Check if ISA-L source is available
    let isal_dir = PathBuf::from("isa-l");
    let isal_build_dir = isal_dir.join("build");

    if isal_build_dir.exists() {
        // Link to the built ISA-L library
        println!(
            "cargo:rustc-link-search=native={}",
            isal_build_dir.display()
        );

        // On macOS, link to dylib; on Linux, link to so
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=dylib=isal");
        } else {
            println!("cargo:rustc-link-lib=isal");
        }

        // Tell Rust where to find the library at runtime
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            isal_build_dir.display()
        );

        // Enable ISA-L feature
        println!("cargo:rustc-cfg=feature=\"isal\"");

        println!("cargo:warning=ISA-L library found and linked");
    } else {
        println!("cargo:warning=ISA-L not found, using libdeflate fallback");
    }

    // Re-run if ISA-L build changes
    println!("cargo:rerun-if-changed=isa-l/build");
    println!("cargo:rerun-if-changed=build.rs");
}
