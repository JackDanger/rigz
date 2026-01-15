# rigz

**Rust Implementation of Gzip with Parallel Compression**

rigz is a drop-in replacement for gzip that uses multiple processors and cores for compression. It is a Rust port of [pigz](https://zlib.net/pigz/) by Mark Adler.

## Features

- **Parallel compression** - Uses all available CPU cores for compression
- **gzip-compatible output** - Works with standard `gunzip` everywhere
- **Drop-in replacement** - Same command-line interface as gzip
- **Cross-platform** - Works on Linux, macOS, Windows, and any platform Rust supports

## Installation

### From source

```bash
cargo install --path .
```

### Debian/Ubuntu

```bash
dpkg-buildpackage -b -us -uc
sudo dpkg -i ../rigz_*.deb
```

## Usage

```bash
# Compress a file (replaces original with .gz)
rigz file.txt

# Compress to stdout
rigz -c file.txt > file.txt.gz

# Decompress
rigz -d file.txt.gz

# Use specific compression level (1=fastest, 9=best)
rigz -9 file.txt

# Use specific number of threads
rigz -p 4 file.txt

# Keep original file
rigz -k file.txt
```

## Options

```
-0 to -9           Compression level (default: 6)
--fast, --best     Compression levels 1 and 9 respectively
-c, --stdout       Write to stdout instead of file
-d, --decompress   Decompress instead of compress
-f, --force        Force overwrite of output file
-k, --keep         Keep original file after compression
-p, --processes n  Use n threads for compression (default: all cores)
-q, --quiet        Suppress all output
-r, --recursive    Recursively compress directories
-v, --verbose      Verbose output
-h, --help         Show help
-V, --version      Show version
```

## Performance

rigz achieves comparable or better performance than pigz:

| File Size | gzip | rigz (1 thread) | rigz (4 threads) | Speedup |
|-----------|------|-----------------|------------------|---------|
| 1 MB      | 0.06s | 0.06s          | 0.035s           | 1.7x    |
| 10 MB     | 0.37s | 0.38s          | 0.11s            | 3.4x    |
| 100 MB    | 3.7s  | 3.8s           | 1.1s             | 3.4x    |

## How It Works

rigz uses a parallel compression strategy similar to pigz:

1. **Block-based compression**: The input is split into 128KB blocks
2. **Parallel compression**: Each block is compressed independently using rayon
3. **Concatenated output**: Compressed blocks are concatenated as valid gzip members (RFC 1952 allows this)
4. **Standard decompression**: Output works with any gzip-compatible decompressor

For single-threaded operation, rigz uses flate2 with system zlib directly for optimal performance.

## Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run benchmarks
make quick       # Fast benchmark (~30 seconds)
make perf-full   # Comprehensive benchmark (~10 minutes)
```

## License

rigz is distributed under the zlib license, the same as pigz and zlib:

```
This software is provided 'as-is', without any express or implied
warranty. In no event will the author be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

## Credits

- [pigz](https://zlib.net/pigz/) by Mark Adler - Original parallel gzip implementation
- [flate2](https://crates.io/crates/flate2) - Rust bindings to zlib
- [rayon](https://crates.io/crates/rayon) - Data parallelism library for Rust

## Contributing

Contributions are welcome! Please see the `.cursorrules` file for development guidelines.
