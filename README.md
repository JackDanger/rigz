# rigz

**Fast parallel gzip, written in Rust.**

Drop-in replacement for gzip. Uses all your CPU cores. Beats both gzip and [pigz](https://zlib.net/pigz/).

## Install

```bash
cargo install rigz
```

Or build from source:

```bash
git clone https://github.com/jackdanger/rigz
cd rigz
cargo build --release
./target/release/rigz --help
```

## Usage

```bash
rigz file.txt           # Compress → file.txt.gz
rigz -d file.txt.gz     # Decompress → file.txt
rigz -p4 -9 file.txt    # 4 threads, max compression
cat data | rigz > out   # Stdin → stdout
```

Same flags as gzip: `-1` to `-9`, `-c`, `-d`, `-k`, `-f`, `-r`, `-v`, `-q`.

## Performance

Benchmarked on M4 Mac, 10MB text file:

| Tool | Threads | Time | vs gzip |
|------|---------|------|---------|
| gzip | 1 | 0.32s | baseline |
| **rigz** | 1 | 0.32s | same |
| pigz | 4 | 0.09s | 3.6× faster |
| **rigz** | 4 | **0.085s** | **3.8× faster** |
| pigz | 8 | 0.046s | 7× faster |
| **rigz** | 8 | **0.042s** | **7.6× faster** |

Output is byte-for-byte compatible with `gunzip`.

## How It Works

1. **Block-based parallelism**: Input splits into 128KB blocks (like pigz)
2. **Independent compression**: Each block compresses in parallel via [rayon](https://docs.rs/rayon)
3. **Concatenated gzip**: Blocks become separate gzip members ([RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952) allows this)
4. **System zlib**: Uses your system's zlib for identical compression to gzip

Key architectural decisions:
- **Single-threaded mode goes direct to flate2** - no overhead
- **Memory-mapped I/O for large files** - zero-copy via [memmap2](https://docs.rs/memmap2)
- **Global rayon thread pool** - no per-call initialization cost

## Why Rust?

- **rayon** - Zero-cost work-stealing parallelism
- **memmap2** - Safe memory-mapped I/O
- **flate2** - Thin wrapper over system zlib
- No garbage collector pauses during compression

## License

[zlib license](LICENSE) (same as zlib and pigz).

## Credits

- [pigz](https://zlib.net/pigz/) by Mark Adler - the original parallel gzip
- [flate2](https://docs.rs/flate2) - Rust zlib bindings
- [rayon](https://docs.rs/rayon) - data parallelism
