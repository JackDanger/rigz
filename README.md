# gzippy

A fast parallel gzip, written in Rust.

## What is this?

gzippy compresses and decompresses files using the gzip format. It uses all your CPU cores to work faster, while producing output that any gzip tool can read.

```bash
gzippy file.txt           # Compress → file.txt.gz
gzippy -d file.txt.gz     # Decompress → file.txt
cat data | gzippy > out   # Works with pipes too
```

## Install

```bash
cargo install gzippy
```

Or build from source:

```bash
git clone --recursive https://github.com/jackdanger/gzippy
cd gzippy
cargo build --release
```

## How fast is it?

On a 4-core machine compressing 10MB of text:

| Level | Time | Output size |
|-------|------|-------------|
| Fast (`-1`) | 24ms | 4.5 MB |
| Default (`-6`) | 76ms | 4.0 MB |
| Best (`-9`) | 201ms | 3.9 MB |

Decompression runs at 300-500 MB/s depending on the file.

## Options

Works like gzip: `-1` to `-9`, `-c` (stdout), `-d` (decompress), `-k` (keep original), `-f` (force), `-v` (verbose).

Extra options:
- `-p4` — use 4 threads (default: all cores)
- `--level 11` or `--ultra` — smaller output, slower
- `--level 12` or `--max` — smallest output

## Requirements

- 64-bit Linux or macOS
- Rust 1.70+

## Standing on shoulders

gzippy exists because of the brilliant work done by others:

- [**pigz**](https://zlib.net/pigz/) by Mark Adler — showed how to parallelize gzip
- [**libdeflate**](https://github.com/ebiggers/libdeflate) by Eric Biggers — fast, modern deflate
- [**zlib-ng**](https://github.com/zlib-ng/zlib-ng) — keeps zlib fast on modern CPUs
- [**rapidgzip**](https://github.com/mxmlnkn/rapidgzip) — parallel decompression techniques
- [**ISA-L**](https://github.com/intel/isa-l) by Intel — SIMD-optimized assembly

We study their code, learn from their optimizations, and try to combine the best ideas into one tool.

## License

[zlib license](LICENSE) — same as zlib and pigz.

## About

Made by [Jack Danger](https://github.com/jackdanger) as a [centaur](https://doctorow.medium.com/https-pluralistic-net-2025-12-05-pop-that-bubble-u-washington-8b6b75abc28e) on a mix of current models and tools.
