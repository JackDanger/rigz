# Contributing to rigz

Welcome! This guide will help you understand the codebase and make your first contribution.

## What is rigz?

rigz is a **parallel gzip** - it compresses files using multiple CPU cores. The key insight is simple:

1. Split input into 128KB chunks
2. Compress each chunk in parallel using [rayon](https://docs.rs/rayon)
3. Concatenate the results (gzip allows this per [RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952))

That's it! The rest is optimization details.

## Quick Start

```bash
# Build and run quick benchmarks
make

# Verify output works with gunzip  
make validate

# Run tests
cargo test
```

## Understanding the Code

### The 30-Second Tour

```
rigz file.txt
     │
     ▼
┌─────────────┐
│   main.rs   │  Parse args, decide compress vs decompress
└──────┬──────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│ compression.rs   │ or  │ decompression.rs    │
│ (orchestration)  │     │ (libdeflate/flate2) │
└────────┬─────────┘     └─────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ parallel_compress.rs    │  ← THE CORE: rayon + flate2
│                         │
│  1. mmap the file       │
│  2. chunk into 128KB    │
│  3. compress in parallel│
│  4. concatenate output  │
└─────────────────────────┘
```

### File-by-File Guide

| File | Purpose | Complexity |
|------|---------|------------|
| `main.rs` | CLI entry, routes to compress/decompress | Simple |
| `cli.rs` | gzip-compatible argument parsing | Boring but thorough |
| `compression.rs` | File handling, decides single vs parallel | Medium |
| `decompression.rs` | libdeflate for speed, flate2 for multi-member | Medium |
| **`parallel_compress.rs`** | **The heart: rayon parallel gzip** | **Read this first** |
| `optimization.rs` | Thread/buffer tuning heuristics | Can ignore initially |
| `simple_optimizations.rs` | Wrapper around parallel_compress | Can ignore initially |
| `error.rs` | Error types | Simple |
| `format.rs` | gzip/zlib format detection | Simple |
| `utils.rs` | File utilities | Simple |

### Start Here: parallel_compress.rs

This is the core algorithm. Read it first:

```rust
// The key function - parallel compression using rayon
pub fn compress_file<P: AsRef<Path>, W: Write>(&self, path: P, writer: W) -> io::Result<u64> {
    // 1. Memory-map the file (zero-copy)
    let mmap = unsafe { Mmap::map(&file)? };
    
    // 2. Split into 128KB chunks
    let blocks: Vec<&[u8]> = mmap.chunks(block_size).collect();
    
    // 3. Compress each chunk in parallel
    let compressed_blocks: Vec<Vec<u8>> = pool.install(|| {
        blocks
            .par_iter()  // rayon parallel iterator
            .map(|block| {
                let mut encoder = GzEncoder::new(...);
                encoder.write_all(block);
                encoder.finish()
            })
            .collect()
    });
    
    // 4. Concatenate (gzip allows this!)
    for block in compressed_blocks {
        writer.write_all(&block)?;
    }
}
```

## First Contribution Ideas

### Easy (Good First Issues)

1. **Add `--no-color` flag** - Disable colored output
2. **Improve error messages** - Make them more helpful
3. **Add `--version` details** - Show zlib version, CPU count

### Medium

1. **Better progress output** - Show compression progress for large files
2. **Benchmark harness** - Add criterion benchmarks
3. **ARM-specific tuning** - Optimize block size for Apple Silicon

### Advanced

1. **Shared dictionary compression** - Use previous block as dictionary (like pigz `-i`)
2. **Intel ISA-L backend** - Hardware-accelerated compression on Intel
3. **Parallel decompression** - Decompress multi-member files in parallel

## How Compression Works

### Single-threaded (fast path)
```
Input → flate2::GzEncoder → Output
```
Just use zlib-ng directly. No overhead.

### Multi-threaded (parallel path)
```
Input → mmap → [chunk1, chunk2, chunk3, ...]
                    ↓ rayon parallel
              [gzip1, gzip2, gzip3, ...]
                    ↓ concatenate
                 Output
```

Each chunk becomes a complete gzip "member". The gzip format allows concatenation.

### Decompression Strategy
- **Single-member gzip**: libdeflate (30-50% faster than zlib)
- **Multi-member gzip**: zlib-ng via flate2 (reliable boundary parsing)

We can't easily parallelize decompression because deflate streams can contain bytes that look like gzip headers (`1f 8b 08`), making boundary detection unreliable.

## Key Dependencies

| Crate | Why |
|-------|-----|
| `flate2` | Rust bindings to zlib-ng (compression) |
| `libdeflater` | Rust bindings to libdeflate (fast decompression) |
| `rayon` | Work-stealing parallelism |
| `memmap2` | Memory-mapped I/O |
| `clap` | CLI parsing |

## Performance Testing

```bash
# Quick iteration (< 30s)
make quick

# Full suite (10+ min, includes 100MB files)
make perf-full

# Specific test
python3 scripts/perf.py --sizes 1,10 --levels 6 --threads 1,4
```

### Benchmarking Rules

| File Size | Runs Needed | Why |
|-----------|-------------|-----|
| 1MB | 30 | High variance (CV can be 20%+) |
| 10MB | 15 | Medium variance |
| 100MB | 7 | Low variance |

**Use median, not min.** Too few runs give false positives.

### Performance Targets

- **Single-threaded**: Beat gzip (we're ~50% faster)
- **Multi-threaded**: Beat pigz (we're ~40-50% faster)

## Pull Request Checklist

- [ ] `cargo test` passes
- [ ] `make validate` passes (output works with gunzip)
- [ ] `make quick` shows no regressions
- [ ] Code is formatted (`cargo fmt`)

## What NOT to Do

These things were tried and failed:

1. **Dynamic block sizing** - Caused 14% performance regression
2. **Manual deflate boundary detection** - Deflate streams contain false gzip headers
3. **gzp crate** - Threading issues; custom rayon is better
4. **Creating thread pools per-call** - Use global `OnceLock<ThreadPool>`

## Questions?

Open an issue! We're happy to help newcomers understand the codebase.
