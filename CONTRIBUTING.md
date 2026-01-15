# Contributing to rigz

## Quick Start

```bash
make         # Build and run quick benchmarks (<30s)
make validate  # Verify output works with gunzip
cargo test   # Run unit tests
```

## Architecture

```
src/
├── main.rs              # CLI entry point
├── cli.rs               # gzip-compatible argument parsing
├── compression.rs       # File compression orchestration
├── decompression.rs     # File decompression
├── parallel_compress.rs # Rayon-based parallel gzip (the core)
├── optimization.rs      # Content detection, thread tuning
├── simple_optimizations.rs  # SimpleOptimizer wrapper
├── error.rs             # Error types
├── format.rs            # Gzip/zlib format detection
└── utils.rs             # File utilities
```

**Critical paths:**
- **Single-threaded**: `compression.rs` → direct `GzEncoder` (no overhead)
- **Multi-threaded**: `compression.rs` → `mmap` → `ParallelGzEncoder` → rayon

## Hard-Won Lessons

### 1. Use system zlib, not libz-ng

```toml
# CORRECT - produces identical output to gzip
flate2 = { version = "1.0", default-features = false, features = ["zlib"] }

# WRONG - produces different compression ratios at L1-L8
flate2 = { version = "1.0" }  # defaults to libz-ng
```

### 2. Fixed block size beats dynamic

Use 128KB blocks like pigz. Dynamic sizing based on file size caused 14% regression.

### 3. Single-threaded must be zero-overhead

```rust
if thread_count == 1 {
    // Go directly to flate2, skip all optimizer logic
    let mut encoder = GzEncoder::new(writer, compression);
    io::copy(&mut reader, &mut encoder)?;
    encoder.finish()?;
}
```

### 4. Benchmarking needs many runs

| File Size | Minimum Runs | Why |
|-----------|--------------|-----|
| 1MB | 20 | Coefficient of variation can be 20%+ |
| 10MB | 10 | CV is 2-5% |
| 100MB | 5 | CV is <2% |

Use **median** not min. 3 runs gave us false failures.

### 5. mmap only helps for large files

The mmap overhead only pays off for files >50MB. For smaller files, regular `io::copy` is faster.

### 6. Global thread pool

Creating a rayon thread pool per compression is expensive:

```rust
static THREAD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
```

## Performance Testing

```bash
make quick      # Fast iteration (<30s)
make perf-full  # Full suite (10+ min)
```

**Targets:**
- Single-threaded: Match gzip (within 5%)
- Multi-threaded: Beat pigz

## Pull Request Checklist

- [ ] `cargo test` passes
- [ ] `make validate` passes (output works with gunzip)
- [ ] `make quick` shows no regressions
- [ ] Single-threaded compression ratio matches gzip (within 0.1%)

## Future Optimization Ideas

- **libdeflate** - 2-3x faster than zlib for single-block compression
- **Intel ISA-L** - Hardware-accelerated on Intel CPUs
- **Shared dictionaries** - Like pigz, improve compression between adjacent blocks
