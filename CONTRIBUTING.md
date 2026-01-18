# Contributing to gzippy

Welcome! This guide will help you understand the codebase.

## What is gzippy?

gzippy is a **parallel gzip** that beats pigz by:

1. Using **libdeflate** for L1-L6 (30-50% faster than zlib)
2. Using **zlib-ng** with dictionary sharing for L7-L9 (pigz's algorithm)
3. Using a **custom scheduler** optimized for the compression workload

## Quick Start

```bash
cargo build --release
cargo test
./target/release/gzippy --help
```

## Architecture

```
gzippy file.txt
     │
     ▼
┌─────────────┐
│   main.rs   │  Parse args, route to compress/decompress
└──────┬──────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│ compression.rs   │ or  │ decompression.rs    │
│ (orchestration)  │     │ (libdeflate)        │
└────────┬─────────┘     └─────────────────────┘
         │
         ▼ L1-L6                    ▼ L7-L9
┌────────────────────┐    ┌─────────────────────┐
│ parallel_compress  │    │ pipelined_compress  │
│ (libdeflate)       │    │ (zlib-ng + dict)    │
└────────────────────┘    └─────────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
           ┌──────────────┐
           │ scheduler.rs │  Pigz-style N+1 threading
           └──────────────┘
```

## File Guide

| File | Purpose |
|------|---------|
| `main.rs` | CLI entry point |
| `cli.rs` | gzip-compatible argument parsing |
| `compression.rs` | Routing to parallel/pipelined path |
| `decompression.rs` | libdeflate for all decompression |
| `parallel_compress.rs` | L1-L6: independent blocks via libdeflate |
| `pipelined_compress.rs` | L7-L9: dictionary sharing via zlib-ng |
| `scheduler.rs` | Pigz-style N+1 thread scheduler |
| `optimization.rs` | Block size and thread count tuning |

## Key Design Decisions

### L1-L6: Independent Blocks (libdeflate)

- Each block is a complete gzip member
- Blocks compress in parallel without dependencies
- BGZF-style markers enable parallel decompression
- 30-50% faster than pigz

### L7-L9: Pipelined Compression (zlib-ng)

- Each block uses the previous block's data as dictionary
- Matches pigz compression ratio (critical for L9)
- Dedicated writer thread prevents compression stalls
- Still 20-30% faster than pigz

### Threading Model

Adopted from pigz:
- **N compress workers** claim blocks via atomic counter
- **1 dedicated writer** writes blocks in order
- Workers never block on I/O
- Brief spin-wait for low-latency handoff

## Performance Testing

```bash
# CI benchmark (specific config)
python3 scripts/benchmark_ci.py --size 10 --level 9 --threads 4

# Full validation matrix
python3 scripts/validate_ci.py --size 10 --trials 5
```

### Thresholds

| Level | Speed vs pigz | Size vs pigz |
|-------|---------------|--------------|
| L1-L6 | Must beat | Within 8% |
| L7-L8 | Must beat | Within 2% |
| L9 | Must beat | Within 0.5% |

## Pre-commit Checks

The repo has a pre-commit hook that runs:
1. `cargo fmt --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`

If you get lint errors, run:
```bash
cargo fmt
cargo clippy --fix --allow-staged
```

## Pull Request Checklist

- [ ] `cargo test` passes
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy` has no warnings
- [ ] CI benchmarks pass (gzippy beats pigz)

## Questions?

Open an issue!
