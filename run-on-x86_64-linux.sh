#!/bin/bash
# Run command on remote x86_64 Linux machine
# Usage: ./run-on-x86_64-linux.sh <command> [args...]
# Example: ./run-on-x86_64-linux.sh cargo test --release bench_cf_silesia -- --nocapture

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 cargo test --release bench_cf_silesia -- --nocapture"
    exit 1
fi

CMD="$*"
ssh -J neurotic root@10.30.0.199 "cd ./gzippy/ && git pull && $CMD"
