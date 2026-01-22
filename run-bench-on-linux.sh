#!/bin/bash
# Run benchmark on remote Linux machine
# Usage: ./run-bench-on-linux.sh [--runs N] [--analyze]

ARGS="${@:---runs 25}"
ssh -J neurotic root@10.30.0.199 "cd ./gzippy/ && git pull && ./bench-decompress.sh $ARGS"
