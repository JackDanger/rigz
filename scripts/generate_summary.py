#!/usr/bin/env python3
"""
Generate PR performance summary from benchmark results.

This script reads benchmark JSON files and generates a Markdown summary
showing how gzippy compares to all alternatives.

Usage:
    python3 scripts/generate_summary.py \
        --system results/system.json \
        --compression results/compression.json \
        --decompression results/decompression.json \
        --output summary.md
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    """Load JSON file, return empty dict if missing."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes > 1_000_000:
        return f"{size_bytes / 1_000_000:.2f} MB"
    elif size_bytes > 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    else:
        return f"{size_bytes} B"


def compare_icon(gzippy_val: float, other_val: float, higher_is_better: bool = True) -> tuple:
    """
    Return (icon, diff_pct) comparing gzippy to another tool.
    
    Icons:
    - ✅ = gzippy wins significantly (>10%)
    - 🟢 = gzippy wins
    - 🟡 = roughly equal (within 5%)
    - 🔴 = gzippy loses
    """
    if other_val == 0:
        return "—", 0
    
    diff_pct = (gzippy_val / other_val - 1) * 100
    
    if higher_is_better:
        if diff_pct > 10:
            icon = "✅"
        elif diff_pct > 0:
            icon = "🟢"
        elif diff_pct > -5:
            icon = "🟡"
        else:
            icon = "🔴"
    else:
        # Lower is better (e.g., size)
        if diff_pct < -5:
            icon = "✅"
        elif diff_pct < 0:
            icon = "🟢"
        elif diff_pct < 5:
            icon = "🟡"
        else:
            icon = "🔴"
    
    return icon, diff_pct


def generate_compression_table(compression: list, threads: int) -> list:
    """Generate compression comparison table for given thread count."""
    lines = []
    levels = sorted(set(r['level'] for r in compression))
    competitors = ['pigz', 'igzip', 'gzip']
    
    lines.append(f"### {threads} Thread{'s' if threads > 1 else ''}")
    lines.append("")
    lines.append("| Level | gzippy Speed | gzippy Size | vs pigz | vs igzip | vs gzip |")
    lines.append("|-------|--------------|-------------|---------|----------|---------|")
    
    for level in levels:
        level_results = [r for r in compression if r['level'] == level and r['threads'] == threads]
        gzippy = next((r for r in level_results if r['tool'] == 'gzippy'), None)
        
        if not gzippy:
            continue
        
        gzippy_speed = f"{gzippy['speed']:.0f} MB/s"
        gzippy_size = format_size(gzippy['size'])
        
        comparisons = []
        for comp in competitors:
            comp_result = next((r for r in level_results if r['tool'] == comp), None)
            if comp_result and comp_result['speed'] > 0:
                icon, diff = compare_icon(gzippy['speed'], comp_result['speed'], higher_is_better=True)
                comparisons.append(f"{icon} {diff:+.0f}%")
            else:
                comparisons.append("—")
        
        lines.append(f"| L{level} | {gzippy_speed} | {gzippy_size} | {' | '.join(comparisons)} |")
    
    lines.append("")
    return lines


def generate_decompression_table(decompression: list) -> list:
    """Generate decompression comparison table."""
    lines = []
    sources = sorted(set(r['source'] for r in decompression))
    
    lines.append("| Source | gzippy Speed | vs pigz | vs igzip | vs gzip | Status |")
    lines.append("|--------|--------------|---------|----------|---------|--------|")
    
    all_pass = True
    for source in sources:
        source_results = [r for r in decompression if r['source'] == source]
        gzippy = next((r for r in source_results if r['tool'] == 'gzippy'), None)
        
        if not gzippy:
            continue
        
        gzippy_speed = f"{gzippy['speed']:.0f} MB/s"
        
        comparisons = []
        wins = 0
        for comp in ['pigz', 'igzip', 'gzip']:
            comp_result = next((r for r in source_results if r['tool'] == comp), None)
            if comp_result and comp_result['speed'] > 0:
                icon, diff = compare_icon(gzippy['speed'], comp_result['speed'], higher_is_better=True)
                if diff >= 0:
                    wins += 1
                comparisons.append(f"{icon} {diff:+.0f}%")
            else:
                comparisons.append("—")
        
        status = "✅ PASS" if wins >= 2 else "🔴 FAIL"
        if wins < 2:
            all_pass = False
        
        lines.append(f"| {source} | {gzippy_speed} | {' | '.join(comparisons)} | {status} |")
    
    lines.append("")
    return lines, all_pass


def generate_key_metrics(compression: list, decompression: list) -> list:
    """Generate key metrics summary."""
    lines = []
    lines.append("## 📊 Key Metrics")
    lines.append("")
    
    gzippy_comp = [r for r in compression if r['tool'] == 'gzippy']
    pigz_comp = [r for r in compression if r['tool'] == 'pigz']
    gzippy_decomp = [r for r in decompression if r['tool'] == 'gzippy']
    
    if gzippy_comp and pigz_comp:
        # L1 multi-thread speed comparison
        gzippy_l1_mt = next((r for r in gzippy_comp if r['level'] == 1 and r['threads'] > 1), None)
        pigz_l1_mt = next((r for r in pigz_comp if r['level'] == 1 and r['threads'] > 1), None)
        
        if gzippy_l1_mt and pigz_l1_mt:
            speed_advantage = (gzippy_l1_mt['speed'] / pigz_l1_mt['speed'] - 1) * 100
            lines.append(f"- **Compression Speed (L1)**: {speed_advantage:+.0f}% vs pigz")
        
        # L9 size comparison
        gzippy_l9 = next((r for r in gzippy_comp if r['level'] == 9), None)
        pigz_l9 = next((r for r in pigz_comp if r['level'] == 9), None)
        
        if gzippy_l9 and pigz_l9:
            size_advantage = (gzippy_l9['size'] / pigz_l9['size'] - 1) * 100
            lines.append(f"- **Compression Ratio (L9)**: {size_advantage:+.1f}% vs pigz")
    
    if gzippy_decomp:
        best_decomp = max(gzippy_decomp, key=lambda r: r['speed'])
        lines.append(f"- **Best Decompression**: {best_decomp['speed']:.0f} MB/s ({best_decomp['source']}-compressed)")
    
    lines.append("")
    return lines


def generate_goals_table(compression: list, decompression: list) -> list:
    """Generate pass/fail goals table."""
    lines = []
    lines.append("## ✅ Performance Goals")
    lines.append("")
    lines.append("| Goal | Status |")
    lines.append("|------|--------|")
    
    gzippy_comp = [r for r in compression if r['tool'] == 'gzippy']
    pigz_comp = [r for r in compression if r['tool'] == 'pigz']
    
    # Compression speed goal
    gzippy_l1_mt = next((r for r in gzippy_comp if r['level'] == 1 and r['threads'] > 1), None)
    pigz_l1_mt = next((r for r in pigz_comp if r['level'] == 1 and r['threads'] > 1), None)
    
    if gzippy_l1_mt and pigz_l1_mt:
        if gzippy_l1_mt['speed'] >= pigz_l1_mt['speed']:
            lines.append("| Compression faster than pigz | ✅ PASS |")
        else:
            lines.append("| Compression faster than pigz | ❌ FAIL |")
    
    # Compression ratio goal
    gzippy_l9 = next((r for r in gzippy_comp if r['level'] == 9), None)
    pigz_l9 = next((r for r in pigz_comp if r['level'] == 9), None)
    
    if gzippy_l9 and pigz_l9:
        if gzippy_l9['size'] <= pigz_l9['size'] * 1.005:  # Within 0.5%
            lines.append("| Compression ratio matches pigz (L9) | ✅ PASS |")
        else:
            lines.append("| Compression ratio matches pigz (L9) | ❌ FAIL |")
    
    # Decompression speed goal
    gzippy_decomp = [r for r in decompression if r['tool'] == 'gzippy' and r.get('source') == 'gzippy']
    if gzippy_decomp:
        if gzippy_decomp[0]['speed'] >= 300:  # Conservative CI threshold
            lines.append("| Decompression ≥300 MB/s (BGZF) | ✅ PASS |")
        else:
            lines.append("| Decompression ≥300 MB/s (BGZF) | ❌ FAIL |")
    
    lines.append("")
    return lines


def generate_summary(system: dict, compression: list, decompression: list) -> str:
    """Generate full Markdown summary."""
    lines = []
    
    # Header
    lines.append("# 🚀 gzippy Performance Summary")
    lines.append("")
    if system:
        lines.append(f"**System**: {system.get('cpu', 'Unknown')} ({system.get('cores', '?')} cores)")
        lines.append(f"**SIMD**: {system.get('simd', 'Unknown')}")
        lines.append("")
    
    # Compression
    if compression:
        lines.append("## 📦 Compression: gzippy vs Alternatives")
        lines.append("")
        
        threads = sorted(set(r['threads'] for r in compression))
        for t in threads:
            lines.extend(generate_compression_table(compression, t))
    
    # Decompression
    if decompression:
        lines.append("## 📤 Decompression: gzippy vs Alternatives")
        lines.append("")
        decomp_lines, _ = generate_decompression_table(decompression)
        lines.extend(decomp_lines)
    
    # Key metrics
    if compression or decompression:
        lines.extend(generate_key_metrics(compression, decompression))
        lines.extend(generate_goals_table(compression, decompression))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate PR performance summary")
    parser.add_argument("--system", type=str, default="results/system.json",
                       help="Path to system.json")
    parser.add_argument("--compression", type=str, default="results/compression.json",
                       help="Path to compression.json")
    parser.add_argument("--decompression", type=str, default="results/decompression.json",
                       help="Path to decompression.json")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Load data
    system = load_json(args.system)
    compression = load_json(args.compression)
    decompression = load_json(args.decompression)
    
    # Handle list vs dict formats
    if isinstance(compression, dict):
        compression = compression.get('benchmarks', compression.get('results', []))
    if isinstance(decompression, dict):
        decompression = decompression.get('benchmarks', decompression.get('results', []))
    
    # Generate summary
    summary = generate_summary(system, compression, decompression)
    
    # Output
    if args.output:
        Path(args.output).write_text(summary)
        print(f"Summary written to {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
