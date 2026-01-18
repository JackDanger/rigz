#!/usr/bin/env python3
"""
Generate markdown summary from validation results.

Handles both validate.py and validate_ci.py JSON formats.

Usage:
    python3 scripts/validation_summary.py validation-results.json
    python3 scripts/validation_summary.py validation-results.json --format github  # For PR comments
"""

import json
import sys
import argparse
from collections import defaultdict


def format_time(seconds):
    """Format time with appropriate precision."""
    if seconds < 0.01:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.1f}s"


def format_size(bytes_val):
    """Format size in human-readable form."""
    mb = bytes_val / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.1f}MB"
    return f"{bytes_val / 1024:.1f}KB"


def detect_format(results):
    """Detect which script generated the JSON."""
    if "summary" in results:
        return "validate"  # validate.py format
    elif "compression_stats" in results:
        return "validate_ci"  # validate_ci.py format
    else:
        raise ValueError("Unknown JSON format")


def get_thresholds(level: int) -> tuple:
    """Returns (max_time_overhead_pct, max_size_overhead_pct) matching benchmark_ci.py."""
    if level >= 9:
        return (5.0, 0.5)
    elif level >= 7:
        return (5.0, 2.0)
    else:
        return (2.0, 8.0)


def generate_summary_from_ci(results, format_type="markdown"):
    """Generate summary from validate_ci.py output."""
    lines = []
    
    passed = results.get('passed', 0)
    failed = results.get('failed', 0)
    total = passed + failed
    size_mb = results.get('config', {}).get('size_mb', 0)
    data_types = results.get('config', {}).get('data_types', ['unknown'])
    test_size = f"{size_mb}MB" if size_mb else "unknown"
    
    # Header
    if format_type == "github":
        lines.append("## 🔄 gzippy Validation Results")
    else:
        lines.append("# Validation Results")
    lines.append("")
    
    # Status
    data_types_str = ", ".join(data_types) if len(data_types) <= 3 else f"{len(data_types)} types"
    if failed == 0:
        lines.append(f"**✅ All {passed} tests passed** ({test_size} × {data_types_str})")
    else:
        lines.append(f"**❌ {failed}/{total} tests failed** ({test_size} × {data_types_str})")
    lines.append("")
    
    # Track overall best stats across all data types
    best_gzip_speedup = 0
    best_pigz_speedup = 0
    best_throughput = 0
    
    # Compression performance - one summary table (text data is most representative)
    # Use first data type for the summary table, or all if only one
    lines.append("### Compression Performance")
    lines.append("")
    
    if len(data_types) > 1:
        # Show just the text results in the main table (most representative)
        primary_type = "text" if "text" in data_types else data_types[0]
        lines.append(f"*Results on {primary_type} data ({test_size})*")
        lines.append("")
    
    lines.append("| Config | gzippy Time | vs gzip | vs pigz | Ratio | Size vs pigz |")
    lines.append("|--------|-----------|---------|---------|-------|--------------|")
    
    # Group all compression stats by config
    all_comp_by_config = defaultdict(lambda: defaultdict(dict))
    for stat in results.get("compression_stats", []):
        key = (stat["level"], stat["threads"])
        dtype = stat.get("data_type", "unknown")
        all_comp_by_config[dtype][key][stat["tool"]] = stat
    
    # Show primary type in main table
    primary_type = "text" if "text" in data_types else (data_types[0] if data_types else "unknown")
    comp_by_config = all_comp_by_config.get(primary_type, {})
    
    for (level, threads), tools in sorted(comp_by_config.items()):
        config = f"L{level}, {threads}t"
        max_time_overhead, max_size_overhead = get_thresholds(level)
        
        gzip_time = tools.get("gzip", {}).get("median_time", 0)
        pigz_time = tools.get("pigz", {}).get("median_time", 0)
        gzippy_time = tools.get("gzippy", {}).get("median_time", 0)
        
        gzippy_size = tools.get("gzippy", {}).get("output_size", 0)
        pigz_size = tools.get("pigz", {}).get("output_size", 0)
        gzippy_input = tools.get("gzippy", {}).get("input_size", 0)
        
        gzippy_str = format_time(gzippy_time) if gzippy_time else "—"
        
        # vs gzip speedup
        if gzippy_time and gzip_time:
            gzip_speedup = gzip_time / gzippy_time
            gzip_speedup_str = f"**{gzip_speedup:.1f}×**"
            best_gzip_speedup = max(best_gzip_speedup, gzip_speedup)
            # Italicize if we're slower than gzip
            if gzip_speedup < 1.0:
                gzip_speedup_str = f"*{gzip_speedup:.2f}×*"
        else:
            gzip_speedup_str = "—"
        
        # vs pigz speedup
        if gzippy_time and pigz_time:
            pigz_speedup = pigz_time / gzippy_time
            pigz_overhead = (gzippy_time / pigz_time - 1) * 100
            best_pigz_speedup = max(best_pigz_speedup, pigz_speedup)
            # Italicize if we fail to beat threshold
            if pigz_overhead > max_time_overhead:
                pigz_speedup_str = f"*{pigz_speedup:.2f}×*"
            else:
                pigz_speedup_str = f"**{pigz_speedup:.1f}×**"
        else:
            pigz_speedup_str = "—"
        
        # Compression ratio
        if gzippy_size and gzippy_input:
            ratio = gzippy_size / gzippy_input * 100
            ratio_str = f"{ratio:.1f}%"
        else:
            ratio_str = "—"
        
        # Size vs pigz
        if gzippy_size and pigz_size:
            size_diff = (gzippy_size / pigz_size - 1) * 100
            if size_diff > max_size_overhead:
                # Italicize if we fail threshold
                size_str = f"*{size_diff:+.1f}%*"
            elif size_diff < 0:
                size_str = f"**{size_diff:+.1f}%**"
            else:
                size_str = f"{size_diff:+.1f}%"
        else:
            size_str = "—"
        
        if gzippy_time and size_mb:
            throughput = size_mb / gzippy_time
            best_throughput = max(best_throughput, throughput)
        
        lines.append(f"| {config} | {gzippy_str} | {gzip_speedup_str} | {pigz_speedup_str} | {ratio_str} | {size_str} |")
    
    lines.append("")
    lines.append("*Italics indicate failing to beat threshold*")
    lines.append("")
    
    # Decompression summary
    decomp_tests = [t for t in results.get("tests", []) if "decompress_tool" in t]
    decomp_passed = sum(1 for t in decomp_tests if t.get("passed"))
    decomp_total = len(decomp_tests)
    
    lines.append("### Decompression")
    lines.append("")
    if decomp_total > 0:
        if decomp_passed == decomp_total:
            lines.append(f"✅ All {decomp_total} cross-tool decompression tests passed")
        else:
            lines.append(f"⚠️ {decomp_passed}/{decomp_total} decompression tests passed")
            for t in decomp_tests:
                if not t.get("passed"):
                    comp = t.get("compress_tool", "?")
                    decomp = t.get("decompress_tool", "?")
                    level = t.get("level", "?")
                    threads = t.get("threads", "?")
                    lines.append(f"  - ❌ {comp} → {decomp} (L{level}, {threads}t)")
    else:
        lines.append("No decompression tests recorded")
    
    lines.append("")
    
    # Key stats
    if best_gzip_speedup > 0:
        lines.append("### Key Stats")
        lines.append("")
        lines.append(f"- **{best_gzip_speedup:.0f}×** faster than gzip (best case)")
        lines.append(f"- **{best_pigz_speedup:.1f}×** faster than pigz (best case)")
        lines.append(f"- **{best_throughput:.0f} MB/s** peak throughput")
        lines.append("")
    
    # Errors
    if results.get("errors"):
        lines.append("### Errors")
        lines.append("")
        for error in results["errors"]:
            lines.append(f"- ❌ {error}")
        lines.append("")
    
    # Footer
    if format_type == "github":
        lines.append("<details>")
        lines.append("<summary>View full validation matrix</summary>")
        lines.append("")
        lines.append("See the uploaded `validation-results.json` artifact for complete data.")
        lines.append("")
        lines.append("</details>")
    
    return "\n".join(lines)


def generate_summary_from_validate(results, format_type="markdown"):
    """Generate summary from validate.py output."""
    lines = []
    
    passed = results['summary']['passed']
    failed = results['summary']['failed']
    total = passed + failed
    test_size = format_size(results.get('test_size_bytes', 0))
    
    # Header
    if format_type == "github":
        lines.append("## 🔄 gzippy Validation Results")
    else:
        lines.append("# Validation Results")
    lines.append("")
    
    # Status badge
    if failed == 0:
        lines.append(f"**✅ All {passed} tests passed** (tested on {test_size})")
    else:
        lines.append(f"**❌ {failed}/{total} tests failed** (tested on {test_size})")
    lines.append("")
    
    # Compression performance table
    lines.append("### Compression Performance")
    lines.append("")
    lines.append("| Config | gzippy Time | vs gzip | vs pigz | Ratio | Size vs pigz |")
    lines.append("|--------|-----------|---------|---------|-------|--------------|")
    
    # Group compression by config
    comp_by_config = defaultdict(dict)
    for r in results.get("compression", []):
        if r.get("success"):
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    test_size_bytes = results.get("test_size_bytes", 0)
    best_gzip_speedup = 0
    best_pigz_speedup = 0
    best_throughput = 0
    
    for (level, threads), tools in sorted(comp_by_config.items()):
        config = f"L{level}, {threads}t"
        max_time_overhead, max_size_overhead = get_thresholds(level)
        
        gzip_time = tools.get("gzip", {}).get("median_seconds", 0)
        pigz_time = tools.get("pigz", {}).get("median_seconds", 0)
        gzippy_time = tools.get("gzippy", {}).get("median_seconds", 0)
        
        gzippy_size = tools.get("gzippy", {}).get("output_size", 0)
        pigz_size = tools.get("pigz", {}).get("output_size", 0)
        
        gzippy_str = format_time(gzippy_time) if gzippy_time else "—"
        
        # vs gzip speedup
        if gzippy_time and gzip_time:
            gzip_speedup = gzip_time / gzippy_time
            best_gzip_speedup = max(best_gzip_speedup, gzip_speedup)
            if gzip_speedup < 1.0:
                gzip_speedup_str = f"*{gzip_speedup:.2f}×*"
            else:
                gzip_speedup_str = f"**{gzip_speedup:.1f}×**"
        else:
            gzip_speedup_str = "—"
        
        # vs pigz speedup
        if gzippy_time and pigz_time:
            pigz_speedup = pigz_time / gzippy_time
            pigz_overhead = (gzippy_time / pigz_time - 1) * 100
            best_pigz_speedup = max(best_pigz_speedup, pigz_speedup)
            if pigz_overhead > max_time_overhead:
                pigz_speedup_str = f"*{pigz_speedup:.2f}×*"
            else:
                pigz_speedup_str = f"**{pigz_speedup:.1f}×**"
        else:
            pigz_speedup_str = "—"
        
        # Compression ratio
        if gzippy_size and test_size_bytes:
            ratio = gzippy_size / test_size_bytes * 100
            ratio_str = f"{ratio:.1f}%"
        else:
            ratio_str = "—"
        
        # Size vs pigz
        if gzippy_size and pigz_size:
            size_diff = (gzippy_size / pigz_size - 1) * 100
            if size_diff > max_size_overhead:
                size_str = f"*{size_diff:+.1f}%*"
            elif size_diff < 0:
                size_str = f"**{size_diff:+.1f}%**"
            else:
                size_str = f"{size_diff:+.1f}%"
        else:
            size_str = "—"
        
        if gzippy_time and test_size_mb:
            throughput = test_size_mb / gzippy_time
            best_throughput = max(best_throughput, throughput)
        
        lines.append(f"| {config} | {gzippy_str} | {gzip_speedup_str} | {pigz_speedup_str} | {ratio_str} | {size_str} |")
    
    lines.append("")
    lines.append("*Italics indicate failing to beat threshold*")
    lines.append("")
    
    # Decompression summary
    decomp_results = results.get("decompression", [])
    decomp_passed = sum(1 for r in decomp_results if r.get("success") and r.get("correct"))
    decomp_total = len(decomp_results)
    
    lines.append("### Decompression")
    lines.append("")
    if decomp_passed == decomp_total:
        lines.append(f"✅ All {decomp_total} cross-tool decompression tests passed")
    else:
        lines.append(f"⚠️ {decomp_passed}/{decomp_total} decompression tests passed")
        for r in decomp_results:
            if not (r.get("success") and r.get("correct")):
                comp = r.get("compressor", "?")
                decomp = r.get("decompressor", "?")
                level = r.get("level", "?")
                threads = r.get("threads", "?")
                lines.append(f"  - ❌ {comp} → {decomp} (L{level}, {threads}t)")
    
    lines.append("")
    
    # Key stats
    lines.append("### Key Stats")
    lines.append("")
    lines.append(f"- **{best_gzip_speedup:.0f}×** faster than gzip (best case)")
    lines.append(f"- **{best_pigz_speedup:.1f}×** faster than pigz (best case)")
    lines.append(f"- **{best_throughput:.0f} MB/s** peak throughput")
    lines.append("")
    
    # Footer
    if format_type == "github":
        lines.append("<details>")
        lines.append("<summary>View full validation matrix</summary>")
        lines.append("")
        lines.append("See the uploaded `validation-results.json` artifact for complete data.")
        lines.append("")
        lines.append("</details>")
    
    return "\n".join(lines)


def generate_summary(results, format_type="markdown"):
    """Generate markdown summary from validation results (auto-detect format)."""
    fmt = detect_format(results)
    if fmt == "validate":
        return generate_summary_from_validate(results, format_type)
    else:
        return generate_summary_from_ci(results, format_type)


def main():
    parser = argparse.ArgumentParser(description="Generate validation summary")
    parser.add_argument("input", help="JSON results file")
    parser.add_argument("--format", choices=["markdown", "github"], default="markdown",
                       help="Output format (github adds PR-friendly formatting)")
    args = parser.parse_args()
    
    with open(args.input) as f:
        results = json.load(f)
    
    print(generate_summary(results, args.format))


if __name__ == "__main__":
    main()
