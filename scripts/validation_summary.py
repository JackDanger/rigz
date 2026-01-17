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
        lines.append("## 🔄 rigz Validation Results")
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
    
    lines.append("| Config | gzip | pigz | rigz | Speedup |")
    lines.append("|--------|------|------|------|---------|")
    
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
        
        gzip_time = tools.get("gzip", {}).get("median_time", 0)
        pigz_time = tools.get("pigz", {}).get("median_time", 0)
        rigz_time = tools.get("rigz", {}).get("median_time", 0)
        
        gzip_str = format_time(gzip_time) if gzip_time else "—"
        pigz_str = format_time(pigz_time) if pigz_time else "—"
        rigz_str = format_time(rigz_time) if rigz_time else "—"
        
        if rigz_time and gzip_time:
            speedup = gzip_time / rigz_time
            speedup_str = f"**{speedup:.1f}×** vs gzip"
            best_gzip_speedup = max(best_gzip_speedup, speedup)
        else:
            speedup_str = "—"
        
        if rigz_time and pigz_time:
            pigz_speedup = pigz_time / rigz_time
            best_pigz_speedup = max(best_pigz_speedup, pigz_speedup)
        
        if rigz_time and size_mb:
            throughput = size_mb / rigz_time
            best_throughput = max(best_throughput, throughput)
        
        lines.append(f"| {config} | {gzip_str} | {pigz_str} | {rigz_str} | {speedup_str} |")
    
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
        lines.append("## 🔄 rigz Validation Results")
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
    lines.append("| Config | gzip | pigz | rigz | Speedup |")
    lines.append("|--------|------|------|------|---------|")
    
    # Group compression by config
    comp_by_config = defaultdict(dict)
    for r in results.get("compression", []):
        if r.get("success"):
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    best_gzip_speedup = 0
    best_pigz_speedup = 0
    best_throughput = 0
    
    for (level, threads), tools in sorted(comp_by_config.items()):
        config = f"L{level}, {threads}t"
        
        gzip_time = tools.get("gzip", {}).get("median_seconds", 0)
        pigz_time = tools.get("pigz", {}).get("median_seconds", 0)
        rigz_time = tools.get("rigz", {}).get("median_seconds", 0)
        
        gzip_str = format_time(gzip_time) if gzip_time else "—"
        pigz_str = format_time(pigz_time) if pigz_time else "—"
        rigz_str = format_time(rigz_time) if rigz_time else "—"
        
        if rigz_time and gzip_time:
            speedup = gzip_time / rigz_time
            speedup_str = f"**{speedup:.1f}×** vs gzip"
            best_gzip_speedup = max(best_gzip_speedup, speedup)
        else:
            speedup_str = "—"
        
        if rigz_time and pigz_time:
            pigz_speedup = pigz_time / rigz_time
            best_pigz_speedup = max(best_pigz_speedup, pigz_speedup)
        
        if rigz_time and test_size_mb:
            throughput = test_size_mb / rigz_time
            best_throughput = max(best_throughput, throughput)
        
        lines.append(f"| {config} | {gzip_str} | {pigz_str} | {rigz_str} | {speedup_str} |")
    
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
