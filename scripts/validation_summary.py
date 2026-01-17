#!/usr/bin/env python3
"""
Generate markdown summary from validation results.

Usage:
    python3 scripts/validation_summary.py validation-results.json
    python3 scripts/validation_summary.py validation-results.json --format github  # For PR comments
"""

import json
import sys
import argparse


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


def generate_summary(results, format_type="markdown"):
    """Generate markdown summary from validation results."""
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
    from collections import defaultdict
    comp_by_config = defaultdict(dict)
    for r in results.get("compression", []):
        if r.get("success"):
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    
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
        else:
            speedup_str = "—"
        
        lines.append(f"| {config} | {gzip_str} | {pigz_str} | {rigz_str} | {speedup_str} |")
    
    lines.append("")
    
    # Decompression summary (just show if all passed)
    decomp_results = results.get("decompression", [])
    decomp_passed = sum(1 for r in decomp_results if r.get("success") and r.get("correct"))
    decomp_total = len(decomp_results)
    
    lines.append("### Decompression")
    lines.append("")
    if decomp_passed == decomp_total:
        lines.append(f"✅ All {decomp_total} cross-tool decompression tests passed")
    else:
        lines.append(f"⚠️ {decomp_passed}/{decomp_total} decompression tests passed")
        
        # List failures
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
    
    # Find best speedups
    best_gzip_speedup = 0
    best_pigz_speedup = 0
    best_throughput = 0
    
    for (level, threads), tools in comp_by_config.items():
        if "rigz" in tools:
            rigz_time = tools["rigz"]["median_seconds"]
            if rigz_time > 0:
                throughput = test_size_mb / rigz_time
                best_throughput = max(best_throughput, throughput)
                
                if "gzip" in tools:
                    speedup = tools["gzip"]["median_seconds"] / rigz_time
                    best_gzip_speedup = max(best_gzip_speedup, speedup)
                if "pigz" in tools:
                    speedup = tools["pigz"]["median_seconds"] / rigz_time
                    best_pigz_speedup = max(best_pigz_speedup, speedup)
    
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
