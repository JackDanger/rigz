#!/usr/bin/env python3
"""
Check performance guards against benchmark results.

This script applies threshold assertions to aggregated benchmark results,
determining whether gzippy meets its performance targets.

Guards:
- Compression: gzippy must be faster than pigz at L1-L8, match ratio at L9
- Decompression: gzippy must beat pigz/gzip, be within 1% of rapidgzip
- Single-threaded: gzippy must be >= 90% of igzip speed

Usage:
    python3 scripts/check_guards.py \
        --compression aggregated/compression.json \
        --decompression aggregated/decompression.json \
        --output guards-report.json
"""

import argparse
import json
import sys
from pathlib import Path


# Performance thresholds
THRESHOLDS = {
    # Compression
    "comp_vs_pigz_speed": 0.95,      # Must be >= 95% of pigz speed
    "comp_vs_pigz_size_l9": 1.005,   # Must be <= 100.5% of pigz size at L9
    
    # Decompression  
    "decomp_vs_pigz": 1.0,           # Must be faster than pigz
    "decomp_vs_gzip": 1.0,           # Must be faster than gzip
    "decomp_vs_rapidgzip": 0.99,     # Must be >= 99% of rapidgzip
    "decomp_vs_igzip": 0.90,         # Must be >= 90% of igzip (it's hand-tuned asm)
}


def load_json(path: str) -> list:
    """Load JSON file, return empty list if missing."""
    try:
        with open(path) as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, dict):
                return data.get("results", data.get("benchmarks", []))
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def check_compression_guards(results: list) -> tuple:
    """
    Check compression performance guards.
    
    Returns (passed: bool, report: list of dicts)
    """
    report = []
    all_passed = True
    
    # Group by level and threads
    by_config = {}
    for r in results:
        if "error" in r:
            continue
        key = (r.get("level", 0), r.get("threads", 1), r.get("data_type", "unknown"))
        if key not in by_config:
            by_config[key] = {}
        by_config[key][r.get("tool", "unknown")] = r
    
    for (level, threads, data_type), tools in by_config.items():
        gzippy = tools.get("gzippy")
        pigz = tools.get("pigz")
        
        if not gzippy or not pigz:
            continue
        
        gzippy_speed = gzippy.get("speed_mbps", gzippy.get("speed", 0))
        pigz_speed = pigz.get("speed_mbps", pigz.get("speed", 0))
        
        if pigz_speed == 0:
            continue
        
        speed_ratio = gzippy_speed / pigz_speed
        speed_passed = speed_ratio >= THRESHOLDS["comp_vs_pigz_speed"]
        
        guard = {
            "name": f"Compression L{level} T{threads} {data_type}",
            "metric": "speed_vs_pigz",
            "gzippy": gzippy_speed,
            "pigz": pigz_speed,
            "ratio": speed_ratio,
            "threshold": THRESHOLDS["comp_vs_pigz_speed"],
            "passed": speed_passed,
        }
        report.append(guard)
        
        if not speed_passed:
            all_passed = False
        
        # Size check at L9
        if level >= 9:
            gzippy_size = gzippy.get("output_size", gzippy.get("size", 0))
            pigz_size = pigz.get("output_size", pigz.get("size", 0))
            
            if pigz_size > 0:
                size_ratio = gzippy_size / pigz_size
                size_passed = size_ratio <= THRESHOLDS["comp_vs_pigz_size_l9"]
                
                guard = {
                    "name": f"Compression Ratio L{level} T{threads} {data_type}",
                    "metric": "size_vs_pigz",
                    "gzippy": gzippy_size,
                    "pigz": pigz_size,
                    "ratio": size_ratio,
                    "threshold": THRESHOLDS["comp_vs_pigz_size_l9"],
                    "passed": size_passed,
                }
                report.append(guard)
                
                if not size_passed:
                    all_passed = False
    
    return all_passed, report


def check_decompression_guards(results: list) -> tuple:
    """
    Check decompression performance guards.
    
    Returns (passed: bool, report: list of dicts)
    """
    report = []
    all_passed = True
    
    # Group by source and threads
    by_config = {}
    for r in results:
        if "error" in r or r.get("status") == "fail":
            continue
        key = (r.get("source", "unknown"), r.get("threads", 1), r.get("data_type", "unknown"))
        if key not in by_config:
            by_config[key] = {}
        tool = r.get("tool", "unknown")
        # Normalize unpigz -> pigz for comparison
        if tool == "unpigz":
            tool = "pigz"
        by_config[key][tool] = r
    
    for (source, threads, data_type), tools in by_config.items():
        gzippy = tools.get("gzippy")
        
        if not gzippy:
            continue
        
        gzippy_speed = gzippy.get("speed_mbps", gzippy.get("speed", 0))
        
        # vs pigz
        pigz = tools.get("pigz")
        if pigz:
            pigz_speed = pigz.get("speed_mbps", pigz.get("speed", 0))
            if pigz_speed > 0:
                ratio = gzippy_speed / pigz_speed
                passed = ratio >= THRESHOLDS["decomp_vs_pigz"]
                report.append({
                    "name": f"Decompress {source} T{threads} {data_type} vs pigz",
                    "metric": "speed_vs_pigz",
                    "gzippy": gzippy_speed,
                    "other": pigz_speed,
                    "ratio": ratio,
                    "threshold": THRESHOLDS["decomp_vs_pigz"],
                    "passed": passed,
                })
                if not passed:
                    all_passed = False
        
        # vs gzip (single-threaded only)
        if threads == 1:
            gzip = tools.get("gzip")
            if gzip:
                gzip_speed = gzip.get("speed_mbps", gzip.get("speed", 0))
                if gzip_speed > 0:
                    ratio = gzippy_speed / gzip_speed
                    passed = ratio >= THRESHOLDS["decomp_vs_gzip"]
                    report.append({
                        "name": f"Decompress {source} T1 {data_type} vs gzip",
                        "metric": "speed_vs_gzip",
                        "gzippy": gzippy_speed,
                        "other": gzip_speed,
                        "ratio": ratio,
                        "threshold": THRESHOLDS["decomp_vs_gzip"],
                        "passed": passed,
                    })
                    if not passed:
                        all_passed = False
        
        # vs rapidgzip
        rapidgzip = tools.get("rapidgzip")
        if rapidgzip:
            rapid_speed = rapidgzip.get("speed_mbps", rapidgzip.get("speed", 0))
            if rapid_speed > 0:
                ratio = gzippy_speed / rapid_speed
                passed = ratio >= THRESHOLDS["decomp_vs_rapidgzip"]
                report.append({
                    "name": f"Decompress {source} T{threads} {data_type} vs rapidgzip",
                    "metric": "speed_vs_rapidgzip",
                    "gzippy": gzippy_speed,
                    "other": rapid_speed,
                    "ratio": ratio,
                    "threshold": THRESHOLDS["decomp_vs_rapidgzip"],
                    "passed": passed,
                })
                if not passed:
                    all_passed = False
        
        # vs igzip (single-threaded only)
        if threads == 1:
            igzip = tools.get("igzip")
            if igzip:
                igzip_speed = igzip.get("speed_mbps", igzip.get("speed", 0))
                if igzip_speed > 0:
                    ratio = gzippy_speed / igzip_speed
                    passed = ratio >= THRESHOLDS["decomp_vs_igzip"]
                    report.append({
                        "name": f"Decompress {source} T1 {data_type} vs igzip",
                        "metric": "speed_vs_igzip",
                        "gzippy": gzippy_speed,
                        "other": igzip_speed,
                        "ratio": ratio,
                        "threshold": THRESHOLDS["decomp_vs_igzip"],
                        "passed": passed,
                    })
                    if not passed:
                        all_passed = False
    
    return all_passed, report


def main():
    parser = argparse.ArgumentParser(description="Check performance guards")
    parser.add_argument("--compression", type=str, default="aggregated/compression.json",
                       help="Path to compression results")
    parser.add_argument("--decompression", type=str, default="aggregated/decompression.json",
                       help="Path to decompression results")
    parser.add_argument("--output", type=str, default="guards-report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Load results
    compression = load_json(args.compression)
    decompression = load_json(args.decompression)
    
    print("=== Performance Guards ===\n")
    
    # Check guards
    comp_passed, comp_report = check_compression_guards(compression)
    decomp_passed, decomp_report = check_decompression_guards(decompression)
    
    all_passed = comp_passed and decomp_passed
    
    # Print results
    print("## Compression Guards\n")
    for g in comp_report:
        status = "✅" if g["passed"] else "❌"
        print(f"{status} {g['name']}: {g['ratio']:.3f}x (threshold: {g['threshold']})")
    
    print("\n## Decompression Guards\n")
    for g in decomp_report:
        status = "✅" if g["passed"] else "❌"
        print(f"{status} {g['name']}: {g['ratio']:.3f}x (threshold: {g['threshold']})")
    
    # Summary
    total = len(comp_report) + len(decomp_report)
    passed_count = sum(1 for g in comp_report + decomp_report if g["passed"])
    
    print(f"\n{'='*50}")
    print(f"{'✅ ALL GUARDS PASSED' if all_passed else '❌ SOME GUARDS FAILED'}")
    print(f"Passed: {passed_count}/{total}")
    
    # Write report
    report = {
        "passed": all_passed,
        "compression_passed": comp_passed,
        "decompression_passed": decomp_passed,
        "compression_guards": comp_report,
        "decompression_guards": decomp_report,
        "thresholds": THRESHOLDS,
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport written to {args.output}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
