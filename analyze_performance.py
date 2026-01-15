#!/usr/bin/env python3
"""
LLM-optimized performance analysis for rigz.
Provides structured feedback for code optimization.
"""

import csv
import sys
from collections import defaultdict


def analyze_performance_results(csv_file):
    """Generate LLM-actionable performance analysis."""
    try:
        with open(csv_file, 'r') as f:
            results = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return False

    if not results:
        print("No performance data found.")
        return False

    # Group by test scenario
    scenarios = defaultdict(list)
    for row in results:
        key = f"{row['file']}_L{row['level']}_T{row['threads']}"
        scenarios[key].append(row)

    performance_gaps = []
    wins = 0
    total = 0

    print("PERFORMANCE ANALYSIS FOR LLM OPTIMIZATION")
    print("=" * 50)
    
    # System info
    print(f"System: {len([r for r in results if r['tool'] == 'rigz'])} test scenarios")
    print()

    for scenario, tests in scenarios.items():
        if len(tests) == 3:  # gzip, pigz, rigz
            total += 1
            rigz = next(t for t in tests if t["tool"] == "rigz")
            pigz = next(t for t in tests if t["tool"] == "pigz") 
            gzip = next(t for t in tests if t["tool"] == "gzip")
            
            rigz_speed = float(rigz["speed"])
            pigz_speed = float(pigz["speed"])
            gzip_speed = float(gzip["speed"])
            
            # Parse scenario details
            parts = scenario.split('_')
            file_info = parts[0]
            level = parts[1][1:]  # Remove 'L'
            threads = parts[2][1:]  # Remove 'T'
            
            # Determine file characteristics
            if 'text' in file_info:
                data_type = 'text'
            elif 'binary' in file_info:
                data_type = 'binary'
            else:
                data_type = 'random'
                
            if '1MB' in file_info:
                file_size = '1MB'
            elif '10MB' in file_info:
                file_size = '10MB'
            elif '100MB' in file_info:
                file_size = '100MB'
            else:
                file_size = 'unknown'

            if rigz_speed > max(pigz_speed, gzip_speed):
                wins += 1
                status = "WIN"
            elif rigz_speed < min(pigz_speed, gzip_speed):
                status = "LOSS"
                # Calculate performance gaps (handle division by zero)
                if rigz_speed > 0:
                    pigz_gap = (pigz_speed - rigz_speed) / rigz_speed * 100
                    gzip_gap = (gzip_speed - rigz_speed) / rigz_speed * 100
                else:
                    pigz_gap = 0.0
                    gzip_gap = 0.0
                
                performance_gaps.append({
                    'scenario': scenario,
                    'data_type': data_type,
                    'file_size': file_size,
                    'level': level,
                    'threads': threads,
                    'rigz_speed': rigz_speed,
                    'pigz_speed': pigz_speed,
                    'gzip_speed': gzip_speed,
                    'pigz_gap': pigz_gap,
                    'gzip_gap': gzip_gap,
                    'compression_ratio': float(rigz['ratio'])
                })
            else:
                status = "MIXED"
            
            print(f"{status:5} {scenario:25} rigz:{rigz_speed:5.1f} pigz:{pigz_speed:5.1f} gzip:{gzip_speed:5.1f} MB/s")

    win_rate = wins / total * 100 if total > 0 else 0
    print(f"\nOVERALL: {wins}/{total} wins ({win_rate:.1f}%)")
    
    if performance_gaps:
        print(f"\nPERFORMANCE GAPS REQUIRING OPTIMIZATION:")
        print("-" * 50)
        
        # Sort by biggest performance gap vs pigz
        performance_gaps.sort(key=lambda x: x['pigz_gap'], reverse=True)
        
        for gap in performance_gaps:
            print(f"SCENARIO: {gap['scenario']}")
            print(f"  Data: {gap['data_type']}, Size: {gap['file_size']}, Level: {gap['level']}, Threads: {gap['threads']}")
            print(f"  rigz: {gap['rigz_speed']:.1f} MB/s")
            print(f"  pigz: {gap['pigz_speed']:.1f} MB/s ({gap['pigz_gap']:+.1f}% gap)")
            print(f"  gzip: {gap['gzip_speed']:.1f} MB/s ({gap['gzip_gap']:+.1f}% gap)")
            print(f"  compression_ratio: {gap['compression_ratio']:.3f}")
            print()
        
        # Pattern analysis for LLM
        print("OPTIMIZATION PATTERNS:")
        print("-" * 30)
        
        # Analyze by data type
        by_type = defaultdict(list)
        for gap in performance_gaps:
            by_type[gap['data_type']].append(gap)
            
        for data_type, gaps in by_type.items():
            avg_gap = sum(g['pigz_gap'] for g in gaps) / len(gaps)
            print(f"{data_type} data: {len(gaps)} failures, avg {avg_gap:.1f}% slower than pigz")
        
        # Analyze by file size
        by_size = defaultdict(list)
        for gap in performance_gaps:
            by_size[gap['file_size']].append(gap)
            
        print("By file size:")
        for size, gaps in by_size.items():
            avg_gap = sum(g['pigz_gap'] for g in gaps) / len(gaps)
            print(f"  {size}: {len(gaps)} failures, avg {avg_gap:.1f}% slower")
        
        # Analyze by thread count
        by_threads = defaultdict(list)
        for gap in performance_gaps:
            by_threads[gap['threads']].append(gap)
            
        print("By thread count:")
        for threads, gaps in by_threads.items():
            avg_gap = sum(g['pigz_gap'] for g in gaps) / len(gaps)
            print(f"  {threads} threads: {len(gaps)} failures, avg {avg_gap:.1f}% slower")
        
        # Analyze by compression level
        by_level = defaultdict(list)
        for gap in performance_gaps:
            by_level[gap['level']].append(gap)
            
        print("By compression level:")
        for level, gaps in by_level.items():
            avg_gap = sum(g['pigz_gap'] for g in gaps) / len(gaps)
            print(f"  Level {level}: {len(gaps)} failures, avg {avg_gap:.1f}% slower")
        
        print("\nACTIONABLE RECOMMENDATIONS:")
        print("-" * 35)
        
        # Priority recommendations based on patterns
        if by_threads.get('8', []):
            print("HIGH PRIORITY: Thread scaling issues at 8 threads")
            print("  - Review lock contention in multi-threaded code")
            print("  - Check memory bandwidth saturation")
            print("  - Optimize work distribution algorithms")
        
        worst_type = max(by_type.items(), key=lambda x: len(x[1]))
        if worst_type:
            print(f"MEDIUM PRIORITY: {worst_type[0]} data optimization")
            print(f"  - {len(worst_type[1])} failures suggest algorithm tuning needed")
            print(f"  - Review compression strategy for {worst_type[0]} content")
        
        if by_level.get('9', []):
            print("LOW PRIORITY: Maximum compression level efficiency")
            print("  - Level 9 performance gaps may be acceptable for quality trade-off")
    
    else:
        print("\nEXCELLENT: rigz outperforms alternatives in all tested scenarios!")
        print("Consider expanding test coverage or increasing optimization targets.")
    
    return True


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "test_results/performance.csv"
    if not analyze_performance_results(csv_file):
        sys.exit(1)