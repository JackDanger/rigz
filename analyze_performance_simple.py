#!/usr/bin/env python3
"""
Simple Performance Analysis Script for rigz vs pigz
Reads test_results/performance.md and identifies where rigz underperforms pigz
Uses only Python standard library - no external dependencies required
"""

import re
import json
from pathlib import Path
from collections import defaultdict, Counter

def parse_performance_data(file_path):
    """Parse the performance.md file and extract comparison data"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the table section
    lines = content.split('\n')
    
    # Look for table header and data
    data = []
    header_found = False
    
    for line in lines:
        if '| File |' in line and '| Tool |' in line:
            header_found = True
            continue
        
        if header_found and line.startswith('|') and ('rigz' in line or 'pigz' in line):
            # Skip the header separator line
            if '---' in line:
                continue
                
            # Parse table row
            parts = [p.strip() for p in line.split('|')]
            # Remove empty parts from beginning and end
            parts = [p for p in parts if p]
            
            if len(parts) >= 8:
                try:
                    filename = parts[0]
                    size = int(parts[1]) if parts[1].isdigit() else 0
                    level = int(parts[2])
                    threads = int(parts[3])
                    tool = parts[4]
                    # Handle time with leading dot
                    time_str = parts[5]
                    if time_str.startswith('.'):
                        time_s = float('0' + time_str)
                    else:
                        time_s = float(time_str)
                    speed_mbs = float(parts[6]) if parts[6] != '0' else 0
                    ratio = parts[7] if len(parts) > 7 else '100%'
                    
                    # Only include rigz and pigz data
                    if tool in ['rigz', 'pigz']:
                        data.append({
                            'filename': filename,
                            'size': size,
                            'level': level,
                            'threads': threads,
                            'tool': tool,
                            'time_s': time_s,
                            'speed_mbs': speed_mbs,
                            'ratio': ratio
                        })
                except (ValueError, IndexError) as e:
                    print(f"Parse error for line: {line} -> {e}")
                    continue
    
    return data

def find_underperformance(data):
    """Find cases where rigz underperforms pigz"""
    
    # Group by test configuration
    grouped = defaultdict(list)
    for entry in data:
        key = (entry['filename'], entry['size'], entry['level'], entry['threads'])
        grouped[key].append(entry)
    
    underperformance_cases = []
    
    for key, group in grouped.items():
        rigz_data = [entry for entry in group if entry['tool'] == 'rigz']
        pigz_data = [entry for entry in group if entry['tool'] == 'pigz']
        
        if len(rigz_data) == 1 and len(pigz_data) == 1:
            rigz_time = rigz_data[0]['time_s']
            pigz_time = pigz_data[0]['time_s']
            
            # Check if rigz is slower (allowing 5% margin for measurement error)
            if rigz_time > pigz_time * 1.05:
                slowdown_pct = ((rigz_time / pigz_time) - 1) * 100
                
                underperformance_cases.append({
                    'filename': key[0],
                    'size': key[1],
                    'size_category': categorize_file_size(key[1]),
                    'level': key[2],
                    'threads': key[3],
                    'rigz_time': rigz_time,
                    'pigz_time': pigz_time,
                    'slowdown_pct': slowdown_pct,
                    'rigz_speed': rigz_data[0]['speed_mbs'],
                    'pigz_speed': pigz_data[0]['speed_mbs']
                })
    
    return underperformance_cases

def categorize_file_size(size):
    """Categorize file sizes for better analysis"""
    if size <= 10240:  # 10KB
        return "Tiny (≤10KB)"
    elif size <= 102400:  # 100KB
        return "Small (≤100KB)"
    elif size <= 1048576:  # 1MB
        return "Medium (≤1MB)"
    elif size <= 10485760:  # 10MB
        return "Large (≤10MB)"
    else:
        return "Very Large (>10MB)"

def analyze_patterns(underperf_cases):
    """Analyze patterns in underperformance cases"""
    
    if not underperf_cases:
        return {}
    
    # Count by different dimensions
    level_counts = Counter()
    thread_counts = Counter()
    size_counts = Counter()
    file_type_counts = Counter()
    pattern_counts = Counter()
    
    level_slowdowns = defaultdict(list)
    thread_slowdowns = defaultdict(list)
    size_slowdowns = defaultdict(list)
    
    for case in underperf_cases:
        level = case['level']
        threads = case['threads']
        size_cat = case['size_category']
        slowdown = case['slowdown_pct']
        
        # Extract file type from filename
        file_type = case['filename'].split('-')[0] if '-' in case['filename'] else 'other'
        
        # Count occurrences
        level_counts[level] += 1
        thread_counts[threads] += 1
        size_counts[size_cat] += 1
        file_type_counts[file_type] += 1
        pattern_counts[f"L{level}-T{threads}"] += 1
        
        # Track slowdowns for averages
        level_slowdowns[level].append(slowdown)
        thread_slowdowns[threads].append(slowdown)
        size_slowdowns[size_cat].append(slowdown)
    
    # Calculate averages
    level_avg = {k: sum(v)/len(v) for k, v in level_slowdowns.items()}
    thread_avg = {k: sum(v)/len(v) for k, v in thread_slowdowns.items()}
    size_avg = {k: sum(v)/len(v) for k, v in size_slowdowns.items()}
    
    return {
        'counts': {
            'level': dict(level_counts),
            'threads': dict(thread_counts),
            'size': dict(size_counts),
            'file_type': dict(file_type_counts),
            'patterns': dict(pattern_counts)
        },
        'averages': {
            'level': level_avg,
            'threads': thread_avg,
            'size': size_avg
        }
    }

def create_simple_plots(underperf_cases, output_dir):
    """Create simple text-based visualizations"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not underperf_cases:
        with open(output_dir / 'underperformance_analysis.txt', 'w') as f:
            f.write("🎉 No underperformance cases found!\n")
            f.write("rigz meets or exceeds pigz performance in all tested scenarios.\n")
        return
    
    patterns = analyze_patterns(underperf_cases)
    
    # Create text-based analysis
    analysis = []
    analysis.append("=" * 80)
    analysis.append("RIGZ vs PIGZ UNDERPERFORMANCE ANALYSIS")
    analysis.append("=" * 80)
    analysis.append("")
    
    # Summary stats
    total_cases = len(underperf_cases)
    avg_slowdown = sum(case['slowdown_pct'] for case in underperf_cases) / total_cases
    max_slowdown = max(case['slowdown_pct'] for case in underperf_cases)
    worst_case = max(underperf_cases, key=lambda x: x['slowdown_pct'])
    
    analysis.append(f"SUMMARY:")
    analysis.append(f"- Total underperformance cases: {total_cases}")
    analysis.append(f"- Average slowdown: {avg_slowdown:.1f}%")
    analysis.append(f"- Maximum slowdown: {max_slowdown:.1f}%")
    analysis.append(f"- Worst case: {worst_case['filename']} L{worst_case['level']} T{worst_case['threads']} ({max_slowdown:.1f}% slower)")
    analysis.append("")
    
    # Top problematic patterns
    analysis.append("TOP PROBLEMATIC PATTERNS:")
    top_patterns = sorted(patterns['counts']['patterns'].items(), key=lambda x: x[1], reverse=True)[:5]
    for pattern, count in top_patterns:
        analysis.append(f"- {pattern}: {count} cases")
    analysis.append("")
    
    # Analysis by compression level
    analysis.append("BY COMPRESSION LEVEL:")
    level_data = [(k, patterns['counts']['level'][k], patterns['averages']['level'][k]) 
                  for k in sorted(patterns['counts']['level'].keys())]
    for level, count, avg_slowdown in level_data:
        analysis.append(f"- Level {level}: {count} cases, avg {avg_slowdown:.1f}% slower")
    analysis.append("")
    
    # Analysis by thread count
    analysis.append("BY THREAD COUNT:")
    thread_data = [(k, patterns['counts']['threads'][k], patterns['averages']['threads'][k]) 
                   for k in sorted(patterns['counts']['threads'].keys())]
    for threads, count, avg_slowdown in thread_data:
        analysis.append(f"- {threads} threads: {count} cases, avg {avg_slowdown:.1f}% slower")
    analysis.append("")
    
    # Analysis by file size
    analysis.append("BY FILE SIZE:")
    size_data = [(k, patterns['counts']['size'][k], patterns['averages']['size'][k]) 
                 for k in patterns['counts']['size'].keys()]
    size_data.sort(key=lambda x: x[1], reverse=True)  # Sort by count
    for size_cat, count, avg_slowdown in size_data:
        analysis.append(f"- {size_cat}: {count} cases, avg {avg_slowdown:.1f}% slower")
    analysis.append("")
    
    # Worst cases detail
    analysis.append("TOP 10 WORST CASES:")
    worst_cases = sorted(underperf_cases, key=lambda x: x['slowdown_pct'], reverse=True)[:10]
    for i, case in enumerate(worst_cases, 1):
        analysis.append(f"{i:2d}. {case['filename']:20s} L{case['level']} T{case['threads']:2d} | "
                       f"{case['slowdown_pct']:6.1f}% slower | "
                       f"rigz: {case['rigz_time']:6.3f}s | pigz: {case['pigz_time']:6.3f}s")
    analysis.append("")
    
    # Create CSV data for detailed analysis
    csv_lines = ["filename,size,size_category,level,threads,rigz_time,pigz_time,slowdown_pct,rigz_speed,pigz_speed"]
    for case in underperf_cases:
        csv_lines.append(f"{case['filename']},{case['size']},{case['size_category']},"
                        f"{case['level']},{case['threads']},{case['rigz_time']:.6f},"
                        f"{case['pigz_time']:.6f},{case['slowdown_pct']:.2f},"
                        f"{case['rigz_speed']:.2f},{case['pigz_speed']:.2f}")
    
    # Write files
    with open(output_dir / 'underperformance_analysis.txt', 'w') as f:
        f.write('\n'.join(analysis))
    
    with open(output_dir / 'underperformance_data.csv', 'w') as f:
        f.write('\n'.join(csv_lines))
    
    with open(output_dir / 'underperformance_data.json', 'w') as f:
        json.dump(underperf_cases, f, indent=2)
    
    print(f"📁 Analysis files saved to: {output_dir}/")
    print(f"   - underperformance_analysis.txt (detailed report)")
    print(f"   - underperformance_data.csv (raw data)")
    print(f"   - underperformance_data.json (structured data)")

def generate_summary_report(data, underperf_cases):
    """Generate a concise summary report"""
    
    total_tests = len(set((entry['filename'], entry['size'], entry['level'], entry['threads']) 
                         for entry in data if entry['tool'] == 'rigz'))
    underperf_count = len(underperf_cases)
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total test configurations: {total_tests}")
    print(f"Cases where rigz underperforms: {underperf_count} ({underperf_count/total_tests*100:.1f}%)")
    print(f"Cases where rigz meets/exceeds pigz: {total_tests - underperf_count} ({(total_tests-underperf_count)/total_tests*100:.1f}%)")
    
    if underperf_cases:
        avg_slowdown = sum(case['slowdown_pct'] for case in underperf_cases) / len(underperf_cases)
        max_slowdown = max(case['slowdown_pct'] for case in underperf_cases)
        worst_case = max(underperf_cases, key=lambda x: x['slowdown_pct'])
        
        print(f"\nUnderperformance Details:")
        print(f"- Average slowdown: {avg_slowdown:.1f}%")
        print(f"- Maximum slowdown: {max_slowdown:.1f}%")
        print(f"- Worst case: {worst_case['filename']} (L{worst_case['level']}, T{worst_case['threads']}) - {max_slowdown:.1f}% slower")
        
        # Quick pattern analysis
        patterns = analyze_patterns(underperf_cases)
        print(f"\nMost problematic scenarios:")
        top_patterns = sorted(patterns['counts']['patterns'].items(), key=lambda x: x[1], reverse=True)[:3]
        for pattern, count in top_patterns:
            print(f"- {pattern}: {count} cases")
    else:
        print("\n🎉 Excellent! No underperformance cases detected.")
        print("rigz meets or exceeds pigz performance in all scenarios.")

def main():
    """Main analysis function"""
    
    # File paths
    performance_file = Path("test_results/performance.md")
    output_dir = Path("performance_analysis")
    
    if not performance_file.exists():
        print(f"❌ Performance file not found: {performance_file}")
        print("Make sure you're running this from the rigz directory and that test_results/performance.md exists.")
        return
    
    print("🔍 Analyzing rigz vs pigz performance data...")
    
    # Parse data
    data = parse_performance_data(performance_file)
    print(f"📊 Loaded {len(data)} performance data points")
    
    # Debug: show sample of parsed data
    if data:
        print(f"Sample data point: {data[0]}")
    else:
        print("⚠️  No data points found - checking file format...")
        with open(performance_file, 'r') as f:
            lines = f.readlines()[:15]
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}: {line.rstrip()}")
    
    # Find underperformance cases
    underperf_cases = find_underperformance(data)
    print(f"⚠️  Found {len(underperf_cases)} underperformance cases")
    
    # Create analysis files
    create_simple_plots(underperf_cases, output_dir)
    
    # Generate summary
    generate_summary_report(data, underperf_cases)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()