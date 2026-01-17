#!/usr/bin/env python3
"""
Generate performance charts from validation results.

Reads JSON output from validate.py and generates:
1. Terminal ASCII bar charts
2. Optional HTML chart (if --html flag used)

Usage:
    python3 scripts/validate.py --json | python3 scripts/validation_chart.py
    python3 scripts/validation_chart.py results.json
    python3 scripts/validation_chart.py results.json --html > chart.html
"""

import json
import sys
import argparse
from collections import defaultdict


# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


TOOL_COLORS = {
    "gzip": Colors.BLUE,
    "pigz": Colors.YELLOW,
    "rigz": Colors.GREEN,
}

TOOL_HTML_COLORS = {
    "gzip": "#3b82f6",  # blue
    "pigz": "#eab308",  # yellow
    "rigz": "#22c55e",  # green
}


def format_time(seconds):
    """Format time with appropriate precision."""
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 10:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds:.1f}s"


def format_size(bytes_val):
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def bar_chart_ascii(label, value, max_value, width=40, color=""):
    """Generate an ASCII bar chart line."""
    if max_value == 0:
        filled = 0
    else:
        filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    reset = Colors.RESET if color else ""
    return f"  {label:20} {color}{bar}{reset} {format_time(value)}"


def print_compression_chart(results, threads_filter=None):
    """Print compression performance chart."""
    print(f"\n{Colors.BOLD}═══ Compression Performance ═══{Colors.RESET}\n")
    
    # Group by level and threads
    by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            if threads_filter is None or r["threads"] == threads_filter:
                by_config[key][r["tool"]] = r
    
    for (level, threads), tools in sorted(by_config.items()):
        print(f"{Colors.BOLD}Level {level}, {threads} thread(s){Colors.RESET}")
        
        # Find max time for scaling
        max_time = max(t["median_seconds"] for t in tools.values())
        
        # Calculate throughput (MB/s)
        test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
        
        for tool in ["gzip", "pigz", "rigz"]:
            if tool in tools:
                r = tools[tool]
                time_s = r["median_seconds"]
                throughput = test_size_mb / time_s if time_s > 0 else 0
                size_str = format_size(r["output_size_bytes"])
                color = TOOL_COLORS.get(tool, "")
                
                print(bar_chart_ascii(f"{tool} ({size_str})", time_s, max_time, color=color))
                print(f"                         └─ {throughput:.1f} MB/s")
        print()


def print_decompression_chart(results, threads_filter=None):
    """Print decompression performance chart."""
    print(f"\n{Colors.BOLD}═══ Decompression Performance ═══{Colors.RESET}\n")
    
    # Group by (level, threads, compressor)
    by_config = defaultdict(lambda: defaultdict(dict))
    for r in results["decompression"]:
        if r["success"] and r["correct"]:
            key = (r["level"], r["threads"], r["compressor"])
            if threads_filter is None or r["threads"] == threads_filter:
                by_config[key][r["decompressor"]] = r
    
    # For brevity, show rigz-compressed files only (most relevant)
    for (level, threads, compressor), decomps in sorted(by_config.items()):
        if compressor != "rigz":
            continue
            
        print(f"{Colors.BOLD}rigz L{level}@{threads}t → decompressor{Colors.RESET}")
        
        max_time = max(d["median_seconds"] for d in decomps.values())
        
        for tool in ["gzip", "pigz", "rigz"]:
            if tool in decomps:
                r = decomps[tool]
                color = TOOL_COLORS.get(tool, "")
                print(bar_chart_ascii(tool, r["median_seconds"], max_time, color=color))
        print()


def print_speedup_summary(results):
    """Print speedup summary comparing rigz to others."""
    print(f"\n{Colors.BOLD}═══ Speedup Summary (rigz vs others) ═══{Colors.RESET}\n")
    
    # Group compression by config
    by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            by_config[key][r["tool"]] = r["median_seconds"]
    
    print(f"  {'Config':<15} {'vs gzip':>12} {'vs pigz':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    
    for (level, threads), tools in sorted(by_config.items()):
        if "rigz" not in tools:
            continue
        
        rigz_time = tools["rigz"]
        gzip_speedup = tools.get("gzip", rigz_time) / rigz_time if rigz_time > 0 else 0
        pigz_speedup = tools.get("pigz", rigz_time) / rigz_time if rigz_time > 0 else 0
        
        config = f"L{level} {threads}t"
        
        # Color based on whether rigz is faster
        gzip_color = Colors.GREEN if gzip_speedup > 1 else Colors.RED
        pigz_color = Colors.GREEN if pigz_speedup > 1 else Colors.RED
        
        print(f"  {config:<15} {gzip_color}{gzip_speedup:>10.1f}x{Colors.RESET}  {pigz_color}{pigz_speedup:>10.1f}x{Colors.RESET}")
    
    print()


def generate_html_chart(results):
    """Generate an HTML page with interactive charts using Chart.js."""
    
    # Prepare compression data
    comp_data = defaultdict(lambda: defaultdict(dict))
    for r in results["compression"]:
        if r["success"]:
            key = f"L{r['level']} {r['threads']}t"
            comp_data[key][r["tool"]] = r["median_seconds"]
    
    # Prepare decompression data (rigz-compressed only)
    decomp_data = defaultdict(lambda: defaultdict(dict))
    for r in results["decompression"]:
        if r["success"] and r["correct"] and r["compressor"] == "rigz":
            key = f"L{r['level']} {r['threads']}t"
            decomp_data[key][r["decompressor"]] = r["median_seconds"]
    
    configs = sorted(comp_data.keys())
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>rigz Validation Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1, h2 {{ color: #22c55e; }}
        .chart-container {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #16213e;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #22c55e;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🚀 rigz Validation Results</h1>
    
    <div class="summary">
        <div class="stat-card">
            <div class="stat-value">{results['summary']['passed']}/{results['summary']['passed'] + results['summary']['failed']}</div>
            <div class="stat-label">Tests Passed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{format_size(results.get('test_size_bytes', 0))}</div>
            <div class="stat-label">Test File Size</div>
        </div>
    </div>

    <h2>Compression Time (lower is better)</h2>
    <div class="chart-container">
        <canvas id="compChart"></canvas>
    </div>

    <h2>Decompression Time (rigz-compressed files)</h2>
    <div class="chart-container">
        <canvas id="decompChart"></canvas>
    </div>

    <script>
        const configs = {json.dumps(configs)};
        
        const compData = {{
            labels: configs,
            datasets: [
                {{
                    label: 'gzip',
                    data: configs.map(c => {json.dumps({k: v.get('gzip', 0) for k, v in comp_data.items()})}[c] || 0),
                    backgroundColor: '{TOOL_HTML_COLORS["gzip"]}',
                }},
                {{
                    label: 'pigz',
                    data: configs.map(c => {json.dumps({k: v.get('pigz', 0) for k, v in comp_data.items()})}[c] || 0),
                    backgroundColor: '{TOOL_HTML_COLORS["pigz"]}',
                }},
                {{
                    label: 'rigz',
                    data: configs.map(c => {json.dumps({k: v.get('rigz', 0) for k, v in comp_data.items()})}[c] || 0),
                    backgroundColor: '{TOOL_HTML_COLORS["rigz"]}',
                }},
            ]
        }};

        const decompData = {{
            labels: configs,
            datasets: [
                {{
                    label: 'gzip',
                    data: configs.map(c => {json.dumps({k: v.get('gzip', 0) for k, v in decomp_data.items()})}[c] || 0),
                    backgroundColor: '{TOOL_HTML_COLORS["gzip"]}',
                }},
                {{
                    label: 'pigz',
                    data: configs.map(c => {json.dumps({k: v.get('pigz', 0) for k, v in decomp_data.items()})}[c] || 0),
                    backgroundColor: '{TOOL_HTML_COLORS["pigz"]}',
                }},
                {{
                    label: 'rigz',
                    data: configs.map(c => {json.dumps({k: v.get('rigz', 0) for k, v in decomp_data.items()})}[c] || 0),
                    backgroundColor: '{TOOL_HTML_COLORS["rigz"]}',
                }},
            ]
        }};

        const chartOptions = {{
            responsive: true,
            plugins: {{
                legend: {{ labels: {{ color: '#eee' }} }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#eee' }}, grid: {{ color: '#333' }} }},
                y: {{ 
                    ticks: {{ color: '#eee' }}, 
                    grid: {{ color: '#333' }},
                    title: {{ display: true, text: 'Seconds', color: '#eee' }}
                }}
            }}
        }};

        new Chart(document.getElementById('compChart'), {{
            type: 'bar',
            data: compData,
            options: chartOptions
        }});

        new Chart(document.getElementById('decompChart'), {{
            type: 'bar',
            data: decompData,
            options: chartOptions
        }});
    </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate charts from validation results")
    parser.add_argument("input", nargs="?", help="JSON results file (or stdin)")
    parser.add_argument("--html", action="store_true", help="Output HTML chart")
    parser.add_argument("--threads", type=int, help="Filter to specific thread count")
    args = parser.parse_args()
    
    # Read JSON input
    if args.input:
        with open(args.input) as f:
            results = json.load(f)
    else:
        results = json.load(sys.stdin)
    
    if args.html:
        print(generate_html_chart(results))
    else:
        print_compression_chart(results, threads_filter=args.threads)
        print_decompression_chart(results, threads_filter=args.threads)
        print_speedup_summary(results)


if __name__ == "__main__":
    main()
