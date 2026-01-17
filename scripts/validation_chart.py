#!/usr/bin/env python3
"""
Generate performance charts from validation results.

Reads JSON output from validate.py and generates:
1. Terminal ASCII bar charts
2. Optional HTML chart (if --html flag used)

Usage:
    python3 scripts/validation_chart.py test_results/validation.json
    python3 scripts/validation_chart.py test_results/validation.json --html > chart.html
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
    """Generate a warm, sophisticated HTML visualization."""
    
    # Process compression data
    comp_by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    # Calculate speedups and throughputs
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    
    # Build data for each config
    race_data = []
    for (level, threads), tools in sorted(comp_by_config.items()):
        config = {"level": level, "threads": threads, "tools": {}}
        for tool, data in tools.items():
            time_s = data["median_seconds"]
            throughput = test_size_mb / time_s if time_s > 0 else 0
            output_mb = data["output_size_bytes"] / (1024 * 1024)
            config["tools"][tool] = {
                "time": time_s,
                "throughput": throughput,
                "output_mb": output_mb,
                "ratio": data["output_size_bytes"] / data["input_size_bytes"] * 100
            }
        race_data.append(config)
    
    # Calculate best speedups for hero stats
    max_gzip_speedup = max(
        (c["tools"]["gzip"]["time"] / c["tools"]["rigz"]["time"] 
         for c in race_data if "gzip" in c["tools"] and "rigz" in c["tools"]),
        default=1
    )
    max_pigz_speedup = max(
        (c["tools"]["pigz"]["time"] / c["tools"]["rigz"]["time"] 
         for c in race_data if "pigz" in c["tools"] and "rigz" in c["tools"]),
        default=1
    )
    max_throughput = max(
        (c["tools"]["rigz"]["throughput"] for c in race_data if "rigz" in c["tools"]),
        default=0
    )
    
    # Summary
    passed = results['summary']['passed']
    failed = results['summary']['failed']
    total = passed + failed
    test_size_str = format_size(results.get('test_size_bytes', 0))
    
    # Convert race_data to JSON
    race_data_json = json.dumps(race_data)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rigz — Performance</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            /* Warm neutral palette */
            --bg: #FAF8F5;
            --bg-warm: #F5F0E8;
            --bg-card: #FFFFFF;
            --border: #E8E2D9;
            --border-dark: #D4CCC0;
            
            --text: #2C2825;
            --text-secondary: #6B635A;
            --text-muted: #9C948A;
            
            /* Earthy accent colors - distinguishable shapes/patterns, not just hue */
            --sage: #7D8B74;
            --sage-light: #E8EBE5;
            --terracotta: #C4856A;
            --terracotta-light: #F5E6E0;
            --stone: #8B8178;
            --stone-light: #EDEAE6;
            
            /* For the tools - using value contrast, not just hue */
            --tool-1: #2C2825;  /* Darkest - gzip */
            --tool-2: #7D8B74;  /* Medium sage - pigz */
            --tool-3: #C4856A;  /* Warm terracotta - rigz (the star) */
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
            font-size: 16px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 4rem 2rem;
        }}
        
        /* Header */
        header {{
            text-align: center;
            margin-bottom: 5rem;
            padding-bottom: 3rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 1.5rem;
        }}
        
        h1 {{
            font-family: 'Cormorant Garamond', serif;
            font-size: clamp(3rem, 10vw, 5rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 1rem;
        }}
        
        .subtitle {{
            font-size: 1.125rem;
            color: var(--text-secondary);
            max-width: 480px;
            margin: 0 auto;
        }}
        
        /* Summary Cards */
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 5rem;
        }}
        
        .summary-item {{
            background: var(--bg-card);
            padding: 2rem 1.5rem;
            text-align: center;
        }}
        
        .summary-value {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.5rem;
            font-weight: 500;
            color: var(--text);
            line-height: 1.1;
            margin-bottom: 0.5rem;
        }}
        
        .summary-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        
        /* Section */
        section {{
            margin-bottom: 4rem;
        }}
        
        h2 {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.75rem;
            font-weight: 400;
            margin-bottom: 2rem;
            color: var(--text);
        }}
        
        /* Legend */
        .legend {{
            display: flex;
            gap: 2rem;
            margin-bottom: 2.5rem;
            padding: 1.25rem 1.5rem;
            background: var(--bg-warm);
            border-radius: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}
        
        .legend-swatch {{
            width: 24px;
            height: 8px;
            border-radius: 4px;
        }}
        
        .legend-swatch.gzip {{
            background: var(--tool-1);
        }}
        
        .legend-swatch.pigz {{
            background: var(--tool-2);
            /* Add pattern for accessibility */
            background: repeating-linear-gradient(
                90deg,
                var(--tool-2) 0px,
                var(--tool-2) 4px,
                transparent 4px,
                transparent 6px
            );
        }}
        
        .legend-swatch.rigz {{
            background: var(--tool-3);
        }}
        
        /* Benchmark Cards */
        .benchmark {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }}
        
        .benchmark-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .benchmark-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            font-weight: 500;
        }}
        
        .benchmark-title .highlight {{
            color: var(--terracotta);
        }}
        
        .benchmark-badge {{
            font-size: 0.75rem;
            color: var(--text-muted);
            padding: 0.35rem 0.75rem;
            background: var(--terracotta-light);
            border-radius: 100px;
            color: var(--terracotta);
        }}
        
        /* Bars */
        .bar-group {{
            margin-bottom: 1.5rem;
        }}
        
        .bar-group:last-child {{
            margin-bottom: 0;
        }}
        
        .bar-row {{
            display: grid;
            grid-template-columns: 60px 1fr 120px;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }}
        
        .bar-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .bar-track {{
            height: 28px;
            background: var(--bg-warm);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }}
        
        .bar-fill {{
            height: 100%;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: white;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .bar-fill.gzip {{
            background: var(--tool-1);
        }}
        
        .bar-fill.pigz {{
            background: var(--tool-2);
            /* Subtle stripe pattern for accessibility */
            background: repeating-linear-gradient(
                -45deg,
                var(--tool-2) 0px,
                var(--tool-2) 8px,
                rgba(255,255,255,0.15) 8px,
                rgba(255,255,255,0.15) 10px
            );
        }}
        
        .bar-fill.rigz {{
            background: var(--tool-3);
        }}
        
        .bar-stats {{
            text-align: right;
        }}
        
        .bar-throughput {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text);
        }}
        
        .bar-ratio {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        
        /* Speedup footer */
        .benchmark-footer {{
            display: flex;
            gap: 3rem;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px dashed var(--border);
        }}
        
        .speedup {{
            text-align: center;
        }}
        
        .speedup-value {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 2rem;
            font-weight: 500;
            color: var(--terracotta);
        }}
        
        .speedup-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        
        /* Footer */
        footer {{
            text-align: center;
            padding-top: 3rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
        
        footer a {{
            color: var(--terracotta);
            text-decoration: none;
        }}
        
        footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{
                padding: 2rem 1rem;
            }}
            
            .summary {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .bar-row {{
                grid-template-columns: 50px 1fr;
            }}
            
            .bar-stats {{
                display: none;
            }}
            
            .legend {{
                flex-wrap: wrap;
                gap: 1rem;
            }}
            
            .benchmark-footer {{
                flex-wrap: wrap;
                gap: 1.5rem;
            }}
        }}
        
        /* Subtle animation */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .benchmark {{
            animation: fadeIn 0.5s ease-out both;
        }}
        
        .benchmark:nth-child(1) {{ animation-delay: 0.1s; }}
        .benchmark:nth-child(2) {{ animation-delay: 0.15s; }}
        .benchmark:nth-child(3) {{ animation-delay: 0.2s; }}
        .benchmark:nth-child(4) {{ animation-delay: 0.25s; }}
        .benchmark:nth-child(5) {{ animation-delay: 0.3s; }}
        .benchmark:nth-child(6) {{ animation-delay: 0.35s; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <p class="eyebrow">{passed} of {total} tests passed</p>
            <h1>rigz</h1>
            <p class="subtitle">
                Parallel gzip compression for Rust. 
                Faster without sacrificing compatibility.
            </p>
        </header>
        
        <div class="summary">
            <div class="summary-item">
                <div class="summary-value">{max_gzip_speedup:.1f}×</div>
                <div class="summary-label">vs gzip</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{max_pigz_speedup:.1f}×</div>
                <div class="summary-label">vs pigz</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{max_throughput:.0f}</div>
                <div class="summary-label">MB/s peak</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{test_size_str}</div>
                <div class="summary-label">test file</div>
            </div>
        </div>
        
        <section>
            <h2>Compression Performance</h2>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-swatch gzip"></div>
                    <span>gzip · standard</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch pigz"></div>
                    <span>pigz · parallel</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch rigz"></div>
                    <span>rigz · this project</span>
                </div>
            </div>
            
            <p style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 2rem;">
                Shorter bars are faster. Time shown inside each bar.
            </p>
            
            <div id="benchmarks"></div>
        </section>
        
        <footer>
            <p>
                Generated by <a href="https://github.com/jackdanger/rigz">rigz</a> validation suite.
                All compressed output is compatible with standard gzip.
            </p>
        </footer>
    </div>
    
    <script>
        const raceData = {race_data_json};
        const testSizeMB = {test_size_mb:.1f};
        
        function formatTime(seconds) {{
            if (seconds < 1) return (seconds * 1000).toFixed(0) + 'ms';
            if (seconds < 10) return seconds.toFixed(2) + 's';
            return seconds.toFixed(1) + 's';
        }}
        
        function formatThroughput(mbps) {{
            return mbps.toFixed(0) + ' MB/s';
        }}
        
        function createBenchmark(config) {{
            const tools = config.tools;
            const maxTime = Math.max(...Object.values(tools).map(t => t.time));
            
            // Find winner (lowest time)
            let winner = null;
            let winnerTime = Infinity;
            for (const [tool, data] of Object.entries(tools)) {{
                if (data.time < winnerTime) {{
                    winnerTime = data.time;
                    winner = tool;
                }}
            }}
            
            const gzipSpeedup = tools.gzip && tools.rigz ? (tools.gzip.time / tools.rigz.time).toFixed(1) : '—';
            const pigzSpeedup = tools.pigz && tools.rigz ? (tools.pigz.time / tools.rigz.time).toFixed(1) : '—';
            
            let barsHTML = '';
            for (const tool of ['gzip', 'pigz', 'rigz']) {{
                if (!tools[tool]) continue;
                const data = tools[tool];
                const widthPercent = (data.time / maxTime) * 100;
                
                barsHTML += `
                    <div class="bar-group">
                        <div class="bar-row">
                            <div class="bar-label">${{tool}}</div>
                            <div class="bar-track">
                                <div class="bar-fill ${{tool}}" style="width: ${{widthPercent}}%">
                                    ${{formatTime(data.time)}}
                                </div>
                            </div>
                            <div class="bar-stats">
                                <div class="bar-throughput">${{formatThroughput(data.throughput)}}</div>
                                <div class="bar-ratio">${{data.ratio.toFixed(1)}}% ratio</div>
                            </div>
                        </div>
                    </div>
                `;
            }}
            
            const threadWord = config.threads === 1 ? 'thread' : 'threads';
            const winnerBadge = winner === 'rigz' ? '<span class="benchmark-badge">rigz fastest</span>' : '';
            
            return `
                <div class="benchmark">
                    <div class="benchmark-header">
                        <div class="benchmark-title">
                            <span class="highlight">Level ${{config.level}}</span> · ${{config.threads}} ${{threadWord}}
                        </div>
                        ${{winnerBadge}}
                    </div>
                    ${{barsHTML}}
                    <div class="benchmark-footer">
                        <div class="speedup">
                            <div class="speedup-value">${{gzipSpeedup}}×</div>
                            <div class="speedup-label">faster than gzip</div>
                        </div>
                        <div class="speedup">
                            <div class="speedup-value">${{pigzSpeedup}}×</div>
                            <div class="speedup-label">faster than pigz</div>
                        </div>
                    </div>
                </div>
            `;
        }}
        
        // Render all benchmarks
        const container = document.getElementById('benchmarks');
        container.innerHTML = raceData.map(createBenchmark).join('');
    </script>
</body>
</html>
'''
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
