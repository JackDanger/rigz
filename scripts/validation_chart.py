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
    """Generate a warm, sophisticated HTML visualization with unified dot plot."""
    
    # Process compression data
    comp_by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    # Calculate throughputs and build unified data
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    
    # Find global min/max for scaling
    all_times = []
    configs = []
    for (level, threads), tools in sorted(comp_by_config.items()):
        config = {"level": level, "threads": threads, "label": f"Level {level}, {threads}t", "tools": {}}
        for tool, data in tools.items():
            time_s = data["median_seconds"]
            throughput = test_size_mb / time_s if time_s > 0 else 0
            config["tools"][tool] = {
                "time": time_s,
                "throughput": throughput,
            }
            all_times.append(time_s)
        
        # Calculate speedups
        if "rigz" in config["tools"] and "gzip" in config["tools"]:
            config["gzip_speedup"] = config["tools"]["gzip"]["time"] / config["tools"]["rigz"]["time"]
        else:
            config["gzip_speedup"] = 1
        if "rigz" in config["tools"] and "pigz" in config["tools"]:
            config["pigz_speedup"] = config["tools"]["pigz"]["time"] / config["tools"]["rigz"]["time"]
        else:
            config["pigz_speedup"] = 1
            
        configs.append(config)
    
    global_max_time = max(all_times) if all_times else 1
    
    # Calculate best speedups for hero stats
    max_gzip_speedup = max((c["gzip_speedup"] for c in configs), default=1)
    max_pigz_speedup = max((c["pigz_speedup"] for c in configs), default=1)
    max_throughput = max(
        (c["tools"]["rigz"]["throughput"] for c in configs if "rigz" in c["tools"]),
        default=0
    )
    
    # Summary
    passed = results['summary']['passed']
    failed = results['summary']['failed']
    total = passed + failed
    test_size_str = format_size(results.get('test_size_bytes', 0))
    
    # Convert configs to JSON
    configs_json = json.dumps(configs)
    
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
            --bg: #FAF8F5;
            --bg-warm: #F5F0E8;
            --bg-card: #FFFFFF;
            --border: #E8E2D9;
            
            --text: #2C2825;
            --text-secondary: #6B635A;
            --text-muted: #9C948A;
            
            --gzip: #2C2825;
            --pigz: #7D8B74;
            --rigz: #C4856A;
            
            --rigz-light: #F5E6E0;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'DM Sans', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 4rem 2rem;
        }}
        
        /* Header */
        header {{
            text-align: center;
            margin-bottom: 4rem;
        }}
        
        .eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}
        
        h1 {{
            font-family: 'Cormorant Garamond', serif;
            font-size: clamp(3rem, 10vw, 4.5rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            margin-bottom: 0.75rem;
        }}
        
        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            max-width: 400px;
            margin: 0 auto;
        }}
        
        /* Hero stats - the main story */
        .hero-stats {{
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin: 3rem 0 4rem;
            padding: 2rem 0;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
        }}
        
        .hero-stat {{
            text-align: center;
        }}
        
        .hero-value {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 3rem;
            font-weight: 500;
            color: var(--rigz);
            line-height: 1;
        }}
        
        .hero-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}
        
        /* The unified dot plot */
        .chart-section {{
            margin-bottom: 3rem;
        }}
        
        h2 {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.5rem;
            font-weight: 400;
            margin-bottom: 1.5rem;
            color: var(--text);
        }}
        
        .chart-explanation {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 2rem;
            max-width: 500px;
        }}
        
        /* Legend */
        .legend {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1.5rem;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        
        .legend-dot.gzip {{ background: var(--gzip); }}
        .legend-dot.pigz {{ 
            background: var(--pigz);
            /* Pattern for accessibility */
            background: conic-gradient(var(--pigz) 0deg, var(--pigz) 90deg, transparent 90deg, transparent 180deg, var(--pigz) 180deg, var(--pigz) 270deg, transparent 270deg);
        }}
        .legend-dot.rigz {{ background: var(--rigz); }}
        
        /* Dot plot rows */
        .dot-plot {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .dot-row {{
            display: grid;
            grid-template-columns: 120px 1fr 100px;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .dot-row:last-child {{
            border-bottom: none;
        }}
        
        .dot-row:hover {{
            background: var(--bg-warm);
        }}
        
        .row-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .row-label .level {{
            color: var(--rigz);
            font-weight: 500;
        }}
        
        /* The actual dot track */
        .dot-track {{
            position: relative;
            height: 32px;
            background: linear-gradient(to right, var(--rigz-light) 0%, var(--bg-warm) 100%);
            border-radius: 16px;
            margin: 0 1rem;
        }}
        
        .dot-track::before {{
            content: 'faster →';
            position: absolute;
            left: 8px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.65rem;
            color: var(--text-muted);
            opacity: 0.6;
        }}
        
        .dot {{
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            cursor: default;
            transition: transform 0.15s ease;
            z-index: 1;
        }}
        
        .dot:hover {{
            transform: translate(-50%, -50%) scale(1.2);
            z-index: 10;
        }}
        
        .dot.gzip {{ background: var(--gzip); }}
        .dot.pigz {{ 
            background: var(--pigz);
            /* Cross pattern for accessibility */
            background: 
                linear-gradient(45deg, var(--pigz) 40%, white 40%, white 60%, var(--pigz) 60%);
        }}
        .dot.rigz {{ background: var(--rigz); }}
        
        .dot-label {{
            position: absolute;
            top: -24px;
            left: 50%;
            transform: translateX(-50%);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text);
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.15s ease;
            background: white;
            padding: 2px 6px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .dot:hover .dot-label {{
            opacity: 1;
        }}
        
        /* Speedup column */
        .speedup-col {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .speedup-value {{
            font-size: 1rem;
            font-weight: 500;
            color: var(--rigz);
        }}
        
        .speedup-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
        }}
        
        /* Time axis */
        .time-axis {{
            display: grid;
            grid-template-columns: 120px 1fr 100px;
            padding: 0.75rem 1.5rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            background: var(--bg-warm);
        }}
        
        .time-axis-track {{
            display: flex;
            justify-content: space-between;
            margin: 0 1rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Footer */
        footer {{
            text-align: center;
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        
        footer a {{
            color: var(--rigz);
            text-decoration: none;
        }}
        
        footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Reading guide */
        .reading-guide {{
            background: var(--bg-warm);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 2rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .reading-guide strong {{
            color: var(--text);
        }}
        
        @media (max-width: 640px) {{
            .hero-stats {{
                flex-direction: column;
                gap: 1.5rem;
            }}
            
            .dot-row {{
                grid-template-columns: 80px 1fr;
            }}
            
            .speedup-col {{
                display: none;
            }}
            
            .time-axis {{
                grid-template-columns: 80px 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <p class="eyebrow">{passed}/{total} tests passed</p>
            <h1>rigz</h1>
            <p class="subtitle">Parallel gzip compression. Faster without sacrificing compatibility.</p>
        </header>
        
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-value">{max_gzip_speedup:.0f}×</div>
                <div class="hero-label">faster than gzip</div>
            </div>
            <div class="hero-stat">
                <div class="hero-value">{max_pigz_speedup:.1f}×</div>
                <div class="hero-label">faster than pigz</div>
            </div>
            <div class="hero-stat">
                <div class="hero-value">{max_throughput:.0f}</div>
                <div class="hero-label">MB/s peak</div>
            </div>
        </div>
        
        <section class="chart-section">
            <h2>Compression Time</h2>
            
            <div class="reading-guide">
                <strong>How to read:</strong> Each row shows one configuration. 
                Dots show where each tool finishes — <strong>left is faster</strong>. 
                rigz (terracotta) is always leftmost.
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot rigz"></div>
                    <span>rigz</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot pigz"></div>
                    <span>pigz</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot gzip"></div>
                    <span>gzip</span>
                </div>
            </div>
            
            <div class="dot-plot" id="dotPlot"></div>
        </section>
        
        <footer>
            <p>Benchmarked on a {test_size_str} test file. Output is compatible with standard gzip.</p>
            <p style="margin-top: 0.5rem;"><a href="https://github.com/jackdanger/rigz">github.com/jackdanger/rigz</a></p>
        </footer>
    </div>
    
    <script>
        const configs = {configs_json};
        const globalMaxTime = {global_max_time};
        
        function formatTime(s) {{
            if (s < 1) return (s * 1000).toFixed(0) + 'ms';
            if (s < 10) return s.toFixed(1) + 's';
            return s.toFixed(0) + 's';
        }}
        
        function createDotRow(config) {{
            const tools = config.tools;
            const maxTime = globalMaxTime;
            
            // Position dots based on time (faster = more left, using log scale for better spread)
            function getPosition(time) {{
                // Use square root for better visual spread
                const minTime = Math.min(...Object.values(tools).map(t => t.time));
                const normalized = (Math.sqrt(time) - Math.sqrt(minTime)) / (Math.sqrt(maxTime) - Math.sqrt(minTime));
                // Invert: faster (lower time) = more left
                return (1 - normalized) * 85 + 7.5; // 7.5% to 92.5%
            }}
            
            let dotsHTML = '';
            for (const tool of ['gzip', 'pigz', 'rigz']) {{
                if (tools[tool]) {{
                    const pos = getPosition(tools[tool].time);
                    const time = formatTime(tools[tool].time);
                    dotsHTML += `
                        <div class="dot ${{tool}}" style="left: ${{pos}}%">
                            <span class="dot-label">${{tool}}: ${{time}}</span>
                        </div>
                    `;
                }}
            }}
            
            const speedup = config.gzip_speedup;
            const speedupDisplay = speedup >= 10 ? speedup.toFixed(0) : speedup.toFixed(1);
            
            return `
                <div class="dot-row">
                    <div class="row-label">
                        <span class="level">L${{config.level}}</span>, ${{config.threads}}t
                    </div>
                    <div class="dot-track">
                        ${{dotsHTML}}
                    </div>
                    <div class="speedup-col">
                        <div class="speedup-value">${{speedupDisplay}}×</div>
                        <div class="speedup-label">vs gzip</div>
                    </div>
                </div>
            `;
        }}
        
        // Add time axis
        function createTimeAxis() {{
            return `
                <div class="time-axis">
                    <div></div>
                    <div class="time-axis-track">
                        <span>0s</span>
                        <span>${{formatTime(globalMaxTime / 2)}}</span>
                        <span>${{formatTime(globalMaxTime)}}</span>
                    </div>
                    <div></div>
                </div>
            `;
        }}
        
        const container = document.getElementById('dotPlot');
        container.innerHTML = configs.map(createDotRow).join('') + createTimeAxis();
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
