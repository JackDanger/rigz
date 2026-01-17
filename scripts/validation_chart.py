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
    
    by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            if threads_filter is None or r["threads"] == threads_filter:
                by_config[key][r["tool"]] = r
    
    for (level, threads), tools in sorted(by_config.items()):
        print(f"{Colors.BOLD}Level {level}, {threads} thread(s){Colors.RESET}")
        max_time = max(t["median_seconds"] for t in tools.values())
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
    
    by_config = defaultdict(lambda: defaultdict(dict))
    for r in results["decompression"]:
        if r["success"] and r["correct"]:
            key = (r["level"], r["threads"], r["compressor"])
            if threads_filter is None or r["threads"] == threads_filter:
                by_config[key][r["decompressor"]] = r
    
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
        gzip_color = Colors.GREEN if gzip_speedup > 1 else Colors.RED
        pigz_color = Colors.GREEN if pigz_speedup > 1 else Colors.RED
        print(f"  {config:<15} {gzip_color}{gzip_speedup:>10.1f}x{Colors.RESET}  {pigz_color}{pigz_speedup:>10.1f}x{Colors.RESET}")
    print()


def generate_html_chart(results):
    """Generate project homepage with performance data."""
    
    # Process compression data
    comp_by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    
    # Build config data
    all_times = []
    configs = []
    for (level, threads), tools in sorted(comp_by_config.items()):
        config = {"level": level, "threads": threads, "tools": {}}
        for tool, data in tools.items():
            time_s = data["median_seconds"]
            throughput = test_size_mb / time_s if time_s > 0 else 0
            config["tools"][tool] = {"time": time_s, "throughput": throughput}
            all_times.append(time_s)
        
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
    
    max_gzip_speedup = max((c["gzip_speedup"] for c in configs), default=1)
    max_pigz_speedup = max((c["pigz_speedup"] for c in configs), default=1)
    max_throughput = max(
        (c["tools"]["rigz"]["throughput"] for c in configs if "rigz" in c["tools"]),
        default=0
    )
    
    passed = results['summary']['passed']
    total = passed + results['summary']['failed']
    test_size_str = format_size(results.get('test_size_bytes', 0))
    
    configs_json = json.dumps(configs)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rigz — Fast Parallel Gzip</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500&family=DM+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
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
            max-width: 720px;
            margin: 0 auto;
            padding: 3rem 1.5rem;
        }}
        
        /* Hero */
        .hero {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}
        
        h1 {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 3.5rem;
            font-weight: 400;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }}
        
        .tagline {{
            font-size: 1.25rem;
            color: var(--text-secondary);
            margin-bottom: 1.5rem;
        }}
        
        .hero-stat {{
            display: inline-block;
            background: var(--rigz-light);
            color: var(--rigz);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 1.1rem;
        }}
        
        /* Install */
        section {{
            margin-bottom: 2.5rem;
        }}
        
        h2 {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.4rem;
            font-weight: 400;
            margin-bottom: 0.75rem;
            color: var(--text);
        }}
        
        .code-block {{
            background: var(--text);
            color: #F5F0E8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            overflow-x: auto;
        }}
        
        .code-block .comment {{
            color: #9C948A;
        }}
        
        .code-block .prompt {{
            color: var(--rigz);
        }}
        
        p {{
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }}
        
        p:last-child {{
            margin-bottom: 0;
        }}
        
        /* Usage examples */
        .usage-grid {{
            display: grid;
            gap: 0.75rem;
        }}
        
        .usage-item {{
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem 1rem;
        }}
        
        .usage-cmd {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        .usage-desc {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        
        /* Performance section */
        .perf-intro {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}
        
        .legend {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            color: var(--text-secondary);
        }}
        
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        
        .legend-dot.gzip {{ background: var(--gzip); }}
        .legend-dot.pigz {{ background: var(--pigz); }}
        .legend-dot.rigz {{ background: var(--rigz); }}
        
        /* Dot plot */
        .dot-plot {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .dot-row {{
            display: grid;
            grid-template-columns: 70px 1fr 60px;
            align-items: center;
            padding: 0.6rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.85rem;
        }}
        
        .dot-row:last-child {{ border-bottom: none; }}
        .dot-row:hover {{ background: var(--bg-warm); }}
        
        .row-label {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}
        
        .row-label .hl {{ color: var(--rigz); font-weight: 500; }}
        
        .dot-track {{
            position: relative;
            height: 24px;
            background: linear-gradient(to right, var(--rigz-light) 0%, var(--bg-warm) 100%);
            border-radius: 12px;
            margin: 0 0.5rem;
        }}
        
        .dot {{
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            cursor: default;
        }}
        
        .dot.gzip {{ background: var(--gzip); }}
        .dot.pigz {{ background: var(--pigz); }}
        .dot.rigz {{ background: var(--rigz); }}
        
        .dot-label {{
            position: absolute;
            top: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            white-space: nowrap;
            opacity: 0;
            background: white;
            padding: 2px 5px;
            border-radius: 3px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            transition: opacity 0.15s;
        }}
        
        .dot:hover .dot-label {{ opacity: 1; }}
        
        .speedup-col {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            color: var(--rigz);
            font-weight: 500;
        }}
        
        /* Links */
        .links {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        
        a {{
            color: var(--rigz);
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .links a {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        
        /* Footer */
        footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        
        @media (max-width: 600px) {{
            h1 {{ font-size: 2.5rem; }}
            .dot-row {{ grid-template-columns: 60px 1fr 50px; }}
            .usage-item {{ grid-template-columns: 1fr; gap: 0.25rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="hero">
            <h1>rigz</h1>
            <p class="tagline">Drop-in gzip replacement. Parallel compression in Rust.</p>
            <span class="hero-stat">{max_gzip_speedup:.0f}× faster than gzip</span>
        </header>
        
        <section>
            <h2>Install</h2>
            <div class="code-block">
<span class="comment"># Build from source</span>
<span class="prompt">$</span> cargo install --path .

<span class="comment"># Or download a release</span>
<span class="prompt">$</span> curl -L https://github.com/jackdanger/rigz/releases/latest/download/rigz-$(uname -m) -o rigz
</div>
        </section>
        
        <section>
            <h2>Usage</h2>
            <p>Works exactly like gzip. Same flags, same behavior.</p>
            <div class="usage-grid">
                <div class="usage-item">
                    <code class="usage-cmd">rigz file.txt</code>
                    <span class="usage-desc">→ file.txt.gz</span>
                </div>
                <div class="usage-item">
                    <code class="usage-cmd">rigz -d file.txt.gz</code>
                    <span class="usage-desc">decompress</span>
                </div>
                <div class="usage-item">
                    <code class="usage-cmd">tar cf - dir/ | rigz &gt; archive.tar.gz</code>
                    <span class="usage-desc">pipe</span>
                </div>
                <div class="usage-item">
                    <code class="usage-cmd">rigz -1</code> / <code class="usage-cmd">-9</code>
                    <span class="usage-desc">fast / best</span>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Performance</h2>
            <p class="perf-intro">
                Tested on {test_size_str}. Left is faster. Hover for times.
            </p>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot rigz"></div> rigz</div>
                <div class="legend-item"><div class="legend-dot pigz"></div> pigz</div>
                <div class="legend-item"><div class="legend-dot gzip"></div> gzip</div>
            </div>
            <div class="dot-plot" id="dotPlot"></div>
        </section>
        
        <section>
            <h2>Links</h2>
            <div class="links">
                <a href="https://github.com/jackdanger/rigz">GitHub</a>
                <a href="https://github.com/jackdanger/rigz/issues">Issues</a>
                <a href="https://github.com/jackdanger/rigz/releases">Releases</a>
                <a href="https://crates.io/crates/rigz">crates.io</a>
            </div>
        </section>
        
        <footer>
            <p>Output is compatible with standard gzip. {passed}/{total} validation tests passing.</p>
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
            
            function getPosition(time) {{
                const minTime = Math.min(...Object.values(tools).map(t => t.time));
                const normalized = (Math.sqrt(time) - Math.sqrt(minTime)) / (Math.sqrt(maxTime) - Math.sqrt(minTime));
                return (1 - normalized) * 80 + 10;
            }}
            
            let dotsHTML = '';
            for (const tool of ['gzip', 'pigz', 'rigz']) {{
                if (tools[tool]) {{
                    const pos = getPosition(tools[tool].time);
                    const time = formatTime(tools[tool].time);
                    dotsHTML += `<div class="dot ${{tool}}" style="left: ${{pos}}%"><span class="dot-label">${{tool}}: ${{time}}</span></div>`;
                }}
            }}
            
            const speedup = config.gzip_speedup;
            const speedupStr = speedup >= 10 ? speedup.toFixed(0) + '×' : speedup.toFixed(1) + '×';
            
            return `
                <div class="dot-row">
                    <div class="row-label"><span class="hl">L${{config.level}}</span> ${{config.threads}}t</div>
                    <div class="dot-track">${{dotsHTML}}</div>
                    <div class="speedup-col">${{speedupStr}}</div>
                </div>
            `;
        }}
        
        document.getElementById('dotPlot').innerHTML = configs.map(createDotRow).join('');
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
