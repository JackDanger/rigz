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
    """Generate a stunning HTML visualization."""
    
    # Process compression data
    comp_by_config = defaultdict(dict)
    for r in results["compression"]:
        if r["success"]:
            key = (r["level"], r["threads"])
            comp_by_config[key][r["tool"]] = r
    
    # Process decompression data (rigz-compressed only for hero section)
    decomp_by_config = defaultdict(dict)
    for r in results["decompression"]:
        if r["success"] and r["correct"] and r["compressor"] == "rigz":
            key = (r["level"], r["threads"])
            decomp_by_config[key][r["decompressor"]] = r
    
    # Calculate speedups and throughputs
    test_size_mb = results.get("test_size_bytes", 0) / (1024 * 1024)
    
    # Build race data for each config
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
        # Calculate speedups
        if "rigz" in config["tools"]:
            rigz_time = config["tools"]["rigz"]["time"]
            for tool in config["tools"]:
                config["tools"][tool]["speedup_vs_rigz"] = config["tools"][tool]["time"] / rigz_time if rigz_time > 0 else 1
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
    <title>rigz — Performance Benchmarks</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-deep: #0a0a0f;
            --bg-card: #12121a;
            --bg-card-hover: #1a1a25;
            --border: #2a2a3a;
            --text: #e4e4e7;
            --text-muted: #71717a;
            --text-dim: #52525b;
            
            /* Tool colors */
            --gzip: #f43f5e;
            --gzip-glow: rgba(244, 63, 94, 0.3);
            --pigz: #f59e0b;
            --pigz-glow: rgba(245, 158, 11, 0.3);
            --rigz: #10b981;
            --rigz-glow: rgba(16, 185, 129, 0.4);
            
            /* Accent */
            --accent: #8b5cf6;
            --accent-glow: rgba(139, 92, 246, 0.3);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-deep);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        /* Animated gradient background */
        .bg-gradient {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(16, 185, 129, 0.15), transparent),
                radial-gradient(ellipse 60% 40% at 100% 50%, rgba(139, 92, 246, 0.1), transparent),
                radial-gradient(ellipse 60% 40% at 0% 80%, rgba(244, 63, 94, 0.08), transparent);
            pointer-events: none;
            z-index: 0;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 3rem 2rem;
            position: relative;
            z-index: 1;
        }}
        
        /* Hero Section */
        .hero {{
            text-align: center;
            margin-bottom: 4rem;
        }}
        
        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 100px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--rigz);
            margin-bottom: 1.5rem;
        }}
        
        .hero h1 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: clamp(3rem, 8vw, 5rem);
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #10b981 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }}
        
        .hero-subtitle {{
            font-size: 1.25rem;
            color: var(--text-muted);
            max-width: 600px;
            margin: 0 auto 2rem;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 4rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-4px);
            border-color: var(--rigz);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 40px var(--rigz-glow);
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--rigz), var(--accent));
        }}
        
        .stat-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--rigz);
            margin-bottom: 0.25rem;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Section Headers */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .section-header h2 {{
            font-size: 1.75rem;
            font-weight: 600;
        }}
        
        .section-header .line {{
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--border), transparent);
        }}
        
        /* Race Track Visualization */
        .race-section {{
            margin-bottom: 4rem;
        }}
        
        .race-track {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }}
        
        .race-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .race-config {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .race-config .level {{
            color: var(--accent);
        }}
        
        .race-config .threads {{
            color: var(--text-muted);
            font-weight: 400;
        }}
        
        .winner-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.8rem;
            background: linear-gradient(135deg, var(--rigz), rgba(16, 185, 129, 0.6));
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #000;
        }}
        
        .race-lanes {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        
        .lane {{
            display: grid;
            grid-template-columns: 80px 1fr 140px;
            align-items: center;
            gap: 1rem;
        }}
        
        .lane-tool {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            font-size: 0.95rem;
        }}
        
        .lane-tool.gzip {{ color: var(--gzip); }}
        .lane-tool.pigz {{ color: var(--pigz); }}
        .lane-tool.rigz {{ color: var(--rigz); }}
        
        .lane-bar-container {{
            height: 40px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }}
        
        .lane-bar {{
            height: 100%;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            color: rgba(0, 0, 0, 0.8);
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }}
        
        .lane-bar.gzip {{
            background: linear-gradient(90deg, var(--gzip), #fb7185);
            box-shadow: 0 0 20px var(--gzip-glow);
        }}
        
        .lane-bar.pigz {{
            background: linear-gradient(90deg, var(--pigz), #fbbf24);
            box-shadow: 0 0 20px var(--pigz-glow);
        }}
        
        .lane-bar.rigz {{
            background: linear-gradient(90deg, var(--rigz), #34d399);
            box-shadow: 0 0 30px var(--rigz-glow);
        }}
        
        .lane-stats {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .lane-throughput {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
        }}
        
        .lane-time {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        
        /* Speedup callout */
        .speedup-callout {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px dashed var(--border);
        }}
        
        .speedup-item {{
            text-align: center;
        }}
        
        .speedup-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--rigz);
        }}
        
        .speedup-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        
        /* Legend */
        .legend {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .legend-dot.gzip {{ background: var(--gzip); box-shadow: 0 0 10px var(--gzip-glow); }}
        .legend-dot.pigz {{ background: var(--pigz); box-shadow: 0 0 10px var(--pigz-glow); }}
        .legend-dot.rigz {{ background: var(--rigz); box-shadow: 0 0 10px var(--rigz-glow); }}
        
        .legend-text {{
            font-size: 0.9rem;
            color: var(--text-muted);
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            color: var(--text-dim);
            font-size: 0.9rem;
        }}
        
        .footer a {{
            color: var(--rigz);
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .animate-in {{
            animation: fadeInUp 0.6s ease-out forwards;
        }}
        
        .race-track {{
            opacity: 0;
            animation: fadeInUp 0.6s ease-out forwards;
        }}
        
        .race-track:nth-child(1) {{ animation-delay: 0.1s; }}
        .race-track:nth-child(2) {{ animation-delay: 0.2s; }}
        .race-track:nth-child(3) {{ animation-delay: 0.3s; }}
        .race-track:nth-child(4) {{ animation-delay: 0.4s; }}
        .race-track:nth-child(5) {{ animation-delay: 0.5s; }}
        .race-track:nth-child(6) {{ animation-delay: 0.6s; }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{
                padding: 2rem 1rem;
            }}
            
            .lane {{
                grid-template-columns: 60px 1fr;
                gap: 0.5rem;
            }}
            
            .lane-stats {{
                display: none;
            }}
            
            .speedup-callout {{
                flex-direction: column;
                gap: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    
    <div class="container">
        <header class="hero animate-in">
            <div class="hero-badge">
                <span>✓</span>
                <span>{passed}/{total} tests passed</span>
            </div>
            <h1>rigz</h1>
            <p class="hero-subtitle">
                Rust parallel gzip. Dramatically faster compression without sacrificing compatibility.
            </p>
        </header>
        
        <div class="stats-grid animate-in" style="animation-delay: 0.1s">
            <div class="stat-card">
                <div class="stat-value">{max_gzip_speedup:.1f}×</div>
                <div class="stat-label">faster than gzip</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max_pigz_speedup:.1f}×</div>
                <div class="stat-label">faster than pigz</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max_throughput:.0f}</div>
                <div class="stat-label">MB/s peak throughput</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{test_size_str}</div>
                <div class="stat-label">test tarball size</div>
            </div>
        </div>
        
        <section class="race-section">
            <div class="section-header animate-in" style="animation-delay: 0.2s">
                <h2>Compression Performance</h2>
                <div class="line"></div>
            </div>
            
            <div class="legend animate-in" style="animation-delay: 0.25s">
                <div class="legend-item">
                    <div class="legend-dot gzip"></div>
                    <span class="legend-text">gzip (standard)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot pigz"></div>
                    <span class="legend-text">pigz (parallel)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot rigz"></div>
                    <span class="legend-text">rigz (this project)</span>
                </div>
            </div>
            
            <div id="race-tracks"></div>
        </section>
        
        <footer class="footer">
            <p>
                Generated by <a href="https://github.com/jackdanger/rigz">rigz</a> validation suite.
                Bars show time (shorter = faster). All output is gzip-compatible.
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
        
        function createRaceTrack(config) {{
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
            
            const gzipSpeedup = tools.gzip ? (tools.gzip.time / tools.rigz.time).toFixed(1) : '-';
            const pigzSpeedup = tools.pigz ? (tools.pigz.time / tools.rigz.time).toFixed(1) : '-';
            
            let lanesHTML = '';
            for (const tool of ['gzip', 'pigz', 'rigz']) {{
                if (!tools[tool]) continue;
                const data = tools[tool];
                const widthPercent = (data.time / maxTime) * 100;
                
                lanesHTML += `
                    <div class="lane">
                        <div class="lane-tool ${{tool}}">${{tool}}</div>
                        <div class="lane-bar-container">
                            <div class="lane-bar ${{tool}}" style="width: ${{widthPercent}}%">
                                ${{formatTime(data.time)}}
                            </div>
                        </div>
                        <div class="lane-stats">
                            <div class="lane-throughput">${{formatThroughput(data.throughput)}}</div>
                            <div class="lane-time">${{data.ratio.toFixed(1)}}% ratio</div>
                        </div>
                    </div>
                `;
            }}
            
            return `
                <div class="race-track">
                    <div class="race-header">
                        <div class="race-config">
                            <span class="level">Level ${{config.level}}</span>
                            <span class="threads">• ${{config.threads}} thread${{config.threads > 1 ? 's' : ''}}</span>
                        </div>
                        ${{winner === 'rigz' ? '<div class="winner-badge">🏆 rigz wins</div>' : ''}}
                    </div>
                    <div class="race-lanes">
                        ${{lanesHTML}}
                    </div>
                    <div class="speedup-callout">
                        <div class="speedup-item">
                            <div class="speedup-value">${{gzipSpeedup}}×</div>
                            <div class="speedup-label">faster than gzip</div>
                        </div>
                        <div class="speedup-item">
                            <div class="speedup-value">${{pigzSpeedup}}×</div>
                            <div class="speedup-label">faster than pigz</div>
                        </div>
                    </div>
                </div>
            `;
        }}
        
        // Render all race tracks
        const container = document.getElementById('race-tracks');
        container.innerHTML = raceData.map(createRaceTrack).join('');
        
        // Animate bars on load
        setTimeout(() => {{
            document.querySelectorAll('.lane-bar').forEach(bar => {{
                bar.style.width = bar.style.width; // Trigger reflow
            }});
        }}, 100);
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
