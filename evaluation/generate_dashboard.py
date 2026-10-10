"""Generate the evaluation dashboard with embedded report data (no file upload needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_dashboard(report_path: str | None = None, output_path: str | None = None) -> str:
    """Read report.json and embed it into dashboard.html as a self-contained file."""
    eval_dir = Path(__file__).parent

    # Load report
    report_file = Path(report_path) if report_path else eval_dir / "report.json"
    if not report_file.exists():
        print(f"Report not found: {report_file}")
        print("Run the evaluation first: python -m evaluation.runner")
        sys.exit(1)

    with open(report_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    # Load dashboard template
    template_file = eval_dir / "dashboard.html"
    with open(template_file, "r", encoding="utf-8") as f:
        html = f.read()

    # Escape the JSON for safe embedding in a <script> tag
    # Replace </script> sequences that would prematurely close the tag
    report_json = json.dumps(report_data, default=str, ensure_ascii=True)
    report_json = report_json.replace("</", "<\\/")

    # Inject auto-load script right before the closing </body>
    # This script runs after all other scripts are parsed
    inject_script = f"""
<script>
// Auto-load embedded report data
var EMBEDDED_REPORT = {report_json};
document.addEventListener('DOMContentLoaded', function() {{
  try {{
    DATA = EMBEDDED_REPORT;
    render();
  }} catch(e) {{
    console.error('Dashboard render error:', e);
    document.body.innerHTML = '<div style="padding:40px;color:#ef4444;font-family:monospace;">' +
      '<h2>Dashboard Error</h2><pre>' + e.message + '</pre></div>';
  }}
}});
</script>
</body>"""

    html = html.replace("</body>", inject_script)

    # Output
    out_file = Path(output_path) if output_path else eval_dir / "evaluation_report.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard generated: {out_file}")
    return str(out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation dashboard HTML")
    parser.add_argument("--report", type=str, default=None, help="Path to report.json")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    path = generate_dashboard(args.report, args.output)

    # Auto-open in browser
    import webbrowser
    webbrowser.open(f"file:///{path}")
