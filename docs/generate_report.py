"""
MODULE 15: Automated Report Generator
Converts markdown reports to HTML or formatted summary document.
"""

import os
import sys

def generate_report():
    md_path = "docs/PROJECT_REPORT.md"
    html_out = "docs/PROJECT_REPORT.html"
    
    if not os.path.exists(md_path):
        print("Markdown report not found.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    try:
        import markdown
        html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    except ImportError:
        # Simple HTML wrapper if markdown package not present
        html_body = f"<pre style='font-family: sans-serif; white-space: pre-wrap;'>{md_text}</pre>"
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PhishGuard AI - Comprehensive Project Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fdfdfd;
        }}
        h1, h2, h3 {{ color: #1e3a8a; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{ background-color: #f1f5f9; }}
        pre {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        code {{ font-family: monospace; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report exported to {html_out}")

if __name__ == "__main__":
    generate_report()
