"""Build the technical report: report/sections/*.md -> report/PathSnap_Report.html -> .pdf

The sections are joined in file-name order and converted with python-markdown.
Images are embedded as base64 so the HTML is self-contained. The PDF is printed
with headless Microsoft Edge (no extra Python dependency).

Run:  python scripts/build_report.py
"""
import base64
import mimetypes
import re
import shutil
import subprocess
from pathlib import Path

import _common  # noqa: F401
import markdown

from pathsnap import config as C

REPORT = C.ROOT / "report"
SECTIONS = REPORT / "sections"
OUT_HTML = REPORT / "PathSnap_Report.html"
OUT_PDF = REPORT / "PathSnap_Report.pdf"
EDGE_CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; border-bottom: 1.5px solid #333; padding-bottom: 3pt; margin-top: 22pt; break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 16pt; break-after: avoid; }
h4 { font-size: 11pt; margin-top: 14pt; break-after: avoid; }
h2.section { break-before: page; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.8pt; break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
figure { margin: 10pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 120mm; border: 1px solid #ccc; }
figcaption { font-size: 9pt; color: #333; margin-top: 4pt; text-align: left; }
.missing { border: 2px dashed #c00; color: #c00; padding: 30pt 10pt; }
li { margin: 2pt 0; }
"""


def fix_indent(md_text):
    """python-markdown needs 4-space indents for nested lists; the sections use 2-3."""
    out, fence = [], False
    item = re.compile(r"^(\d+\.|[-*]) ")
    for line in md_text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence:
            m = re.match(r"^( {1,3})(\S.*)$", line)
            if m:
                line = " " * (len(m.group(1)) * 2) + m.group(2)
            # a top-level list straight after a paragraph line needs a blank line first
            prev = out[-1] if out else ""
            if item.match(line) and prev.strip() and not item.match(prev) and not prev.startswith((" ", "|")):
                out.append("")
        out.append(line)
    return "\n".join(out)


def figure(match, base):
    alt, src = match.group(1), match.group(2)
    path = (base / src).resolve()
    if not path.exists():
        print("MISSING FIGURE:", src)
        body = f'<div class="missing">Missing figure: {src}</div>'
    else:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        body = f'<img src="data:{mime};base64,{data}" alt="">'
    cap = markdown.markdown(alt).removeprefix("<p>").removesuffix("</p>")
    return f'\n<figure>{body}<figcaption>{cap}</figcaption></figure>\n'


parts = []
for i, f in enumerate(sorted(SECTIONS.glob("*.md"))):
    text = f.read_text(encoding="utf-8")
    text = re.sub(r"^!\[(.*?)\]\((.+?)\)\s*$", lambda m: figure(m, f.parent), text, flags=re.M)
    # lazy_ol=False keeps list numbers such as "5." (the question numbers from the brief)
    html = markdown.markdown(fix_indent(text), extensions=["tables", "fenced_code", "sane_lists"],
                             lazy_ol=False)
    if i == 0:
        html = re.sub(r"<h2>", "<h1>", html, count=1).replace("</h2>", "</h1>", 1)
    else:
        html = html.replace("<h2>", '<h2 class="section">', 1)
    parts.append(html)

OUT_HTML.write_text(
    f'<!doctype html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title>'
    f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
print("wrote", OUT_HTML)

edge = next((p for p in EDGE_CANDIDATES if Path(p).exists()), None) or shutil.which("msedge")
if edge is None:
    print("Microsoft Edge not found: open the HTML in a browser and print it to PDF.")
else:
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    print("wrote", OUT_PDF)
