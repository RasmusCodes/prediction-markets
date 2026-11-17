"""Render paper/paper.md to paper/paper.html (self-contained, figures embedded) and paper/paper.pdf (headless
Microsoft Edge or Chrome/Chromium; the PDF step is skipped if neither is found).

    python paper/build.py [--no-pdf]        # needs `pip install markdown-it-py`
"""
import base64
import os
import re
import shutil
import subprocess
import sys

from markdown_it import MarkdownIt

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #f9f9f7; color: #0b0b0b; font: 15px/1.6 Georgia, 'Times New Roman', serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 24px 64px; background: #fcfcfb; }
h1 { font: 700 26px/1.25 system-ui, 'Segoe UI', sans-serif; margin: 0 0 6px; }
h2 { font: 650 19px/1.3 system-ui, 'Segoe UI', sans-serif; margin: 34px 0 10px; border-bottom: 1px solid #e1e0d9; padding-bottom: 4px; }
h3 { font: 600 16px system-ui, 'Segoe UI', sans-serif; }
p, li { text-align: justify; hyphens: auto; }
em { color: #52514e; }
img { max-width: 100%; display: block; margin: 14px auto 4px; border: 1px solid #e1e0d9; border-radius: 6px; }
table { border-collapse: collapse; margin: 12px 0 18px; font: 13px/1.4 system-ui, 'Segoe UI', sans-serif; width: 100%;
  font-variant-numeric: tabular-nums; }
th { text-align: left; color: #52514e; border-bottom: 1.5px solid #c3c2b7; padding: 5px 6px; font-weight: 600; }
td { border-bottom: 1px solid #e1e0d9; padding: 4px 6px; vertical-align: top; }
code, pre { font: 12.5px/1.45 Consolas, 'Courier New', monospace; }
pre { background: #f0efec; padding: 12px; border-radius: 6px; overflow-x: auto; }
@media print { body { background: #fff; } main { padding: 0; max-width: none; } img { break-inside: avoid; }
  h2 { break-after: avoid; } table { break-inside: avoid; } }
@media (max-width: 640px) { main { padding: 20px 16px; } table { font-size: 12px; } }
"""


def main():
    src = open(os.path.join(HERE, "paper.md"), encoding="utf-8").read()
    src = re.sub(r"^---\n.*?\n---\n", "", src, flags=re.S)       # drop YAML front matter
    md = MarkdownIt("commonmark", {"typographer": True}).enable(["table", "replacements", "smartquotes"])
    html = md.render(src)

    def embed(m):
        path = os.path.join(HERE, m.group(1))
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'src="data:image/png;base64,{b64}"'
    html = re.sub(r'src="(figures/[^"]+\.png)"', embed, html)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width, initial-scale=1'><title>Prediction Markets Paper</title><style>{CSS}</style>"
           f"</head><body><main>{html}</main></body></html>")
    out = os.path.join(HERE, "paper.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("wrote", out, f"{len(doc) / 1e6:.1f} MB")
    if "--no-pdf" in sys.argv:
        return
    browser = find_browser()
    if browser is None:
        print("no Edge/Chrome found: skipped paper.pdf")
        return
    pdf = os.path.join(HERE, "paper.pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "file:///" + out.replace("\\", "/").lstrip("/")], timeout=180, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", pdf, os.path.exists(pdf))


def find_browser():
    fixed = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for p in fixed:
        if os.path.exists(p):
            return p
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    return None


if __name__ == "__main__":
    main()
