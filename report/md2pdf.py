"""Render a Markdown report to PDF (A4) with headless Chromium:  python md2pdf.py part2_writeup.md"""
import pathlib, shutil, subprocess, sys
import markdown

CSS = """@page{size:A4;margin:11mm 13mm}body{font-family:'Latin Modern Roman','CMU Serif',Georgia,serif;font-size:9.6pt;
line-height:1.22;color:#111}h1{font-size:14pt;margin:0 0 6px}p{margin:4px 0;text-align:justify}ul{margin:3px 0;padding-left:16px}
li{margin:1px 0}table{border-collapse:collapse;width:100%;margin:5px 0;font-size:8.5pt}th,td{border-top:1px solid #999;
border-bottom:1px solid #999;padding:2px 5px;text-align:left}th{background:#f2f2f2}code{font-size:8.5pt;background:#f4f4f4;padding:0 2px}"""
src = pathlib.Path(sys.argv[1]).resolve()
html = markdown.markdown(src.read_text(), extensions=["tables"])
out_html = src.with_suffix(".html")
out_html.write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>")
pdf = src.with_suffix(".pdf")
browser = next(b for b in ["brave", "google-chrome", "chromium"] if shutil.which(b))
subprocess.run([browser, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True)
out_html.unlink()
print(pdf)
