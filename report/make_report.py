"""Build the report in two passes so the static table of contents gets the
right page numbers, then export DOCX -> PDF with Microsoft Word.

Pass 1 writes the TOC with placeholder numbers and exports a PDF. The page of
every heading is then read from that PDF, and pass 2 writes the real numbers.
The placeholders have the same width, so pagination is identical.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pypdf

HERE = Path(__file__).resolve().parent
DOCX = HERE / "Report_Load_Forecast_Procurement.docx"
PDF = DOCX.with_suffix(".pdf")
PAGES = HERE / "toc_pages.json"
FIRST_BODY_TEXT = "Electricity cannot be stored cheaply"


def run(*cmd):
    subprocess.run(cmd, check=True, cwd=HERE)


def export():
    run("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(HERE / "export_pdf.ps1"), str(DOCX))


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def find_pages():
    heads = json.load(open(HERE / "toc_headings.json"))
    texts = [norm(p.extract_text() or "") for p in pypdf.PdfReader(PDF).pages]
    start = next(i for i, t in enumerate(texts) if norm(FIRST_BODY_TEXT) in t)
    pages, cur = {}, 0
    for h in heads:
        key = norm(h)
        if h.startswith("1 "):           # body starts; skip the TOC page
            cur = start
        for i in range(cur, len(texts)):
            if key in texts[i]:
                pages[h] = i + 1
                cur = i
                break
        else:
            print("not found:", h)
    return pages


def main():
    if PAGES.exists():
        PAGES.unlink()
    run(sys.executable, "build_report.py")
    export()
    pages = find_pages()
    json.dump(pages, open(PAGES, "w"), indent=1)
    run(sys.executable, "build_report.py")
    export()
    check = find_pages()
    print("TOC consistent:", check == pages, "| pages:", len(pypdf.PdfReader(PDF).pages))


if __name__ == "__main__":
    main()
