#!/usr/bin/env python3
"""Turn one attached file into something a local model can actually read.

Why this exists
---------------
Attachments used to go straight to the vision model, so anything that was not
a PNG, JPEG or WebP was refused — including PDF, which is the format people
actually attach. Forcing a PDF through OCR would also be wrong: most PDFs
carry a text layer, and rendering pages to images to read text that is already
present is slower, lossier, and needlessly uses the GPU.

So this routes by what the file *is*:

    text-native (txt, md, csv, json)  ->  decoded, never OCR'd
    pdf with a text layer             ->  pypdf, never OCR'd
    pdf without one (a scan)          ->  pages rendered to PNG for the OCR model
    docx                              ->  paragraphs and tables, never OCR'd
    xlsx                              ->  sheet cells, never OCR'd
    anything else                     ->  refused by name, with the reason

Only a scanned PDF reaches the OCR model. That is the point.

Dependencies
------------
Nothing new. `pypdf`, `fitz` (PyMuPDF) and `docx` are already installed for
the document sidecar. PyMuPDF is what renders pages — a self-contained wheel,
so there is no Poppler or Ghostscript to install and nothing reaches the
network.

Contract
--------
    python attachment_extract.py <path> <out_dir> [--max-pages N]

Prints one JSON object on stdout:

    {"kind": "pdf-text"|"pdf-scan"|"text"|"docx"|"xlsx",
     "text": "...",          # empty for pdf-scan
     "pages": 6,             # real page/sheet count, or 1
     "pageImages": [...],    # absolute PNG paths, only for pdf-scan
     "truncated": false}

or, on refusal:

    {"error": "..."}

Every number reported is counted, never estimated: the caller shows them to
the person as "6 pages", and a guess there would be a lie.
"""

import json
import os
import sys
import zipfile

# A page is rendered at this width when it has to go to the OCR model. The
# model normalises coordinates against whatever it is given, and its encoder
# works at roughly this scale; larger costs time for no more detail.
RENDER_WIDTH = 1000

# Rendering is the expensive path, so it is bounded. A hundred-page scan is a
# batch job, not a chat attachment, and saying so beats a silent long wait.
DEFAULT_MAX_PAGES = 12

# Enough of a text layer to call it a text PDF. Below this the "text layer" is
# usually a header or a stray watermark on an otherwise scanned page.
MIN_TEXT_CHARS_PER_PAGE = 24

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".tsv"}


def fail(message):
    print(json.dumps({"error": message}))
    return 0


def read_text_native(path):
    """Decode a text file without guessing wildly at its encoding.

    UTF-8 first because that is what these files almost always are; then
    UTF-16, which is what Notepad and Excel still emit; then latin-1, which
    cannot fail and at least preserves byte structure.
    """
    raw = open(path, "rb").read()
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("latin-1", "replace")


def extract_pdf(path, out_dir, max_pages):
    import pypdf

    reader = pypdf.PdfReader(path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return {"error": "This PDF is password protected."}

    total = len(reader.pages)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n\n".join(p.strip() for p in parts if p.strip())

    # A real text layer means no OCR at all.
    if len(text) >= MIN_TEXT_CHARS_PER_PAGE * max(1, min(total, 3)):
        return {"kind": "pdf-text", "text": text, "pages": total,
                "pageImages": [], "truncated": False}

    # No usable text layer: it is a scan. Render pages for the OCR model.
    import fitz

    doc = fitz.open(path)
    rendered = []
    limit = min(total, max_pages)
    for index in range(limit):
        page = doc.load_page(index)
        rect = page.rect
        scale = RENDER_WIDTH / rect.width if rect.width else 1.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        target = os.path.join(out_dir, "page-%d.png" % (index + 1))
        pix.save(target)
        rendered.append(target)
    doc.close()
    return {"kind": "pdf-scan", "text": "", "pages": total,
            "pageImages": rendered, "truncated": limit < total}


def extract_docx(path):
    import docx

    document = docx.Document(path)
    blocks = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                blocks.append(" | ".join(cells))
    return {"kind": "docx", "text": "\n\n".join(blocks), "pages": 1,
            "pageImages": [], "truncated": False}


def extract_xlsx(path, max_rows=2000):
    """Read sheet values straight from the package.

    `openpyxl` is not installed, and adding it for one format would be a new
    dependency. An .xlsx is a zip of XML: the shared-string table plus each
    sheet's cell values is all that is needed, and the standard library
    already reads both.
    """
    from xml.etree import ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("%ssi" % ns):
                shared.append("".join(t.text or "" for t in si.iter("%st" % ns)))
        sheets = sorted(n for n in z.namelist()
                        if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        out, count = [], 0
        for name in sheets:
            root = ET.fromstring(z.read(name))
            for row in root.iter("%srow" % ns):
                if count >= max_rows:
                    return {"kind": "xlsx", "text": "\n".join(out),
                            "pages": len(sheets), "pageImages": [], "truncated": True}
                values = []
                for c in row.findall("%sc" % ns):
                    v = c.find("%sv" % ns)
                    if v is None or v.text is None:
                        values.append("")
                        continue
                    if c.get("t") == "s":
                        idx = int(v.text)
                        values.append(shared[idx] if idx < len(shared) else "")
                    else:
                        values.append(v.text)
                if any(x.strip() for x in values):
                    out.append(" | ".join(values))
                count += 1
        return {"kind": "xlsx", "text": "\n".join(out), "pages": len(sheets),
                "pageImages": [], "truncated": False}


def main():
    if len(sys.argv) < 3:
        return fail("usage: attachment_extract.py <path> <out_dir> [--max-pages N]")
    path, out_dir = sys.argv[1], sys.argv[2]
    max_pages = DEFAULT_MAX_PAGES
    if "--max-pages" in sys.argv:
        try:
            max_pages = int(sys.argv[sys.argv.index("--max-pages") + 1])
        except (IndexError, ValueError):
            return fail("--max-pages needs a number")

    if not os.path.isfile(path):
        return fail("that file is not there any more")
    os.makedirs(out_dir, exist_ok=True)
    suffix = os.path.splitext(path)[1].lower()

    try:
        if suffix in TEXT_SUFFIXES:
            result = {"kind": "text", "text": read_text_native(path), "pages": 1,
                      "pageImages": [], "truncated": False}
        elif suffix == ".pdf":
            result = extract_pdf(path, out_dir, max_pages)
        elif suffix == ".docx":
            result = extract_docx(path)
        elif suffix == ".xlsx":
            result = extract_xlsx(path)
        else:
            return fail("%s is not a document ARJUN can read."
                        % (suffix or "this file type"))
    except Exception as error:  # noqa: BLE001 - the caller shows this to a person
        return fail("%s could not be read: %s" % (os.path.basename(path), error))

    if "error" in result:
        return fail(result["error"])
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
