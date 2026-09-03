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
    pptx                              ->  slide text and speaker notes, never OCR'd
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

    {"kind": "pdf-text"|"pdf-scan"|"text"|"docx"|"xlsx"|"pptx",
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


def extract_pptx(path, max_slides=500):
    """Read slide text straight from the package.

    Same reasoning as `extract_xlsx`: `python-pptx` is not installed and adding
    it for one format would be a new dependency in an air-gapped deployment. A
    .pptx is a zip of XML — each slide is `ppt/slides/slideN.xml` and every run
    of text in it is an `<a:t>` element, which the standard library reads.

    Slides are emitted in numeric order and separated by a heading, because a
    reviewer asking "what does slide 7 say" needs the boundary to survive into
    the text. Speaker notes are included: on an approval deck the note under
    the slide is often where the actual commitment is written.
    """
    from xml.etree import ElementTree as ET

    drawing = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

    def slide_number(name):
        digits = "".join(c for c in os.path.basename(name) if c.isdigit())
        return int(digits) if digits else 0

    def text_of(member, archive):
        # A slide that will not parse costs that slide, not the deck. One
        # corrupt part in a 60-slide package used to raise out of the whole
        # extraction, so a reviewer got "could not be read" for a file that
        # was 59/60ths readable -- and no way to tell which slide was bad.
        try:
            root = ET.fromstring(archive.read(member))
        except (ET.ParseError, KeyError):
            return None
        runs = [t.text for t in root.iter("%st" % drawing) if t.text]
        return "\n".join(run.strip() for run in runs if run.strip())

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        slides = sorted(
            (n for n in names
             if n.startswith("ppt/slides/slide") and n.endswith(".xml")),
            key=slide_number,
        )
        notes = {
            slide_number(n): n
            for n in names
            if n.startswith("ppt/notesSlides/notesSlide") and n.endswith(".xml")
        }

        blocks, truncated, unreadable = [], False, 0
        for index, member in enumerate(slides, start=1):
            if index > max_slides:
                truncated = True
                break
            body = text_of(member, z)
            note = notes.get(slide_number(member))
            note_text = text_of(note, z) if note else ""
            section = ["Slide %d" % index]
            if body is None:
                # Named rather than skipped silently. A gap a reader can
                # see is recoverable; one they cannot is a summary with a
                # hole in it.
                section.append("[this slide could not be read]")
                unreadable += 1
            elif body:
                section.append(body)
            if note_text:
                section.append("Speaker notes: %s" % note_text)
            # A slide with a diagram and no text still gets its heading, so the
            # numbering stays aligned with the deck a person is looking at.
            blocks.append("\n".join(section))

        # Every slide unreadable is a failure, not a result. Returning a
        # document of nothing but placeholders would let a summary be
        # written about a deck that was never read.
        if slides and unreadable == len(blocks):
            return {"error": "no slide in this deck could be read; the file may be corrupt"}

        return {"kind": "pptx", "text": "\n\n".join(blocks),
                "pages": len(slides), "pageImages": [], "truncated": truncated}


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
        elif suffix == ".pptx":
            result = extract_pptx(path)
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
