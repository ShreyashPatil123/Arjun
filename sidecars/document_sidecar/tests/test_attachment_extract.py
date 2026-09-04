"""Tests for the per-page PDF routing in `attachment_extract`.

The bug these are written against
---------------------------------
`extract_pdf` used to weigh the *whole file's* text against one threshold and
call the result `pdf-text` or `pdf-scan`. Two things followed, both silent:

  - A digital cover sheet in front of scanned pages carried the whole file
    over the line on its own. The scans were never rendered, never OCR'd and
    never mentioned; `truncated` came back `False`.
  - The threshold was `MIN_TEXT_CHARS_PER_PAGE * min(total, 3)`, capped at
    three pages' worth however long the document was. One readable page
    anywhere in a hundred-page scan declared the whole thing text.

Every test below builds a real PDF with PyMuPDF and runs the real extractor.
Nothing is stubbed, because the thing under test is a judgement about what is
actually on a page, and a stub would only assert the judgement back at itself.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from attachment_extract import MIN_TEXT_CHARS_PER_PAGE, extract_pdf  # noqa: E402


# Comfortably over the per-page threshold, so a page carrying it is one the
# text layer can be trusted for.
COVER_TEXT = "Q3 Seal Inspection Report for the maintenance review board"

assert len(COVER_TEXT) >= MIN_TEXT_CHARS_PER_PAGE, "the fixture must clear the threshold"


def text_page(doc, body):
    """A digitally produced page: a real text layer, no image."""
    page = doc.new_page()
    page.insert_text((72, 100), body, fontsize=11)


def scanned_page(doc):
    """A scanned page: an image and no text layer at all."""
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 600, 800), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)


class PerPageRouting(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "out")
        os.makedirs(self.out, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, *pages):
        """Writes a PDF from a list of "text"/"scan" markers and returns it."""
        doc = fitz.open()
        for index, kind in enumerate(pages):
            if kind == "text":
                text_page(doc, "%s (page %d)" % (COVER_TEXT, index + 1))
            else:
                scanned_page(doc)
        path = os.path.join(self.dir, "doc.pdf")
        doc.save(path)
        doc.close()
        return path

    def extract(self, *pages, **kwargs):
        return extract_pdf(self.build(*pages), self.out, kwargs.get("max_pages", 12))

    # -- The reported failure --------------------------------------------

    def test_a_digital_cover_does_not_swallow_the_scan_behind_it(self):
        """The exact case from the report: page 1 digital, page 2 scanned."""
        result = self.extract("text", "scan")

        self.assertEqual(result["kind"], "pdf-mixed")
        self.assertEqual(result["pages"], 2)
        # The scanned page must actually be queued for the model.
        self.assertEqual(len(result["pageImages"]), 1, "the scanned page was not rendered")
        sources = [d["source"] for d in result["pageDetail"]]
        self.assertEqual(sources, ["text", "ocr"])

    def test_the_rendered_image_is_named_for_its_real_page(self):
        """A citation to page 2 has to mean page 2 of the file."""
        result = self.extract("text", "scan")
        ocr = [d for d in result["pageDetail"] if d["source"] == "ocr"][0]
        self.assertEqual(ocr["page"], 2)
        self.assertTrue(os.path.basename(ocr["image"]).endswith("page-2.png"), ocr["image"])
        self.assertTrue(os.path.exists(ocr["image"]))

    def test_one_readable_page_does_not_declare_a_long_scan_readable(self):
        """The capped threshold, which never grew with the document.

        Twenty pages, one of them digital. The old rule compared the cover's
        text against three pages' worth and stopped there.
        """
        result = self.extract("text", *["scan"] * 19, max_pages=25)

        self.assertEqual(result["kind"], "pdf-mixed")
        self.assertEqual(len(result["pageImages"]), 19,
                         "nineteen scanned pages should all be queued")

    # -- Every page accounted for ----------------------------------------

    def test_every_page_appears_in_the_detail_exactly_once(self):
        """A page missing from the record cannot be told from a blank one
        afterwards, which is how the original bug hid."""
        result = self.extract("text", "scan", "text", "scan", "scan")
        numbers = [d["page"] for d in result["pageDetail"]]
        self.assertEqual(numbers, [1, 2, 3, 4, 5])

    def test_pages_over_the_render_limit_are_marked_unread(self):
        result = self.extract(*["scan"] * 5, max_pages=2)

        self.assertEqual(len(result["pageImages"]), 2)
        self.assertEqual(result["unreadPages"], [3, 4, 5])
        self.assertTrue(result["truncated"],
                        "dropping three pages is exactly what truncated means")
        for detail in result["pageDetail"][2:]:
            self.assertEqual(detail["source"], "unread")
            self.assertIn("limit", detail["why"])

    def test_a_fully_read_document_is_not_marked_truncated(self):
        result = self.extract("text", "scan")
        self.assertEqual(result["unreadPages"], [])
        self.assertFalse(result["truncated"])

    # -- The unmixed cases still behave ----------------------------------

    def test_an_all_text_pdf_needs_no_model(self):
        result = self.extract("text", "text", "text")
        self.assertEqual(result["kind"], "pdf-text")
        self.assertEqual(result["pageImages"], [])
        self.assertEqual(result["unreadPages"], [])
        self.assertTrue(all(d["source"] == "text" for d in result["pageDetail"]))

    def test_an_all_scan_pdf_goes_entirely_to_the_model(self):
        result = self.extract("scan", "scan")
        self.assertEqual(result["kind"], "pdf-scan")
        self.assertEqual(len(result["pageImages"]), 2)
        self.assertTrue(all(d["source"] == "ocr" for d in result["pageDetail"]))

    def test_a_page_below_the_threshold_keeps_its_text_as_a_fallback(self):
        """A page with a stray header over an unreadable scan.

        It goes to the model, but the little text it did carry is kept so the
        caller can report it if OCR comes back with nothing.
        """
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Fig 3", fontsize=9)
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 600, 800), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
        path = os.path.join(self.dir, "stray.pdf")
        doc.save(path)
        doc.close()

        result = extract_pdf(path, self.out, 12)
        self.assertEqual(result["kind"], "pdf-scan")
        self.assertEqual(result["pageDetail"][0]["layerText"], "Fig 3")


class Contract(unittest.TestCase):
    """The script is spawned as a process, so its stdout is the real contract."""

    def test_stdout_is_one_json_object_the_caller_can_parse(self):
        directory = tempfile.mkdtemp()
        try:
            doc = fitz.open()
            text_page(doc, COVER_TEXT)
            scanned_page(doc)
            path = os.path.join(directory, "mixed.pdf")
            doc.save(path)
            doc.close()

            script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "attachment_extract.py",
            )
            out = subprocess.run(
                [sys.executable, script, path, os.path.join(directory, "out")],
                capture_output=True, text=True, check=True,
            ).stdout
            parsed = json.loads(out.strip().splitlines()[-1])

            self.assertEqual(parsed["kind"], "pdf-mixed")
            self.assertEqual(parsed["pages"], 2)
            # The field names the Rust side deserialises.
            for field in ("kind", "text", "pages", "pageImages",
                          "pageDetail", "unreadPages", "truncated"):
                self.assertIn(field, parsed)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
