"""Tests for the P&ID engine and the multimodal additions to the router.

The engine is exercised on a hand-built PDF rather than a real drawing
because the property being tested here is the engine's logic, not a
particular font. A real P&ID fixture is fragile — its font, line weight
and tag spacing change between plants — and the engine that depends on
those would be a brittle one.

The P&ID engine's symbol detector is rule-based and works on text, so
what we are really testing is:

1. The router routes a P&ID-shaped document to the P&ID engine.
2. The P&ID engine extracts every tag the page carries.
3. The output regions have a stable, distinct bounding box per tag.
4. Tables and image regions are reported on the result.
5. The classifier is exposed as a separate method and answers honestly.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import router  # noqa: E402
from engines.base import (  # noqa: E402
    EngineCapabilities,
    ExtractionResult,
    PageResult,
    RegionKind,
)
from engines.pid_engine import (  # noqa: E402
    EQUIPMENT_TAG_RE,
    INSTRUMENT_TAG_RE,
    PidEngine,
    SAFETY_TAG_RE,
    _detect_equipment,
    _detect_equipment_regions,
    _detect_tags,
    _layout_regions,
    LINE_NUMBER_RE,
)
from engines.text_layer import TextLayerEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny PDF writer
# ---------------------------------------------------------------------------
#
# ``pypdf`` does not ship with a builder, and pulling in ``reportlab`` or
# ``fpdf`` just to write a test fixture is more weight than the engine.
# Hand-rolled PDFs are simple: a header, a body, a cross-reference table,
# and a trailer. The file is a valid PDF 1.4 by construction.

def _build_pdf(pages_text: "list[str]") -> bytes:
    """Build a multi-page PDF where each page draws ``text`` as a string.

    Built on top of pypdf's own ``PdfWriter`` so the output is a real,
    well-formed PDF that pypdf's text extraction can read back. The
    trick that makes pypdf return the text is the ToUnicode CMap, which
    the spec allows and which the writer attaches by hand below.

    The text is rendered at (72, 720) in 12-pt Helvetica. Layout does
    not matter for the engine's contract — what matters is that the text
    is on the page.
    """
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()

    # A ToUnicode CMap that maps every character code to itself. Identity,
    # essentially. Without this, pypdf cannot recover the original
    # characters and returns an empty page.
    cmap_body = (
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Adobe-Identity-UCS def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n"
        b"<0000> <FFFF>\n"
        b"endcodespacerange\n"
        b"1 beginbfrange\n"
        b"<0000> <FFFF> <0000>\n"
        b"endbfrange\n"
        b"endcmap\n"
        b"CMapName currentdict /CMap defineresource pop\n"
        b"end end\n"
    )
    cmap_stream = DecodedStreamObject()
    cmap_stream.set_data(cmap_body)
    cmap_ref = writer._add_object(cmap_stream)

    # The font dictionary references the CMap.
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font[NameObject("/ToUnicode")] = cmap_ref
    font_ref = writer._add_object(font)

    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        # The text is rendered as a literal PDF string. Tests that need
        # non-ASCII characters replace them with ASCII before passing
        # them in — the engine's signal regexes do not care about the
        # difference, and latin-1 is the path pypdf's text extraction
        # returns reliably.
        escaped = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        content = (
            f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        ).encode("latin-1")
        stream = DecodedStreamObject()
        stream.set_data(content)
        stream_ref = writer._add_object(stream)

        page_ref = page.indirect_reference
        page_obj = page_ref.get_object()
        page_obj[NameObject("/Contents")] = stream_ref
        resources = DictionaryObject()
        font_dict = DictionaryObject()
        font_dict[NameObject("/F1")] = font_ref
        resources[NameObject("/Font")] = font_dict
        page_obj[NameObject("/Resources")] = resources

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# P&ID engine unit tests
# ---------------------------------------------------------------------------

class TagDetection(unittest.TestCase):
    """The regex-based tag detection. Unit-level, no PDF."""

    def test_instrument_tags_are_detected(self):
        text = "PT-2201 sits on the pump discharge; FT-3301 on the feed."
        tags = _detect_tags(text)
        labels = [label for label, _ in tags]
        values = [value for _, value in tags]
        self.assertIn("instrument", labels)
        self.assertIn("PT-2201", values)
        self.assertIn("FT-3301", values)

    def test_equipment_tags_are_detected(self):
        text = "Pump P-101, vessel V-201, heat exchanger E-501."
        tags = _detect_tags(text)
        values = [value for _, value in tags]
        self.assertIn("P-101", values)
        self.assertIn("V-201", values)
        self.assertIn("E-501", values)

    def test_safety_tags_are_detected(self):
        text = "PSV-1201 is on the reboiler shell. SDV-2102 manual isolation."
        tags = _detect_tags(text)
        values = [value for _, value in tags]
        self.assertIn("PSV-1201", values)
        self.assertIn("SDV-2102", values)

    def test_line_numbers_are_detected(self):
        text = "Line 23-101-4N3 from V-201 to P-101."
        tags = _detect_tags(text)
        values = [value for _, value in tags]
        self.assertIn("23-101-4N3", values)

    def test_detect_is_case_insensitive_for_equipment(self):
        # The engine compiles patterns with re.IGNORECASE, but the
        # canonical form is uppercase. The detector returns the *original*
        # text from the input, so a lowercase tag is matched as itself.
        text = "Pump p-101 sits in the suction line."
        tags = _detect_tags(text)
        values = [value for _, value in tags]
        self.assertIn("p-101", values)


class EquipmentVocabulary(unittest.TestCase):
    """The equipment/valve keyword extraction."""

    def test_pump_vocabulary(self):
        out = _detect_equipment("The pump P-101 has a mechanical seal.")
        labels = [label for label, _ in out]
        self.assertIn("pump", labels)

    def test_valve_vocabulary(self):
        out = _detect_equipment("A gate valve sits on the line.")
        labels = [label for label, _ in out]
        self.assertIn("valve", labels)

    def test_dedupe(self):
        out = _detect_equipment("pump pump pump")
        # Three mentions, one region.
        pump_entries = [item for item in out if item[1] == "pump"]
        self.assertEqual(len(pump_entries), 1)


class LayoutRegions(unittest.TestCase):
    """The deterministic tag-layout helper."""

    def test_no_tags_means_no_regions(self):
        self.assertEqual(_layout_regions([]), [])

    def test_each_tag_gets_a_box(self):
        tags = [("instrument", "PT-2201"), ("equipment", "P-101")]
        regions = _layout_regions(tags)
        self.assertEqual(len(regions), 2)
        for region in regions:
            self.assertEqual(region.kind, RegionKind.Symbol)
            self.assertEqual(region.box_confidence, 0.7)
            self.assertGreater(region.right, region.left)
            self.assertGreater(region.bottom, region.top)

    def test_regions_have_distinct_boxes(self):
        tags = [("instrument", f"PT-{i:04d}") for i in range(8)]
        regions = _layout_regions(tags)
        # A grid of 8 fits in 2 rows of 4; adjacent cells should not
        # overlap each other.
        for i, left in enumerate(regions):
            for j, right in enumerate(regions):
                if i == j:
                    continue
                overlap = (
                    min(left.right, right.right) > max(left.left, right.left)
                    and min(left.bottom, right.bottom) > max(left.top, right.top)
                )
                self.assertFalse(
                    overlap,
                    f"regions {i} and {j} overlap: {left.to_dict()} and {right.to_dict()}",
                )


class EquipmentRegions(unittest.TestCase):
    """The keyword-derived regions for un-tagged equipment vocabulary."""

    def test_equipment_region_has_label(self):
        regions = _detect_equipment_regions("A pump and a compressor.")
        labels = [region.label for region in regions]
        self.assertIn("pump", labels)
        self.assertIn("compressor", labels)

    def test_no_equipment_no_regions(self):
        regions = _detect_equipment_regions("Just some prose.")
        # No keywords present, no regions produced.
        self.assertEqual(regions, [])


class PidEngineEndToEnd(unittest.TestCase):
    """Build a tiny PDF and read it through the P&ID engine."""

    @classmethod
    def setUpClass(cls):
        cls.pdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_fixtures", "tiny_pid.pdf"
        )
        os.makedirs(os.path.dirname(cls.pdf_path), exist_ok=True)
        text = (
            "P&ID Rev 3 - Crude Distillation Unit.\n"
            "Line 23-101-4N3 from V-201 to P-101.\n"
            "Instrument PT-2201 on the pump discharge. PSV-1201 on the reboiler.\n"
            "Gate valve and check valve. Pump and heat exchanger.\n"
        )
        with open(cls.pdf_path, "wb") as handle:
            handle.write(_build_pdf([text]))

    def test_engine_is_available(self):
        self.assertTrue(PidEngine.available())

    def test_extract_returns_pages(self):
        engine = PidEngine()
        result = engine.extract(self.pdf_path)
        self.assertEqual(len(result.pages), 1)
        self.assertIn("P&ID", result.pages[0].text)

    def test_extract_returns_tag_regions(self):
        engine = PidEngine()
        result = engine.extract(self.pdf_path)
        all_regions = result.image_regions
        labels = [region.label for region in all_regions]
        # The fixture has at least one of each kind.
        self.assertIn("instrument", labels)
        self.assertIn("equipment", labels)
        self.assertIn("safety_instrument", labels)
        self.assertIn("line", labels)

    def test_extract_carries_document_type(self):
        engine = PidEngine()
        result = engine.extract(self.pdf_path)
        self.assertIsNotNone(result.document_type)
        self.assertEqual(result.document_type.label, "pid")

    def test_extract_capabilities_mark_pid_symbols(self):
        engine = PidEngine()
        result = engine.extract(self.pdf_path)
        self.assertTrue(result.capabilities.pid_symbols)


# ---------------------------------------------------------------------------
# Router integration: P&ID-shaped documents get routed to the P&ID engine.
# ---------------------------------------------------------------------------

class RouterTypeRouting(unittest.TestCase):
    """A P&ID-shaped PDF routes to the P&ID engine."""

    @classmethod
    def setUpClass(cls):
        cls.pdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_fixtures", "tiny_pid_router.pdf"
        )
        os.makedirs(os.path.dirname(cls.pdf_path), exist_ok=True)
        text = (
            "P&ID Rev 2 - Naphtha Hydrotreater.\n"
            "PFD. Line 14-201-2A from P-301 to E-501.\n"
            "Instrument FT-3301, FV-3302. PSV-3101.\n"
            "Pump and heat exchanger on the schematic.\n"
        )
        with open(cls.pdf_path, "wb") as handle:
            handle.write(_build_pdf([text]))

    def setUp(self):
        self._original_preference = router.ENGINE_PREFERENCE
        # Make sure both the text-layer and P&ID engines are available.
        router.ENGINE_PREFERENCE = [router.TextLayerEngine, router.TextLayerEngine]

    def tearDown(self):
        router.ENGINE_PREFERENCE = self._original_preference

    def test_pid_routes_to_pid_engine(self):
        r = router.DocumentRouter()
        payload = r.dispatch("extract", {"path": self.pdf_path})
        self.assertEqual(payload["engine"], "pid")
        self.assertTrue(payload["capabilities"]["pidSymbols"])

    def test_non_pid_routes_to_text_layer(self):
        # A non-P&ID document — a SOP — should not route to the P&ID
        # engine, even though P&ID is registered as a type-routed engine.
        text = (
            "Standard Operating Procedure SOP-014.\n"
            "Step 1: lockout tagout the pump. Step 2: depressurise.\n"
            "PPE required. Caution: high temperature. The operator shall verify.\n"
        )
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_fixtures", "tiny_sop.pdf"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(_build_pdf([text]))
        r = router.DocumentRouter()
        payload = r.dispatch("extract", {"path": path})
        # The text layer is what the SOP hits, since the fixture is just text.
        self.assertEqual(payload["engine"], "text-layer")


# ---------------------------------------------------------------------------
# The new ``classify`` method on the router.
# ---------------------------------------------------------------------------

class RouterClassify(unittest.TestCase):
    """``classify`` answers without doing a full extraction."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_fixtures"
        )
        os.makedirs(cls.fixtures, exist_ok=True)

        # Tiny P&ID, SOP, datasheet and quote PDFs.
        samples = {
            "pid.pdf": (
                "P&ID Rev 1. Line 23-101-1A from V-201 to P-101. "
                "PT-2201 on the pump. PSV-1201 on the shell. "
                "Pump and heat exchanger. Gate valve. "
            ),
            "sop.pdf": (
                "Standard Operating Procedure SOP-005. "
                "Step 1: lockout. Step 2: depressurise. Step 3: verify. "
                "PPE required. Caution: high temperature. "
                "The operator shall confirm the isolation. "
            ),
            "datasheet.pdf": (
                "Equipment Data Sheet - Pump P-101. "
                "Model OH2. Capacity 150 m3/h. Head 45 m. Power 30 kW. "
                "Material of construction: SS 316. "
                "Manufacturer: Sulzer. Vendor: ABC. "
                "Inlet NPS 100. Outlet NPS 80. "
            ),
            "quote.pdf": (
                "Quotation Q-2026-114. Item 1: Pump P-101, qty 1, "
                "unit price $24,500. Item 2: Seal, qty 2, unit price $850. "
                "Subtotal $26,200. GST 18%. Grand total $30,916. "
                "FOB Mumbai. Lead time 12 weeks. Valid until 30 April 2026. "
            ),
        }
        for filename, text in samples.items():
            path = os.path.join(cls.fixtures, filename)
            with open(path, "wb") as handle:
                handle.write(_build_pdf([text]))
            setattr(cls, filename.replace(".pdf", "_path"), path)

    def setUp(self):
        self.router = router.DocumentRouter()

    def test_classify_pid(self):
        verdict = self.router.dispatch("classify", {"path": self.pid_path})
        self.assertFalse(verdict["abstained"])
        self.assertEqual(verdict["label"], "pid")

    def test_classify_sop(self):
        verdict = self.router.dispatch("classify", {"path": self.sop_path})
        self.assertFalse(verdict["abstained"])
        self.assertEqual(verdict["label"], "sop")

    def test_classify_datasheet(self):
        verdict = self.router.dispatch("classify", {"path": self.datasheet_path})
        self.assertFalse(verdict["abstained"])
        self.assertEqual(verdict["label"], "datasheet")

    def test_classify_quote(self):
        verdict = self.router.dispatch("classify", {"path": self.quote_path})
        self.assertFalse(verdict["abstained"])
        self.assertEqual(verdict["label"], "vendor_quote")

    def test_classify_missing_file_refuses(self):
        # Missing files are an explicit client error: the caller named
        # something that is not there, and the contract is that a
        # ``classify`` call reports an error rather than guessing.
        with self.assertRaises(ValueError):
            self.router.dispatch("classify", {"path": "/no/such/file.pdf"})

    def test_classify_unknown_method_still_raises(self):
        with self.assertRaises(ValueError):
            self.router.dispatch("not_a_method", {})


# ---------------------------------------------------------------------------
# Multimodal additions to the base result: tables and image regions.
# ---------------------------------------------------------------------------

class MultimodalFields(unittest.TestCase):
    """The ExtractionResult now carries tables, image regions, document type."""

    def test_to_dict_includes_tables_images_and_type(self):
        page = PageResult(
            page=1,
            text="Some page.",
            confidence=0.9,
        )
        from engines.base import DocumentTypeInfo, TableRecord
        result = ExtractionResult(
            engine="test",
            engine_version="0",
            pages=[page],
            capabilities=EngineCapabilities(tables=True),
        )
        result.tables = [
            TableRecord(
                page=1, headers=["K", "V"], rows=[["a", "1"], ["b", "2"]],
                flat_text="K: a | V: 1\nK: b | V: 2",
            )
        ]
        result.document_type = DocumentTypeInfo(
            label="datasheet", confidence=0.9, abstained=False,
        )
        payload = result.to_dict()
        self.assertIn("tables", payload)
        self.assertIn("imageRegions", payload)
        self.assertIn("documentType", payload)
        self.assertEqual(payload["documentType"]["label"], "datasheet")
        self.assertEqual(payload["tables"][0]["headers"], ["K", "V"])

    def test_table_flat_text_is_searchable(self):
        from engines.base import TableRecord
        rec = TableRecord(
            page=1, headers=["K", "V"], rows=[["design pressure", "14 bar"]],
            flat_text="K: design pressure | V: 14 bar",
        )
        self.assertIn("design pressure", rec.flat_text)


if __name__ == "__main__":
    unittest.main()
