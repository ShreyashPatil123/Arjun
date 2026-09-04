"""Tests for the workbook reader in `attachment_extract`.

The bug these are written against
---------------------------------
OOXML keeps a cell's value in more than one place, and the reader looked in
exactly one of them: `<v>`, read as a shared-string index when `t="s"`. That is
what Excel writes. It is not what *ARJUN* writes — `src-tauri/src/artifacts/
xlsx.rs` emits every label as an inline string, `<c t="inlineStr"><is><t>`,
which carries no `<v>` element at all.

So a row reading `Inspection result | 9.0` came back as ` | 9.0`: the number
kept, the word for it dropped. And because the reader also laid cells out by
their order in the XML rather than by the square each one names, a row of
label-gap-value collapsed to `" | "` and was discarded as blank — which is
what happened to ARJUN's own calculation workbook, every row of it.

`src-tauri/src/artifacts/xlsx.rs` holds the writer-to-reader roundtrip that
runs the real writer into the real reader. These are the unit tests underneath
it, covering the shapes Excel and other tools produce that the roundtrip
cannot reach.
"""

import os
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attachment_extract import column_index, extract_xlsx  # noqa: E402

MAIN = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
RELS = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'
RNS = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def workbook_xml(names):
    sheets = "".join(
        '<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (name, i + 1, i + 1)
        for i, name in enumerate(names)
    )
    return "<workbook %s %s><sheets>%s</sheets></workbook>" % (MAIN, RNS, sheets)


def rels_xml(parts):
    entries = "".join(
        '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/worksheet" Target="%s"/>' % (i + 1, part)
        for i, part in enumerate(parts)
    )
    return "<Relationships %s>%s</Relationships>" % (RELS, entries)


def sheet_xml(rows):
    return "<worksheet %s><sheetData>%s</sheetData></worksheet>" % (MAIN, "".join(rows))


class WorkbookBuilder(unittest.TestCase):
    """Builds real .xlsx packages; nothing here is stubbed."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, rows, names=("Calculation",), shared=None, parts=None):
        """One sheet of `rows`, unless `parts` gives several sheets' rows."""
        path = os.path.join(self.dir, "book.xlsx")
        bodies = parts if parts is not None else [rows]
        targets = ["worksheets/sheet%d.xml" % (i + 1) for i in range(len(bodies))]
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("xl/workbook.xml", workbook_xml(names))
            z.writestr("xl/_rels/workbook.xml.rels", rels_xml(targets))
            for target, body in zip(targets, bodies):
                z.writestr("xl/" + target, sheet_xml(body))
            if shared is not None:
                items = "".join("<si><t>%s</t></si>" % s for s in shared)
                z.writestr(
                    "xl/sharedStrings.xml",
                    '<sst %s count="%d">%s</sst>' % (MAIN, len(shared), items),
                )
        return path

    def read(self, rows, **kwargs):
        max_rows = kwargs.pop("max_rows", 2000)
        return extract_xlsx(self.build(rows, **kwargs), max_rows)


class Strings(WorkbookBuilder):
    def test_the_reported_row_keeps_its_label(self):
        """`Inspection result | 9.0`, exactly as reported."""
        rows = [
            '<row r="1">'
            '<c r="A1" t="inlineStr"><is><t xml:space="preserve">Inspection result</t></is></c>'
            '<c r="B1"><v>9.0</v></c>'
            "</row>"
        ]
        self.assertIn("Inspection result | 9.0", self.read(rows)["text"])

    def test_a_shared_string_is_still_read(self):
        """What Excel itself writes must keep working."""
        rows = ['<row r="1"><c r="A1" t="s"><v>1</v></c></row>']
        self.assertIn("Seal wear", self.read(rows, shared=["unused", "Seal wear"])["text"])

    def test_a_shared_index_out_of_range_is_blank_not_a_crash(self):
        rows = ['<row r="1"><c r="A1" t="s"><v>99</v></c><c r="B1"><v>3</v></c></row>']
        self.assertIn("| 3", self.read(rows, shared=["only one"])["text"])

    def test_an_inline_string_split_across_runs_is_joined(self):
        """Formatting splits a string into several <t> runs."""
        rows = [
            '<row r="1"><c r="A1" t="inlineStr"><is>'
            "<r><t>Inspection </t></r><r><t>result</t></r></is></c></row>"
        ]
        self.assertIn("Inspection result", self.read(rows)["text"])


class Coordinates(WorkbookBuilder):
    def test_a_gap_stays_a_gap(self):
        """ARJUN's writer emits `Text, Empty, Text` and omits the empty cell.

        Laid out by document order, the column-C value lands in column B and
        sits under the wrong heading — which looks entirely fine.
        """
        rows = [
            '<row r="1">'
            '<c r="A1" t="inlineStr"><is><t>Expression</t></is></c>'
            '<c r="C1" t="inlineStr"><is><t>(9.0 - 8.2) / 9.0</t></is></c>'
            "</row>"
        ]
        self.assertIn("Expression |  | (9.0 - 8.2) / 9.0", self.read(rows)["text"])

    def test_a_row_that_starts_late_keeps_its_leading_blanks(self):
        rows = ['<row r="1"><c r="C1"><v>7</v></c></row>']
        self.assertIn(" |  | 7", self.read(rows)["text"])

    def test_cells_out_of_order_land_in_their_own_columns(self):
        rows = ['<row r="1"><c r="C1"><v>3</v></c><c r="A1"><v>1</v></c></row>']
        self.assertIn("1 |  | 3", self.read(rows)["text"])

    def test_column_index_runs_past_z(self):
        self.assertEqual(column_index("A1"), 0)
        self.assertEqual(column_index("Z9"), 25)
        self.assertEqual(column_index("AA1"), 26)
        self.assertEqual(column_index("AB12"), 27)
        self.assertIsNone(column_index(""))


class FormulasAndTypes(WorkbookBuilder):
    def test_a_formula_reports_both_its_cache_and_its_working(self):
        """A calculation workbook exists to show the working."""
        rows = ['<row r="1"><c r="A1"><f>120 * 0.85</f><v>102</v></c></row>']
        text = self.read(rows)["text"]
        self.assertIn("102", text)
        self.assertIn("=120 * 0.85", text)

    def test_an_uncached_formula_is_reported_rather_than_read_as_empty(self):
        rows = ['<row r="1"><c r="A1"><f>SUM(B1:B4)</f></c></row>']
        self.assertIn("=SUM(B1:B4)", self.read(rows)["text"])

    def test_a_formula_returning_a_string_keeps_the_string(self):
        rows = ['<row r="1"><c r="A1" t="str"><f>UPPER(B1)</f><v>PASS</v></c></row>']
        self.assertIn("PASS", self.read(rows)["text"])

    def test_a_boolean_reads_as_a_word_not_a_digit(self):
        rows = ['<row r="1"><c r="A1" t="b"><v>1</v></c><c r="B1" t="b"><v>0</v></c></row>']
        self.assertIn("TRUE | FALSE", self.read(rows)["text"])

    def test_an_error_cell_is_shown_rather_than_blanked(self):
        """A reader deciding whether to trust a workbook needs to see this."""
        rows = ['<row r="1"><c r="A1" t="e"><v>#DIV/0!</v></c></row>']
        self.assertIn("#DIV/0!", self.read(rows)["text"])


class Sheets(WorkbookBuilder):
    def test_a_single_sheet_workbook_gets_no_header(self):
        """There is nothing to disambiguate, so the line would be noise.

        Most attachments are one sheet, and ARJUN's own workbook is too.
        """
        rows = ['<row r="1"><c r="A1"><v>1</v></c></row>']
        self.assertNotIn("--- sheet:", self.read(rows)["text"])

    def test_a_multi_sheet_workbook_names_each_sheet(self):
        parts = [
            ['<row r="1"><c r="A1"><v>1</v></c></row>'],
            ['<row r="1"><c r="A1"><v>2</v></c></row>'],
        ]
        text = extract_xlsx(self.build(None, names=["Inputs", "Results"], parts=parts))["text"]
        self.assertIn("--- sheet: Inputs ---", text)
        self.assertIn("--- sheet: Results ---", text)

    def test_a_sheet_with_no_rows_contributes_no_header(self):
        """A header over nothing reads as a tab that was checked and empty."""
        parts = [
            ['<row r="1"><c r="A1" t="inlineStr"><is><t>only here</t></is></c></row>'],
            [],
        ]
        text = extract_xlsx(self.build(None, names=["Full", "Blank"], parts=parts))["text"]
        self.assertIn("--- sheet: Full ---", text)
        self.assertNotIn("Blank", text)

    def test_sheets_come_back_in_workbook_order_not_alphabetical(self):
        """Sorting part names as text puts sheet10 ahead of sheet2."""
        names = ["First", "Second", "Third"]
        parts = [
            ['<row r="1"><c r="A1" t="inlineStr"><is><t>one</t></is></c></row>'],
            ['<row r="1"><c r="A1" t="inlineStr"><is><t>two</t></is></c></row>'],
            ['<row r="1"><c r="A1" t="inlineStr"><is><t>three</t></is></c></row>'],
        ]
        result = extract_xlsx(self.build(None, names=names, parts=parts))
        text = result["text"]
        self.assertEqual(result["pages"], 3)
        self.assertLess(text.index("First"), text.index("Second"))
        self.assertLess(text.index("Second"), text.index("Third"))

    def test_a_package_with_no_workbook_part_still_reads(self):
        """A stripped or unusual package falls back to the sheet files."""
        path = os.path.join(self.dir, "bare.xlsx")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(
                "xl/worksheets/sheet1.xml",
                sheet_xml(['<row r="1"><c r="A1" t="inlineStr"><is><t>bare</t></is></c></row>']),
            )
        self.assertIn("bare", extract_xlsx(path)["text"])


class Truncation(WorkbookBuilder):
    def test_a_long_workbook_is_cut_off_and_says_so(self):
        rows = ['<row r="%d"><c r="A%d"><v>%d</v></c></row>' % (i, i, i)
                for i in range(1, 11)]
        result = self.read(rows, max_rows=4)
        self.assertTrue(result["truncated"])
        self.assertIn("4", result["text"])
        self.assertNotIn("9", result["text"])

    def test_a_short_workbook_is_not_marked_truncated(self):
        rows = ['<row r="1"><c r="A1"><v>1</v></c></row>']
        self.assertFalse(self.read(rows)["truncated"])


if __name__ == "__main__":
    unittest.main()
