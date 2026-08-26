"""Tests for the two-tier reading policy.

The case that matters most is the one this machine is actually in: no vision
engine installed. The plan still has to be produced, and the gap still has to be
visible — a document whose scanned pages were never read must not look like a
document that simply had little on it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import escalation  # noqa: E402

TEXT_ONLY = {"ocr": False, "layout": False, "tables": False}
FULL = {"ocr": True, "layout": True, "tables": True}


def page(number, text, confidence=1.0):
    return {"page": number, "text": text, "confidence": confidence}


GOOD_PAGE = "Asset PV-2201 inspected 12 March. Wall thickness 8.2 mm against a minimum of 9.0 mm."


class WhichPagesEscalate(unittest.TestCase):
    def test_a_well_read_page_is_settled(self):
        result = escalation.plan([page(1, GOOD_PAGE)], TEXT_ONLY)
        self.assertEqual(result.settled, [1])
        self.assertEqual(result.candidates, [])

    def test_an_empty_page_needs_ocr(self):
        result = escalation.plan([page(1, "", confidence=0.0)], TEXT_ONLY)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].needs, "ocr")

    def test_a_poorly_decoded_page_needs_a_vision_pass(self):
        result = escalation.plan(
            [page(1, "□□□ garbled but long enough to pass the length check", 0.3)],
            TEXT_ONLY,
        )
        self.assertEqual(result.candidates[0].needs, "vision")

    def test_only_the_bad_pages_escalate(self):
        """The whole point of two tiers: a 200-page report should not cost 200
        vision passes because three pages were scanned."""
        pages = [page(i, GOOD_PAGE) for i in range(1, 20)]
        pages.append(page(20, "", confidence=0.0))

        result = escalation.plan(pages, TEXT_ONLY)
        self.assertEqual(len(result.settled), 19)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].page, 20)

    def test_the_plan_names_what_is_required(self):
        result = escalation.plan([page(1, "", confidence=0.0)], TEXT_ONLY)
        self.assertEqual(result.required_capabilities, ["ocr"])


class WhenTheEngineAlreadyTried(unittest.TestCase):
    """Running OCR twice does not help. That page needs a person."""

    def test_an_empty_page_from_an_ocr_capable_engine_needs_a_human(self):
        result = escalation.plan([page(1, "", confidence=0.0)], FULL)
        self.assertEqual(result.candidates[0].needs, "human")
        self.assertIn("person", result.candidates[0].reason)


class WhenNothingIsInstalled(unittest.TestCase):
    """The state this machine is in today."""

    def test_unmet_needs_are_reported_with_the_pages_and_the_remedy(self):
        result = escalation.plan(
            [page(1, GOOD_PAGE), page(2, "", 0.0), page(3, "", 0.0)], TEXT_ONLY
        )
        warnings = escalation.describe_unmet(result, available=[])

        self.assertEqual(len(warnings), 1)
        self.assertIn("2, 3", warnings[0])
        self.assertIn("OCR model", warnings[0])
        # The consequence is stated, not implied.
        self.assertIn("not included in anything downstream", warnings[0])

    def test_nothing_is_warned_about_when_every_page_read(self):
        result = escalation.plan([page(1, GOOD_PAGE)], TEXT_ONLY)
        self.assertEqual(escalation.describe_unmet(result, available=[]), [])

    def test_a_met_need_produces_no_warning(self):
        result = escalation.plan([page(1, "", 0.0)], TEXT_ONLY)
        self.assertEqual(escalation.describe_unmet(result, available=["ocr"]), [])

    def test_a_page_needing_a_human_is_always_warned_about(self):
        """No installed engine can clear this one, so it is reported whatever
        else is available."""
        result = escalation.plan([page(1, "", 0.0)], FULL)
        warnings = escalation.describe_unmet(result, available=["ocr", "vision"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("need a person", warnings[0])


class EmptyInput(unittest.TestCase):
    def test_a_document_with_no_pages_plans_nothing(self):
        result = escalation.plan([], TEXT_ONLY)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.settled, [])
        self.assertEqual(escalation.describe_unmet(result), [])


if __name__ == "__main__":
    unittest.main()
