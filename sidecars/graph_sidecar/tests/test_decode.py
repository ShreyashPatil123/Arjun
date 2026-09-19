"""Decoder tests, pinned to output this machine actually got out of REBEL.

Every `RAW_*` constant below was produced by running Babelscape/rebel-large over
the passage in its docstring and printing the decoded string verbatim, special
tokens kept. None of it is illustrative. The repository already holds this
discipline on the Rust side -- `knowledge/graph/typing.rs` pins real replies from
Nemotron3-Nano-4B, including two the model invented -- and the reason is the
same: a parser tested against output someone imagined passes until the day it
meets the model.

These tests import only `rebel`, never torch. They run with no weights present.

Written against `unittest`, not pytest, for the same reason `_sentences` in
`rebel.py` is hand-rolled: pytest is a third-party import, and an air-gapped
install that cannot run its own test suite cannot be verified in the field.
The document sidecar next door holds the same line. Before this was a
`TestCase`, `python -m unittest discover` over this directory collected zero
tests and exited 0 -- a green gate asserting nothing.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebel import decode_triplets, split_for_encoder  # noqa: E402


# Input: "Punta Cana is a resort town in the municipality of Higuey, in La
# Altagracia Province, the eastern most province of the Dominican Republic."
#
# The important case in this string is `Punta Cana`, which carries *two*
# (tail, relation) pairs under one `<triplet>`. An earlier version of the
# decoder merged them into one triplet whose relation was the two labels
# concatenated -- "located in the administrative territorial entity country" --
# and silently dropped `La Altagracia Province`. That is what this fixture is
# here to prevent.
RAW_MULTI_PAIR = (
    "<s><triplet> Punta Cana <subj> La Altagracia Province <obj> located in the "
    "administrative territorial entity <subj> Dominican Republic <obj> country "
    "<triplet> Higuey <subj> La Altagracia Province <obj> located in the "
    "administrative territorial entity <subj> Dominican Republic <obj> country "
    "<triplet> La Altagracia Province <subj> Dominican Republic <obj> country "
    "<triplet> Dominican Republic <subj> La Altagracia Province <obj> contains "
    "administrative territorial entity</s>"
)

# Input: "Pump PV-2201 was supplied by Acme Pumps Ltd under contract CT-4471.
# Acme Pumps Ltd is headquartered in Pune and maintains the unit under an annual
# service agreement with the Refining Division."
#
# REBEL emitted the same triplet twice. It does this often enough that
# deduplication is part of the decoder rather than the caller.
RAW_REPEATED = (
    "<s><triplet> Acme Pumps Ltd <subj> Pune <obj> headquarters location "
    "<triplet> Acme Pumps Ltd <subj> Pune <obj> headquarters location</s>"
)

# Input: "PT-2201 | 0-25 barg | Rosemount 3051 | Loop 2201"
#
# Kept because it is wrong. Rosemount is the instrument manufacturer, not a
# territory, and REBEL asserts a geographic relation because its training corpus
# is Wikipedia abstracts. The decoder's job is to return it faithfully; refusing
# it is the relation allowlist's job, in `knowledge/graph/relations.rs`. A
# decoder that quietly dropped implausible triplets would hide the measurement
# that justifies the allowlist.
RAW_WRONG_DOMAIN = (
    "<s><triplet> PT-2201 <subj> Rosemount <obj> located in the administrative "
    "territorial entity</s>"
)


class DecodeTripletsTest(unittest.TestCase):
    """The decoder and the splitter, with no model present."""

    def test_multi_pair_head_yields_one_triplet_per_pair(self) -> None:
        triplets = decode_triplets(RAW_MULTI_PAIR, "c-wiki")
        got = [(t.subject, t.relation, t.object) for t in triplets]

        self.assertEqual(
            got,
            [
                ("Punta Cana", "located in the administrative territorial entity", "La Altagracia Province"),
                ("Punta Cana", "country", "Dominican Republic"),
                ("Higuey", "located in the administrative territorial entity", "La Altagracia Province"),
                ("Higuey", "country", "Dominican Republic"),
                ("La Altagracia Province", "country", "Dominican Republic"),
                ("Dominican Republic", "contains administrative territorial entity", "La Altagracia Province"),
            ],
        )

    def test_every_triplet_carries_the_chunk_it_came_from(self) -> None:
        # The chunk id is the whole basis of verification on the Rust side. A
        # triplet that lost it cannot be checked against anything.
        for triplet in decode_triplets(RAW_MULTI_PAIR, "c-wiki"):
            self.assertEqual(triplet.chunk_id, "c-wiki")

    def test_repeated_triplet_collapses_to_one(self) -> None:
        triplets = decode_triplets(RAW_REPEATED, "c-0007")
        self.assertEqual(len(triplets), 1)
        self.assertEqual(
            (triplets[0].subject, triplets[0].relation, triplets[0].object),
            ("Acme Pumps Ltd", "headquarters location", "Pune"),
        )

    def test_wrong_domain_triplet_is_returned_not_hidden(self) -> None:
        triplets = decode_triplets(RAW_WRONG_DOMAIN, "c-0044")
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0].relation, "located in the administrative territorial entity")

    def test_empty_and_malformed_output_yields_nothing(self) -> None:
        self.assertEqual(decode_triplets("", "c-1"), [])
        self.assertEqual(decode_triplets("<s></s>", "c-1"), [])
        # A head with no relation: the model stopped mid-span.
        self.assertEqual(decode_triplets("<s><triplet> Pump PV-2201 <subj> Acme Pumps Ltd</s>", "c-1"), [])
        # A relation with no tail.
        self.assertEqual(decode_triplets("<s><triplet> Pump PV-2201 <obj> manufacturer</s>", "c-1"), [])

    def test_short_passage_is_not_split(self) -> None:
        self.assertEqual(split_for_encoder("One sentence only."), ["One sentence only."])
        self.assertEqual(split_for_encoder("   "), [])

    def test_long_passage_splits_on_sentence_boundaries(self) -> None:
        sentence = "Pump PV-2201 was supplied by Acme Pumps Ltd under contract CT-4471. "
        pieces = split_for_encoder(sentence * 60, max_chars=400)

        self.assertTrue(len(pieces) > 1)
        # No piece may exceed the window, or the tokenizer truncates and the
        # relation stated in the truncated tail vanishes without a word.
        self.assertTrue(all(len(piece) <= 400 for piece in pieces))
        # Nothing may be cut mid-sentence: a subject in one piece and its object in
        # the next fails verification on the Rust side and is reported as a
        # fabrication, which it is not.
        self.assertTrue(all(piece.endswith(".") for piece in pieces))

    def test_split_preserves_every_sentence(self) -> None:
        text = "Alpha one. Bravo two. Charlie three. Delta four."
        pieces = split_for_encoder(text, max_chars=20)
        rejoined = " ".join(pieces)
        for sentence in ["Alpha one.", "Bravo two.", "Charlie three.", "Delta four."]:
            self.assertIn(sentence, rejoined)


if __name__ == "__main__":
    unittest.main()
