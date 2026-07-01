import json
import tempfile
import unittest
from pathlib import Path

from helpers.broll_slots import build_slots_from_edl, choose_slot_type, phrase_score, suggest_slots_for_phrases


class TestBrollSlots(unittest.TestCase):
    def test_slot_scoring_and_type(self):
        self.assertGreater(phrase_score("Revenue grew 42 percent", 2.0), 2.0)
        self.assertEqual(choose_slot_type(["dashboard", "click"]), "screen-demo")
        self.assertEqual(choose_slot_type(["42", "percent"]), "data-callout")

    def test_suggest_slots_for_phrases(self):
        phrases = [
            {"start": 0.0, "end": 1.5, "text": "Open the dashboard and click settings"},
            {"start": 2.0, "end": 2.4, "text": "okay"},
        ]
        suggestions = suggest_slots_for_phrases("clipA", phrases, min_score=2.0)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["type"], "screen-demo")

    def test_build_slots_from_edl_maps_output_offsets(self):
        payload = {
            "words": [
                {"type": "word", "text": "Revenue", "start": 1.0, "end": 1.3},
                {"type": "word", "text": "grew", "start": 1.3, "end": 1.6},
                {"type": "word", "text": "42", "start": 1.6, "end": 1.8},
                {"type": "word", "text": "percent", "start": 1.8, "end": 2.2},
            ]
        }
        edl = {
            "ranges": [{"source": "clipA", "start": 1.0, "end": 2.2}],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            edit_dir = Path(tmp_dir) / "edit"
            transcripts_dir = edit_dir / "transcripts"
            transcripts_dir.mkdir(parents=True)
            (transcripts_dir / "clipA.json").write_text(json.dumps(payload), encoding="utf-8")
            slots = build_slots_from_edl(edl, edit_dir, min_score=1.0)
            self.assertEqual(len(slots), 1)
            self.assertIn("start_in_output", slots[0])
            self.assertAlmostEqual(slots[0]["start_in_output"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
