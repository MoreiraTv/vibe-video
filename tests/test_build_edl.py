import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from helpers.build_edl import build_edl, is_false_start_phrase, is_filler_phrase, merge_adjacent_ranges


class TestBuildEdl(unittest.TestCase):
    def _write_test_video(self, path: Path, moving: bool) -> None:
        size = (96, 96)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            size,
        )
        if not writer.isOpened():
            self.fail("could not create test video")
        for idx in range(40):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            if moving:
                x = 5 + (idx * 2 % 60)
                cv2.rectangle(frame, (x, 20), (x + 20, 60), (255, 255, 255), -1)
            else:
                cv2.rectangle(frame, (20, 20), (40, 60), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

    def test_filler_and_false_start_detection(self):
        self.assertTrue(is_filler_phrase("umm uh"))
        self.assertTrue(is_false_start_phrase("we need", "we need to ship", 0.8))
        self.assertFalse(is_false_start_phrase("we should explain this carefully", "totally different thought", 1.2))

    def test_merge_adjacent_ranges(self):
        ranges = [
            {"source": "take1", "start": 0.0, "end": 1.0, "quote": "Hello"},
            {"source": "take1", "start": 1.2, "end": 2.0, "quote": "world"},
            {"source": "take2", "start": 0.0, "end": 1.0, "quote": "new"},
        ]
        merged = merge_adjacent_ranges(ranges, max_gap=0.25)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["end"], 2.0)
        self.assertIn("world", merged[0]["quote"])

    def test_build_edl_from_transcripts(self):
        payload = {
            "words": [
                {"type": "word", "text": "umm", "start": 0.0, "end": 0.2, "speaker_id": "speaker_0"},
                {"type": "spacing", "start": 0.2, "end": 0.9},
                {"type": "word", "text": "we", "start": 0.9, "end": 1.1, "speaker_id": "speaker_0"},
                {"type": "word", "text": "need", "start": 1.1, "end": 1.3, "speaker_id": "speaker_0"},
                {"type": "spacing", "start": 1.3, "end": 1.9},
                {"type": "word", "text": "we", "start": 1.9, "end": 2.1, "speaker_id": "speaker_0"},
                {"type": "word", "text": "need", "start": 2.1, "end": 2.3, "speaker_id": "speaker_0"},
                {"type": "word", "text": "to", "start": 2.3, "end": 2.4, "speaker_id": "speaker_0"},
                {"type": "word", "text": "ship", "start": 2.4, "end": 2.7, "speaker_id": "speaker_0"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            edit_dir = root / "edit"
            transcripts_dir = edit_dir / "transcripts"
            transcripts_dir.mkdir(parents=True)
            (root / "clipA.mp4").write_bytes(b"video")
            (transcripts_dir / "clipA.json").write_text(json.dumps(payload), encoding="utf-8")

            edl = build_edl(transcripts_dir)
            self.assertEqual(edl["version"], 1)
            self.assertEqual(edl["sources"]["clipA"], str((root / "clipA.mp4").resolve()))
            self.assertEqual(len(edl["ranges"]), 1)
            self.assertAlmostEqual(edl["ranges"][0]["start"], 1.9, places=2)
            self.assertAlmostEqual(edl["ranges"][0]["end"], 2.7, places=2)

    def test_build_edl_keep_policy_preserves_silence_gap(self):
        payload = {
            "words": [
                {"type": "word", "text": "first", "start": 0.0, "end": 0.5, "speaker_id": "speaker_0"},
                {"type": "spacing", "start": 0.5, "end": 2.0},
                {"type": "word", "text": "second", "start": 2.0, "end": 2.4, "speaker_id": "speaker_0"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            edit_dir = root / "edit"
            transcripts_dir = edit_dir / "transcripts"
            transcripts_dir.mkdir(parents=True)
            (root / "clipB.mp4").write_bytes(b"video")
            (transcripts_dir / "clipB.json").write_text(json.dumps(payload), encoding="utf-8")

            edl = build_edl(transcripts_dir, silence_policy="keep")
            self.assertEqual(edl["silence_policy"], "keep")
            self.assertEqual(len(edl["ranges"]), 1)
            self.assertAlmostEqual(edl["ranges"][0]["start"], 0.0, places=2)
            self.assertAlmostEqual(edl["ranges"][0]["end"], 2.4, places=2)

    def test_build_edl_visual_policy_preserves_moving_silence(self):
        payload = {
            "words": [
                {"type": "word", "text": "look", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
                {"type": "spacing", "start": 0.4, "end": 2.0},
                {"type": "word", "text": "wow", "start": 2.0, "end": 2.4, "speaker_id": "speaker_0"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            edit_dir = root / "edit"
            transcripts_dir = edit_dir / "transcripts"
            transcripts_dir.mkdir(parents=True)
            video_path = root / "clipC.mp4"
            self._write_test_video(video_path, moving=True)
            (transcripts_dir / "clipC.json").write_text(json.dumps(payload), encoding="utf-8")

            edl = build_edl(
                transcripts_dir,
                silence_policy="visual",
                motion_threshold=0.005,
                max_visual_gap=3.0,
            )
            self.assertEqual(len(edl["ranges"]), 1)
            self.assertAlmostEqual(edl["ranges"][0]["start"], 0.0, places=2)
            self.assertAlmostEqual(edl["ranges"][0]["end"], 2.4, places=2)
            self.assertIn("visual track", edl["ranges"][0]["reason"])

    def test_build_edl_visual_policy_cuts_static_silence(self):
        payload = {
            "words": [
                {"type": "word", "text": "look", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
                {"type": "spacing", "start": 0.4, "end": 2.0},
                {"type": "word", "text": "wow", "start": 2.0, "end": 2.4, "speaker_id": "speaker_0"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            edit_dir = root / "edit"
            transcripts_dir = edit_dir / "transcripts"
            transcripts_dir.mkdir(parents=True)
            video_path = root / "clipD.mp4"
            self._write_test_video(video_path, moving=False)
            (transcripts_dir / "clipD.json").write_text(json.dumps(payload), encoding="utf-8")

            edl = build_edl(
                transcripts_dir,
                silence_policy="visual",
                motion_threshold=0.005,
                max_visual_gap=3.0,
            )
            self.assertEqual(len(edl["ranges"]), 2)
            self.assertAlmostEqual(edl["ranges"][0]["end"], 0.4, places=2)
            self.assertAlmostEqual(edl["ranges"][1]["start"], 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
