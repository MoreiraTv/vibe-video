import tempfile
import unittest
from pathlib import Path

from PIL import Image

from helpers.timeline_view import render_edl_overview, segment_offsets


class TestTimelineView(unittest.TestCase):
    def test_segment_offsets(self):
        edl = {
            "ranges": [
                {"source": "A", "start": 1.0, "end": 3.0},
                {"source": "B", "start": 5.0, "end": 6.5},
            ]
        }
        offsets = segment_offsets(edl)
        self.assertEqual(offsets[0]["output_start"], 0.0)
        self.assertEqual(offsets[0]["output_end"], 2.0)
        self.assertEqual(offsets[1]["output_start"], 2.0)
        self.assertEqual(offsets[1]["output_end"], 3.5)

    def test_render_edl_overview(self):
        edl = {
            "ranges": [
                {"source": "A", "start": 0.0, "end": 2.0, "beat": "HOOK"},
                {"source": "B", "start": 1.0, "end": 4.0, "beat": "DETAIL"},
            ],
            "overlays": [{"file": "edit/animations/slot_1/render.mp4", "start_in_output": 0.5, "duration": 1.2}],
            "subtitles": "edit/master.srt",
            "total_duration_s": 5.0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "overview.png"
            render_edl_overview(edl, out_path)
            self.assertTrue(out_path.exists())
            with Image.open(out_path) as img:
                self.assertEqual(img.size, (2000, 760))


if __name__ == "__main__":
    unittest.main()
