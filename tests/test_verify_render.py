import unittest

from helpers.verify_render import build_verification_windows, cut_boundaries, duration_check, expected_duration


class TestVerifyRender(unittest.TestCase):
    def test_expected_duration_and_boundaries(self):
        edl = {
            "ranges": [
                {"start": 0.0, "end": 2.0},
                {"start": 4.0, "end": 5.5},
                {"start": 10.0, "end": 11.0},
            ]
        }
        self.assertEqual(expected_duration(edl), 4.5)
        self.assertEqual(cut_boundaries(edl), [2.0, 3.5])

    def test_build_verification_windows(self):
        windows = build_verification_windows(9.0, [2.0, 5.0], radius=1.0, midpoint_count=2)
        labels = [item["label"] for item in windows]
        self.assertIn("cut_01", labels)
        self.assertIn("cut_02", labels)
        self.assertIn("opening", labels)
        self.assertIn("ending", labels)
        self.assertIn("mid_01", labels)
        self.assertIn("mid_02", labels)

    def test_duration_check(self):
        good = duration_check(10.0, 10.08, tolerance=0.1)
        bad = duration_check(10.0, 10.3, tolerance=0.1)
        self.assertTrue(good["ok"])
        self.assertFalse(bad["ok"])


if __name__ == "__main__":
    unittest.main()
