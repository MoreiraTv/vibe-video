import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers.templates import (
    apply_setup_to_edl,
    build_template_setup,
    create_template_from_edit,
    infer_defaults_from_edit,
    load_template,
    merge_template_answers,
)
from helpers.template_wizard import collect_answers


class TestTemplates(unittest.TestCase):
    def test_load_template_resolves_inheritance(self):
        template = load_template("speaker-vertical")
        self.assertEqual(template["id"], "speaker-vertical")
        self.assertTrue(template["questions"])
        self.assertEqual(template["defaults"]["silence_policy"], "keep")

    def test_build_template_setup_derives_render_hints(self):
        template = load_template("vertical-gameplay-facecam")
        answers = merge_template_answers(
            template,
            answers={
                "aspect_ratio": "9:16",
                "subtitle_style": "karaoke",
                "tracking_enabled": True,
                "tracking_mode": "vlog",
                "silence_policy": "visual",
            },
        )
        setup = build_template_setup(template, answers)
        self.assertTrue(setup["render"]["vertical"])
        self.assertEqual(setup["render"]["subtitles_path"], "edit/master.ass")
        self.assertEqual(setup["edl_overrides"]["tracking"]["mode"], "vlog")
        self.assertEqual(setup["editing"]["silence_policy"], "visual")

    def test_apply_setup_to_edl_updates_render_related_fields(self):
        template = load_template("horizontal-best-moments")
        answers = merge_template_answers(
            template,
            answers={
                "aspect_ratio": "16:9",
                "subtitle_style": "plain",
                "silence_policy": "audio",
                "tracking_enabled": False,
                "grade_preset": "neutral_punch",
            },
        )
        setup = build_template_setup(template, answers)
        edl = {"ranges": [], "sources": {}, "grade": "auto"}
        updated = apply_setup_to_edl(setup, edl)
        self.assertFalse(updated["vertical"])
        self.assertEqual(updated["silence_policy"], "audio")
        self.assertEqual(updated["subtitles"], "edit/master.srt")
        self.assertEqual(updated["grade"], "neutral_punch")

    def test_create_template_from_edit_uses_existing_setup_answers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            edit_dir = Path(tmp_dir) / "edit"
            edit_dir.mkdir(parents=True)
            (edit_dir / "template_setup.json").write_text(
                json.dumps(
                    {
                        "template_id": "vertical-vlog-blur-bg",
                        "answers": {
                            "subtitle_style": "karaoke",
                            "subtitle_position": "top",
                            "silence_policy": "visual",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (edit_dir / "edl.json").write_text(
                json.dumps(
                    {
                        "vertical": True,
                        "silence_policy": "visual",
                        "subtitles": "edit/master.ass",
                        "grade": "auto",
                    }
                ),
                encoding="utf-8",
            )
            out_path = create_template_from_edit(edit_dir, "My Vlog Template")
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["id"], "my-vlog-template")
            self.assertEqual(payload["defaults"]["subtitle_position"], "top")
            self.assertEqual(payload["defaults"]["subtitle_style"], "karaoke")
            out_path.unlink()

    def test_infer_defaults_from_edit_reads_edl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            edit_dir = Path(tmp_dir) / "edit"
            edit_dir.mkdir(parents=True)
            (edit_dir / "edl.json").write_text(
                json.dumps(
                    {
                        "vertical": True,
                        "silence_policy": "keep",
                        "subtitles": "edit/master.ass",
                        "grade": "warm_cinematic",
                        "tracking": {"enabled": True, "mode": "stage"},
                    }
                ),
                encoding="utf-8",
            )
            inferred = infer_defaults_from_edit(edit_dir)
            self.assertEqual(inferred["aspect_ratio"], "9:16")
            self.assertEqual(inferred["silence_policy"], "keep")
            self.assertEqual(inferred["subtitle_style"], "karaoke")
            self.assertTrue(inferred["tracking_enabled"])

    def test_collect_answers_accepts_prefilled_without_prompting(self):
        template = load_template("vertical-vlog-blur-bg")
        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            answers = collect_answers(
                template,
                prefilled_answers={
                    "goal": "best_moments",
                    "split_mode": "multi_clip",
                    "aspect_ratio": "9:16",
                    "layout_style": "center-blur-bg",
                    "facecam_position": "auto",
                    "background_style": "blur-fill",
                    "title_mode": "auto_suggest",
                    "logo_mode": "none",
                    "tracking_enabled": False,
                    "silence_policy": "visual",
                    "must_preserve_reactions": True,
                    "pacing": "balanced",
                    "subtitle_position": "center",
                    "subtitle_grouping": "3",
                    "subtitle_style": "karaoke",
                    "subtitle_highlight_color": "yellow",
                    "subtitle_outline": "strong",
                    "subtitle_font": "clean-modern",
                    "subtitle_case": "title",
                    "grade_preset": "auto",
                },
            )
        self.assertEqual(answers["subtitle_style"], "karaoke")
        self.assertFalse(answers["tracking_enabled"])
        self.assertIsNone(answers["tracking_mode"])


if __name__ == "__main__":
    unittest.main()
