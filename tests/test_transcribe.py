import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from helpers import transcribe


class TestTranscribe(unittest.TestCase):
    def test_load_env_reads_repo_style_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "VIBE_VIDEO_TRANSCRIBE_PROVIDER=elevenlabs\nELEVENLABS_API_KEY=test-key\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                transcribe.load_env_file(env_path)
                self.assertEqual(os.environ["VIBE_VIDEO_TRANSCRIBE_PROVIDER"], "elevenlabs")
                self.assertEqual(os.environ["ELEVENLABS_API_KEY"], "test-key")

    def test_resolve_provider_defaults_to_local(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(transcribe.resolve_provider(None), "local")

    def test_normalize_language_code_for_local_provider(self):
        self.assertEqual(transcribe.normalize_language_code("pt-br", "local"), "pt")
        self.assertEqual(transcribe.normalize_language_code("PT_BR", "local"), "pt")
        self.assertEqual(transcribe.normalize_language_code("en", "local"), "en")

    def test_normalize_language_code_for_elevenlabs_preserves_locale(self):
        self.assertEqual(transcribe.normalize_language_code("pt-br", "elevenlabs"), "pt-br")

    @patch("helpers.transcribe.requests.post")
    def test_call_elevenlabs_uses_api_key_and_returns_payload(self, mock_post):
        response = MagicMock()
        response.json.return_value = {"words": [{"text": "hello", "start": 0.0, "end": 0.5, "type": "word"}]}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"ELEVENLABS_API_KEY": "secret", "ELEVENLABS_SPEECH_TO_TEXT_MODEL": "scribe_v2"},
            clear=True,
        ):
            video = Path(tmp_dir) / "clip.mp4"
            video.write_bytes(b"fake video")
            payload = transcribe.call_elevenlabs(video, language="en", num_speakers=2)

        self.assertIn("words", payload)
        self.assertEqual(payload["words"][0]["text"], "hello")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["xi-api-key"], "secret")
        self.assertEqual(kwargs["data"]["model_id"], "scribe_v2")
        self.assertEqual(kwargs["data"]["language_code"], "en")
        self.assertEqual(kwargs["data"]["num_speakers"], "2")
        self.assertEqual(kwargs["data"]["diarize"], "true")


if __name__ == "__main__":
    unittest.main()
