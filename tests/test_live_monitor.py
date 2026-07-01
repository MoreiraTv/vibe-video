import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers.live_monitor import (
    LiveTarget,
    SessionRecord,
    build_page_url,
    create_session_record,
    format_session_brief,
    get_session_record,
    is_live,
    list_session_records,
    load_registry,
    normalize_target,
    read_session_transcript_entries,
    should_process_segment,
    upsert_session_record,
    write_transcript_markdown,
)


class TestLiveMonitor(unittest.TestCase):
    def test_build_page_url(self):
        self.assertEqual(build_page_url("kick", "jonvlogs"), "https://kick.com/jonvlogs")
        self.assertEqual(build_page_url("twitch", "gaules"), "https://twitch.tv/gaules")

    def test_normalize_target_from_provider_and_channel(self):
        target = normalize_target("kick", "jonvlogs")
        self.assertEqual(target, LiveTarget(provider="kick", channel="jonvlogs", page_url="https://kick.com/jonvlogs"))

    def test_normalize_target_from_url(self):
        target = normalize_target("https://kick.com/jonvlogs")
        self.assertEqual(target.provider, "kick")
        self.assertEqual(target.channel, "jonvlogs")

    @patch("helpers.live_monitor.get_stream_url")
    def test_is_live_success(self, mock_get_stream_url):
        mock_get_stream_url.return_value = "https://example.com/live.m3u8"
        target = normalize_target("kick", "jonvlogs")
        online, stream_url, error = is_live(target)
        self.assertTrue(online)
        self.assertEqual(stream_url, "https://example.com/live.m3u8")
        self.assertIsNone(error)

    @patch("helpers.live_monitor.get_stream_url")
    def test_is_live_failure(self, mock_get_stream_url):
        mock_get_stream_url.side_effect = RuntimeError("offline")
        target = normalize_target("kick", "jonvlogs")
        online, stream_url, error = is_live(target)
        self.assertFalse(online)
        self.assertIsNone(stream_url)
        self.assertIn("offline", error)

    def test_registry_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            record = create_session_record(
                base_dir=base_dir,
                target=LiveTarget(provider="kick", channel="jonvlogs", page_url="https://kick.com/jonvlogs"),
                quality="best",
                language="pt",
                transcribe_provider="local",
                transcribe_model="large-v3-turbo",
                num_speakers=None,
                poll_seconds=30,
                segment_time=6,
            )
            upsert_session_record(base_dir, record)
            payload = load_registry(base_dir)
            self.assertIn(record.id, payload["sessions"])
            loaded = get_session_record(base_dir, record.id)
            self.assertEqual(loaded.channel, "jonvlogs")
            self.assertEqual(len(list_session_records(base_dir)), 1)

    def test_write_transcript_markdown_and_read_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            record = SessionRecord(
                id="live_test",
                provider="kick",
                channel="jonvlogs",
                page_url="https://kick.com/jonvlogs",
                created_at=1,
                updated_at=1,
                status="monitoring",
                desired_status="running",
                pid=123,
                quality="best",
                language=None,
                transcribe_provider="local",
                transcribe_model="large-v3-turbo",
                num_speakers=None,
                poll_seconds=30,
                segment_time=6,
                base_dir=str(session_dir.parent),
                session_dir=str(session_dir),
                segments_dir=str(session_dir / "segments"),
                transcripts_dir=str(session_dir / "transcripts"),
                transcript_buffer_path=str(session_dir / "transcript_buffer.md"),
                transcript_jsonl_path=str(session_dir / "transcript.jsonl"),
                events_path=str(session_dir / "events.jsonl"),
                log_path=str(session_dir / "worker.log"),
            )
            entries = [
                {
                    "timestamp": 1,
                    "captured_at_iso": "2026-06-30 10:00:00",
                    "segment": "segment_00001.mp4",
                    "segment_path": "x",
                    "transcript_path": "y",
                    "phrases": [{"start": "000.00", "end": "002.50", "text": "hello world"}],
                }
            ]
            transcript_jsonl = Path(record.transcript_jsonl_path)
            transcript_jsonl.write_text(json.dumps(entries[0]) + "\n", encoding="utf-8")
            write_transcript_markdown(Path(record.transcript_buffer_path), entries)
            content = Path(record.transcript_buffer_path).read_text(encoding="utf-8")
            self.assertIn("hello world", content)
            self.assertEqual(read_session_transcript_entries(record)[0]["segment"], "segment_00001.mp4")

    def test_should_process_segment_requires_stable_size(self):
        self.assertFalse(should_process_segment(None, 5000))
        self.assertFalse(should_process_segment(3000, 5000))
        self.assertTrue(should_process_segment(5000, 5000))

    def test_format_session_brief(self):
        record = SessionRecord(
            id="live_demo",
            provider="kick",
            channel="jonvlogs",
            page_url="https://kick.com/jonvlogs",
            created_at=1,
            updated_at=2,
            status="monitoring",
            desired_status="running",
            pid=999,
            quality="best",
            language=None,
            transcribe_provider="local",
            transcribe_model="large-v3-turbo",
            num_speakers=None,
            poll_seconds=30,
            segment_time=6,
            base_dir="base",
            session_dir="session",
            segments_dir="segments",
            transcripts_dir="transcripts",
            transcript_buffer_path="buffer.md",
            transcript_jsonl_path="transcript.jsonl",
            events_path="events.jsonl",
            log_path="worker.log",
            processed_segments=4,
        )
        brief = format_session_brief(record)
        self.assertEqual(brief["id"], "live_demo")
        self.assertEqual(brief["processed_segments"], 4)
        self.assertEqual(brief["transcript_buffer_path"], "buffer.md")


if __name__ == "__main__":
    unittest.main()
