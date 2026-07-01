"""Monitor live streams with persistent session IDs and continuous transcription.

This helper supports multiple concurrent live-monitor sessions. Each session gets
its own ID, directory, transcript files, and worker process so the agent can
inspect the saved transcript directly from disk without polling the CLI.

Main commands:
    python helpers/live_monitor.py start https://kick.com/jonvlogs
    python helpers/live_monitor.py list
    python helpers/live_monitor.py show live_20260630_ab12cd
    python helpers/live_monitor.py read live_20260630_ab12cd
    python helpers/live_monitor.py stop live_20260630_ab12cd
    python helpers/live_monitor.py delete live_20260630_ab12cd
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import string
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from pack_transcripts import group_into_phrases
    from transcribe import DEFAULT_ELEVEN_MODEL, DEFAULT_LOCAL_MODEL, resolve_provider, transcribe_one
except Exception:
    from helpers.pack_transcripts import group_into_phrases
    from helpers.transcribe import DEFAULT_ELEVEN_MODEL, DEFAULT_LOCAL_MODEL, resolve_provider, transcribe_one


ENV_STREAMLINK_CMD = "VIBE_VIDEO_STREAMLINK_CMD"
ENV_FFMPEG_CMD = "VIBE_VIDEO_FFMPEG_CMD"
ENV_BASE_DIR = "VIBE_VIDEO_LIVE_MONITOR_DIR"
SUPPORTED_PROVIDERS = {"kick", "twitch", "youtube"}
REGISTRY_FILENAME = "registry.json"
SCAN_INTERVAL_SECONDS = 2.0
MIN_READY_BYTES = 1024
DEFAULT_SEGMENT_TIME = 6
DEFAULT_POLL_SECONDS = 30


@dataclass(frozen=True)
class LiveTarget:
    provider: str
    channel: str
    page_url: str


@dataclass
class SessionRecord:
    id: str
    provider: str
    channel: str
    page_url: str
    created_at: int
    updated_at: int
    status: str
    desired_status: str
    pid: int | None
    quality: str
    language: str | None
    transcribe_provider: str
    transcribe_model: str
    num_speakers: int | None
    poll_seconds: int
    segment_time: int
    base_dir: str
    session_dir: str
    segments_dir: str
    transcripts_dir: str
    transcript_buffer_path: str
    transcript_jsonl_path: str
    events_path: str
    log_path: str
    last_error: str | None = None
    stream_url: str | None = None
    processed_segments: int = 0


def now_ts() -> int:
    return int(time.time())


def _which_or_env(env_name: str, fallback: str) -> str | None:
    override = os.getenv(env_name)
    if override:
        return override
    return shutil.which(fallback)


def resolve_streamlink_cmd() -> str:
    cmd = _which_or_env(ENV_STREAMLINK_CMD, "streamlink")
    if not cmd:
        raise RuntimeError(
            "streamlink not found on PATH. Install Streamlink or set VIBE_VIDEO_STREAMLINK_CMD."
        )
    return cmd


def resolve_ffmpeg_cmd() -> str:
    cmd = _which_or_env(ENV_FFMPEG_CMD, "ffmpeg")
    if not cmd:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or set VIBE_VIDEO_FFMPEG_CMD."
        )
    return cmd


def build_page_url(provider: str, channel: str) -> str:
    provider = provider.lower()
    if provider == "kick":
        return f"https://kick.com/{channel}"
    if provider == "twitch":
        return f"https://twitch.tv/{channel}"
    return channel if channel.startswith("http") else f"https://youtube.com/watch?v={channel}"


def normalize_target(provider_or_input: str, channel: str | None = None) -> LiveTarget:
    if channel is not None:
        provider = provider_or_input.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported provider: {provider}")
        safe_channel = channel.strip().strip("/")
        if not safe_channel:
            raise ValueError("channel name cannot be empty")
        return LiveTarget(provider=provider, channel=safe_channel, page_url=build_page_url(provider, safe_channel))

    raw = provider_or_input.strip()
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("when provider is omitted, pass a full live URL like https://kick.com/jonvlogs")

    host = parsed.netloc.lower()
    parts = [part for part in parsed.path.split("/") if part]
    if "kick.com" in host:
        if not parts:
            raise ValueError("Kick URL does not include a channel")
        channel_name = parts[0]
        return LiveTarget(provider="kick", channel=channel_name, page_url=f"https://kick.com/{channel_name}")
    if "twitch.tv" in host:
        if not parts:
            raise ValueError("Twitch URL does not include a channel")
        channel_name = parts[0]
        return LiveTarget(provider="twitch", channel=channel_name, page_url=f"https://twitch.tv/{channel_name}")
    if "youtube.com" in host or "youtu.be" in host:
        return LiveTarget(provider="youtube", channel=raw, page_url=raw)
    raise ValueError(f"unsupported live URL host: {parsed.netloc}")


def get_base_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    env_dir = os.getenv(ENV_BASE_DIR)
    if env_dir:
        return Path(env_dir).resolve()
    return (Path.cwd() / "edit" / "live_monitor").resolve()


def registry_path(base_dir: Path) -> Path:
    return base_dir / REGISTRY_FILENAME


def ensure_base_layout(base_dir: Path) -> None:
    (base_dir / "sessions").mkdir(parents=True, exist_ok=True)


def load_registry(base_dir: Path) -> dict[str, Any]:
    ensure_base_layout(base_dir)
    path = registry_path(base_dir)
    if not path.exists():
        return {"sessions": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "sessions" not in payload or not isinstance(payload["sessions"], dict):
        return {"sessions": {}}
    return payload


def save_registry(base_dir: Path, payload: dict[str, Any]) -> None:
    ensure_base_layout(base_dir)
    registry_path(base_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_session_records(base_dir: Path) -> list[SessionRecord]:
    payload = load_registry(base_dir)
    sessions = payload.get("sessions", {})
    return [SessionRecord(**record) for record in sessions.values()]


def get_session_record(base_dir: Path, session_id: str) -> SessionRecord:
    payload = load_registry(base_dir)
    record = payload.get("sessions", {}).get(session_id)
    if not record:
        raise KeyError(f"session not found: {session_id}")
    return SessionRecord(**record)


def upsert_session_record(base_dir: Path, record: SessionRecord) -> None:
    payload = load_registry(base_dir)
    payload.setdefault("sessions", {})
    payload["sessions"][record.id] = asdict(record)
    save_registry(base_dir, payload)


def remove_session_record(base_dir: Path, session_id: str) -> None:
    payload = load_registry(base_dir)
    payload.setdefault("sessions", {})
    payload["sessions"].pop(session_id, None)
    save_registry(base_dir, payload)


def new_session_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"live_{stamp}_{suffix}"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_transcript_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = ["# Live transcript buffer", ""]
    for entry in entries:
        segment = entry["segment"]
        lines.append(f"## {segment} ({entry['captured_at_iso']})")
        for phrase in entry["phrases"]:
            lines.append(f"  [{phrase['start']}-{phrase['end']}] {phrase['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_time(seconds: float) -> str:
    return f"{seconds:07.2f}"


def words_to_phrases(transcript_path: Path, silence_threshold: float = 0.5) -> list[dict[str, Any]]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    phrases = group_into_phrases(data.get("words", []), silence_threshold)
    return [
        {
            "start": format_time(float(item["start"])),
            "end": format_time(float(item["end"])),
            "text": item["text"],
        }
        for item in phrases
    ]


def create_session_record(
    base_dir: Path,
    target: LiveTarget,
    quality: str,
    language: str | None,
    transcribe_provider: str,
    transcribe_model: str,
    num_speakers: int | None,
    poll_seconds: int,
    segment_time: int,
) -> SessionRecord:
    session_id = new_session_id()
    session_dir = base_dir / "sessions" / session_id
    segments_dir = session_dir / "segments"
    transcripts_dir = session_dir / "transcripts"
    session_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    created = now_ts()
    return SessionRecord(
        id=session_id,
        provider=target.provider,
        channel=target.channel,
        page_url=target.page_url,
        created_at=created,
        updated_at=created,
        status="starting",
        desired_status="running",
        pid=None,
        quality=quality,
        language=language,
        transcribe_provider=transcribe_provider,
        transcribe_model=transcribe_model,
        num_speakers=num_speakers,
        poll_seconds=poll_seconds,
        segment_time=segment_time,
        base_dir=str(base_dir),
        session_dir=str(session_dir),
        segments_dir=str(segments_dir),
        transcripts_dir=str(transcripts_dir),
        transcript_buffer_path=str(session_dir / "transcript_buffer.md"),
        transcript_jsonl_path=str(session_dir / "transcript.jsonl"),
        events_path=str(session_dir / "events.jsonl"),
        log_path=str(session_dir / "worker.log"),
    )


def run_streamlink(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [resolve_streamlink_cmd(), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def get_stream_url(target: LiveTarget, quality: str = "best") -> str:
    result = run_streamlink(["--stream-url", target.page_url, quality])
    stdout = result.stdout.strip()
    if result.returncode != 0 or not stdout:
        stderr = result.stderr.strip() or stdout
        raise RuntimeError(stderr or f"streamlink failed to resolve stream for {target.page_url}")
    return stdout


def is_live(target: LiveTarget, quality: str = "best") -> tuple[bool, str | None, str | None]:
    try:
        stream_url = get_stream_url(target, quality=quality)
        return True, stream_url, None
    except Exception as exc:
        return False, None, str(exc)


def wait_until_live(target: LiveTarget, poll_seconds: int, timeout_seconds: int | None, quality: str = "best") -> str:
    started = time.time()
    while True:
        online, stream_url, _error = is_live(target, quality=quality)
        if online and stream_url:
            return stream_url
        if timeout_seconds is not None and (time.time() - started) >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for {target.provider}:{target.channel} to go live."
            )
        time.sleep(max(1, poll_seconds))


def capture_clip(stream_url: str, output_path: Path, duration: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = resolve_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i",
        stream_url,
        "-t",
        str(duration),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def spawn_segmenter(stream_url: str, segments_dir: Path, segment_time: int) -> subprocess.Popen[str]:
    ffmpeg_cmd = resolve_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i",
        stream_url,
        "-f",
        "segment",
        "-segment_time",
        str(segment_time),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(segments_dir / "segment_%05d.mp4"),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def pid_is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate_pid(pid: int | None) -> None:
    if not pid:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def remove_tree_with_retries(path: Path, attempts: int = 8, delay_seconds: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    if last_error:
        raise last_error


def refresh_record(base_dir: Path, session_id: str, **updates: Any) -> SessionRecord:
    record = get_session_record(base_dir, session_id)
    for key, value in updates.items():
        setattr(record, key, value)
    record.updated_at = now_ts()
    upsert_session_record(base_dir, record)
    return record


def _detached_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]


def spawn_worker(base_dir: Path, session_id: str) -> int:
    record = get_session_record(base_dir, session_id)
    log_path = Path(record.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        session_id,
        "--base-dir",
        str(base_dir),
    ]
    popen_kwargs: dict[str, Any] = {
        "stdout": log_handle,
        "stderr": log_handle,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = _detached_creation_flags()
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)
    refresh_record(base_dir, session_id, pid=proc.pid, status="starting", desired_status="running")
    log_handle.close()
    return proc.pid


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def read_session_transcript_entries(record: SessionRecord) -> list[dict[str, Any]]:
    return read_jsonl(Path(record.transcript_jsonl_path))


def append_event(record: SessionRecord, event_type: str, message: str, **payload: Any) -> None:
    append_jsonl(
        Path(record.events_path),
        {
            "timestamp": now_ts(),
            "type": event_type,
            "message": message,
            **payload,
        },
    )


def transcribe_segment(record: SessionRecord, segment_path: Path) -> dict[str, Any]:
    session_dir = Path(record.session_dir)
    transcript_path = transcribe_one(
        video=segment_path,
        edit_dir=session_dir,
        model_name=record.transcribe_model,
        language=record.language,
        provider=record.transcribe_provider,
        num_speakers=record.num_speakers,
        verbose=False,
    )
    phrases = words_to_phrases(transcript_path)
    entry = {
        "timestamp": now_ts(),
        "captured_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "segment": segment_path.name,
        "segment_path": str(segment_path),
        "transcript_path": str(transcript_path),
        "phrases": phrases,
    }
    append_jsonl(Path(record.transcript_jsonl_path), entry)
    transcript_entries = read_session_transcript_entries(record)
    write_transcript_markdown(Path(record.transcript_buffer_path), transcript_entries)
    return entry


def should_process_segment(previous_size: int | None, current_size: int) -> bool:
    return previous_size is not None and previous_size == current_size and current_size >= MIN_READY_BYTES


def run_worker_loop(base_dir: Path, session_id: str) -> None:
    record = get_session_record(base_dir, session_id)
    target = LiveTarget(provider=record.provider, channel=record.channel, page_url=record.page_url)
    segments_dir = Path(record.segments_dir)
    size_cache: dict[Path, int] = {}
    processed: set[str] = set()
    segmenter: subprocess.Popen[str] | None = None

    append_event(record, "worker_started", "Live monitor worker started.")
    refresh_record(base_dir, session_id, status="connecting")

    while True:
        record = get_session_record(base_dir, session_id)
        if record.desired_status != "running":
            break

        if segmenter is None or segmenter.poll() is not None:
            try:
                stream_url = get_stream_url(target, quality=record.quality)
                record = refresh_record(base_dir, session_id, status="monitoring", stream_url=stream_url, last_error=None)
                append_event(record, "stream_online", "Live stream is online and capture is starting.", stream_url=stream_url)
                segmenter = spawn_segmenter(stream_url, segments_dir, record.segment_time)
            except Exception as exc:
                record = refresh_record(base_dir, session_id, status="offline", last_error=str(exc))
                append_event(record, "stream_offline", "Live stream is offline or unavailable.", error=str(exc))
                time.sleep(max(1, record.poll_seconds))
                continue

        for segment_path in sorted(segments_dir.glob("segment_*.mp4")):
            if segment_path.name in processed:
                continue
            try:
                current_size = segment_path.stat().st_size
            except FileNotFoundError:
                continue
            previous_size = size_cache.get(segment_path)
            size_cache[segment_path] = current_size
            if not should_process_segment(previous_size, current_size):
                continue

            try:
                entry = transcribe_segment(record, segment_path)
                processed.add(segment_path.name)
                record = refresh_record(
                    base_dir,
                    session_id,
                    processed_segments=record.processed_segments + 1,
                    status="monitoring",
                )
                append_event(
                    record,
                    "segment_transcribed",
                    "Segment transcribed successfully.",
                    segment=segment_path.name,
                    phrases=len(entry["phrases"]),
                )
            except Exception as exc:
                record = refresh_record(base_dir, session_id, status="error", last_error=str(exc))
                append_event(
                    record,
                    "segment_error",
                    "Segment transcription failed.",
                    segment=segment_path.name,
                    error=str(exc),
                )

        time.sleep(SCAN_INTERVAL_SECONDS)

    if segmenter and segmenter.poll() is None:
        segmenter.terminate()
        try:
            segmenter.wait(timeout=10)
        except subprocess.TimeoutExpired:
            segmenter.kill()

    record = refresh_record(base_dir, session_id, status="stopped", desired_status="stopped")
    append_event(record, "worker_stopped", "Live monitor worker stopped.")


def monitor_stream(
    stream_url: str,
    out_dir: Path,
    segment_time: int,
    max_duration: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = resolve_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i",
        stream_url,
        "-f",
        "segment",
        "-segment_time",
        str(segment_time),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(segments_dir / "segment_%05d.mp4"),
    ]
    if max_duration is not None:
        cmd[4:4] = ["-t", str(max_duration)]
    subprocess.run(cmd, check=True)
    return segments_dir


def format_session_brief(record: SessionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "provider": record.provider,
        "channel": record.channel,
        "status": record.status,
        "desired_status": record.desired_status,
        "pid": record.pid,
        "processed_segments": record.processed_segments,
        "transcribe_provider": record.transcribe_provider,
        "transcript_buffer_path": record.transcript_buffer_path,
        "events_path": record.events_path,
        "session_dir": record.session_dir,
        "last_error": record.last_error,
        "updated_at": record.updated_at,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor live streams with persistent session IDs")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_target_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("provider_or_url", type=str, help="Provider name (kick/twitch/youtube) or full live URL")
        parser.add_argument("channel", type=str, nargs="?", help="Channel or video ID when provider is passed separately")
        parser.add_argument("--quality", type=str, default="best", help="Streamlink quality (default: best)")

    status_ap = sub.add_parser("status", help="Check whether a live stream target is online right now")
    add_target_args(status_ap)

    wait_ap = sub.add_parser("wait", help="Poll until a live stream target is online")
    add_target_args(wait_ap)
    wait_ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS, help="Polling interval in seconds")
    wait_ap.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")

    capture_ap = sub.add_parser("capture", help="Capture a fixed-duration clip from the live stream")
    add_target_args(capture_ap)
    capture_ap.add_argument("--duration", type=int, required=True, help="Clip duration in seconds")
    capture_ap.add_argument("-o", "--output", type=Path, default=None, help="Output mp4 path")

    monitor_ap = sub.add_parser("monitor", help="Segment the live stream into files for later editing")
    add_target_args(monitor_ap)
    monitor_ap.add_argument("--segment-time", type=int, default=DEFAULT_SEGMENT_TIME, help="Segment duration in seconds")
    monitor_ap.add_argument("--max-duration", type=int, default=None, help="Optional total monitoring duration in seconds")
    monitor_ap.add_argument("--out-dir", type=Path, default=None, help="Output directory for live segments")

    start_ap = sub.add_parser("start", help="Start a persistent live-monitor session with continuous transcription")
    add_target_args(start_ap)
    start_ap.add_argument("--base-dir", type=Path, default=None, help="Base live-monitor directory")
    start_ap.add_argument("--segment-time", type=int, default=DEFAULT_SEGMENT_TIME, help="Segment duration in seconds")
    start_ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS, help="Offline polling interval in seconds")
    start_ap.add_argument("--language", type=str, default=None, help="Optional transcription language code")
    start_ap.add_argument("--transcribe-provider", choices=["local", "elevenlabs"], default=None, help="Continuous transcription provider")
    start_ap.add_argument("--model", type=str, default=None, help="Model name for the chosen transcription provider")
    start_ap.add_argument("--num-speakers", type=int, default=None, help="Optional diarization hint for ElevenLabs")

    list_ap = sub.add_parser("list", help="List persistent live-monitor sessions")
    list_ap.add_argument("--base-dir", type=Path, default=None, help="Base live-monitor directory")

    show_ap = sub.add_parser("show", help="Show one persistent live-monitor session by ID")
    show_ap.add_argument("session_id", type=str)
    show_ap.add_argument("--base-dir", type=Path, default=None, help="Base live-monitor directory")

    read_ap = sub.add_parser("read", help="Read saved transcript info for one session")
    read_ap.add_argument("session_id", type=str)
    read_ap.add_argument("--base-dir", type=Path, default=None, help="Base live-monitor directory")
    read_ap.add_argument("--path-only", action="store_true", help="Print only the transcript buffer path")
    read_ap.add_argument("--jsonl", action="store_true", help="Print transcript entries as JSON")

    stop_ap = sub.add_parser("stop", help="Stop one persistent live-monitor session by ID")
    stop_ap.add_argument("session_id", type=str)
    stop_ap.add_argument("--base-dir", type=Path, default=None, help="Base live-monitor directory")

    delete_ap = sub.add_parser("delete", help="Delete one persistent live-monitor session and its files")
    delete_ap.add_argument("session_id", type=str)
    delete_ap.add_argument("--base-dir", type=Path, default=None, help="Base live-monitor directory")

    worker_ap = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker_ap.add_argument("session_id", type=str)
    worker_ap.add_argument("--base-dir", type=Path, required=True)

    args = ap.parse_args()

    if args.command == "_worker":
        run_worker_loop(args.base_dir.resolve(), args.session_id)
        return

    if args.command in {"status", "wait", "capture", "monitor", "start"}:
        target = normalize_target(args.provider_or_url, args.channel)

    if args.command == "status":
        online, stream_url, error = is_live(target, quality=args.quality)
        payload = {
            "provider": target.provider,
            "channel": target.channel,
            "page_url": target.page_url,
            "online": online,
            "stream_url": stream_url,
            "error": error,
            "checked_at": now_ts(),
        }
        print(json.dumps(payload, indent=2))
        if not online:
            sys.exit(1)
        return

    if args.command == "wait":
        stream_url = wait_until_live(
            target,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout,
            quality=args.quality,
        )
        print(stream_url)
        return

    if args.command == "capture":
        stream_url = get_stream_url(target, quality=args.quality)
        output_path = (args.output or ((Path.cwd() / "edit" / "live" / f"{target.provider}_{target.channel}") / f"{now_ts()}_capture.mp4")).resolve()
        capture_clip(stream_url, output_path, args.duration)
        print(output_path)
        return

    if args.command == "monitor":
        stream_url = get_stream_url(target, quality=args.quality)
        out_dir = (args.out_dir or ((Path.cwd() / "edit" / "live" / f"{target.provider}_{target.channel}"))).resolve()
        segments_dir = monitor_stream(
            stream_url,
            out_dir=out_dir,
            segment_time=args.segment_time,
            max_duration=args.max_duration,
        )
        print(segments_dir)
        return

    if args.command == "start":
        base_dir = get_base_dir(args.base_dir)
        transcribe_provider = resolve_provider(args.transcribe_provider)
        transcribe_model = args.model or (DEFAULT_ELEVEN_MODEL if transcribe_provider == "elevenlabs" else DEFAULT_LOCAL_MODEL)
        record = create_session_record(
            base_dir=base_dir,
            target=target,
            quality=args.quality,
            language=args.language,
            transcribe_provider=transcribe_provider,
            transcribe_model=transcribe_model,
            num_speakers=args.num_speakers,
            poll_seconds=args.poll_seconds,
            segment_time=args.segment_time,
        )
        upsert_session_record(base_dir, record)
        pid = spawn_worker(base_dir, record.id)
        record = get_session_record(base_dir, record.id)
        print(json.dumps(format_session_brief(record), indent=2))
        print(f"worker_pid={pid}")
        return

    if args.command == "list":
        base_dir = get_base_dir(args.base_dir)
        items = [format_session_brief(record) for record in list_session_records(base_dir)]
        print(json.dumps(items, indent=2))
        return

    if args.command == "show":
        base_dir = get_base_dir(args.base_dir)
        record = get_session_record(base_dir, args.session_id)
        print(json.dumps(asdict(record), indent=2))
        return

    if args.command == "read":
        base_dir = get_base_dir(args.base_dir)
        record = get_session_record(base_dir, args.session_id)
        transcript_path = Path(record.transcript_buffer_path)
        if args.path_only:
            print(transcript_path)
            return
        if args.jsonl:
            print(json.dumps(read_session_transcript_entries(record), indent=2, ensure_ascii=False))
            return
        if transcript_path.exists():
            print(transcript_path.read_text(encoding="utf-8"))
            return
        print("")
        return

    if args.command == "stop":
        base_dir = get_base_dir(args.base_dir)
        record = refresh_record(base_dir, args.session_id, desired_status="stopped")
        terminate_pid(record.pid)
        time.sleep(1)
        try:
            record = refresh_record(base_dir, args.session_id, status="stopped", pid=None)
        except KeyError:
            record.status = "stopped"
            record.pid = None
            record.desired_status = "stopped"
            record.updated_at = now_ts()
            upsert_session_record(base_dir, record)
        append_event(record, "manual_stop", "Session stopped by CLI.")
        print(json.dumps(format_session_brief(record), indent=2))
        return

    if args.command == "delete":
        base_dir = get_base_dir(args.base_dir)
        record = get_session_record(base_dir, args.session_id)
        terminate_pid(record.pid)
        session_dir = Path(record.session_dir)
        if session_dir.exists():
            remove_tree_with_retries(session_dir)
        remove_session_record(base_dir, args.session_id)
        print(json.dumps({"deleted": args.session_id, "session_dir": str(session_dir)}, indent=2))
        return


if __name__ == "__main__":
    main()
