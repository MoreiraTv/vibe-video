"""Transcribe a video using faster-whisper locally or ElevenLabs Scribe.

Writes a JSON response compatible with the Scribe format to
<edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --model large-v3-turbo
    python helpers/transcribe.py <video_path> --provider elevenlabs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


DEFAULT_LOCAL_MODEL = "large-v3-turbo"
DEFAULT_ELEVEN_MODEL = "scribe_v2"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ENV_PROVIDER = "VIBE_VIDEO_TRANSCRIBE_PROVIDER"
ENV_ELEVEN_KEY = "ELEVENLABS_API_KEY"
ENV_ELEVEN_MODEL = "ELEVENLABS_SPEECH_TO_TEXT_MODEL"


def normalize_language_code(language: str | None, provider: str) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower().replace("_", "-")
    if provider == "local" and "-" in normalized:
        return normalized.split("-", 1)[0]
    return normalized


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_env() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")
    load_env_file(Path.cwd() / ".env")


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_whisper(
    audio_path: Path,
    model_name: str = DEFAULT_LOCAL_MODEL,
    language: str | None = None,
) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install it or use --provider elevenlabs with ELEVENLABS_API_KEY."
        ) from exc

    # Try to load on GPU first, fallback to CPU
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"  [Warning] CUDA failed ({e}), falling back to CPU with int8...", flush=True)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True
    )
    
    words = []
    for segment in segments:
        for w in segment.words:
            words.append({
                "type": "word",
                "text": w.word,
                "start": w.start,
                "end": w.end,
                "speaker_id": "speaker_0"  # Dummy speaker ID
            })

    return {"words": words}


def call_elevenlabs(
    video_path: Path,
    language: str | None = None,
    num_speakers: int | None = None,
    model_name: str | None = None,
) -> dict:
    api_key = os.getenv(ENV_ELEVEN_KEY)
    if not api_key:
        raise RuntimeError(
            f"{ENV_ELEVEN_KEY} is not set. Add it to .env or your shell environment to use ElevenLabs transcription."
        )

    model_id = model_name or os.getenv(ENV_ELEVEN_MODEL, DEFAULT_ELEVEN_MODEL)
    data: dict[str, str] = {
        "model_id": model_id,
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers is not None:
        data["num_speakers"] = str(num_speakers)
        data["diarize"] = "true"

    with video_path.open("rb") as handle:
        response = requests.post(
            ELEVENLABS_STT_URL,
            headers={"xi-api-key": api_key},
            data=data,
            files={"file": (video_path.name, handle, "application/octet-stream")},
            timeout=1800,
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"ElevenLabs transcription failed ({response.status_code}): {response.text[:400]}"
        ) from exc

    payload = response.json()
    if not isinstance(payload, dict) or "words" not in payload:
        raise RuntimeError("ElevenLabs response did not include a 'words' field.")
    return payload


def resolve_provider(provider: str | None) -> str:
    chosen = (provider or os.getenv(ENV_PROVIDER, "local")).strip().lower()
    if chosen not in {"local", "elevenlabs"}:
        raise ValueError(f"unsupported transcription provider: {chosen}")
    return chosen


def transcribe_one(
    video: Path,
    edit_dir: Path,
    model_name: str = DEFAULT_LOCAL_MODEL,
    language: str | None = None,
    provider: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    load_env()
    resolved_provider = resolve_provider(provider)
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    t0 = time.time()
    if resolved_provider == "local":
        normalized_language = normalize_language_code(language, resolved_provider)
        if verbose:
            print(f"  extracting audio from {video.name}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / f"{video.stem}.wav"
            extract_audio(video, audio)
            size_mb = audio.stat().st_size / (1024 * 1024)
            if verbose:
                print(
                    f"  transcribing {video.stem}.wav ({size_mb:.1f} MB) locally with {model_name}...",
                    flush=True,
                )
            payload = call_whisper(audio, model_name, normalized_language)
    else:
        normalized_language = normalize_language_code(language, resolved_provider)
        if verbose:
            eleven_model = model_name or os.getenv(ENV_ELEVEN_MODEL, DEFAULT_ELEVEN_MODEL)
            print(f"  transcribing {video.name} with ElevenLabs ({eleven_model})...", flush=True)
        payload = call_elevenlabs(
            video_path=video,
            language=normalized_language,
            num_speakers=num_speakers,
            model_name=model_name,
        )

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s via {resolved_provider}")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with faster-whisper or ElevenLabs")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en', 'pt'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name for the selected provider. Defaults to large-v3-turbo for local "
            "and ELEVENLABS_SPEECH_TO_TEXT_MODEL or scribe_v2 for ElevenLabs."
        ),
    )
    ap.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["local", "elevenlabs"],
        help=(
            "Transcription provider. Defaults to .env VIBE_VIDEO_TRANSCRIBE_PROVIDER if set, "
            "otherwise local."
        ),
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional diarization hint. Used by ElevenLabs; ignored by local faster-whisper.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        model_name=args.model or (DEFAULT_ELEVEN_MODEL if resolve_provider(args.provider) == "elevenlabs" else DEFAULT_LOCAL_MODEL),
        language=args.language,
        provider=args.provider,
        num_speakers=args.num_speakers,
    )


if __name__ == "__main__":
    main()
