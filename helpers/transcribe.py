"""Transcribe a video using faster-whisper locally.

Extracts mono 16kHz audio via ffmpeg, runs faster-whisper with word-level timestamps,
and writes a JSON response compatible with the Scribe format to
<edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --model large-v3-turbo
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

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("faster-whisper is not installed. Please install it using: pip install faster-whisper")


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_whisper(
    audio_path: Path,
    model_name: str = "large-v3-turbo",
    language: str | None = None,
) -> dict:
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


def transcribe_one(
    video: Path,
    edit_dir: Path,
    model_name: str = "large-v3-turbo",
    language: str | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  transcribing {video.stem}.wav ({size_mb:.1f} MB) locally with {model_name}...", flush=True)
        
        payload = call_whisper(audio, model_name, language)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally with faster-whisper")
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
        default="large-v3-turbo",
        help="Model name for faster-whisper (default: large-v3-turbo)",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Ignored. Kept for CLI compatibility.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        model_name=args.model,
        language=args.language,
    )


if __name__ == "__main__":
    main()
