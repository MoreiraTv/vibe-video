"""Batch-transcribe every video in a directory with local or ElevenLabs providers.

Walks <videos_dir> for common video extensions, runs faster-whisper on
each, writes transcripts to <videos_dir>/edit/transcripts/<name>.json.

Cached per-file: any source that already has a transcript is skipped.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --workers 1
    python helpers/transcribe_batch.py <videos_dir> --model large-v3-turbo
    python helpers/transcribe_batch.py <videos_dir> --edit-dir /custom/edit
    python helpers/transcribe_batch.py <videos_dir> --provider elevenlabs
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import DEFAULT_ELEVEN_MODEL, DEFAULT_LOCAL_MODEL, resolve_provider, transcribe_one


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel batch transcription of a videos directory")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1 for local GPU memory)")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for the selected provider",
    )
    ap.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["local", "elevenlabs"],
        help="Transcription provider. Defaults to .env VIBE_VIDEO_TRANSCRIBE_PROVIDER or local.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional diarization hint. Used by ElevenLabs; ignored by local faster-whisper.",
    )
    args = ap.parse_args()

    provider = resolve_provider(args.provider)
    model_name = args.model or (DEFAULT_ELEVEN_MODEL if provider == "elevenlabs" else DEFAULT_LOCAL_MODEL)

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    already_cached = [v for v in videos if (edit_dir / "transcripts" / f"{v.stem}.json").exists()]
    pending = [v for v in videos if v not in already_cached]

    print(f"found {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    print(f"transcribing {len(pending)} files with {args.workers} parallel workers via {provider}")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                video=v,
                edit_dir=edit_dir,
                model_name=model_name,
                language=args.language,
                provider=provider,
                num_speakers=args.num_speakers,
                verbose=False,
            ): v
            for v in pending
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  →  {out.name}")
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
