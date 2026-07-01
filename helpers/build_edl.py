"""Build a starter EDL from transcript JSON files.

Creates speech-first cut ranges by grouping transcript words into phrases,
dropping filler-only and obvious false-start phrases, then merging nearby
phrases into edit-ready ranges.

Usage:
    python helpers/build_edl.py --edit-dir /path/to/edit
    python helpers/build_edl.py --edit-dir /path/to/edit --videos-dir /path/to/raw
    python helpers/build_edl.py --edit-dir /path/to/edit -o /path/to/edit/edl.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from pack_transcripts import group_into_phrases
except Exception:
    from helpers.pack_transcripts import group_into_phrases


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".MP4", ".MOV", ".MKV", ".AVI", ".M4V"}
FILLER_WORDS = {
    "uh", "umm", "um", "erm", "ah", "like", "hmm", "hm", "mm", "so",
}


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", text.lower())


def tokenize_text(text: str) -> list[str]:
    return [tok for tok in (normalize_word(part) for part in text.split()) if tok]


def count_meaningful_tokens(tokens: list[str]) -> int:
    return sum(1 for token in tokens if token not in FILLER_WORDS)


def is_filler_phrase(text: str) -> bool:
    tokens = tokenize_text(text)
    if not tokens:
        return True
    meaningful = count_meaningful_tokens(tokens)
    return meaningful == 0


def is_false_start_phrase(current_text: str, next_text: str, current_duration: float) -> bool:
    current_tokens = tokenize_text(current_text)
    next_tokens = tokenize_text(next_text)
    if not current_tokens or not next_tokens:
        return False
    if len(current_tokens) > 6 or current_duration > 2.5:
        return False
    if current_tokens == next_tokens[: len(current_tokens)]:
        return True
    if len(current_tokens) >= 2 and current_tokens[0] == next_tokens[0]:
        overlap = sum(1 for a, b in zip(current_tokens, next_tokens) if a == b)
        return overlap >= min(2, len(current_tokens))
    return False


def phrase_to_range(
    phrase: dict,
    source_name: str,
    beat_index: int,
) -> dict:
    text = re.sub(r"\s+", " ", phrase["text"]).strip()
    beat = "HOOK" if beat_index == 1 else f"BEAT_{beat_index:02d}"
    return {
        "source": source_name,
        "start": round(float(phrase["start"]), 3),
        "end": round(float(phrase["end"]), 3),
        "beat": beat,
        "quote": text[:160],
        "reason": "Speech-first starter range from transcript phrases.",
    }


def sample_motion_score(
    video_path: Path,
    start: float,
    end: float,
    sample_fps: float = 2.0,
    max_samples: int = 24,
    resize_width: int = 160,
) -> float:
    """Estimate visual activity inside a time range using frame deltas.

    Returns a normalized score in [0, 1]. Higher means more movement or
    stronger visual change across the sampled frames.
    """
    if end <= start or not video_path.exists():
        return 0.0

    duration = end - start
    sample_count = max(2, min(max_samples, int(duration * max(sample_fps, 0.5)) + 1))
    sample_times = np.linspace(start, end, num=sample_count, endpoint=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    frames: list[np.ndarray] = []
    try:
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if w > resize_width:
                new_h = max(1, int(h * (resize_width / w)))
                gray = cv2.resize(gray, (resize_width, new_h), interpolation=cv2.INTER_AREA)
            frames.append(gray)
    finally:
        cap.release()

    if len(frames) < 2:
        return 0.0

    deltas: list[float] = []
    prev = frames[0]
    for current in frames[1:]:
        delta = cv2.absdiff(prev, current)
        deltas.append(float(np.mean(delta)) / 255.0)
        prev = current
    if not deltas:
        return 0.0
    return float(sum(deltas) / len(deltas))


def should_preserve_silence_gap(
    gap: float,
    source_path: Path | None,
    prev_phrase: dict,
    next_phrase: dict,
    silence_policy: str,
    motion_threshold: float,
    max_visual_gap: float | None = None,
) -> bool:
    if gap <= 0:
        return True
    if silence_policy == "audio":
        return False
    if silence_policy == "keep":
        return True
    if silence_policy != "visual":
        return False
    if source_path is None or not source_path.exists():
        return False
    if max_visual_gap is not None and gap > max_visual_gap:
        return False

    score = sample_motion_score(
        source_path,
        float(prev_phrase["end"]),
        float(next_phrase["start"]),
    )
    return score >= motion_threshold


def merge_adjacent_ranges(
    ranges: list[dict],
    max_gap: float = 0.35,
    max_duration: float = 12.0,
) -> list[dict]:
    if not ranges:
        return []
    merged: list[dict] = [dict(ranges[0])]
    for current in ranges[1:]:
        prev = merged[-1]
        same_source = current["source"] == prev["source"]
        gap = float(current["start"]) - float(prev["end"])
        merged_duration = float(current["end"]) - float(prev["start"])
        if same_source and gap <= max_gap and merged_duration <= max_duration:
            prev["end"] = current["end"]
            prev["quote"] = f"{prev['quote']} {current['quote']}".strip()[:160]
            prev["reason"] = "Merged adjacent clean phrases into one edit-ready range."
        else:
            merged.append(dict(current))
    return merged


def build_ranges_for_source(
    phrases: list[dict],
    source_name: str,
    source_path: Path | None,
    max_gap: float,
    max_duration: float,
    silence_policy: str,
    motion_threshold: float,
    max_visual_gap: float | None,
) -> list[dict]:
    if not phrases:
        return []

    ranges: list[dict] = []
    current_phrase = dict(phrases[0])
    current_start = float(current_phrase["start"])
    current_end = float(current_phrase["end"])
    current_quote = re.sub(r"\s+", " ", current_phrase["text"]).strip()
    current_reason = "Speech-first starter range from transcript phrases."
    beat_index = 1

    for phrase in phrases[1:]:
        next_start = float(phrase["start"])
        next_end = float(phrase["end"])
        next_text = re.sub(r"\s+", " ", phrase["text"]).strip()
        gap = next_start - current_end
        merged_duration = next_end - current_start

        preserve_gap = should_preserve_silence_gap(
            gap=gap,
            source_path=source_path,
            prev_phrase=current_phrase,
            next_phrase=phrase,
            silence_policy=silence_policy,
            motion_threshold=motion_threshold,
            max_visual_gap=max_visual_gap,
        )

        if gap <= max_gap and merged_duration <= max_duration:
            current_end = next_end
            current_quote = f"{current_quote} {next_text}".strip()[:160]
            current_reason = "Merged adjacent clean phrases into one edit-ready range."
            current_phrase = dict(phrase)
            continue

        if preserve_gap and merged_duration <= max_duration:
            current_end = next_end
            current_quote = f"{current_quote} {next_text}".strip()[:160]
            if silence_policy == "keep":
                current_reason = "Preserved a silent beat between phrases because silence cutting was disabled."
            else:
                current_reason = "Preserved a silent beat because the visual track still showed activity or reaction."
            current_phrase = dict(phrase)
            continue

        ranges.append({
            "source": source_name,
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "beat": "HOOK" if beat_index == 1 else f"BEAT_{beat_index:02d}",
            "quote": current_quote[:160],
            "reason": current_reason,
        })
        beat_index += 1
        current_phrase = dict(phrase)
        current_start = float(current_phrase["start"])
        current_end = float(current_phrase["end"])
        current_quote = next_text
        current_reason = "Speech-first starter range from transcript phrases."

    ranges.append({
        "source": source_name,
        "start": round(current_start, 3),
        "end": round(current_end, 3),
        "beat": "HOOK" if beat_index == 1 else f"BEAT_{beat_index:02d}",
        "quote": current_quote[:160],
        "reason": current_reason,
    })
    return ranges


def prune_phrases(
    phrases: list[dict],
    min_duration: float = 0.25,
    drop_fillers: bool = True,
) -> list[dict]:
    kept: list[dict] = []
    for idx, phrase in enumerate(phrases):
        duration = float(phrase["end"]) - float(phrase["start"])
        if duration < min_duration:
            continue
        text = phrase.get("text", "")
        if drop_fillers and is_filler_phrase(text):
            continue
        next_text = phrases[idx + 1]["text"] if idx + 1 < len(phrases) else ""
        if next_text and is_false_start_phrase(text, next_text, duration):
            continue
        kept.append(phrase)
    return kept


def build_source_map(transcripts_dir: Path, videos_dir: Path | None = None) -> dict[str, str]:
    base_dir = videos_dir.resolve() if videos_dir else transcripts_dir.parent.parent.resolve()
    source_map: dict[str, str] = {}
    for transcript_path in sorted(transcripts_dir.glob("*.json")):
        stem = transcript_path.stem
        candidates = [
            p for p in base_dir.iterdir()
            if p.is_file() and p.stem == stem and p.suffix in VIDEO_EXTS
        ] if base_dir.exists() else []
        if candidates:
            source_map[stem] = str(candidates[0].resolve())
    return source_map


def build_edl(
    transcripts_dir: Path,
    videos_dir: Path | None = None,
    silence_threshold: float = 0.5,
    max_gap: float = 0.35,
    min_phrase_duration: float = 0.25,
    max_duration: float = 12.0,
    silence_policy: str = "audio",
    motion_threshold: float = 0.02,
    max_visual_gap: float | None = 6.0,
) -> dict:
    source_map = build_source_map(transcripts_dir, videos_dir)
    ranges: list[dict] = []

    for transcript_path in sorted(transcripts_dir.glob("*.json")):
        source_name = transcript_path.stem
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        phrases = group_into_phrases(data.get("words", []), silence_threshold)
        phrases = prune_phrases(phrases, min_duration=min_phrase_duration)
        source_path_str = source_map.get(source_name)
        source_path = Path(source_path_str) if source_path_str else None
        ranges.extend(
            build_ranges_for_source(
                phrases=phrases,
                source_name=source_name,
                source_path=source_path,
                max_gap=max_gap,
                max_duration=max_duration,
                silence_policy=silence_policy,
                motion_threshold=motion_threshold,
                max_visual_gap=max_visual_gap,
            )
        )

    total_duration = round(
        sum(float(entry["end"]) - float(entry["start"]) for entry in ranges),
        3,
    )
    return {
        "version": 1,
        "sources": source_map,
        "ranges": ranges,
        "grade": "auto",
        "overlays": [],
        "subtitles": "edit/master.srt",
        "total_duration_s": total_duration,
        "silence_policy": silence_policy,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a starter EDL from transcripts")
    ap.add_argument("--edit-dir", type=Path, required=True, help="Edit directory containing transcripts/")
    ap.add_argument("--videos-dir", type=Path, default=None, help="Raw videos directory for source path resolution")
    ap.add_argument(
        "--silence-threshold",
        type=float,
        default=0.5,
        help="Phrase break threshold in seconds (default 0.5)",
    )
    ap.add_argument(
        "--max-gap",
        type=float,
        default=0.35,
        help="Merge neighboring clean phrases when the gap is <= this many seconds",
    )
    ap.add_argument(
        "--min-phrase-duration",
        type=float,
        default=0.25,
        help="Drop phrases shorter than this many seconds",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=12.0,
        help="Maximum duration of one auto-built EDL range",
    )
    ap.add_argument(
        "--silence-policy",
        choices=("audio", "keep", "visual"),
        default="audio",
        help=(
            "How to handle silence gaps between kept phrases: "
            "'audio' cuts by transcript only, "
            "'keep' preserves silent beats, "
            "'visual' preserves only gaps with visible activity."
        ),
    )
    ap.add_argument(
        "--motion-threshold",
        type=float,
        default=0.02,
        help="Visual activity score needed to preserve a silence gap in 'visual' mode",
    )
    ap.add_argument(
        "--max-visual-gap",
        type=float,
        default=6.0,
        help="Do not auto-preserve silence gaps longer than this in 'visual' mode",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output edl.json path")
    args = ap.parse_args()

    edit_dir = args.edit_dir.resolve()
    transcripts_dir = edit_dir / "transcripts"
    if not transcripts_dir.is_dir():
        sys.exit(f"no transcripts directory at {transcripts_dir}")

    edl = build_edl(
        transcripts_dir=transcripts_dir,
        videos_dir=args.videos_dir,
        silence_threshold=args.silence_threshold,
        max_gap=args.max_gap,
        min_phrase_duration=args.min_phrase_duration,
        max_duration=args.max_duration,
        silence_policy=args.silence_policy,
        motion_threshold=args.motion_threshold,
        max_visual_gap=args.max_visual_gap,
    )
    out_path = (args.output or (edit_dir / "edl.json")).resolve()
    out_path.write_text(json.dumps(edl, indent=2), encoding="utf-8")
    print(f"saved starter EDL -> {out_path}")
    print(f"  ranges: {len(edl['ranges'])}")
    print(f"  mapped sources: {len(edl['sources'])}")
    print(f"  total duration: {edl['total_duration_s']:.2f}s")


if __name__ == "__main__":
    main()
