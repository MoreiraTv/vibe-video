"""Suggest B-roll or overlay slots from transcripts or an EDL.

Usage:
    python helpers/broll_slots.py --edit-dir /path/to/edit
    python helpers/broll_slots.py --edit-dir /path/to/edit --edl /path/to/edit/edl.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from pack_transcripts import group_into_phrases
except Exception:
    from helpers.pack_transcripts import group_into_phrases


DATA_WORDS = {"percent", "revenue", "growth", "chart", "graph", "faster", "slower", "drop", "increase"}
SCREEN_WORDS = {"screen", "dashboard", "page", "app", "button", "settings", "demo", "click", "workflow"}
TEXT_WORDS = {"important", "remember", "key", "biggest", "problem", "solution", "before", "after", "instead"}


def tokenize(text: str) -> list[str]:
    return [part for part in re.findall(r"[a-z0-9%']+", text.lower()) if part]


def choose_slot_type(tokens: list[str]) -> str:
    if any(token.isdigit() or "%" in token for token in tokens) or any(token in DATA_WORDS for token in tokens):
        return "data-callout"
    if any(token in SCREEN_WORDS for token in tokens):
        return "screen-demo"
    if any(token in TEXT_WORDS for token in tokens):
        return "text-emphasis"
    return "visual-cutaway"


def phrase_score(text: str, duration: float) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    score = 0.0
    score += min(duration, 4.0) * 0.4
    score += sum(1.5 for token in tokens if token.isdigit() or "%" in token)
    score += sum(1.0 for token in tokens if token in DATA_WORDS | SCREEN_WORDS | TEXT_WORDS)
    if "?" in text or "!" in text:
        score += 0.5
    return round(score, 2)


def anchor_word(text: str) -> str:
    tokens = tokenize(text)
    if not tokens:
        return ""
    ranked = sorted(tokens, key=lambda token: (token.isdigit() or "%" in token, len(token)), reverse=True)
    return ranked[0]


def phrase_to_slot(
    source: str,
    phrase: dict,
    index: int,
    output_offset: float | None = None,
) -> dict:
    start = float(phrase["start"])
    end = float(phrase["end"])
    duration = end - start
    text = re.sub(r"\s+", " ", phrase["text"]).strip()
    tokens = tokenize(text)
    slot = {
        "id": f"{source}_slot_{index:02d}",
        "source": source,
        "start": round(max(0.0, start - 0.2), 3),
        "end": round(min(end + 0.35, start + 4.0), 3),
        "phrase_start": round(start, 3),
        "phrase_end": round(end, 3),
        "score": phrase_score(text, duration),
        "type": choose_slot_type(tokens),
        "anchor_word": anchor_word(text),
        "reason": f"Useful visual support for: {text[:120]}",
        "text": text[:180],
    }
    if output_offset is not None:
        slot["start_in_output"] = round(output_offset + max(0.0, start - float(phrase["segment_start"])), 3)
        slot["end_in_output"] = round(output_offset + min(end + 0.35, float(phrase["segment_end"])) - float(phrase["segment_start"]), 3)
    return slot


def suggest_slots_for_phrases(
    source: str,
    phrases: list[dict],
    min_score: float = 2.0,
    output_offset: float | None = None,
) -> list[dict]:
    suggestions: list[dict] = []
    for idx, phrase in enumerate(phrases, start=1):
        score = phrase_score(phrase["text"], float(phrase["end"]) - float(phrase["start"]))
        if score < min_score:
            continue
        slot = phrase_to_slot(source, phrase, idx, output_offset=output_offset)
        suggestions.append(slot)
    return suggestions


def load_transcript_phrases(transcript_path: Path, silence_threshold: float) -> list[dict]:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    return group_into_phrases(data.get("words", []), silence_threshold)


def build_slots_from_edit_dir(
    edit_dir: Path,
    silence_threshold: float = 0.5,
    min_score: float = 2.0,
) -> list[dict]:
    transcripts_dir = edit_dir / "transcripts"
    if not transcripts_dir.is_dir():
        raise FileNotFoundError(f"no transcripts directory at {transcripts_dir}")
    suggestions: list[dict] = []
    for transcript_path in sorted(transcripts_dir.glob("*.json")):
        phrases = load_transcript_phrases(transcript_path, silence_threshold)
        suggestions.extend(
            suggest_slots_for_phrases(
                source=transcript_path.stem,
                phrases=phrases,
                min_score=min_score,
            )
        )
    return sorted(suggestions, key=lambda item: item["score"], reverse=True)


def _phrases_for_range(phrases: list[dict], start: float, end: float) -> list[dict]:
    out: list[dict] = []
    for phrase in phrases:
        if float(phrase["end"]) <= start or float(phrase["start"]) >= end:
            continue
        clipped = dict(phrase)
        clipped["start"] = max(start, float(phrase["start"]))
        clipped["end"] = min(end, float(phrase["end"]))
        clipped["segment_start"] = start
        clipped["segment_end"] = end
        out.append(clipped)
    return out


def build_slots_from_edl(
    edl: dict,
    edit_dir: Path,
    silence_threshold: float = 0.5,
    min_score: float = 2.0,
) -> list[dict]:
    transcripts_dir = edit_dir / "transcripts"
    transcript_cache: dict[str, list[dict]] = {}
    suggestions: list[dict] = []
    offset = 0.0
    for segment in edl.get("ranges", []):
        source = segment["source"]
        if source not in transcript_cache:
            transcript_cache[source] = load_transcript_phrases(transcripts_dir / f"{source}.json", silence_threshold)
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        phrases = _phrases_for_range(transcript_cache[source], seg_start, seg_end)
        suggestions.extend(
            suggest_slots_for_phrases(
                source=source,
                phrases=phrases,
                min_score=min_score,
                output_offset=offset,
            )
        )
        offset += seg_end - seg_start
    return sorted(suggestions, key=lambda item: item["score"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest B-roll or overlay slots from transcripts")
    ap.add_argument("--edit-dir", type=Path, required=True, help="Edit directory containing transcripts/")
    ap.add_argument("--edl", type=Path, default=None, help="Optional edl.json for output-timeline mapping")
    ap.add_argument(
        "--silence-threshold",
        type=float,
        default=0.5,
        help="Phrase break threshold in seconds (default 0.5)",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=2.0,
        help="Minimum phrase score to keep as a suggestion (default 2.0)",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path")
    args = ap.parse_args()

    edit_dir = args.edit_dir.resolve()
    if not edit_dir.is_dir():
        sys.exit(f"edit dir not found: {edit_dir}")

    if args.edl:
        edl_path = args.edl.resolve()
        if not edl_path.exists():
            sys.exit(f"edl not found: {edl_path}")
        edl = json.loads(edl_path.read_text(encoding="utf-8"))
        slots = build_slots_from_edl(
            edl=edl,
            edit_dir=edit_dir,
            silence_threshold=args.silence_threshold,
            min_score=args.min_score,
        )
    else:
        slots = build_slots_from_edit_dir(
            edit_dir=edit_dir,
            silence_threshold=args.silence_threshold,
            min_score=args.min_score,
        )

    payload = {"version": 1, "slots": slots}
    out_path = (args.output or (edit_dir / "broll_slots.json")).resolve()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved B-roll suggestions -> {out_path}")
    print(f"  slots: {len(slots)}")


if __name__ == "__main__":
    main()
