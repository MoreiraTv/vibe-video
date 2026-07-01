"""Verify a rendered output against an EDL.

Checks duration drift, computes cut-boundary verification windows, and can
optionally render PNG drill-downs for those windows using timeline_view.

Usage:
    python helpers/verify_render.py <edl.json> <final.mp4>
    python helpers/verify_render.py <edl.json> <preview.mp4> --emit-json
    python helpers/verify_render.py <edl.json> <final.mp4> --no-images
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from timeline_view import render_timeline
except Exception:
    from helpers.timeline_view import render_timeline


def expected_duration(edl: dict) -> float:
    return round(sum(float(r["end"]) - float(r["start"]) for r in edl.get("ranges", [])), 3)


def cut_boundaries(edl: dict) -> list[float]:
    boundaries: list[float] = []
    offset = 0.0
    ranges = edl.get("ranges", [])
    for idx, segment in enumerate(ranges):
        duration = float(segment["end"]) - float(segment["start"])
        offset += duration
        if idx < len(ranges) - 1:
            boundaries.append(round(offset, 3))
    return boundaries


def _window(label: str, start: float, end: float, kind: str) -> dict:
    return {
        "label": label,
        "kind": kind,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
    }


def build_verification_windows(
    total_duration: float,
    boundaries: list[float],
    radius: float = 1.5,
    midpoint_count: int = 3,
) -> list[dict]:
    windows: list[dict] = []
    for idx, boundary in enumerate(boundaries, start=1):
        windows.append(
            _window(
                f"cut_{idx:02d}",
                max(0.0, boundary - radius),
                min(total_duration, boundary + radius),
                "cut",
            )
        )

    if total_duration > 0:
        windows.append(_window("opening", 0.0, min(total_duration, 2.0), "sample"))
        windows.append(
            _window("ending", max(0.0, total_duration - 2.0), total_duration, "sample")
        )
        for idx in range(midpoint_count):
            center = total_duration * (idx + 1) / (midpoint_count + 1)
            windows.append(
                _window(
                    f"mid_{idx + 1:02d}",
                    max(0.0, center - 1.0),
                    min(total_duration, center + 1.0),
                    "sample",
                )
            )

    deduped: list[dict] = []
    seen: set[tuple[str, float, float]] = set()
    for item in windows:
        key = (item["kind"], item["start"], item["end"])
        if item["duration"] <= 0 or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def probe_duration(video_path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def duration_check(expected: float, actual: float, tolerance: float = 0.15) -> dict:
    drift = round(actual - expected, 3)
    return {
        "name": "duration_match",
        "expected": expected,
        "actual": round(actual, 3),
        "drift": drift,
        "tolerance": tolerance,
        "ok": abs(drift) <= tolerance,
    }


def verify_render(
    edl: dict,
    render_path: Path,
    images_dir: Path | None,
    radius: float = 1.5,
    midpoint_count: int = 3,
    tolerance: float = 0.15,
) -> dict:
    expected = expected_duration(edl)
    actual = probe_duration(render_path)
    boundaries = cut_boundaries(edl)
    windows = build_verification_windows(
        total_duration=actual,
        boundaries=boundaries,
        radius=radius,
        midpoint_count=midpoint_count,
    )
    report = {
        "render": str(render_path),
        "expected_duration": expected,
        "actual_duration": round(actual, 3),
        "checks": [duration_check(expected, actual, tolerance=tolerance)],
        "windows": windows,
    }

    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        generated: list[str] = []
        for window in windows:
            out_path = images_dir / f"{window['label']}.png"
            render_timeline(
                video=render_path,
                start=float(window["start"]),
                end=float(window["end"]),
                out_path=out_path,
                n_frames=8 if window["kind"] == "cut" else 6,
                transcript=None,
            )
            generated.append(str(out_path))
        report["generated_images"] = generated
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a rendered output against an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("render", type=Path, help="Rendered output to verify")
    ap.add_argument(
        "--radius",
        type=float,
        default=1.5,
        help="Seconds of context around cut boundaries (default 1.5)",
    )
    ap.add_argument(
        "--midpoints",
        type=int,
        default=3,
        help="How many non-cut middle samples to include (default 3)",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Allowed duration drift in seconds (default 0.15)",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Skip generating timeline PNGs for verification windows",
    )
    ap.add_argument(
        "--emit-json",
        action="store_true",
        help="Write a JSON report alongside printing the summary",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    render_path = args.render.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")
    if not render_path.exists():
        sys.exit(f"render not found: {render_path}")

    edl = json.loads(edl_path.read_text(encoding="utf-8"))
    verify_dir = edl_path.parent / "verify"
    images_dir = None if args.no_images else (verify_dir / render_path.stem)
    report = verify_render(
        edl=edl,
        render_path=render_path,
        images_dir=images_dir,
        radius=args.radius,
        midpoint_count=args.midpoints,
        tolerance=args.tolerance,
    )

    ok = all(check["ok"] for check in report["checks"])
    for check in report["checks"]:
        verdict = "OK" if check["ok"] else "FAIL"
        print(
            f"{verdict} {check['name']}: expected {check['expected']:.3f}s, "
            f"actual {check['actual']:.3f}s, drift {check['drift']:+.3f}s"
        )
    print(f"verification windows: {len(report['windows'])}")

    if args.emit_json:
        out_path = verify_dir / f"{render_path.stem}_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {out_path}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
