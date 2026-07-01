"""Filmstrip + waveform composite PNG for a time range of a video.

The only visual drill-down tool. Given a video and a [start, end] range,
extracts N evenly spaced frames via ffmpeg, composites them into a
horizontal filmstrip, and renders a waveform ribbon below with word
labels overlaid from the transcript (if available) and silence gaps
shaded.

Use this at decision points — ambiguous pauses, retake disambiguation,
cut-point sanity checks. Do NOT call it in a scan loop over every
utterance; it's an on-demand drill-down, not a background index.

Usage:
    python helpers/timeline_view.py <video> <start> <end>
    python helpers/timeline_view.py <video> <start> <end> -o out.png
    python helpers/timeline_view.py <video> <start> <end> --n-frames 12
    python helpers/timeline_view.py <video> <start> <end> --transcript <path>
    python helpers/timeline_view.py --edl <edl.json>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -------- Frame extraction ---------------------------------------------------


def extract_frames(video: Path, start: float, end: float, n: int, dest_dir: Path) -> list[Path]:
    """Extract N frames evenly spaced across [start, end]. Returns paths in order."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if n < 1:
        n = 1
    if n == 1:
        times = [(start + end) / 2.0]
    else:
        step = (end - start) / (n - 1)
        times = [start + i * step for i in range(n)]

    paths: list[Path] = []
    for i, t in enumerate(times):
        out = dest_dir / f"f_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{t:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", "4",
            "-vf", "scale=320:-2",
            str(out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        paths.append(out)
    return paths


# -------- Audio envelope (librosa if available, ffmpeg fallback) ------------


def compute_envelope(video: Path, start: float, end: float, samples: int = 2000) -> np.ndarray:
    """Extract the audio segment and return an RMS envelope of length `samples`.

    Uses ffmpeg to dump mono 16kHz PCM to a temp wav, then computes a
    windowed RMS. Falls back gracefully if the source has no audio.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video),
            "-t", f"{(end - start):.3f}",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(wav),
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
            return np.zeros(samples)

        # Read the WAV manually — avoid librosa as a hard dep
        import wave
        with wave.open(str(wav), "rb") as w:
            frames = w.readframes(w.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return np.zeros(samples)

        # Windowed RMS → envelope of length `samples`
        n = pcm.size
        window = max(1, n // samples)
        usable = (n // window) * window
        reshaped = pcm[:usable].reshape(-1, window)
        env = np.sqrt(np.mean(reshaped ** 2, axis=1))
        if env.size < samples:
            env = np.pad(env, (0, samples - env.size))
        elif env.size > samples:
            env = env[:samples]
        # Normalize to [0, 1]
        if env.max() > 0:
            env = env / env.max()
        return env
    finally:
        wav.unlink(missing_ok=True)


# -------- Transcript word overlays ------------------------------------------


def words_in_range(transcript_path: Path, start: float, end: float) -> list[dict]:
    if not transcript_path.exists():
        return []
    data = json.loads(transcript_path.read_text())
    out: list[dict] = []
    for w in data.get("words", []):
        t = w.get("type", "word")
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= start or ws >= end:
            continue
        out.append(w)
    return out


def find_silences(words: list[dict], start: float, end: float, threshold: float = 0.4) -> list[tuple[float, float]]:
    """Find gaps >= threshold seconds inside [start, end] between kept tokens."""
    gaps: list[tuple[float, float]] = []
    prev_end = start
    for w in words:
        if w.get("type") == "spacing":
            continue
        ws = max(start, w.get("start", start))
        if ws - prev_end >= threshold:
            gaps.append((prev_end, ws))
        prev_end = max(prev_end, w.get("end", ws))
    if end - prev_end >= threshold:
        gaps.append((prev_end, end))
    return gaps


# -------- Font loading -------------------------------------------------------


FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# -------- Composite ----------------------------------------------------------


BG = (18, 18, 22)
FG = (235, 235, 235)
DIM = (110, 110, 120)
ACCENT = (255, 140, 60)
SILENCE = (50, 80, 120, 120)  # muted blue, semi-transparent
WAVE = (140, 180, 255)
SOURCE_SWATCHES = [
    (255, 140, 60),
    (90, 190, 255),
    (120, 220, 140),
    (255, 205, 90),
    (230, 120, 210),
]


def render_timeline(
    video: Path,
    start: float,
    end: float,
    out_path: Path,
    n_frames: int,
    transcript: Path | None,
) -> None:
    # Frame extraction
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"extracting {n_frames} frames from {start:.2f}s to {end:.2f}s")
        frame_paths = extract_frames(video, start, end, n_frames, tmp_dir)

        # Layout metrics
        canvas_width = 1920
        frame_h = 180
        filmstrip_y = 50
        filmstrip_h = frame_h
        wave_y = filmstrip_y + filmstrip_h + 20
        wave_h = 220
        label_y = wave_y + wave_h + 10
        canvas_height = label_y + 60

        # Load + resize frames to uniform height and compute total width
        imgs: list[Image.Image] = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            aspect = img.width / img.height
            new_w = int(frame_h * aspect)
            imgs.append(img.resize((new_w, frame_h), Image.LANCZOS))

        total_frame_w = sum(img.width for img in imgs) + (len(imgs) - 1) * 4
        content_w = max(1400, total_frame_w)
        canvas_width = max(canvas_width, content_w + 100)

        canvas = Image.new("RGB", (canvas_width, canvas_height), BG)
        draw = ImageDraw.Draw(canvas, "RGBA")

        header_font = load_font(22)
        label_font = load_font(14)
        small_font = load_font(12)

        # Header — time range
        draw.text(
            (50, 12),
            f"{video.name}   {start:.2f}s → {end:.2f}s   ({(end - start):.2f}s, {n_frames} frames)",
            fill=FG,
            font=header_font,
        )

        # Filmstrip
        x = 50
        strip_width = canvas_width - 100
        if total_frame_w <= strip_width:
            cursor = 50
            for img in imgs:
                canvas.paste(img, (cursor, filmstrip_y))
                cursor += img.width + 4
            draw_width = cursor - 50
        else:
            scale = strip_width / total_frame_w
            new_h = int(frame_h * scale)
            cursor = 50
            for img in imgs:
                new_w = int(img.width * scale)
                scaled = img.resize((new_w, new_h), Image.LANCZOS)
                canvas.paste(scaled, (cursor, filmstrip_y + (filmstrip_h - new_h) // 2))
                cursor += new_w + max(2, int(4 * scale))
            draw_width = cursor - 50

        strip_x0 = 50
        strip_x1 = 50 + draw_width
        strip_span = strip_x1 - strip_x0

        def time_to_x(t: float) -> int:
            frac = (t - start) / max(1e-6, (end - start))
            return int(strip_x0 + frac * strip_span)

        # Waveform background
        draw.rectangle((strip_x0, wave_y, strip_x1, wave_y + wave_h), fill=(28, 28, 34))

        # Silence shading (under the waveform)
        words = words_in_range(transcript, start, end) if transcript else []
        silences = find_silences(words, start, end, threshold=0.4) if words else []
        for a, b in silences:
            xa = time_to_x(a)
            xb = time_to_x(b)
            draw.rectangle((xa, wave_y, xb, wave_y + wave_h), fill=SILENCE)

        # Waveform envelope
        env = compute_envelope(video, start, end, samples=max(strip_span, 200))
        mid_y = wave_y + wave_h // 2
        max_amp = wave_h // 2 - 8
        points_top: list[tuple[int, int]] = []
        points_bot: list[tuple[int, int]] = []
        for i, v in enumerate(env):
            xi = strip_x0 + int(i * strip_span / max(1, len(env) - 1))
            a = int(v * max_amp)
            points_top.append((xi, mid_y - a))
            points_bot.append((xi, mid_y + a))
        if points_top:
            draw.line(points_top, fill=WAVE, width=1, joint="curve")
            draw.line(points_bot, fill=WAVE, width=1, joint="curve")
            # Fill between
            poly = points_top + list(reversed(points_bot))
            draw.polygon(poly, fill=(*WAVE, 60))

        # Word labels above the waveform (only words lasting ≥ 120ms to avoid clutter)
        last_label_x = -9999
        for w in words:
            if w.get("type") != "word":
                continue
            ws = w.get("start")
            we = w.get("end")
            text = (w.get("text") or "").strip()
            if not text or ws is None or we is None:
                continue
            if (we - ws) < 0.05:
                continue
            cx = (time_to_x(ws) + time_to_x(we)) // 2
            if cx - last_label_x < 28:
                continue
            # Tiny tick on the waveform
            draw.line((cx, wave_y - 4, cx, wave_y), fill=DIM, width=1)
            # Text above the waveform
            draw.text((cx + 2, wave_y - 18), text, fill=FG, font=small_font)
            last_label_x = cx

        # Time ruler below waveform
        ruler_y = wave_y + wave_h + 2
        n_ticks = 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            t = start + frac * (end - start)
            xi = strip_x0 + int(frac * strip_span)
            draw.line((xi, ruler_y, xi, ruler_y + 6), fill=DIM, width=1)
            draw.text((xi - 20, ruler_y + 8), f"{t:.2f}s", fill=DIM, font=label_font)

        # Silences legend if any
        if silences:
            txt = f"shaded bands = silences ≥ 400ms ({len(silences)} gap(s))"
            draw.text((strip_x0, label_y + 30), txt, fill=DIM, font=label_font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, "PNG", optimize=True)
        print(f"saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")


def segment_offsets(edl: dict) -> list[dict]:
    offsets: list[dict] = []
    cursor = 0.0
    for segment in edl.get("ranges", []):
        duration = float(segment["end"]) - float(segment["start"])
        offsets.append(
            {
                **segment,
                "output_start": cursor,
                "output_end": cursor + duration,
                "duration": duration,
            }
        )
        cursor += duration
    return offsets


def _fit_text(text: str, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def render_edl_overview(edl: dict, out_path: Path) -> None:
    segments = segment_offsets(edl)
    total_duration = max(float(edl.get("total_duration_s") or 0.0), sum(seg["duration"] for seg in segments))
    if total_duration <= 0:
        total_duration = 1.0

    canvas_width = 2000
    canvas_height = 760
    left = 80
    right = canvas_width - 80
    timeline_y = 170
    bar_h = 84
    overlay_y = 360
    overlay_h = 36
    footer_y = 520

    canvas = Image.new("RGB", (canvas_width, canvas_height), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")
    header_font = load_font(28)
    label_font = load_font(16)
    small_font = load_font(13)

    draw.text((left, 35), "EDL overview", fill=FG, font=header_font)
    draw.text(
        (left, 75),
        f"segments: {len(segments)}   overlays: {len(edl.get('overlays') or [])}   total: {total_duration:.2f}s",
        fill=DIM,
        font=label_font,
    )

    draw.rectangle((left, timeline_y, right, timeline_y + bar_h), fill=(28, 28, 34), outline=(60, 60, 70))

    source_names = sorted({segment["source"] for segment in segments})
    swatches = {name: SOURCE_SWATCHES[idx % len(SOURCE_SWATCHES)] for idx, name in enumerate(source_names)}

    def time_to_x(value: float) -> int:
        frac = max(0.0, min(1.0, value / total_duration))
        return int(left + frac * (right - left))

    for idx, segment in enumerate(segments, start=1):
        x0 = time_to_x(segment["output_start"])
        x1 = max(x0 + 2, time_to_x(segment["output_end"]))
        fill = swatches[segment["source"]]
        draw.rounded_rectangle((x0, timeline_y, x1, timeline_y + bar_h), radius=8, fill=fill)
        draw.text((x0 + 8, timeline_y + 10), f"{idx:02d} {segment['source']}", fill=BG, font=label_font)
        label = segment.get("beat") or segment.get("quote") or ""
        if label:
            draw.text((x0 + 8, timeline_y + 38), _fit_text(label), fill=BG, font=small_font)
        if idx < len(segments):
            cut_x = x1
            draw.line((cut_x, timeline_y - 10, cut_x, timeline_y + bar_h + 10), fill=FG, width=1)

    draw.text((left, overlay_y - 34), "overlays", fill=FG, font=label_font)
    draw.rectangle((left, overlay_y, right, overlay_y + overlay_h), fill=(24, 24, 28), outline=(60, 60, 70))
    for overlay in edl.get("overlays") or []:
        start = float(overlay["start_in_output"])
        end = start + float(overlay["duration"])
        x0 = time_to_x(start)
        x1 = max(x0 + 2, time_to_x(end))
        draw.rounded_rectangle((x0, overlay_y, x1, overlay_y + overlay_h), radius=6, fill=ACCENT)
        draw.text((x0 + 6, overlay_y + 9), _fit_text(Path(overlay["file"]).stem, 20), fill=BG, font=small_font)

    draw.text((left, footer_y - 30), "source legend", fill=FG, font=label_font)
    legend_x = left
    legend_y = footer_y
    for source in source_names:
        fill = swatches[source]
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 22, legend_y + 22), radius=5, fill=fill)
        draw.text((legend_x + 30, legend_y + 2), source, fill=FG, font=label_font)
        legend_x += 180

    tick_y = timeline_y + bar_h + 18
    for tick in range(7):
        frac = tick / 6
        t = total_duration * frac
        x = int(left + frac * (right - left))
        draw.line((x, tick_y, x, tick_y + 8), fill=DIM, width=1)
        draw.text((x - 18, tick_y + 12), f"{t:.1f}s", fill=DIM, font=small_font)

    if edl.get("subtitles"):
        draw.text((left, overlay_y + 70), f"subtitles: {edl['subtitles']}", fill=DIM, font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)
    print(f"saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filmstrip + waveform composite for a video range")
    ap.add_argument("video", type=Path, nargs="?", help="Source video")
    ap.add_argument("start", type=float, nargs="?", help="Start time in seconds")
    ap.add_argument("end", type=float, nargs="?", help="End time in seconds")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    ap.add_argument("--n-frames", type=int, default=10, help="Number of frames in the filmstrip (default 10)")
    ap.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Path to transcript.json for word labels + silence shading. "
             "If omitted, will auto-resolve to <video_parent>/edit/transcripts/<video_stem>.json",
    )
    ap.add_argument(
        "--edl",
        type=Path,
        default=None,
        help="Render a full-project timeline from an EDL",
    )
    args = ap.parse_args()

    if args.edl:
        edl_path = args.edl.resolve()
        if not edl_path.exists():
            sys.exit(f"edl not found: {edl_path}")
        edl = json.loads(edl_path.read_text(encoding="utf-8"))
        out_path = args.output
        if out_path is None:
            out_dir = edl_path.parent / "verify"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "edl_overview.png"
        render_edl_overview(edl, out_path.resolve())
        return

    if not args.video or args.start is None or args.end is None:
        ap.error("video, start, and end are required")

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    if args.end <= args.start:
        sys.exit("end must be > start")

    # Auto-resolve transcript if not given
    transcript = args.transcript
    if transcript is None:
        auto = video.parent / "edit" / "transcripts" / f"{video.stem}.json"
        if auto.exists():
            transcript = auto

    out_path = args.output
    if out_path is None:
        out_dir = video.parent / "edit" / "verify"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video.stem}_{args.start:.2f}-{args.end:.2f}.png"

    render_timeline(
        video=video,
        start=args.start,
        end=args.end,
        out_path=out_path,
        n_frames=args.n_frames,
        transcript=transcript,
    )


if __name__ == "__main__":
    main()
