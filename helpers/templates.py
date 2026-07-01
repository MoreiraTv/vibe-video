"""Template helpers for guided edit setup.

Provides:
- template discovery / loading
- setup assembly from defaults + answers
- saving reusable template presets from an existing edit
- applying a saved template setup onto an EDL

Usage:
    python helpers/templates.py list
    python helpers/templates.py show vertical-gameplay-facecam
    python helpers/templates.py create-from-edit --edit-dir /path/to/edit --name my-template
    python helpers/templates.py apply-setup --setup /path/to/edit/template_setup.json --edl /path/to/edit/edl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT_DIR / "templates"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_templates(templates_dir: Path | None = None) -> list[dict[str, Any]]:
    base = (templates_dir or TEMPLATES_DIR).resolve()
    templates: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.json")):
        payload = resolve_template_payload(path, base)
        payload["_path"] = str(path)
        templates.append(payload)
    return templates


def load_template(template_id: str, templates_dir: Path | None = None) -> dict[str, Any]:
    for template in discover_templates(templates_dir):
        if template.get("id") == template_id:
            return template
    raise KeyError(f"template not found: {template_id}")


def resolve_template_payload(path: Path, templates_dir: Path | None = None) -> dict[str, Any]:
    base_dir = (templates_dir or path.parent).resolve()
    payload = read_json(path)
    extends_id = payload.get("extends")
    if not extends_id:
        return payload

    parent_path = base_dir / f"{extends_id}.json"
    if not parent_path.exists():
        raise FileNotFoundError(f"parent template not found: {parent_path}")
    parent = resolve_template_payload(parent_path, base_dir)
    resolved = deepcopy(parent)
    resolved.update({k: v for k, v in payload.items() if k not in {"defaults", "questions"}})
    merged_defaults = dict(parent.get("defaults") or {})
    merged_defaults.update(payload.get("defaults") or {})
    resolved["defaults"] = merged_defaults
    resolved["questions"] = payload.get("questions") or parent.get("questions") or []
    return resolved


def question_index(template: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {question["id"]: question for question in template.get("questions", [])}


def option_values(question: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for option in question.get("choices", []):
        values.add(str(option["value"]))
        values.add(str(option.get("label", "")))
    return values


def normalize_choice_value(question: dict[str, Any], raw: Any) -> Any:
    if raw is None:
        return None
    for option in question.get("choices", []):
        if raw == option["value"] or str(raw).strip().lower() == str(option["value"]).strip().lower():
            return option["value"]
        if str(raw).strip().lower() == str(option.get("label", "")).strip().lower():
            return option["value"]
    if question.get("allow_custom"):
        return raw
    raise ValueError(f"invalid value '{raw}' for question '{question['id']}'")


def coerce_answer(question: dict[str, Any], raw: Any) -> Any:
    if raw is None:
        return None
    qtype = question.get("type", "text")
    if qtype == "choice":
        return normalize_choice_value(question, raw)
    if qtype == "boolean":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"y", "yes", "true", "1", "on", "sim", "s"}:
            return True
        if text in {"n", "no", "false", "0", "off", "nao", "não"}:
            return False
        raise ValueError(f"invalid boolean '{raw}' for question '{question['id']}'")
    if qtype == "integer":
        return int(raw)
    return str(raw).strip()


def merge_template_answers(
    template: dict[str, Any],
    saved_defaults: dict[str, Any] | None = None,
    answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(template.get("defaults") or {})
    if saved_defaults:
        resolved.update(saved_defaults)
    idx = question_index(template)
    for key, value in (answers or {}).items():
        if key not in idx:
            resolved[key] = value
            continue
        resolved[key] = coerce_answer(idx[key], value)
    for question in template.get("questions", []):
        key = question["id"]
        if key not in resolved and "default" in question:
            resolved[key] = coerce_answer(question, question["default"])
    return resolved


def build_template_setup(
    template: dict[str, Any],
    answers: dict[str, Any],
) -> dict[str, Any]:
    subtitle_style = answers.get("subtitle_style", "bold")
    aspect_ratio = answers.get("aspect_ratio", "16:9")
    vertical = aspect_ratio == "9:16"
    subtitles_path = "edit/master.ass" if subtitle_style == "karaoke" else "edit/master.srt"
    tracking_enabled = bool(answers.get("tracking_enabled"))

    setup = {
        "template_id": template["id"],
        "template_label": template.get("label", template["id"]),
        "template_description": template.get("description", ""),
        "answers": deepcopy(answers),
        "video": {
            "goal": answers.get("goal"),
            "split_mode": answers.get("split_mode"),
            "aspect_ratio": aspect_ratio,
            "title_mode": answers.get("title_mode"),
            "logo_mode": answers.get("logo_mode"),
        },
        "layout": {
            "style": answers.get("layout_style"),
            "facecam_position": answers.get("facecam_position"),
            "background_style": answers.get("background_style"),
        },
        "subtitle": {
            "position": answers.get("subtitle_position"),
            "grouping": answers.get("subtitle_grouping"),
            "style": subtitle_style,
            "font_preset": answers.get("subtitle_font"),
            "outline": answers.get("subtitle_outline"),
            "case": answers.get("subtitle_case"),
            "highlight_color": answers.get("subtitle_highlight_color"),
            "format": "ass" if subtitle_style == "karaoke" else "srt",
        },
        "editing": {
            "silence_policy": answers.get("silence_policy"),
            "pacing": answers.get("pacing"),
            "must_preserve_reactions": answers.get("must_preserve_reactions"),
        },
        "tracking": {
            "enabled": tracking_enabled,
            "mode": answers.get("tracking_mode") if tracking_enabled else None,
        },
        "grade": {
            "preset": answers.get("grade_preset"),
        },
        "render": {
            "vertical": vertical,
            "subtitles_path": subtitles_path,
        },
        "edl_overrides": {
            "vertical": vertical,
            "silence_policy": answers.get("silence_policy"),
            "subtitles": subtitles_path,
            "grade": answers.get("grade_preset", "auto"),
            "tracking": {
                "enabled": tracking_enabled,
                "mode": answers.get("tracking_mode") if tracking_enabled else None,
                "files": {},
            },
        },
    }
    return setup


def infer_base_template(edit_dir: Path, explicit_template_id: str | None = None) -> str:
    if explicit_template_id:
        return explicit_template_id

    template_setup_path = edit_dir / "template_setup.json"
    if template_setup_path.exists():
        setup = read_json(template_setup_path)
        template_id = setup.get("template_id")
        if template_id:
            return str(template_id)

    edl_path = edit_dir / "edl.json"
    if edl_path.exists():
        edl = read_json(edl_path)
        if edl.get("vertical"):
            tracking = edl.get("tracking") or {}
            if tracking.get("enabled"):
                return "vertical-gameplay-facecam"
            return "vertical-vlog-blur-bg"
    return "horizontal-best-moments"


def infer_defaults_from_edit(edit_dir: Path) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    template_setup_path = edit_dir / "template_setup.json"
    if template_setup_path.exists():
        setup = read_json(template_setup_path)
        inferred.update(setup.get("answers") or {})

    edl_path = edit_dir / "edl.json"
    if edl_path.exists():
        edl = read_json(edl_path)
        if "silence_policy" in edl:
            inferred["silence_policy"] = edl["silence_policy"]
        if "grade" in edl:
            inferred["grade_preset"] = edl["grade"]
        if edl.get("vertical") is True:
            inferred["aspect_ratio"] = "9:16"
        elif "aspect_ratio" not in inferred:
            inferred["aspect_ratio"] = "16:9"
        subtitles_path = str(edl.get("subtitles") or "")
        if subtitles_path.lower().endswith(".ass"):
            inferred["subtitle_style"] = "karaoke"
        tracking = edl.get("tracking") or {}
        if tracking.get("enabled") is not None:
            inferred["tracking_enabled"] = bool(tracking.get("enabled"))
            inferred["tracking_mode"] = tracking.get("mode")
    return inferred


def create_template_from_edit(
    edit_dir: Path,
    name: str,
    base_template_id: str | None = None,
    description: str | None = None,
    templates_dir: Path | None = None,
) -> Path:
    edit_dir = edit_dir.resolve()
    base_id = infer_base_template(edit_dir, explicit_template_id=base_template_id)
    base_template = load_template(base_id, templates_dir)
    new_template = deepcopy(base_template)
    safe_id = name.strip().lower().replace(" ", "-").replace("_", "-")
    inferred_defaults = infer_defaults_from_edit(edit_dir)
    new_template["id"] = safe_id
    new_template["label"] = name.strip()
    new_template["description"] = description or f"Generated from edit at {edit_dir}"
    merged_defaults = dict(new_template.get("defaults") or {})
    merged_defaults.update(inferred_defaults)
    new_template["defaults"] = merged_defaults
    new_template["derived_from"] = {
        "edit_dir": str(edit_dir),
        "base_template_id": base_id,
    }
    out_dir = (templates_dir or TEMPLATES_DIR).resolve()
    out_path = out_dir / f"{safe_id}.json"
    write_json(out_path, new_template)
    return out_path


def apply_setup_to_edl(
    setup: dict[str, Any],
    edl: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(edl)
    overrides = setup.get("edl_overrides") or {}
    for key in ("vertical", "silence_policy", "subtitles", "grade"):
        if key in overrides and overrides[key] is not None:
            updated[key] = overrides[key]

    tracking = overrides.get("tracking") or {}
    if tracking.get("enabled"):
        existing = updated.get("tracking") or {}
        updated["tracking"] = {
            "enabled": True,
            "mode": tracking.get("mode") or existing.get("mode"),
            "files": existing.get("files") or {},
        }
    elif tracking.get("enabled") is False and "tracking" in updated:
        updated["tracking"]["enabled"] = False
    return updated


def print_template_summary(template: dict[str, Any]) -> None:
    print(f"{template['id']} - {template.get('label', template['id'])}")
    if template.get("description"):
        print(f"  {template['description']}")
    defaults = template.get("defaults") or {}
    if defaults:
        print("  defaults:")
        for key, value in defaults.items():
            print(f"    - {key}: {value}")
    questions = template.get("questions") or []
    if questions:
        print("  questions:")
        for question in questions:
            print(f"    - {question['id']} ({question.get('type', 'text')})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Template helpers for vibe-video")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available templates")

    show_ap = sub.add_parser("show", help="Show one template")
    show_ap.add_argument("template_id")

    create_ap = sub.add_parser("create-from-edit", help="Create a reusable template from an existing edit/")
    create_ap.add_argument("--edit-dir", type=Path, required=True)
    create_ap.add_argument("--name", required=True)
    create_ap.add_argument("--base-template", default=None)
    create_ap.add_argument("--description", default=None)

    apply_ap = sub.add_parser("apply-setup", help="Apply a template setup onto an EDL")
    apply_ap.add_argument("--setup", type=Path, required=True)
    apply_ap.add_argument("--edl", type=Path, required=True)
    apply_ap.add_argument("--output", type=Path, default=None)

    args = ap.parse_args()

    if args.command == "list":
        for template in discover_templates():
            print(f"{template['id']}\t{template.get('label', template['id'])}")
        return

    if args.command == "show":
        try:
            template = load_template(args.template_id)
        except KeyError as exc:
            sys.exit(str(exc))
        print_template_summary(template)
        return

    if args.command == "create-from-edit":
        out_path = create_template_from_edit(
            edit_dir=args.edit_dir,
            name=args.name,
            base_template_id=args.base_template,
            description=args.description,
        )
        print(f"saved template -> {out_path}")
        return

    if args.command == "apply-setup":
        setup = read_json(args.setup.resolve())
        edl_path = args.edl.resolve()
        edl = read_json(edl_path)
        updated = apply_setup_to_edl(setup, edl)
        out_path = (args.output or edl_path).resolve()
        write_json(out_path, updated)
        print(f"applied template setup -> {out_path}")
        return


if __name__ == "__main__":
    main()
