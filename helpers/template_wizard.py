"""Interactive wizard for starting an edit from a reusable template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from templates import (
        ROOT_DIR,
        build_template_setup,
        load_template,
        merge_template_answers,
        question_index,
        read_json,
        write_json,
    )
except Exception:
    from helpers.templates import (
        ROOT_DIR,
        build_template_setup,
        load_template,
        merge_template_answers,
        question_index,
        read_json,
        write_json,
    )


def load_saved_defaults(defaults_path: Path, template_id: str) -> dict[str, Any]:
    if not defaults_path.exists():
        return {}
    payload = read_json(defaults_path)
    return payload.get("templates", {}).get(template_id, {})


def save_defaults(defaults_path: Path, template_id: str, answers: dict[str, Any]) -> None:
    payload: dict[str, Any]
    if defaults_path.exists():
        payload = read_json(defaults_path)
    else:
        payload = {"templates": {}}
    payload.setdefault("templates", {})[template_id] = answers
    write_json(defaults_path, payload)


def prompt_choice(question: dict[str, Any], default_value: Any) -> Any:
    print("")
    print(f"[{question.get('section', 'Setup')}] {question['prompt']}")
    choices = question.get("choices", [])
    default_label = str(default_value) if default_value is not None else None
    for idx, option in enumerate(choices, start=1):
        label = option.get("label", option["value"])
        desc = option.get("description")
        suffix = " (default)" if option["value"] == default_value else ""
        line = f"  {idx}. {label}{suffix}"
        if desc:
            line += f" - {desc}"
        print(line)
    if question.get("allow_custom"):
        print("  or type your own value")

    raw = input("> ").strip()
    if not raw and default_value is not None:
        return default_value
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]["value"]
    for option in choices:
        if raw.lower() == str(option["value"]).lower() or raw.lower() == str(option.get("label", "")).lower():
            return option["value"]
    if question.get("allow_custom") and raw:
        return raw
    raise ValueError(f"invalid option for {question['id']}")


def prompt_boolean(question: dict[str, Any], default_value: Any) -> bool:
    default_hint = "Y/n" if default_value is True else "y/N"
    print("")
    print(f"[{question.get('section', 'Setup')}] {question['prompt']} ({default_hint})")
    raw = input("> ").strip().lower()
    if not raw:
        return bool(default_value)
    if raw in {"y", "yes", "s", "sim"}:
        return True
    if raw in {"n", "no", "nao", "não"}:
        return False
    raise ValueError(f"invalid yes/no for {question['id']}")


def prompt_text(question: dict[str, Any], default_value: Any) -> str:
    print("")
    suffix = f" [{default_value}]" if default_value not in (None, "") else ""
    print(f"[{question.get('section', 'Setup')}] {question['prompt']}{suffix}")
    raw = input("> ")
    raw = raw.strip()
    if not raw:
        return "" if default_value is None else str(default_value)
    return raw


def collect_answers(
    template: dict[str, Any],
    saved_defaults: dict[str, Any] | None = None,
    prefilled_answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = merge_template_answers(template, saved_defaults=saved_defaults, answers=prefilled_answers)
    idx = question_index(template)
    prefilled_answers = prefilled_answers or {}

    for question in template.get("questions", []):
        qid = question["id"]
        default_value = current.get(qid)
        if qid in prefilled_answers:
            current[qid] = current.get(qid)
            if qid == "subtitle_style" and current.get(qid) != "karaoke":
                current["subtitle_highlight_color"] = "none"
            if qid == "tracking_enabled" and not current.get(qid):
                current["tracking_mode"] = None
            continue
        while True:
            try:
                if qid == "tracking_mode" and not current.get("tracking_enabled"):
                    current[qid] = None
                    break
                qtype = question.get("type", "text")
                if qtype == "choice":
                    current[qid] = prompt_choice(question, default_value)
                elif qtype == "boolean":
                    current[qid] = prompt_boolean(question, default_value)
                else:
                    current[qid] = prompt_text(question, default_value)
                current = merge_template_answers(template, answers=current)
                break
            except ValueError as exc:
                print(f"  {exc}")
                continue

        if qid == "subtitle_style" and current.get(qid) != "karaoke":
            current["subtitle_highlight_color"] = "none"

        if qid == "tracking_enabled" and not current.get(qid):
            current["tracking_mode"] = None

    # Ensure final coercion after conditional fields
    return merge_template_answers(template, answers=current)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a guided edit template wizard")
    ap.add_argument("template_id", help="Template id to start from")
    ap.add_argument("--edit-dir", type=Path, required=True, help="Edit directory where template_setup.json is saved")
    ap.add_argument(
        "--defaults-path",
        type=Path,
        default=None,
        help="Path to defaults JSON (default: <edit-dir>/template_defaults.json)",
    )
    ap.add_argument(
        "--answers-json",
        type=Path,
        default=None,
        help="Optional JSON file with prefilled answers before prompting",
    )
    ap.add_argument(
        "--no-save-defaults",
        action="store_true",
        help="Do not persist answers back into the defaults file",
    )
    args = ap.parse_args()

    try:
        template = load_template(args.template_id)
    except KeyError as exc:
        sys.exit(str(exc))

    edit_dir = args.edit_dir.resolve()
    edit_dir.mkdir(parents=True, exist_ok=True)
    defaults_path = (args.defaults_path or (edit_dir / "template_defaults.json")).resolve()
    saved_defaults = load_saved_defaults(defaults_path, template["id"])
    prefilled_answers = read_json(args.answers_json.resolve()) if args.answers_json else {}

    print(f"Starting template: {template.get('label', template['id'])}")
    if template.get("description"):
        print(template["description"])

    answers = collect_answers(template, saved_defaults=saved_defaults, prefilled_answers=prefilled_answers)
    setup = build_template_setup(template, answers)
    setup_path = edit_dir / "template_setup.json"
    write_json(setup_path, setup)

    if not args.no_save_defaults:
        save_defaults(defaults_path, template["id"], answers)

    print("")
    print(f"saved setup -> {setup_path}")
    print(f"  vertical: {'yes' if setup['render']['vertical'] else 'no'}")
    print(f"  subtitles: {setup['render']['subtitles_path']}")
    print(f"  silence policy: {setup['editing']['silence_policy']}")


if __name__ == "__main__":
    main()
