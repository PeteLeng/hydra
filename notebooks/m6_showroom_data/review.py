"""Local arbiter helpers. Mechanical checks support, but do not perform, semantic review."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import cast
import unicodedata

if __package__:
    from . import build
else:
    import build

ALLOWED = {intent: set() for intent in build.INTENTS}
ALLOWED.update(greet={"name"}, inform=set(build.SLOTS), request_flow={"flow_ref", "part", "theme"})
VALUES = {"flow_ref": {"dlp_headlights", "cabin_cinema", "massage_seats", "privacy_window"},
          "part": {"inside", "outside"}, "theme": {"family", "entertainment", "privacy"}}


def validate_record(raw: object) -> None:
    """Check the annotation contract, not the utterance's meaning or alias interpretation."""
    required = {"text", "intent", "slots"}
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - (required | {"alias_proposals", "source", "group_id"}):
        raise ValueError("expected text, intent, slots; optional alias_proposals/source/group_id")
    text, intent, slots = raw["text"], raw["intent"], raw["slots"]
    if (not isinstance(text, str) or not text.strip() or text != text.strip()
            or any(char in text for char in "\r\n\t\v\f\x85\u2028\u2029")):
        raise ValueError("text must be a nonempty single line without surrounding whitespace")
    if not isinstance(intent, str) or intent not in ALLOWED:
        raise ValueError(f"unknown intent: {intent!r}")
    if not isinstance(slots, list) or (intent == "inform" and len(slots) != 1):
        raise ValueError("slots must be a list; inform requires exactly one slot")
    seen: set[str] = set()
    previous_end = 0
    for slot in slots:
        if not isinstance(slot, dict) or slot.keys() != {"type", "start", "end"}:
            raise ValueError("each slot needs only type, start, end")
        kind, start, end = slot["type"], slot["start"], slot["end"]
        if not isinstance(kind, str) or kind not in ALLOWED[intent] or kind in seen:
            raise ValueError(f"duplicate or disallowed slot {kind!r} for {intent}")
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(text):
            raise ValueError("need ordered, non-overlapping, in-range integer character offsets")
        if text[start:end] != text[start:end].strip() or any(unicodedata.category(text[i]).startswith("P") for i in (start, end - 1)):
            raise ValueError("exclude surrounding whitespace and boundary punctuation from spans")
        previous_end = end
        seen.add(kind)
    proposals = raw.get("alias_proposals", [])
    if not isinstance(proposals, list):
        raise ValueError("alias_proposals must be a list")
    indices: set[int] = set()
    for proposal in proposals:
        if not isinstance(proposal, dict) or proposal.keys() != {"slot", "value"}:
            raise ValueError("each proposal needs slot (index) and value")
        index, value = proposal["slot"], proposal["value"]
        if (type(index) is not int or not 0 <= index < len(slots) or index in indices
                or not isinstance(value, str) or value not in VALUES.get(slots[index]["type"], set())):
            raise ValueError("proposal must reference one finite-vocabulary slot and a known target value")
        indices.add(index)
    for field in ("source", "group_id"):
        if field in raw and (not isinstance(raw[field], str) or not raw[field].strip()):
            raise ValueError(f"{field} must be a nonempty string")
    build.to_bio(cast(build.Record, raw))


def validate_records(records: list[build.Record]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for index, record in enumerate(records):
        try:
            validate_record(record)
            if "source" in record:
                if record["source"] in seen_sources:
                    raise ValueError("duplicate source metadata")
                seen_sources.add(record["source"])
        except ValueError as error:
            source = record.get("source") if isinstance(record, dict) else None
            issues.append({"index": index, "source": source, "error": str(error)})
    return issues


def find_duplicates(records: list[build.Record]) -> list[dict[str, object]]:
    """Flag equal text after case/whitespace folding; caller resolves it, never auto-drop."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[" ".join(record["text"].casefold().split())].append(index)
    return [{"text": text, "indices": indices, "sources": [records[i].get("source") for i in indices],
             "conflicting_labels": len({(records[i]["intent"], tuple(build.to_bio(records[i])[1])) for i in indices}) > 1}
            for text, indices in groups.items() if len(indices) > 1]


def summarize(records: list[build.Record]) -> dict[str, object]:
    intents = Counter(row["intent"] for row in records)
    slots = Counter(slot["type"] for row in records for slot in row["slots"])
    return {"records": len(records), "intents": dict(intents), "slots": dict(slots),
            "missing_intents": sorted(set(build.INTENTS) - intents.keys()),
            "missing_slots": sorted(set(build.SLOTS) - slots.keys())}


def load_candidates(path: Path) -> tuple[list[build.Record], list[dict[str, object]]]:
    batches: list[list[build.Record]] = []
    issues: list[dict[str, object]] = []
    for file in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        rows: list[build.Record] = []
        name = file.resolve().relative_to(build.PROJECT).as_posix() if file.resolve().is_relative_to(build.PROJECT) else file.resolve().as_posix()
        for line_number, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            origin = f"{name}:{line_number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                row.setdefault("source", origin)
                rows.append(row)
            except ValueError as error:
                issues.append({"source": origin, "error": str(error)})
        batches.append(rows)
    return build.combine(batches), issues


def write_reviewed(records: list[build.Record], output: Path = build.DATA / "reviewed.jsonl") -> None:
    """Call only after semantic review. Refuse broken or duplicate records; don't auto-correct."""
    if not records:
        raise ValueError("no accepted records to save")
    issues = validate_records(records)
    if issues:
        raise ValueError(json.dumps(issues, ensure_ascii=False))
    if find_duplicates(records):
        raise ValueError("resolve duplicate utterances before saving reviewed records")
    build.write_jsonl(output, records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=build.DATA / "raw", help="JSONL file or batch directory")
    parser.add_argument("--draft", type=Path, help="save parsed candidates for the arbiter to edit")
    parser.add_argument("--report", type=Path, help="save the mechanical-check report")
    parser.add_argument("--save-reviewed", type=Path, help="save records the arbiter has already semantically reviewed")
    args = parser.parse_args(argv)
    try:
        inputs = {p.resolve() for p in args.input.glob("*.jsonl")} if args.input.is_dir() else {args.input.resolve()}
        outputs = [p.resolve() for p in (args.draft, args.report, args.save_reviewed) if p is not None]
        if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
            raise ValueError("input, draft, report, and accepted-output paths must be distinct")
        records, parse_issues = load_candidates(args.input)
        issues = validate_records(records)
        invalid = {issue["index"] for issue in issues}
        valid_indices = [index for index in range(len(records)) if index not in invalid]
        valid = [records[index] for index in valid_indices]
        duplicates = find_duplicates(valid)
        for group in duplicates:
            group["indices"] = [valid_indices[i] for i in group["indices"]]
        report = {**summarize(valid), "input_records": len(records), "parse_errors": parse_issues,
                  "annotation_errors": issues, "duplicates": duplicates}
        if args.draft is not None:
            build.write_jsonl(args.draft, records)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if parse_issues or issues or report["duplicates"] or not records:
            parser.exit(1, "Review required: inspect errors/duplicates, or supply records.\n")
        if args.save_reviewed is not None:
            write_reviewed(records, args.save_reviewed)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
