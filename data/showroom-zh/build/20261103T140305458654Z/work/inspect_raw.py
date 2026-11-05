"""Render all selected raw records for semantic review, with literal slot slices."""
from collections import Counter
from pathlib import Path
import sys

BUILD = Path(__file__).resolve().parents[1]
PROJECT = BUILD.parents[3]
sys.path.insert(0, str(PROJECT / "notebooks" / "m6_showroom_data"))
from workspace import read_json, read_jsonl, verify_hashes, write_json
from review import validate_record


def short_id(record):
    parts = record["record_id"].split("-")
    return f"{int(parts[-2])}:{int(parts[-1]):03d}"


def render(record):
    text = record["text"]
    hints = {hint["slot"]: hint["value"] for hint in record.get("slot_hints", [])}
    slices = "; ".join(
        f"{s['type']}={text[s['start']:s['end']]}@{s['start']}:{s['end']}"
        + (f"->{hints[i]}" if i in hints else "")
        for i, s in enumerate(record["slots"])
    )
    err = validate_record(record)
    return f"{short_id(record)} | {record['intent']} | {text} | {slices}" + (
        f" | ERROR {err.description}" if err else "")


def load_raw():
    rows = []
    dataset = BUILD.parents[1]
    for selected in read_json(BUILD / "manifest.json")["inputs"]:
        raw = dataset / selected["path"]
        verify_hashes(raw, selected["files"])
        for file in sorted(selected["files"]):
            if file.endswith(".jsonl"):
                rows.extend(read_jsonl(raw / file))
    return rows


if __name__ == "__main__":
    records = load_raw()
    summary = {}
    for attempt in (1, 2, 3):
        rows = [r for r in records if short_id(r).startswith(f"{attempt}:")]
        (BUILD / "work" / f"raw-{attempt}.txt").write_text(
            "\n".join(map(render, rows)) + "\n", encoding="utf-8")
        summary[str(attempt)] = {"records": len(rows), "intents": dict(Counter(r["intent"] for r in rows)),
                                 "invalid_records": sum(validate_record(r) is not None for r in rows)}
    write_json(BUILD / "work" / "raw-summary.json", summary)
    print(summary)
