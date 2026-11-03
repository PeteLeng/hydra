"""Compile the arbiter's final records into a timestamped training dataset."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import re

from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .schema import AcceptedRecord, INTENTS, RATIOS, Record, SLOT_LABELS, SLOTS, SPLITS
    from . import review, workspace
else:
    from schema import AcceptedRecord, INTENTS, RATIOS, Record, SLOT_LABELS, SLOTS, SPLITS
    import review
    import workspace


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """English showroom input units with original offsets; not BERT WordPieces."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+(?:['’\-]\w+)*|[^\w\s]", text)]


def to_bio(record: Record) -> tuple[list[str], list[str]]:
    """Convert original-text spans to complete input-unit BIO labels."""
    units = tokenize(record["text"])
    starts = {start: i for i, (_, start, _) in enumerate(units)}
    ends = {end: i for i, (_, _, end) in enumerate(units)}
    tags = ["O"] * len(units)
    for slot in record["slots"]:
        if slot["start"] not in starts or slot["end"] not in ends:
            raise ValueError(f"{record['record_id']}: slot must align to complete input units")
        first, last = starts[slot["start"]], ends[slot["end"]]
        if first > last or any(tag != "O" for tag in tags[first:last + 1]):
            raise ValueError("invalid or overlapping slot units")
        tags[first] = f"B-{slot['type']}"
        tags[first + 1:last + 1] = [f"I-{slot['type']}"] * (last - first)
    return [word for word, _, _ in units], tags


def split(records: list[AcceptedRecord], seed: int = 1234) -> dict[str, list[AcceptedRecord]]:
    """One fixed ~80/10/10 split: balance intents while keeping groups intact."""
    groups = [row.get("group_id") for row in records]
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Every accepted record needs a group_id")
    if len(set(groups)) < 10:
        raise ValueError("The 80/10/10 splitter needs at least 10 independent groups")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [held_out for _, held_out in splitter.split(
        X=range(len(records)), y=[row["intent"] for row in records], groups=groups)]
    indices = ([i for fold in folds[:8] for i in fold], folds[8], folds[9])
    return {name: [records[i] for i in part] for name, part in zip(SPLITS, indices)}


def export_dataset(splits: dict[str, list[AcceptedRecord]], output: Path, format: str = "snips") -> dict:
    """Write SNIPS units/tags or unchanged span records, plus shared coverage metadata."""
    if format not in {"snips", "span-jsonl"}:
        raise ValueError(f"Unknown dataset format: {format}")
    files: dict[Path, list[str]] = {}
    report = {}
    for name in SPLITS:
        rows = splits[name]
        if format == "span-jsonl":
            files[output / name / "records.jsonl"] = [json.dumps(row, ensure_ascii=False) for row in rows]
        else:
            converted = [to_bio(row) for row in rows]
            files[output / name / "seq.in"] = [" ".join(words) for words, _ in converted]
            files[output / name / "seq.out"] = [" ".join(tags) for _, tags in converted]
            files[output / name / "label"] = [row["intent"] for row in rows]
        intents = Counter(row["intent"] for row in rows)
        slots = Counter(slot["type"] for row in rows for slot in row["slots"])
        report[name] = {
            "records": len(rows), "intents": dict(intents), "slots": dict(slots),
            "missing_intents": sorted(set(INTENTS) - intents.keys()), "missing_slots": sorted(set(SLOTS) - slots.keys()),
            "groups": len({row["group_id"] for row in rows}),
            "fraction": len(rows) / sum(map(len, splits.values())),
            "rows": [{"record_id": row["record_id"], "group_id": row["group_id"]} for row in rows],
        }
    files[output / "intent_label.txt"] = list(INTENTS)
    files[output / "slot_label.txt"] = list(SLOT_LABELS)
    # Check all alignments before writing any export file.
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    return report


def compile_build(directory: Path, seed: int = 1234, format: str = "snips") -> dict:
    """Load -> validate -> allocate groups -> export -> mark this build complete."""
    metadata = workspace.read_json(directory / "manifest.json")
    if metadata.get("kind") != "build" or metadata.get("status") != "draft":
        raise ValueError("Compilation requires a draft build")
    accepted = directory / "review" / "accepted.jsonl"
    records = workspace.read_jsonl(accepted)
    issues = review.validate_records(records)
    if issues:
        raise ValueError(json.dumps(issues, ensure_ascii=False, default=lambda err:
                                    {"type": err.type.name, "description": err.description}))
    splits = split(records, seed)
    output = directory / metadata["dataset"]
    report = export_dataset(splits, output, format)
    split_files = ("records.jsonl",) if format == "span-jsonl" else ("seq.in", "seq.out", "label")
    files = [output / name / file for name in SPLITS for file in split_files]
    files += [output / "intent_label.txt", output / "slot_label.txt"]
    metadata.update(status="complete", format=format,
                    tokenization="showroom" if format == "snips" else None, splits=report,
                    dataset_files=workspace.hashes(directory, files),
                    review_files=workspace.hashes(directory, [accepted]),
                    labels={"intents": list(INTENTS), "slots": list(SLOT_LABELS)},
                    compilation={"seed": seed, "ratios": dict(zip(SPLITS, RATIOS)),
                                 "strategy": "StratifiedGroupKFold", "stratify": "intent", "n_splits": 10,
                                 "scikit_learn_version": version("scikit-learn")})
    workspace.write_json(directory / "manifest.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, help="timestamped build directory")
    parser.add_argument("--seed", type=int, default=1234, help="dataset split seed (not generation)")
    parser.add_argument("--format", choices=("snips", "span-jsonl"), default="snips",
                        help="span-jsonl preserves text/character spans without tokenization")
    args = parser.parse_args(argv)
    try:
        metadata = compile_build(args.build, args.seed, args.format)
        print(json.dumps({name: {key: value for key, value in info.items() if key != "rows"}
                          for name, info in metadata["splits"].items()}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
