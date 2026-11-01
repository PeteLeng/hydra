"""Combine, split, and export already-reviewed records. Review helpers live in review.py."""

import argparse
from collections import Counter
from collections.abc import Iterable
import json
import math
from pathlib import Path
import random
import re
from typing import NotRequired, TypedDict

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "showroom"
INTENTS = ("UNK", "greet", "affirm", "deny", "inform", "request_flow",
           "request_stop", "request_resume", "request_quit")
SLOTS = ("name", "flow_ref", "part", "theme", "choice")
SLOT_LABELS = ("PAD", "UNK", "O") + tuple(f"{prefix}-{slot}" for slot in SLOTS for prefix in ("B", "I"))
SPLITS = ("train", "dev", "test")
RATIOS = (0.8, 0.1, 0.1)
SPLIT_TRIALS = 2048
REFINE_TRIALS = 8192


class Slot(TypedDict):
    type: str
    start: int
    end: int


class AliasProposal(TypedDict):
    slot: int
    value: str


class Record(TypedDict):
    text: str
    intent: str
    slots: list[Slot]
    alias_proposals: NotRequired[list[AliasProposal]]
    source: NotRequired[str]
    group_id: NotRequired[str]


def combine(batches: Iterable[Iterable[Record]]) -> list[Record]:
    return [record for batch in batches for record in batch]


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Input units with original character offsets; reuse this at prediction time."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+(?:['’\-]\w+)*|[^\w\s]", text)]


def to_bio(record: Record) -> tuple[list[str], list[str]]:
    units = tokenize(record["text"])
    starts = {start: i for i, (_, start, _) in enumerate(units)}
    ends = {end: i for i, (_, _, end) in enumerate(units)}
    tags = ["O"] * len(units)
    for slot in record["slots"]:
        if slot["start"] not in starts or slot["end"] not in ends:
            raise ValueError("slot must align to complete input units")
        first, last = starts[slot["start"]], ends[slot["end"]]
        if first > last or any(tag != "O" for tag in tags[first:last + 1]):
            raise ValueError("invalid or overlapping slot units")
        tags[first] = f"B-{slot['type']}"
        tags[first + 1:last + 1] = [f"I-{slot['type']}"] * (last - first)
    return [word for word, _, _ in units], tags


def split(records: list[Record], seed: int = 1234,
          ratios: tuple[float, float, float] = RATIOS) -> dict[str, list[Record]]:
    """Choose a seeded whole-group split for coverage, then count/label balance.

    Search bounded shuffled allocations, then refine with whole-group moves/swaps.
    Prefer retaining all observed labels in training, then covering them in dev
    and test. Never break a group to fill a quota. Coverage can be impossible
    when a label occurs in fewer than three independent groups.
    """
    if not records:
        raise ValueError("no reviewed records to split")
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("need three positive split ratios summing to one")
    buckets: dict[tuple[str, str], list[Record]] = {}
    for index, record in enumerate(records):
        key = ("group", record["group_id"]) if "group_id" in record else ("row", str(index))
        buckets.setdefault(key, []).append(record)
    grouped_rows = list(buckets.values())
    group_counts: list[Counter[tuple[str, str]]] = []
    for rows in grouped_rows:
        counts: Counter[tuple[str, str]] = Counter()
        for row in rows:
            counts[("intent", row["intent"])] += 1
            counts.update(("slot", slot["type"]) for slot in row["slots"])
            if row["intent"] in {"inform", "greet", "request_flow"}:
                kinds = {slot["type"] for slot in row["slots"]}
                pattern = "flow_ref" if "flow_ref" in kinds else "+".join(sorted(kinds)) or "no_slots"
                counts[("pattern", row["intent"] + "/" + pattern)] += 1
        group_counts.append(counts)
    totals: Counter[tuple[str, str]] = Counter()
    for counts in group_counts:
        totals.update(counts)

    def score_allocation(counts_by_split: list[Counter[tuple[str, str]]],
                         sizes: list[int]) -> tuple[int, int, int, float]:
        missing = []
        missing_patterns = 0
        balance = 0.0
        for target, counts, size in zip(ratios, counts_by_split, sizes):
            missing.append(sum(counts[label] == 0 for label in totals if label[0] != "pattern"))
            missing_patterns += sum(counts[label] == 0 for label in totals if label[0] == "pattern")
            balance += 10 * (size / len(records) - target) ** 2
            balance += sum((counts[label] / total - target) ** 2
                           for label, total in totals.items()) / len(totals)
        return missing[0], sum(missing[1:]), missing_patterns, balance

    groups = list(range(len(grouped_rows)))
    rng = random.Random(seed)
    best_score: tuple[int, int, int, float] | None = None
    best_groups, best_cuts = groups[:], [0, 0, 0, len(groups)]
    for _ in range(SPLIT_TRIALS if len(groups) >= 3 else 1):
        rng.shuffle(groups)
        cumulative = [0]
        for group in groups:
            cumulative.append(cumulative[-1] + len(grouped_rows[group]))
        cuts = [0]
        for boundary, target in enumerate((ratios[0], ratios[0] + ratios[1])):
            # Keep all splits nonempty when at least three groups are available.
            low = cuts[-1] + int(len(groups) >= 3)
            high = len(groups) - (2 - boundary if len(groups) >= 3 else 0)
            cuts.append(min(range(low, high + 1), key=lambda i: abs(cumulative[i] - len(records) * target)))
        cuts.append(len(groups))
        counts_by_split = []
        sizes = []
        for start, end in zip(cuts, cuts[1:]):
            counts: Counter[tuple[str, str]] = Counter()
            for group in groups[start:end]:
                counts.update(group_counts[group])
            counts_by_split.append(counts)
            sizes.append(cumulative[end] - cumulative[start])
        score = score_allocation(counts_by_split, sizes)
        if best_score is None or score < best_score:
            best_score, best_groups, best_cuts = score, groups[:], cuts

    assignments = {group: part for part, (start, end) in enumerate(zip(best_cuts, best_cuts[1:]))
                   for group in best_groups[start:end]}
    counts_by_split = [Counter() for _ in SPLITS]
    sizes, group_sizes = [0, 0, 0], [0, 0, 0]
    for group, part in assignments.items():
        counts_by_split[part].update(group_counts[group])
        sizes[part] += len(grouped_rows[group])
        group_sizes[part] += 1
    # Improve sparse-label coverage without splitting related examples. Moves
    # adjust row counts; swaps can exchange coverage without growing a holdout.
    for trial in range(REFINE_TRIALS if len(groups) >= 3 else 0):
        first, second = rng.sample(groups, 2)
        left, right = assignments[first], assignments[second]
        if left == right:
            continue
        changes = [(first, left, right)]
        if trial % 2:
            changes.append((second, right, left))
        elif group_sizes[left] == 1:
            continue
        proposed_counts = [counts.copy() for counts in counts_by_split]
        proposed_sizes = sizes[:]
        for group, old, new in changes:
            proposed_counts[old].subtract(group_counts[group])
            proposed_counts[new].update(group_counts[group])
            proposed_sizes[old] -= len(grouped_rows[group])
            proposed_sizes[new] += len(grouped_rows[group])
        score = score_allocation(proposed_counts, proposed_sizes)
        if best_score is not None and score < best_score:
            best_score, counts_by_split, sizes = score, proposed_counts, proposed_sizes
            for group, old, new in changes:
                assignments[group] = new
                group_sizes[old] -= 1
                group_sizes[new] += 1
    result = {name: combine(grouped_rows[group] for group, part in assignments.items() if part == index)
              for index, name in enumerate(SPLITS)}
    for rows in result.values():
        rng.shuffle(rows)
    return result


def write_jsonl(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8", newline="\n")


def export_snips(splits: dict[str, list[Record]], output: Path) -> dict[str, dict[str, object]]:
    """Write files and return row/count metadata. No semantic filtering or deduplication."""
    files: dict[Path, list[str]] = {}
    report: dict[str, dict[str, object]] = {}
    for name in SPLITS:
        rows = splits[name]
        converted = [to_bio(row) for row in rows]
        files[output / name / "seq.in"] = [" ".join(words) for words, _ in converted]
        files[output / name / "seq.out"] = [" ".join(tags) for _, tags in converted]
        files[output / name / "label"] = [row["intent"] for row in rows]
        report[name] = {
            "records": len(rows), "intents": dict(Counter(row["intent"] for row in rows)),
            "slots": dict(Counter(slot["type"] for row in rows for slot in row["slots"])),
            "rows": [{"source": row.get("source"), "group_id": row.get("group_id")} for row in rows],
        }
    files[output / "intent_label.txt"] = list(INTENTS)
    files[output / "slot_label.txt"] = list(SLOT_LABELS)
    # Convert every row before writing anything, so bad alignment cannot half-export a dataset.
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "reviewed.jsonl")
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    try:
        records = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        destinations = [args.output / "manifest.json", args.output / "intent_label.txt", args.output / "slot_label.txt"]
        destinations += [args.output / name / file for name in SPLITS for file in ("seq.in", "seq.out", "label")]
        if args.input.resolve() in {path.resolve() for path in destinations}:
            raise ValueError("input cannot be an export destination")
        report = export_snips(split(records, args.seed), args.output)
        manifest = {"seed": args.seed, "ratios": dict(zip(SPLITS, RATIOS)),
                    "strategy": "coverage-aware-group-search", "search_trials": SPLIT_TRIALS,
                    "refinement_trials": REFINE_TRIALS,
                    "splits": report}
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
        print(json.dumps({name: {key: value for key, value in info.items() if key != "rows"}
                          for name, info in report.items()}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\nExpected reviewed JSONL; use review.py to inspect candidates.\n")


if __name__ == "__main__":
    main()
