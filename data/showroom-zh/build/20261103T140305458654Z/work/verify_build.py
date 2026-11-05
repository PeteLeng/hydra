"""Independent provenance, split isolation, span and vocabulary audit for this build."""
from collections import Counter
import json
import re
import sys

from inspect_raw import BUILD, load_raw, short_id
from arbitrate import family_key, surface_key
from workspace import read_json, read_jsonl, verify_hashes, write_json
from review import validate_records
from schema import INTENTS, SLOTS
from build import split

VOCABULARY = {
    "flow_ref": {
        "dlp_headlights": "DLP大灯 DLP车灯 大灯 车头大灯 前大灯 DLP前大灯 DLP灯".split(),
        "cabin_cinema": "座舱影院 车载影院 移动影院 车载电影院 车内影院 座舱电影院 车里的电影院 移动电影院".split(),
        "massage_seats": "按摩座椅 座椅按摩 带按摩的座椅 按摩椅".split(),
        "privacy_window": "隐私车窗 隐私玻璃 私密车窗 防窥车窗 隐私窗".split(),
    },
    "part": {"inside": "车内 内饰 座舱里 车里 座舱内 车厢里".split(),
             "outside": "车外 外饰 车身外部 车外面".split()},
    "theme": {"family": "家庭 全家 家用 全家人 一家人".split(),
              "entertainment": "娱乐 休闲娱乐 影音娱乐".split(),
              "privacy": "隐私 私密 私密性 隐私保护".split()},
}
LOOKUP = {kind: {surface: value for value, surfaces in values.items() for surface in surfaces}
          for kind, values in VOCABULARY.items()}


def coverage(rows):
    intents = Counter(r["intent"] for r in rows)
    slots = Counter(s["type"] for r in rows for s in r["slots"])
    values = {kind: Counter() for kind in LOOKUP}
    choices = Counter()
    forms = Counter()
    for row in rows:
        for slot in row["slots"]:
            surface = row["text"][slot["start"]:slot["end"]]
            kind = slot["type"]
            if kind in LOOKUP:
                assert surface in LOOKUP[kind], (row["record_id"], kind, surface)
                values[kind][LOOKUP[kind][surface]] += 1
            elif kind == "choice":
                numeric = re.search(r"\d+|[一二三四五六七八九十]+", surface).group()
                value = int(numeric) if numeric.isascii() else "一二三四五六七八九十".index(numeric) + 1
                assert 1 <= value <= 10
                choices[value] += 1
                forms["arabic" if numeric.isascii() else "chinese"] += 1
    return {"records": len(rows), "groups": len({r["group_id"] for r in rows}),
            "intents": dict(intents), "slots": dict(slots), "canonical_values": values,
            "missing_intents": sorted(set(INTENTS) - intents.keys()),
            "missing_slots": sorted(set(SLOTS) - slots.keys()),
            "missing_values": {kind: sorted(set(VOCABULARY[kind]) - counts.keys()) for kind, counts in values.items()},
            "choices": dict(sorted(choices.items())), "choice_forms": dict(forms)}


def main():
    raw = load_raw()  # Also verifies every selected source hash.
    source = {r["record_id"]: r for r in raw}
    assert len(source) == 1500
    for attempt in (1, 2, 3):
        assert {short_id(r) for r in raw if short_id(r).startswith(f"{attempt}:")} == {
            f"{attempt}:{i:03d}" for i in range(1, 501)}
    rows = read_jsonl(BUILD / "review" / "accepted.jsonl")
    assert not validate_records(rows)
    assert len({surface_key(r["text"]) for r in rows}) == len(rows)
    decisions = read_json(BUILD / "work" / "decisions.json")
    duplicates = read_jsonl(BUILD / "work" / "duplicates.jsonl")
    accepted = {r["record_id"]: r for r in rows}
    discarded_ids = {r["record_id"] for r in raw if short_id(r) in decisions["reject"]}
    duplicate_ids = {r["removed"] for r in duplicates}
    assert not (accepted.keys() & discarded_ids or accepted.keys() & duplicate_ids or discarded_ids & duplicate_ids)
    assert accepted.keys() | discarded_ids | duplicate_ids == source.keys()
    for row in rows:
        original = source[row["record_id"]]
        edit = decisions["correct"].get(short_id(row), {})
        assert row["text"] == edit.get("text", original["text"])
        assert row["intent"] == edit.get("intent", original["intent"])
        if "spans" in edit:
            assert [[s["type"], row["text"][s["start"]:s["end"]]] for s in row["slots"]] == edit["spans"]
        else:
            assert row["slots"] == original["slots"]
        assert set(row) == {"record_id", "text", "intent", "slots", "group_id"}
    group_for_source = {short_id(r): r["group_id"] for r in rows}
    for duplicate in duplicates:
        group_for_source[short_id(source[duplicate["removed"]])] = accepted[duplicate["kept"]]["group_id"]
    for family in read_json(BUILD / "work" / "families.json")["links"]:
        assert len({group_for_source[rid] for rid in family["members"]}) == 1, family
    keys = {}
    for row in rows:
        key = family_key(row)
        assert keys.setdefault(key, row["group_id"]) == row["group_id"]
    manifest = read_json(BUILD / "manifest.json")
    preflight = "--preflight" in sys.argv
    if preflight:
        assert manifest["status"] == "draft"
        splits = split(rows, 1234)
    else:
        assert manifest["status"] == "complete"
        assert manifest["format"] == "span-jsonl" and manifest["tokenization"] is None
        assert manifest["compilation"]["seed"] == 1234
        verify_hashes(BUILD, manifest["dataset_files"])
        verify_hashes(BUILD, manifest["review_files"])
        splits = {name: read_jsonl(BUILD / "dataset" / name / "records.jsonl") for name in ("train", "dev", "test")}
        assert splits == split(rows, 1234), "Export differs from the fixed seeded group partition"
    observed_ids, observed_groups = set(), set()
    for name, part in splits.items():
        ids = {r["record_id"] for r in part}
        groups = {r["group_id"] for r in part}
        assert not (ids & observed_ids or groups & observed_groups)
        assert all(row == accepted[row["record_id"]] for row in part)
        assert not validate_records(part)
        observed_ids.update(ids)
        observed_groups.update(groups)
    assert observed_ids == accepted.keys()
    report = {"checks": "passed", "source_records": len(raw), "accepted": len(rows),
              "unique_groups": len(observed_groups), "record_overlap": 0, "group_overlap": 0,
              "exact_or_punctuation_duplicate_text": 0, "splits": {name: coverage(part) for name, part in splits.items()},
              "overall": coverage(rows)}
    for name, info in report["splits"].items():
        info["fraction"] = info["records"] / len(rows)
        info["sources"] = dict(Counter(short_id(r).split(":")[0] for r in splits[name]))
    write_json(BUILD / "work" / ("preflight.json" if preflight else "verification.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "overall"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
