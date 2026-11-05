"""Apply this build's human-reviewed decisions and explicit linguistic families.

Raw examples were read in raw-{1,2,3}.txt. The rules below implement grouping
decisions, not semantic acceptance or automatic fuzzy annotation correction.
Run before compilation; completed builds are immutable.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import re
import unicodedata

from inspect_raw import BUILD, load_raw, short_id, render
from workspace import read_json, write_json, write_jsonl
from review import validate_records

MARKERS = {"name": "N", "flow_ref": "F", "part": "P", "theme": "T", "choice": "C"}


def surface_key(text):
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def family_key(row):
    text = row["text"]
    for slot in reversed(row["slots"]):
        text = text[:slot["start"]] + MARKERS[slot["type"]] + text[slot["end"]:]
    text = surface_key(text)
    # Sentence particles, courtesy and reduplication do not create new families.
    text = re.sub(r"[吧呀啊呢哦啦嘞了]+$", "", text)
    text = re.sub(r"^(?:嗯嗯|嗯)[，,]?", "", text) or "嗯"
    text = re.sub(r"^(?:请|麻烦)(?:你)?", "", text)
    for old, new in (("看一看", "看"), ("看看", "看"), ("试一试", "试"), ("试试", "试"),
                     ("停一停", "停"), ("停停", "停"), ("停一下", "停"), ("一下", ""),
                     ("您好", "你好"), ("可以", "能"), ("展示", "演示")):
        text = text.replace(old, new)
    text = re.sub(r"^(?:先|现在)", "", text)
    if row["intent"] in {"affirm", "deny"}:
        text = re.sub(r"的$", "", text)
    if row["intent"] == "request_flow":
        for old, new in (("什么", "哪些"), ("啥", "哪些"), ("有没有", "有"),
                         ("帮我", "给我"), ("带我", "给我"), ("让我", "给我"),
                         ("我打算", "我想"), ("我要", "我想"), ("看", "体验")):
            text = text.replace(old, new)
        text = re.sub(r"^想", "我想", text)
        text = re.sub(r"[吗不]$", "", text)
        text = text.replace("都", "")
    if row["intent"] == "inform":
        text = re.sub(r"^(?:我想选|我选|就选)", "选", text)
        text = re.sub(r"^(?:那就|还是)", "就", text)
        text = re.sub(r"我(?:更|比较)?(?:倾向于|偏向)", "我偏好", text)
    if row["intent"] in {"inform", "request_flow"}:
        for old, new in (("我叫做N", "我叫N"), ("我的名字叫N", "我的名字是N"),
                         ("我名字叫N", "我的名字是N"), ("我名字是N", "我的名字是N"),
                         ("名字叫N", "我的名字是N"), ("体验下", "体验"), ("感受", "体验"),
                         ("试", "体验"), ("配置", "功能"), ("项目", "功能"), ("设计", "功能")):
            text = text.replace(old, new)
        text = re.sub(r"P(?:方面|部分|这块|那块|这边|那边|那部分|这部分|里)", "P", text)
        text = re.sub(r"(?:适合|面向)T(?:适用|能用|一起用|一起用的那种)?", "T", text)
        text = re.sub(r"(?:跟|关于|有关|偏向|偏|有关的|相关的)T", "T", text)
        text = re.sub(r"T(?:有关|相关|方面|方向|类|向|一点|好一点|部分|那种)", "T", text)
        text = re.sub(r"(?:更|比较|挺|很)(?=喜欢|感兴趣|偏向|偏好|想看|想了解)", "", text)
        text = text.replace("的", "")
        text = re.sub(r"[吧呀啊呢哦啦嘞了]+$", "", text)
    return text


class Families:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def root(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.root(self.parent[key])
        return self.parent[key]

    def join(self, left, right):
        a, b = self.root(left), self.root(right)
        self.parent[max(a, b)] = min(a, b)


def main():
    assert read_json(BUILD / "manifest.json")["status"] == "draft"
    decisions = read_json(BUILD / "work" / "decisions.json")
    assert decisions["reviewed"] == ["1:001-500", "2:001-500", "3:001-500"]
    raw = load_raw()
    rows = []
    edits = []
    for original in raw:
        rid = short_id(original)
        if rid in decisions["reject"]:
            continue
        row = deepcopy(original)
        row.pop("slot_hints", None)  # Reviewed hints are not canonical-value targets.
        if change := decisions["correct"].get(rid):
            if "intent" in change:
                row["intent"] = change["intent"]
            if "text" in change:
                row["text"] = change["text"]
                row["slots"] = []
                cursor = 0
                for kind, surface in change["spans"]:
                    start = row["text"].index(surface, cursor)
                    cursor = start + len(surface)
                    row["slots"].append({"type": kind, "start": start, "end": cursor})
            edits.append({"record_id": row["record_id"], "before": original, "after": deepcopy(row),
                          "reason": change["reason"]})
        rows.append(row)
    by_id = {short_id(row): row for row in rows}
    assert len(by_id) == len(rows)
    families = Families(by_id)
    seen_family, seen_text = {}, {}
    duplicates, conflicts = [], []
    keep = []
    for row in rows:
        rid = short_id(row)
        key = family_key(row)
        if key in seen_family:
            families.join(rid, seen_family[key])
        seen_family.setdefault(key, rid)
        # Preserve family links before removing repeated or punctuation-only text.
        text_key = surface_key(row["text"])
        if text_key in seen_text:
            previous = seen_text[text_key]
            families.join(rid, short_id(previous))
            payload = lambda r: (r["intent"], [(s["type"], surface_key(r["text"][s["start"]:s["end"]]))
                                              for s in r["slots"]])
            if payload(row) != payload(previous):
                conflicts.append([previous, row])
            duplicates.append({"removed": row["record_id"], "kept": previous["record_id"],
                               "reason": "exact text" if row["text"] == previous["text"] else "punctuation-only variant"})
        else:
            seen_text[text_key] = row
            keep.append(row)
    write_json(BUILD / "work" / "annotation-conflicts.json", {"conflicts": conflicts})
    assert not conflicts, "Resolve annotation conflicts explicitly before acceptance"
    # Explicit cross-template paraphrase links are reviewed in families.json.
    for family in read_json(BUILD / "work" / "families.json")["links"]:
        for rid in family["members"][1:]:
            families.join(family["members"][0], rid)
    groups = defaultdict(list)
    for row in keep:
        groups[families.root(short_id(row))].append(row)
    report = []
    for number, (_, members) in enumerate(sorted(groups.items()), 1):
        gid = f"zh-g{number:04d}"
        report.append(f"{gid} ({len(members)}) " + " / ".join(dict.fromkeys(family_key(r) for r in members)))
        for row in members:
            row["group_id"] = gid
            report.append(f"  {short_id(row)} {row['intent']} {row['text']}")
    errors = validate_records(keep)
    assert not errors, [(e["id"], e["err"].description) for e in errors]
    write_jsonl(BUILD / "review" / "accepted.jsonl", keep)
    write_jsonl(BUILD / "work" / "corrections.jsonl", edits)
    write_jsonl(BUILD / "work" / "duplicates.jsonl", duplicates)
    (BUILD / "work" / "families.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (BUILD / "work" / "family-index.txt").write_text("\n".join(
        f"{short_id(v[0])} ({len(v)}) {v[0]['intent']} | " + " / ".join(dict.fromkeys(family_key(r) for r in v))
        for _, v in sorted(groups.items())) + "\n", encoding="utf-8")
    (BUILD / "work" / "corrected.txt").write_text("\n".join(render(e["after"]) for e in edits) + "\n", encoding="utf-8")
    summary = {"raw": len(raw), "reviewed": len(raw), "rejected": len(decisions["reject"]),
               "corrected_before_dedup": len(edits), "duplicates_removed": len(duplicates),
               "duplicate_reasons": dict(Counter(d["reason"] for d in duplicates)),
               "accepted": len(keep), "groups": len(groups),
               "accepted_by_attempt": dict(Counter(short_id(r).split(":")[0] for r in keep)),
               "corrected_retained": sum(short_id(r) in decisions["correct"] for r in keep),
               "intents": dict(Counter(r["intent"] for r in keep)),
               "largest_groups": sorted((len(v) for v in groups.values()), reverse=True)[:20]}
    write_json(BUILD / "work" / "review-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
