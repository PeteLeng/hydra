"""Unit and local workflow tests for the data pipeline; no model or agent calls."""

from copy import deepcopy
from itertools import count
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import build
import review
import schema
import workspace

IDS = count(1)


def example(text: str, intent: str, kind: str | None = None, surface: str = "") -> schema.Record:
    slots = [] if kind is None else [{"type": kind, "start": text.index(surface), "end": text.index(surface) + len(surface)}]
    return {"record_id": f"fixture-{next(IDS):06d}", "text": text, "intent": intent, "slots": slots}


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        root = workspace.PROJECT / "notebooks/.artifacts/m6_showroom_data_tests"
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.dataset = Path(temporary.name) / "showroom-en"

    def raw_attempt(self, records: list[dict], job: int = 1) -> Path:
        raw = workspace.create_raw(self.dataset, "fixture", count=500, stamp="attempt", job=job)
        for i, row in enumerate(records, 1):
            row["record_id"] = f"{raw.name}-{i:06d}"
        workspace.write_jsonl(raw / "part-01.jsonl", records)
        return raw

    def test_generation_brief_examples(self) -> None:
        for name in ("gen.md", "gen-zh.md"):
            with self.subTest(brief=name):
                brief = Path(__file__).with_name(name).read_text(encoding="utf-8")
                records = [json.loads(line) for block in re.findall(r"```(?:json|jsonl)\n(.*?)\n```", brief, re.DOTALL)
                           for line in block.splitlines()]
                self.assertEqual(review.validate_records(review.deduplicate(records)), [])
                quotas = re.findall(r"\| `([^`]+)` \| (\d+) \|", brief.split("## 6.", 1)[1])
                self.assertEqual({label for label, _ in quotas}, set(schema.INTENTS))
                self.assertEqual(sum(int(count) for _, count in quotas), 500)
                if name == "gen-zh.md":
                    self.assertEqual([row["text"][slot["start"]:slot["end"]] for row in records for slot in row["slots"]],
                                     ["小王", "李明", "第二个", "第2个", "车内", "车内", "家庭", "座舱影院",
                                      "按摩座椅", "DLP大灯", "车内", "娱乐", "车里的电影院"])

    def test_spans_and_export_alignment(self) -> None:
        row = example("show me the mobile theater, please", "request_flow", "flow_ref", "mobile theater")
        self.assertEqual(build.to_bio(row), (["show", "me", "the", "mobile", "theater", ",", "please"],
                                           ["O", "O", "O", "B-flow_ref", "I-flow_ref", "O", "O"]))
        unicode_row = example("hi 🙂 I'm Zoë", "greet", "name", "Zoë")
        self.assertEqual(unicode_row["slots"][0]["start"], 9)
        self.assertIsNone(review.validate_record(unicode_row))
        for row in (example("我想体验按摩座椅", "request_flow", "flow_ref", "按摩座椅"),
                    example("family-friendly", "inform", "theme", "family")):
            self.assertIsNone(review.validate_record(row))  # Independent of English input units.
            with self.assertRaises(ValueError):
                build.to_bio(row)
        for start, end in ((-1, 14), (4, 40), (4, 4), (True, 14)):
            broken = example("the second one", "inform", "choice", "second one")
            broken["slots"][0].update(start=start, end=end)
            self.assertEqual(review.validate_record(broken).type, review.ErrType.INVALID_SPAN)
        self.assertEqual(review.validate_record(example("hi I'm Sam", "affirm", "name", "Sam")).type,
                         review.ErrType.INVALID_LABEL)

    def test_inform_accepts_multiple_slot_types(self) -> None:
        row = example("I prefer family features inside", "inform", "theme", "family")
        row["slots"].append({"type": "part", "start": 25, "end": 31})
        self.assertIsNone(review.validate_record(row))
        self.assertEqual(build.to_bio(row), (["I", "prefer", "family", "features", "inside"],
                                           ["O", "O", "B-theme", "O", "B-part"]))
        for invalid in ([], None, {}):
            with self.subTest(slots=invalid):
                self.assertIsInstance(review.validate_record(dict(row, slots=invalid)), review.Err)

    def test_hint_errors(self) -> None:
        row = example("show me the movie setup", "request_flow", "flow_ref", "movie setup")
        row["slot_hints"] = [{"slot": 0, "value": "cabin_cinema"}]
        self.assertIsNone(review.validate_record(row))
        row["slot_hints"][0]["value"] = "invented_feature"
        self.assertEqual(review.validate_record(row).type, review.ErrType.INVALID_HINT)
        row["slot_hints"] = [{"slot": 3, "value": "cabin_cinema"}]
        self.assertEqual(review.validate_record(row).type, review.ErrType.INVALID_HINT)

    def test_errors_keep_record_context_separate_and_stop_at_first_problem(self) -> None:
        good = example("hello", "greet")
        malformed = {"text": "hello", "intent": "bad_label", "slots": None}
        issues = review.validate_records([good, malformed, dict(good, text="hi"), example("hello", "greet")])
        self.assertEqual([issue["index"] for issue in issues], [1, 2, 3])
        self.assertTrue(all(set(issue) == {"index", "id", "err"} for issue in issues))
        self.assertIsNone(issues[0]["id"])
        self.assertEqual(issues[0]["err"].type, review.ErrType.INVALID_RECORD)
        self.assertEqual(issues[0]["err"].description, "missing record_id")
        self.assertEqual(issues[1]["id"], good["record_id"])
        self.assertEqual(issues[1]["err"].type, review.ErrType.DUPLICATE_ID)
        self.assertEqual(issues[2]["err"].type, review.ErrType.DUPLICATE_TEXT)
        self.assertEqual(review.validate_records([good]), [])
        self.assertEqual(review.validate_record(dict(good, slots=None)).description, "slots must be a list")

    def test_deduplication_preserves_conflicts_order_and_original_ids(self) -> None:
        first = example("show me the theater", "request_flow", "flow_ref", "theater")
        identical = dict(first, record_id="another-id")
        conflicts = [example(first["text"], "UNK"),
                     dict(first, record_id="hint-id", slot_hints=[{"slot": 0, "value": "cabin_cinema"}]),
                     dict(first, record_id="group-a-id", group_id="group-a"),
                     dict(first, record_id="group-b-id", group_id="group-b"),
                     dict(first, record_id="case-id", text=first["text"].upper())]
        records = [first, identical, *conflicts]
        before = deepcopy(records)
        result = review.deduplicate(records)
        self.assertEqual(result, [first, *conflicts])
        self.assertIs(result[0], first)
        self.assertEqual(records, before)

    def test_intent_split_is_repeatable_and_keeps_whole_groups(self) -> None:
        records = [dict(example(f"fixture {i}", "greet" if i < 60 else "affirm"), group_id=f"family-{i // 2}")
                   for i in range(100)]
        splits = build.split(records, seed=1234)
        self.assertEqual(splits, build.split(records, seed=1234))
        self.assertEqual([len(rows) for rows in splits.values()], [80, 10, 10])
        seen = set()
        for rows in splits.values():
            groups = {row["group_id"] for row in rows}
            self.assertFalse(seen & groups)
            seen.update(groups)
            self.assertEqual({row["intent"] for row in rows}, {"greet", "affirm"})
        self.assertCountEqual([row for rows in splits.values() for row in rows], records)
        for invalid in ([], records[:10], [dict(row, group_id="") for row in records]):
            with self.assertRaises(ValueError):
                build.split(invalid)

    def test_build_consumes_only_final_records_and_preserves_ids(self) -> None:
        records = [example(f"hi I'm Person{i}", "greet", "name", f"Person{i}") for i in range(20)]
        records += [example(f"unsupported fixture {i}", "UNK") for i in range(20)]
        raw = self.raw_attempt(records)
        copied = deepcopy(records[0])
        duplicate = self.raw_attempt([copied], job=2)
        directory = workspace.create_build(self.dataset, [raw, duplicate], stamp="build-one")
        kept = review.deduplicate([*records, copied])
        accepted = [dict(row, group_id=f"group-{i // 2}") for i, row in enumerate(kept)]
        accepted_path = directory / "review/accepted.jsonl"
        broken = deepcopy(accepted)
        broken[0]["slots"][0]["end"] = 999
        workspace.write_jsonl(accepted_path, broken)
        with self.assertRaises(ValueError) as caught:
            build.compile_build(directory)
        issue = json.loads(str(caught.exception))[0]
        self.assertEqual(issue["index"], 0)
        self.assertEqual(issue["id"], accepted[0]["record_id"])
        self.assertEqual(issue["err"]["type"], "INVALID_SPAN")
        workspace.write_jsonl(accepted_path, [{key: value for key, value in row.items() if key != "group_id"} for row in accepted])
        with self.assertRaisesRegex(ValueError, "group_id"):
            build.compile_build(directory)
        workspace.write_jsonl(accepted_path, accepted)
        before = workspace.hashes(directory, [accepted_path])
        (directory / "review/arbiter.md").unlink()
        shutil.rmtree(directory / "work")
        self.assertEqual([path.name for path in (directory / "review").iterdir()], ["accepted.jsonl"])
        metadata = build.compile_build(directory, seed=42)
        workspace.verify_hashes(directory, before)
        self.assertEqual(metadata["review_files"], before)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["format"], "snips")
        self.assertEqual(metadata["compilation"]["stratify"], "intent")
        self.assertEqual(sum(info["records"] for info in metadata["splits"].values()), len(records))
        by_id = {row["record_id"]: row for row in records}
        for name in schema.SPLITS:
            folder = directory / "dataset" / name
            texts, tags, labels = [(folder / file).read_text().splitlines() for file in ("seq.in", "seq.out", "label")]
            self.assertEqual(len(texts), len(tags))
            self.assertEqual(len(texts), len(labels))
            self.assertEqual(len(texts), len(metadata["splits"][name]["rows"]))
            self.assertIn("choice", metadata["splits"][name]["missing_slots"])
            for text, tag, label, provenance in zip(texts, tags, labels, metadata["splits"][name]["rows"]):
                self.assertEqual((text.split(), tag.split()), build.to_bio(by_id[provenance["record_id"]]))
                self.assertEqual(label, by_id[provenance["record_id"]]["intent"])
        self.assertEqual((directory / "dataset/intent_label.txt").read_text().splitlines(), list(schema.INTENTS))
        with self.assertRaises(ValueError):
            build.compile_build(directory)
        workspace.verify_hashes(directory, metadata["dataset_files"])

    def test_chinese_span_export_preserves_original_records_without_tokenization(self) -> None:
        self.dataset = self.dataset.with_name("showroom-zh")
        brief = Path(__file__).with_name("gen-zh.md")
        raw = workspace.create_raw(self.dataset, "fixture", brief=brief)
        self.assertEqual((raw / "gen.md").read_bytes(), brief.read_bytes())
        records = []
        for i in range(20):
            row = example(f"我想体验DLP大灯，选项{i}", "request_flow", "flow_ref", "DLP大灯")
            row["slot_hints"] = [{"slot": 0, "value": "dlp_headlights"}]
            records.append(row)
            row = example(f"你好🙂我叫𠮷野 Alex{i}", "greet", "name", f"𠮷野 Alex{i}")
            records.append(row)
        for i, row in enumerate(records, 1):
            row.update(record_id=f"{raw.name}-{i:06d}", group_id=f"family-{(i - 1) // 2}")
        workspace.write_jsonl(raw / "part-01.jsonl", records)
        directory = workspace.create_build(self.dataset, [raw])
        accepted = directory / "review/accepted.jsonl"
        workspace.write_jsonl(accepted, records)
        before = accepted.read_bytes()
        with patch.object(build, "tokenize", side_effect=AssertionError("span export must not tokenize")):
            metadata = build.compile_build(directory, format="span-jsonl")
        self.assertEqual(accepted.read_bytes(), before)
        self.assertEqual(metadata["format"], "span-jsonl")
        self.assertIsNone(metadata["tokenization"])
        exported, seen_groups = [], set()
        for name in schema.SPLITS:
            folder = directory / "dataset" / name
            self.assertEqual([path.name for path in folder.iterdir()], ["records.jsonl"])
            rows = workspace.read_jsonl(folder / "records.jsonl")
            self.assertEqual([row["record_id"] for row in rows], [row["record_id"] for row in metadata["splits"][name]["rows"]])
            groups = {row["group_id"] for row in rows}
            self.assertFalse(seen_groups & groups)
            seen_groups.update(groups)
            exported.extend(rows)
        self.assertCountEqual(exported, records)
        self.assertEqual(len({row["record_id"] for row in exported}), len(records))
        workspace.verify_hashes(directory, metadata["dataset_files"])
        self.assertEqual((directory / "dataset/slot_label.txt").read_text().splitlines(), list(schema.SLOT_LABELS))

    def test_repeated_slot_type_remains_invalid(self) -> None:
        row = example("家庭和娱乐都可以", "inform", "theme", "家庭")
        row["slots"].append({"type": "theme", "start": 3, "end": 5})
        self.assertEqual(review.validate_record(row).type, review.ErrType.INVALID_LABEL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
