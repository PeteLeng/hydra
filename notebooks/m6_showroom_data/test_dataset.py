"""Developer tests of build/review helpers; these do not arbitrate generated samples."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

import build
import review


def example(text: str, intent: str, kind: str | None = None, surface: str = "") -> build.Record:
    slots = [] if kind is None else [{"type": kind, "start": text.index(surface), "end": text.index(surface) + len(surface)}]
    return {"text": text, "intent": intent, "slots": slots}


class DatasetTests(unittest.TestCase):
    def test_brief_examples_and_labels(self) -> None:
        brief = Path(__file__).with_name("gen.md").read_text(encoding="utf-8")
        for block in re.findall(r"```(?:json|jsonl)\n(.*?)\n```", brief, re.DOTALL):
            for line in block.splitlines():
                review.validate_record(json.loads(line))
        quotas = re.findall(r"\| `([^`]+)` \| (\d+) \|", brief.split("## 6.", 1)[1])
        self.assertEqual({label for label, _ in quotas}, set(build.INTENTS))
        self.assertEqual(sum(int(count) for _, count in quotas), 500)
        self.assertEqual((build.DATA / "intent_label.txt").read_text().splitlines(), list(build.INTENTS))
        self.assertEqual((build.DATA / "slot_label.txt").read_text().splitlines(), list(build.SLOT_LABELS))

    def test_alignment_and_mechanical_rejection(self) -> None:
        row = example("show me the mobile theater, please", "request_flow", "flow_ref", "mobile theater")
        self.assertEqual(build.to_bio(row), (["show", "me", "the", "mobile", "theater", ",", "please"],
                                           ["O", "O", "O", "B-flow_ref", "I-flow_ref", "O", "O"]))
        unicode_row = example("hi 🙂 I'm Zoë", "greet", "name", "Zoë")
        self.assertEqual(unicode_row["slots"][0]["start"], 9)
        self.assertFalse(review.validate_records([unicode_row]))
        bad = [example("yes", "not_an_intent"), example("hi I'm Sam", "affirm", "name", "Sam"),
               example("anything", "inform"), example("the second one, please", "inform", "choice", "second one,"),
               example("family-friendly", "inform", "theme", "family")]
        for start, end in ((-1, 14), (4, 40), (4, 4), (True, 14)):
            broken = example("the second one", "inform", "choice", "second one")
            broken["slots"][0].update(start=start, end=end)
            bad.append(broken)
        overlap = deepcopy(row)
        overlap["slots"].append(dict(overlap["slots"][0], type="part"))
        bad.append(overlap)
        self.assertEqual(len(review.validate_records(bad)), len(bad))

    def test_new_aliases_and_meaning_belong_to_arbiter(self) -> None:
        row = example("show me the movie setup", "request_flow", "flow_ref", "movie setup")
        review.validate_record(row)  # Unlisted alias without a proposal is allowed.
        row["alias_proposals"] = [{"slot": 0, "value": "cabin_cinema"}]
        review.validate_record(row)
        row["alias_proposals"][0]["value"] = "invented_feature"
        self.assertTrue(review.validate_records([row]))
        review.validate_record(example("my name is Sam", "inform", "choice", "Sam"))  # Structurally valid, semantically wrong.

    def test_duplicates_and_explicit_group_split(self) -> None:
        one = example("show me the theater", "request_flow", "flow_ref", "theater")
        duplicate = example("SHOW  me the theater", "request_flow", "flow_ref", "theater")
        self.assertFalse(review.find_duplicates([one, duplicate])[0]["conflicting_labels"])
        self.assertTrue(review.find_duplicates([one, example(one["text"], "UNK")])[0]["conflicting_labels"])
        records = [{"text": f"fixture {i}", "intent": "greet", "slots": [], "group_id": f"family-{i // 2}"} for i in range(100)]
        splits = build.split(records, seed=1234)
        self.assertEqual(splits, build.split(records, seed=1234))
        self.assertEqual([len(rows) for rows in splits.values()], [80, 10, 10])
        seen: set[str] = set()
        for rows in splits.values():
            groups = {row["group_id"] for row in rows}
            self.assertFalse(seen & groups)
            seen.update(groups)
        self.assertCountEqual(build.combine(splits.values()), records)
        self.assertEqual(len(build.combine(build.split([one], 1).values())), 1)  # No per-label/value coverage gate.

    def test_review_and_export_workflow(self) -> None:
        root = build.PROJECT / "notebooks" / ".artifacts" / "m6_showroom_data_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            directory = Path(temporary)
            raw, draft, report, approved = [directory / name for name in ("raw", "draft.jsonl", "report.json", "reviewed.jsonl")]
            records = [example("hi I'm Sam", "greet", "name", "Sam"),
                       example("show me the movie setup", "request_flow", "flow_ref", "movie setup"),
                       example("book a restaurant", "UNK")]
            build.write_jsonl(raw / "one.jsonl", records[:1])
            build.write_jsonl(raw / "two.jsonl", records[1:])
            with redirect_stdout(io.StringIO()):
                review.main(["--input", str(raw), "--draft", str(draft), "--report", str(report)])
            candidates, errors = review.load_candidates(draft)
            self.assertFalse(errors)
            self.assertTrue(all("source" in row for row in candidates))
            review.write_reviewed(candidates, approved)
            with self.assertRaises(ValueError):
                review.write_reviewed(candidates + [deepcopy(candidates[0])], directory / "bad.jsonl")
            with redirect_stdout(io.StringIO()):
                build.main(["--input", str(approved), "--output", str(directory / "export"), "--seed", "42"])
            manifest = json.loads((directory / "export" / "manifest.json").read_text())
            self.assertEqual(sum(info["records"] for info in manifest["splits"].values()), len(records))
            for name in build.SPLITS:
                folder = directory / "export" / name
                texts, tags, labels = [(folder / file).read_text().splitlines() for file in ("seq.in", "seq.out", "label")]
                self.assertEqual(len(texts), len(labels))
                self.assertEqual(len(texts), len(tags))
                self.assertTrue(all(len(text.split()) == len(tag.split()) for text, tag in zip(texts, tags)))
                self.assertTrue(all(tag in build.SLOT_LABELS for line in tags for tag in line.split()))
            before = approved.read_bytes()
            (raw / "broken.jsonl").write_text('{"text":', encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                review.main(["--input", str(raw), "--save-reviewed", str(approved)])
            self.assertEqual(approved.read_bytes(), before)
            build.write_jsonl(raw / "mixed.jsonl", [example("invalid", "bad_label"), *records])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                review.main(["--input", str(raw), "--report", str(report)])
            inputs, _ = review.load_candidates(raw)
            for group in json.loads(report.read_text())["duplicates"]:
                self.assertEqual([inputs[i]["source"] for i in group["indices"]], group["sources"])

    def test_grouped_split_covers_rare_labels(self) -> None:
        records = [dict(example(f"greeting {i}", "greet"), group_id=f"greeting-{i // 2}")
                   for i in range(80)]
        # Exactly three independent families carry choice; every split needs one.
        for group in range(3):
            records.extend(dict(example(f"pick {group + 1} variant {i}", "inform", "choice", str(group + 1)),
                                group_id=f"choice-{group}") for i in range(2))
        splits = build.split(records, seed=1234)
        self.assertEqual(splits, build.split(records, seed=1234))
        for rows in splits.values():
            self.assertEqual({row["intent"] for row in rows}, {"greet", "inform"})
            self.assertTrue(any(slot["type"] == "choice" for row in rows for slot in row["slots"]))
        assignments: dict[str, str] = {}
        for name, rows in splits.items():
            for row in rows:
                self.assertEqual(assignments.setdefault(row["group_id"], name), name)
        self.assertCountEqual(build.combine(splits.values()), records)

    def test_unsplittable_label_stays_in_training(self) -> None:
        records = [dict(example(f"hello {i}", "greet"), group_id=f"greeting-{i}") for i in range(40)]
        records.extend(dict(example(f"my name is Person {i}", "inform", "name", f"Person {i}"),
                            group_id="only-name-family") for i in range(3))
        splits = build.split(records)
        self.assertTrue(all(splits.values()))
        self.assertEqual(sum(row["intent"] == "inform" for row in splits["train"]), 3)
        self.assertFalse(any(row["intent"] == "inform" for name in ("dev", "test") for row in splits[name]))

    def test_greeting_names_are_not_hidden_by_inform_names(self) -> None:
        records = [dict(example(f"hello variant {i}", "greet"), group_id=f"hello-{i}") for i in range(30)]
        for i in range(18):
            records.append(dict(example(f"my name is Person {i}", "inform", "name", f"Person {i}"),
                                group_id=f"inform-name-{i // 2}"))
        for i in range(6):
            records.append(dict(example(f"hi I am Guest {i}", "greet", "name", f"Guest {i}"),
                                group_id=f"greet-name-{i // 2}"))
        for rows in build.split(records).values():
            self.assertTrue(any(row["intent"] == "greet" and row["slots"] for row in rows))
            self.assertTrue(any(row["intent"] == "greet" and not row["slots"] for row in rows))
            self.assertTrue(any(row["intent"] == "inform" for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
