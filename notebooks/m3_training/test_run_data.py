"""Check showroom integration and legacy SNIPS report compatibility offline."""

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest

import torch
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from snips_data import (Labels, SlotSpan, SpanExample, read_examples, read_labels, read_run_dataset,
                       read_span_examples, resolve_dataset, text_units, tokenize_span_batch)


class RunDataChecks(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / ".artifacts" / "checks"
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.showroom = "data/showroom-en/build/20261103T115913654808Z/dataset"
        for path, intents, slots in ((self.showroom, "UNK\ngreet\n", "PAD\nUNK\nO\nB-name\nI-name\n"),
                                     ("repos/JointBERT/data/snips", "UNK\nBookRestaurant\n", "PAD\nUNK\nO\nB-city\nI-city\n")):
            directory = self.root / path
            directory.mkdir(parents=True)
            (directory / "intent_label.txt").write_text(intents, encoding="utf-8")
            (directory / "slot_label.txt").write_text(slots, encoding="utf-8")

    def test_unknown_intent_is_a_showroom_target_only_when_requested(self) -> None:
        directory = self.root / self.showroom
        self.assertEqual(read_labels(directory), Labels(["greet"], ["O", "B-name", "I-name"]))
        labels = read_labels(directory, include_unk_intent=True)
        self.assertEqual(labels.intent_names, ["UNK", "greet"])
        for name, value in (("seq.in", "nonsense\n"), ("seq.out", "O\n"), ("label", "UNK\n")):
            (directory / name).write_text(value, encoding="utf-8")
        self.assertEqual(read_examples(directory, labels)[0].intent_id, 0)

    def test_metadata_and_legacy_runs_resolve_their_own_dataset(self) -> None:
        for path, convention, intents, slots in (
            (self.showroom, "showroom", ["UNK", "greet"], ["O", "B-name", "I-name"]),
            ("repos/JointBERT/data/snips", "whitespace", ["BookRestaurant"], ["O", "B-city", "I-city"]),
        ):
            report = {"labels": {"intents": intents, "slots": slots}}
            legacy = read_run_dataset(report, self.root)
            self.assertEqual(legacy.directory, self.root / path)
            self.assertEqual(legacy.tokenization, convention)
            report["dataset"] = {"path": path, "tokenization": convention}
            self.assertEqual(read_run_dataset(report, self.root), legacy)
            report["dataset"]["path"] = "missing"
            with self.assertRaises(ValueError):
                read_run_dataset(report, self.root)
            report["dataset"]["path"] = path
            report["labels"]["intents"] = ["wrong-label"]
            with self.assertRaises(ValueError):
                read_run_dataset(report, self.root)

    def test_custom_text_preserves_dataset_input_units_and_offsets(self) -> None:
        text = "hi 🙂 I'm Zoë, family-friendly!"
        units = text_units(text)
        self.assertEqual([word for word, _, _ in units], ["hi", "🙂", "I'm", "Zoë", ",", "family-friendly", "!"])
        for word, start, end in units:
            self.assertEqual(text[start:end], word)
        self.assertEqual([word for word, _, _ in text_units(text, "whitespace")], text.split())
        with self.assertRaises(ValueError):
            text_units(text, "unknown")

    def test_build_identity_is_not_inferred_from_labels(self) -> None:
        resolved = []
        for name in ("one", "two"):
            build = self.root / "build" / name
            data = build / "dataset"
            shutil.copytree(self.root / self.showroom, data)
            manifest = {"version": 1, "kind": "build", "status": "complete", "dataset": "dataset",
                        "tokenization": "showroom", "dataset_files": {
                            path.relative_to(build).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in data.iterdir()}}
            path = build / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            report = {"dataset": {"path": build.relative_to(self.root).as_posix(), "tokenization": "showroom",
                                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                      "labels": {"intents": ["UNK", "greet"], "slots": ["O", "B-name", "I-name"]}}
            resolved.append(read_run_dataset(report, self.root))
            manifest["status"] = "reviewing"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_dataset(build)
            with self.assertRaises(ValueError):
                read_run_dataset(report, self.root)
        self.assertNotEqual(*resolved)


class SpanDataChecks(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / ".artifacts" / "checks"
        root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        vocabulary = self.directory / "vocab.txt"
        vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n我\n想\n看\n车\n内\n的\n"
                              "大\n灯\n，\n。\ndl\n##p\nalex\n𠮷\n野\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
        self.labels = Labels(["UNK", "request_flow", "greet"],
                             ["O", "B-part", "I-part", "B-flow_ref", "I-flow_ref", "B-name", "I-name"])

    def test_original_text_becomes_all_token_targets_with_specials_and_padding_masked(self) -> None:
        row = {"record_id": "zh-1", "text": "我想看车内的DLP大灯。", "intent": "request_flow",
               "slots": [{"type": "part", "start": 3, "end": 5}, {"type": "flow_ref", "start": 6, "end": 11}]}
        other = {"record_id": "zh-2", "text": "我", "intent": "UNK", "slots": []}
        (self.directory / "records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in (row, other)) + "\n", encoding="utf-8")
        examples = read_span_examples(self.directory, self.labels)
        self.assertEqual(examples[0], SpanExample(1, "zh-1", row["text"], 1,
                                                [SlotSpan("part", 3, 5), SlotSpan("flow_ref", 6, 11)]))
        self.assertEqual(examples[1].line_id, 2)
        batch = tokenize_span_batch(examples, self.tokenizer, self.labels, max_length=15)
        # CLS, 我 想 看 车 内 的 dl ##p 大 灯 。, SEP, PAD, PAD.
        self.assertEqual(batch.slot_ids.tolist(), [
            [-100, 0, 0, 0, 1, 2, 0, 3, 4, 4, 4, 0, -100, -100, -100],
            [-100, 0, -100] + [-100] * 12,
        ])
        self.assertEqual(batch.attention_mask.tolist(), [[1] * 13 + [0] * 2, [1] * 3 + [0] * 12])
        self.assertEqual(batch.intent_ids.tolist(), [1, 0])
        self.assertEqual(batch.line_ids, [1, 2])
        self.assertIsNone(batch.word_ids)
        for tensor in (batch.input_ids, batch.attention_mask, batch.intent_ids, batch.slot_ids):
            self.assertEqual(tensor.dtype, torch.long)
        torch.testing.assert_close(batch.to("cpu").slot_ids, batch.slot_ids)

    def test_unicode_offsets_and_internal_spaces_preserve_original_span(self) -> None:
        text = "我🙂𠮷野 Alex。"
        example = SpanExample(7, "unicode", text, 2, [SlotSpan("name", 2, 9)])
        batch = tokenize_span_batch([example], self.tokenizer, self.labels, max_length=9)
        self.assertEqual(text[2:9], "𠮷野 Alex")
        # CLS, 我, UNK(emoji), 𠮷, 野, alex, 。, SEP, PAD.
        self.assertEqual(batch.slot_ids.tolist(), [[-100, 0, 0, 5, 6, 6, 0, -100, -100]])

    def test_unrepresentable_boundaries_and_missing_text_are_rejected(self) -> None:
        cases = [("Alex", 1, 4),          # Boundary inside a known token.
                 ("xyz", 0, 1),          # Boundary inside an UNK token.
                 (" Alex", 0, 5),         # Leading space has no token offset.
                 ("Alex ", 0, 5),         # Trailing space has no token offset.
                 ("e\u0301", 0, 2),       # Trailing combining accent is normalized away.
                 ("车\x00内", 0, 3)]      # Non-whitespace gap between Chinese tokens.
        for text, start, end in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, r"bad-span \(line 9\): span"):
                tokenize_span_batch([SpanExample(9, "bad-span", text, 2, [SlotSpan("name", start, end)])],
                                    self.tokenizer, self.labels, max_length=10)

    def test_length_counts_specials_and_never_truncates_even_without_slots(self) -> None:
        example = SpanExample(1, "length", "我想看车内", 1, [])
        self.assertEqual(tokenize_span_batch([example], self.tokenizer, self.labels, max_length=7).input_ids.shape,
                         (1, 7))
        with self.assertRaisesRegex(ValueError, r"length \(line 1\).*exceeds 6 tokens"):
            tokenize_span_batch([SpanExample(2, "short", "我", 0, []), example],
                                self.tokenizer, self.labels, max_length=6)
        with self.assertRaisesRegex(ValueError, "no text tokens"):
            tokenize_span_batch([SpanExample(1, "empty-encoding", "\x00", 0, [])],
                                self.tokenizer, self.labels, max_length=7)

    def test_reader_rejects_invalid_spans_and_labels_with_row_identity(self) -> None:
        row = {"record_id": "bad-row", "text": "车内", "intent": "request_flow",
               "slots": [{"type": "part", "start": 0, "end": 2}]}
        invalid = [dict(row, intent="unknown"), dict(row, slots=None),
                   dict(row, slots=[{"type": "part", "start": True, "end": 2}]),
                   dict(row, slots=[{"type": "part", "start": 0, "end": 3}]),
                   dict(row, slots=[{"type": "bad", "start": 0, "end": 2}]),
                   dict(row, slots=row["slots"] * 2)]
        for bad in invalid:
            with self.subTest(row=bad):
                (self.directory / "records.jsonl").write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"records.jsonl:1 \(bad-row\)"):
                    read_span_examples(self.directory, self.labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
