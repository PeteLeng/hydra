"""Check showroom integration and legacy SNIPS report compatibility offline."""

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest

from snips_data import Labels, read_examples, read_labels, read_run_dataset, resolve_dataset, text_units


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
