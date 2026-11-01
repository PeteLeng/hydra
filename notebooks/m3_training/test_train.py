"""Exercise complete training runs with tiny local assets and real optimizer updates."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

import train
from joint_model import load_checkpoint


class TrainingChecks(unittest.TestCase):
    def setUp(self) -> None:
        root = train.PROJECT / ".artifacts" / "checks"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data = self.directory / "data"
        self.data.mkdir()
        (self.directory / "uv.lock").write_text("# local fixture\n", encoding="utf-8")
        (self.data / "intent_label.txt").write_text("UNK\ngreet\naffirm\n", encoding="utf-8")
        (self.data / "slot_label.txt").write_text("PAD\nUNK\nO\nB-name\nI-name\n", encoding="utf-8")
        for name, size in (("train", 5), ("dev", 2), ("test", 2)):
            folder = self.data / name
            folder.mkdir()
            (folder / "seq.in").write_text("hello sam\n" * size, encoding="utf-8")
            (folder / "seq.out").write_text("O B-name\n" * size, encoding="utf-8")
            (folder / "label").write_text("greet\n" * size, encoding="utf-8")
        vocabulary = self.directory / "vocab.txt"
        vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nsam\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
        config = BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=16, max_position_embeddings=64)
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(train, "PROJECT", self.directory))
        stack.enter_context(patch.object(train, "DATA", self.data))
        stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=False))
        stack.enter_context(patch.object(train.BertTokenizerFast, "from_pretrained", return_value=tokenizer))
        stack.enter_context(patch.object(train.BertModel, "from_pretrained", side_effect=lambda *a, **k: BertModel(config)))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_epochs_and_partial_epoch_keep_global_schedule(self) -> None:
        for limit, updates, processed in ((None, 6, 10), (4, 4, 7)):
            with self.subTest(steps=limit), redirect_stdout(io.StringIO()):
                directory = train.train(epochs=2, steps=limit, batch_size=2, eval_every=3, test=True)
                report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(report["epochs"], 2)
                self.assertEqual([row["step"] for row in report["history"]], list(range(1, updates + 1)))
                self.assertEqual([row["epoch"] for row in report["history"]], [1, 1, 1] + [2] * (updates - 3))
                self.assertEqual([(row["step"], row["epoch"]) for row in report["evaluations"]],
                                 [(0, 0), (3, 1), (updates, 2)])
                self.assertEqual(len(report["used_train_lines"]), processed)
                self.assertCountEqual(report["used_train_lines"][:5], range(1, 6))
                if limit is None:
                    self.assertCountEqual(report["used_train_lines"][5:], range(1, 6))
                for index, row in enumerate(report["history"]):
                    self.assertAlmostEqual(row["learning_rate"], 5e-5 * (1 - index / updates))
                self.assertIn(report["best_step"], (0, 3, updates))
                self.assertIsNotNone(report["test"])
                self.assertEqual(load_checkpoint(directory / "last.pt").labels.intent_names,
                                 report["labels"]["intents"])
                self.assertTrue((directory / "tokenizer" / "tokenizer.json").is_file())

    def test_rejects_invalid_epoch_and_step_limits(self) -> None:
        for settings in ({"epochs": 0}, {"epochs": 2, "steps": 0}, {"epochs": 2, "steps": 7}):
            with self.subTest(settings=settings), redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                train.train(batch_size=2, **settings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
