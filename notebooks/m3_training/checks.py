"""Offline checks using a tiny BERT and local fixtures, not downloaded weights."""

from pathlib import Path
import tempfile
import unittest

import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from joint_model import JointModel, compute_loss, load_checkpoint, save_checkpoint
from snips_data import Example, IGNORE_SLOT, Labels, read_examples, read_labels, tokenize_batch
from train import PROJECT, evaluate


class PipelineChecks(unittest.TestCase):
    def setUp(self) -> None:
        root = PROJECT / ".artifacts" / "checks"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.directory = Path(self.temporary.name)
        vocabulary = self.directory / "vocab.txt"
        vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbook\nwash\n##ington\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
        self.labels = Labels(intent_names=["BookRestaurant", "GetWeather"], slot_names=["O", "B-city", "I-city"])
        self.example = Example(line_id=1, words=["book", "washington"], intent_id=0, slot_ids=[0, 1])
        self.batch = tokenize_batch([self.example], self.tokenizer, max_length=8)
        config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=32, max_position_embeddings=16)
        torch.manual_seed(1234)
        self.model = JointModel(BertModel(config), self.labels)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reader_and_alignment(self) -> None:
        (self.directory / "intent_label.txt").write_text("UNK\nBookRestaurant\nGetWeather\n")
        (self.directory / "slot_label.txt").write_text("PAD\nUNK\nO\nB-city\nI-city\n")
        (self.directory / "seq.in").write_text("book washington\n")
        (self.directory / "label").write_text("BookRestaurant\n")
        (self.directory / "seq.out").write_text("O B-city\n")
        self.assertEqual(read_labels(self.directory), self.labels)
        self.assertEqual(read_examples(self.directory, self.labels), [self.example])
        self.assertEqual(self.batch.slot_ids.tolist(), [[-100, 0, 1, -100, -100, -100, -100, -100]])
        self.assertEqual(self.batch.attention_mask.tolist(), [[1, 1, 1, 1, 1, 0, 0, 0]])
        (self.directory / "seq.out").write_text("O\n")
        with self.assertRaises(ValueError):
            read_examples(self.directory, self.labels)

    def test_gradients_update_and_checkpoint(self) -> None:
        self.model.train()
        predictions = self.model(input_ids=self.batch.input_ids, attention_mask=self.batch.attention_mask)
        losses = compute_loss(predictions, intent_ids=self.batch.intent_ids, slot_ids=self.batch.slot_ids)
        torch.testing.assert_close(losses.total, losses.intent + losses.slot)
        shared = self.model.encoder.encoder.layer[0].output.LayerNorm.weight
        for loss in (losses.intent, losses.slot):
            self.assertGreater(torch.autograd.grad(loss, shared, retain_graph=True)[0].norm().item(), 0)
        watched = [shared, self.model.intent_head.weight, self.model.slot_head.weight]
        before = [parameter.detach().clone() for parameter in watched]
        predictions.slot_logits.retain_grad()
        losses.total.backward()
        ignored = self.batch.slot_ids == IGNORE_SLOT
        self.assertEqual(torch.count_nonzero(predictions.slot_logits.grad[ignored]).item(), 0)
        self.assertGreater(torch.count_nonzero(predictions.slot_logits.grad[~ignored]).item(), 0)
        torch.optim.AdamW(self.model.parameters(), lr=1e-3).step()
        for old, new in zip(before, watched):
            self.assertFalse(torch.equal(old, new))
        self.model.eval()
        path = self.directory / "model.pt"
        save_checkpoint(self.model, path)
        loaded = load_checkpoint(path)
        self.assertEqual(loaded.labels, self.labels)
        with torch.no_grad():
            expected = self.model(input_ids=self.batch.input_ids, attention_mask=self.batch.attention_mask)
            actual = loaded(input_ids=self.batch.input_ids, attention_mask=self.batch.attention_mask)
        torch.testing.assert_close(actual.intent_logits, expected.intent_logits)
        torch.testing.assert_close(actual.slot_logits, expected.slot_logits)

    def test_evaluation_mode_and_batch_weighting(self) -> None:
        examples = [self.example, Example(line_id=2, words=["book"], intent_id=1, slot_ids=[0]), self.example]
        together = [tokenize_batch(examples, self.tokenizer, 8)]
        separate = [tokenize_batch(examples[:2], self.tokenizer, 8), tokenize_batch(examples[2:], self.tokenizer, 8)]
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        first = evaluate(self.model, together, torch.device("cpu"))
        second = evaluate(self.model, separate, torch.device("cpu"))
        self.assertAlmostEqual(first.loss, second.loss, places=5)
        self.assertEqual(first.intent_accuracy, second.intent_accuracy)
        self.assertEqual(first.slot_f1, second.slot_f1)
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.encoder.training)
        self.assertFalse(self.model.dropout.training)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
