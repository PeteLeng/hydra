"""SNIPS text files -> examples -> token-aligned batches. No JointBERT imports."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

IGNORE_SLOT = -100


@dataclass
class Labels:
    """Ordered class names; each list index is the corresponding class ID."""

    intent_names: list[str]     # [num_intents]
    slot_names: list[str]       # [num_slots]


@dataclass
class RunDataset:
    directory: Path
    labels: Labels
    tokenization: str


@dataclass
class Example:
    """One utterance; W = number of original dataset words."""

    line_id: int               # Scalar; 1-based line in the original split
    words: list[str]           # [W]
    intent_id: int             # Scalar; class ID into Labels.intent_names
    slot_ids: list[int]        # [W]; one class ID into Labels.slot_names per word


@dataclass
class Batch:
    """B = batch size; L = padded/truncated token length, including special tokens."""

    input_ids: Tensor          # int64 [B, L]; tokenizer vocabulary IDs
    attention_mask: Tensor     # int64 [B, L]; 1 for real tokens, 0 for padding
    intent_ids: Tensor         # int64 [B]; one intent-class ID per example
    slot_ids: Tensor           # int64 [B, L]; token-aligned, IGNORE_SLOT (-100) where unsupervised
    line_ids: list[int]        # [B]; original split line numbers
    word_ids: list[list[int | None]]  # [B, L]; token -> word index, None for specials/padding

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device), attention_mask=self.attention_mask.to(device),
            intent_ids=self.intent_ids.to(device), slot_ids=self.slot_ids.to(device),
            line_ids=self.line_ids, word_ids=self.word_ids,
        )


def read_labels(directory: Path, *, include_unk_intent: bool = False) -> Labels:
    intent_names = (directory / "intent_label.txt").read_text(encoding="utf-8").splitlines()
    slot_names = (directory / "slot_label.txt").read_text(encoding="utf-8").splitlines()
    # Showroom uses UNK as a real intent; SNIPS uses it only for bookkeeping.
    return Labels(intent_names=[name for name in intent_names if include_unk_intent or name != "UNK"],
                  slot_names=[name for name in slot_names if name not in {"PAD", "UNK"}])


def text_units(text: str, convention: str = "showroom") -> list[tuple[str, int, int]]:
    """Original-text input units; showroom matches m6_showroom_data.build.tokenize."""
    patterns = {"showroom": r"\w+(?:['’\-]\w+)*|[^\w\s]", "whitespace": r"\S+"}
    if convention not in patterns:
        raise ValueError(f"Unknown input tokenization: {convention}")
    return [(match.group(), match.start(), match.end()) for match in re.finditer(patterns[convention], text)]


def read_run_dataset(report: dict[str, Any], root: Path) -> RunDataset:
    """Resolve new run metadata or match the label vocabulary of a legacy run."""
    labels = Labels(report["labels"]["intents"], report["labels"]["slots"])
    if "dataset" in report:
        metadata = report["dataset"]
        candidates = [(root / metadata["path"], metadata["tokenization"])]
    else:
        candidates = [(root / "data/showroom", "showroom"),
                      (root / "repos/JointBERT/data/snips", "whitespace")]
    for directory, convention in candidates:
        if not (directory / "intent_label.txt").is_file():
            continue
        available = read_labels(directory, include_unk_intent="UNK" in labels.intent_names)
        if available == labels:
            text_units("", convention)  # Validate the saved convention before using it.
            return RunDataset(directory, labels, convention)
    raise ValueError("No local dataset matches this run's recorded path and label vocabulary")


def read_examples(directory: Path, labels: Labels) -> list[Example]:
    texts = (directory / "seq.in").read_text(encoding="utf-8").splitlines()
    intents = (directory / "label").read_text(encoding="utf-8").splitlines()
    tags = (directory / "seq.out").read_text(encoding="utf-8").splitlines()
    if not len(texts) == len(intents) == len(tags):
        raise ValueError(f"Unaligned files in {directory}")
    intent_name_to_id = {name: i for i, name in enumerate(labels.intent_names)}
    slot_name_to_id = {name: i for i, name in enumerate(labels.slot_names)}
    examples: list[Example] = []
    for line_id, (text, intent, slot_text) in enumerate(zip(texts, intents, tags), start=1):
        words, word_tags = text.split(), slot_text.split()
        if not words or len(words) != len(word_tags):
            raise ValueError(f"Word/tag mismatch at {directory}:{line_id}")
        examples.append(Example(line_id=line_id, words=words, intent_id=intent_name_to_id[intent.strip()],
                                slot_ids=[slot_name_to_id[tag] for tag in word_tags]))
    return examples


def tokenize_batch(
    examples: list[Example], tokenizer: BertTokenizerFast, max_length: int = 50,
) -> Batch:
    """Supervise the first subtoken of each word; ignore specials, padding, and continuations."""
    if not examples or max_length < 3:
        raise ValueError("Need examples and room for at least one token plus [CLS]/[SEP]")
    encoded = tokenizer(
        [example.words for example in examples], is_split_into_words=True,
        padding="max_length", truncation=True, max_length=max_length, return_tensors="pt",
    )
    slot_ids = torch.full_like(encoded["input_ids"], IGNORE_SLOT)  # int64 [B, L]
    word_ids = [encoded.word_ids(batch_index=i) for i in range(len(examples))]
    for i, example in enumerate(examples):
        previous_word_id = None
        for position, word_id in enumerate(word_ids[i]):
            if word_id is not None and word_id != previous_word_id:
                slot_ids[i, position] = example.slot_ids[word_id]
            previous_word_id = word_id
    return Batch(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
        intent_ids=torch.tensor([example.intent_id for example in examples], dtype=torch.long),
        slot_ids=slot_ids, line_ids=[example.line_id for example in examples], word_ids=word_ids,
    )
