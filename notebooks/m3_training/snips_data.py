"""SNIPS files or original-text spans -> token-aligned batches. No JointBERT imports."""

from dataclasses import dataclass
import hashlib
import json
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
    tokenization: str | None
    format: str = "snips"


@dataclass
class Example:
    """One utterance; W = number of original dataset words."""

    line_id: int               # Scalar; 1-based line in the original split
    words: list[str]           # [W]
    intent_id: int             # Scalar; class ID into Labels.intent_names
    slot_ids: list[int]        # [W]; one class ID into Labels.slot_names per word


@dataclass
class SlotSpan:
    type: str
    start: int                # Inclusive Unicode code-point offset in original text
    end: int                  # Exclusive


@dataclass
class SpanExample:
    line_id: int              # 1-based line in the compiled split's records.jsonl
    record_id: str
    text: str                 # Original text, without segmentation or normalization
    intent_id: int
    slots: list[SlotSpan]


@dataclass
class Batch:
    """B = batch size; L = padded/truncated token length, including special tokens."""

    input_ids: Tensor          # int64 [B, L]; tokenizer vocabulary IDs
    attention_mask: Tensor     # int64 [B, L]; 1 for real tokens, 0 for padding
    intent_ids: Tensor         # int64 [B]; one intent-class ID per example
    slot_ids: Tensor           # int64 [B, L]; token-aligned, IGNORE_SLOT (-100) where unsupervised
    line_ids: list[int]        # [B]; original split line numbers
    word_ids: list[list[int | None]] | None = None  # [B, L] for word inputs; absent for span inputs

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


def resolve_dataset(directory: Path, convention: str | None = "showroom", *,
                    format: str = "snips") -> tuple[Path, str | None, str]:
    """Resolve a completed build, or a legacy directory containing label files."""
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("kind") == "build":
            if metadata.get("version") != 1 or metadata.get("status") != "complete":
                raise ValueError(f"Dataset build is not complete: {directory}")
            for name, digest in metadata["dataset_files"].items():
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Dataset build file changed: {directory / name}")
            convention = metadata["tokenization"]
            format = metadata.get("format", "snips")
            directory = directory / metadata["dataset"]
    if not (directory / "intent_label.txt").is_file() or not (directory / "slot_label.txt").is_file():
        raise ValueError(f"No compiled label inventories: {directory}")
    if format == "snips":
        text_units("", convention)
    elif format != "span-jsonl" or convention is not None:
        raise ValueError(f"Unsupported dataset format/tokenization: {format!r}/{convention!r}")
    return directory, convention, format


def read_run_dataset(report: dict[str, Any], root: Path) -> RunDataset:
    """Resolve new run metadata or match the label vocabulary of a legacy run."""
    labels = Labels(report["labels"]["intents"], report["labels"]["slots"])
    if "dataset" in report:
        metadata = report["dataset"]
        if "manifest_sha256" in metadata:
            path = root / metadata["path"] / "manifest.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
                raise ValueError("Dataset build manifest differs from this training run")
        candidates = [(root / metadata["path"], metadata["tokenization"], metadata.get("format", "snips"))]
    else:
        candidates = [(root / "data/showroom-en/build/20261103T115913654808Z/dataset", "showroom", "snips"),
                      (root / "repos/JointBERT/data/snips", "whitespace", "snips")]
    for directory, convention, format in candidates:
        if not directory.is_dir() or ("dataset" not in report and not (directory / "intent_label.txt").is_file()):
            continue
        directory, actual_convention, actual_format = resolve_dataset(directory, convention, format=format)
        if (actual_convention, actual_format) != (convention, format):
            raise ValueError("Dataset format/tokenization differs from this training run")
        available = read_labels(directory, include_unk_intent="UNK" in labels.intent_names)
        if available == labels:
            return RunDataset(directory, labels, convention, format)
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


def read_span_examples(directory: Path, labels: Labels) -> list[SpanExample]:
    """Read compiled span records; retain original text, annotations, and row identity."""
    path = directory / "records.jsonl"
    examples: list[SpanExample] = []
    for line_id, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        context = f"{path}:{line_id}"
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Record must be an object")
            record_id, text = row["record_id"], row["text"]
            context += f" ({record_id})"
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("Need a nonempty record_id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Need nonempty original text")
            intent_id = labels.intent_names.index(row["intent"])
            if not isinstance(row["slots"], list):
                raise ValueError("slots must be a list")
            slots = []
            previous_end = 0
            for slot in row["slots"]:
                span = SlotSpan(slot["type"], slot["start"], slot["end"])
                if (type(span.start) is not int or type(span.end) is not int
                        or not previous_end <= span.start < span.end <= len(text)):
                    raise ValueError(f"Invalid, unordered, or overlapping span: {span}")
                if any(f"{prefix}-{span.type}" not in labels.slot_names for prefix in ("B", "I")):
                    raise ValueError(f"Unknown slot type: {span.type!r}")
                slots.append(span)
                previous_end = span.end
            examples.append(SpanExample(line_id, record_id, text, intent_id, slots))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{context}: {error}") from error
    return examples


def tokenize_span_batch(
    examples: list[SpanExample], tokenizer: BertTokenizerFast, labels: Labels, max_length: int = 50,
) -> Batch:
    """Align spans to every text token; reject lossy boundaries and overlength inputs."""
    if not examples or max_length < 3:
        raise ValueError("Need examples and room for at least one token plus [CLS]/[SEP]")
    # Keep lists until length validation: overlength rows must not be truncated or tensorized.
    encoded = tokenizer(
        [example.text for example in examples], padding="max_length", truncation=False,
        max_length=max_length, return_offsets_mapping=True, return_special_tokens_mask=True,
    )
    tag_ids = {name: i for i, name in enumerate(labels.slot_names)}
    targets = []
    for i, example in enumerate(examples):
        context = f"{example.record_id} (line {example.line_id})"
        if len(encoded["input_ids"][i]) > max_length:
            raise ValueError(f"{context}: input exceeds {max_length} tokens including specials")
        tags = [IGNORE_SLOT] * len(encoded["input_ids"][i])
        tokens = []
        previous_end = 0
        for position, ((start, end), special, attended) in enumerate(zip(
            encoded["offset_mapping"][i], encoded["special_tokens_mask"][i], encoded["attention_mask"][i],
        )):
            if special or not attended:
                continue
            if not previous_end <= start < end <= len(example.text):
                raise ValueError(f"{context}: invalid token offsets [{start}, {end})")
            tokens.append((position, start, end))
            tags[position] = tag_ids["O"]
            previous_end = end
        if not tokens:
            raise ValueError(f"{context}: tokenizer produced no text tokens")
        for span in example.slots:
            covered = [(p, start, end) for p, start, end in tokens if start < span.end and end > span.start]
            if not covered or covered[0][1] != span.start or covered[-1][2] != span.end:
                raise ValueError(f"{context}: span {span} must align to complete model tokens")
            cursor = span.start
            for j, (position, start, end) in enumerate(covered):
                # Spaces inside names/phrases have no tokens; slicing the original span preserves them.
                if example.text[cursor:start].strip():
                    raise ValueError(f"{context}: span {span} contains text missing from token offsets")
                tags[position] = tag_ids[f"{'B' if j == 0 else 'I'}-{span.type}"]
                cursor = end
        targets.append(tags)
    return Batch(
        input_ids=torch.tensor(encoded["input_ids"], dtype=torch.long),
        attention_mask=torch.tensor(encoded["attention_mask"], dtype=torch.long),
        intent_ids=torch.tensor([example.intent_id for example in examples], dtype=torch.long),
        slot_ids=torch.tensor(targets, dtype=torch.long), line_ids=[example.line_id for example in examples],
    )


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
