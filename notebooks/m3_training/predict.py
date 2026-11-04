"""Sentence -> intent and character-based slot spans, using a saved checkpoint."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from joint_model import JointModel, load_checkpoint
from snips_data import text_units


@dataclass
class PredictedSlot:
    type: str
    text: str
    start: int
    end: int


@dataclass
class Prediction:
    intent: str
    confidence: float
    slots: list[PredictedSlot]


@torch.inference_mode()
def predict(text: str, model: JointModel, tokenizer: BertTokenizerFast,
            max_length: int = 50, *, tokenization: str | None = "showroom", format: str = "snips") -> Prediction:
    if not text.strip():
        raise ValueError("Provide a nonempty sentence")
    positions: list[tuple[int | None, int, int]]  # Token position and original-text [start, end).
    if format == "span-jsonl":
        encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                            return_tensors="pt", truncation=False)
        positions = [(i, start, end) for i, ((start, end), special) in enumerate(zip(
            encoded["offset_mapping"][0].tolist(), encoded["special_tokens_mask"][0].tolist(),
        )) if not special and start < end]
    else:
        units = text_units(text, tokenization)
        encoded = tokenizer([word for word, _, _ in units], is_split_into_words=True,
                            return_tensors="pt", truncation=False)
        positions = []
        for index, (_, start, end) in enumerate(units):
            span = encoded.word_to_tokens(index)
            positions.append((span.start if span is not None else None, start, end))
    if not positions:
        raise ValueError("Tokenizer produced no text tokens")
    if encoded["input_ids"].shape[1] > max_length:
        raise ValueError(f"Sentence exceeds the checkpoint's {max_length}-token input limit")
    encoded = encoded.to(next(model.parameters()).device)
    model.eval()
    output = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    probabilities = output.intent_logits.softmax(-1)[0]
    intent_id = int(probabilities.argmax())
    slot_ids = output.slot_logits.argmax(-1)[0].tolist()
    slots: list[PredictedSlot] = []
    active: PredictedSlot | None = None
    for position, start, end in positions:
        tag = model.labels.slot_names[slot_ids[position]] if position is not None else "O"
        if tag == "O":
            active = None
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "B" or active is None or active.type != kind:
            active = PredictedSlot(kind, text[start:end], start, end)
            slots.append(active)
        else:
            active.end = end
            active.text = text[active.start:active.end]
    return Prediction(model.labels.intent_names[intent_id], float(probabilities[intent_id]), slots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Sentence to classify")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to best.pt from the latest completed run")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        runs = Path(__file__).resolve().parents[1] / ".artifacts/m3_training"
        candidates = sorted(p for p in runs.glob("fresh-*/best.pt")
                            if (p.parent / "run.json").is_file() and (p.parent / "tokenizer").is_dir())
        if not candidates:
            parser.error("No completed training run found; supply --checkpoint")
        checkpoint = candidates[-1]
    try:
        settings = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
        tokenizer = BertTokenizerFast.from_pretrained(checkpoint.parent / "tokenizer", local_files_only=True)
        model = load_checkpoint(checkpoint)
        # Legacy showroom runs retain UNK; the standalone SNIPS trainer excluded it.
        convention = settings.get("dataset", {}).get("tokenization",
                     "showroom" if "UNK" in model.labels.intent_names else "whitespace")
        result = predict(args.text, model, tokenizer, settings["max_length"], tokenization=convention,
                         format=settings.get("dataset", {}).get("format", "snips"))
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
