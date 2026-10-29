"""Read top to bottom: data -> model -> updates -> evaluation -> checkpoint/report."""

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "repos" / "JointBERT" / "data" / "snips"
CACHE = PROJECT / ".artifacts" / "huggingface" / "hub"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
os.environ["MPLCONFIGDIR"] = str(PROJECT / ".artifacts" / "matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seqeval.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from joint_model import JointModel, compute_loss, load_checkpoint, save_checkpoint
from snips_data import Batch, IGNORE_SLOT, read_examples, read_labels, tokenize_batch


@dataclass
class Metrics:
    loss: float
    intent_accuracy: float
    slot_f1: float
    frame_accuracy: float


@torch.no_grad()
def evaluate(model: JointModel, batches: Iterable[Batch], device: torch.device) -> Metrics:
    """Forward passes and metrics only. Loss averages over examples/supervised words."""
    model.eval()
    examples = words = intent_correct = frame_correct = 0
    intent_sum = slot_sum = 0.0
    references: list[list[str]] = []
    guesses: list[list[str]] = []
    for batch in batches:
        batch = batch.to(device)
        predictions = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        losses = compute_loss(predictions, batch.intent_ids, batch.slot_ids)
        active = batch.slot_ids != IGNORE_SLOT
        correct_intents = predictions.intent_logits.argmax(-1) == batch.intent_ids
        predicted_slots = predictions.slot_logits.argmax(-1)
        correct_slots = ((predicted_slots == batch.slot_ids) | ~active).all(dim=1)
        examples += len(batch.line_ids)
        words += int(active.sum())
        intent_correct += int(correct_intents.sum())
        frame_correct += int((correct_intents & correct_slots).sum())
        intent_sum += losses.intent.item() * len(batch.line_ids)
        slot_sum += losses.slot.item() * int(active.sum())
        for targets, predicted in zip(batch.slot_ids.tolist(), predicted_slots.tolist()):
            references.append([model.labels.slot_names[t] for t in targets if t != IGNORE_SLOT])
            guesses.append([model.labels.slot_names[p] for t, p in zip(targets, predicted) if t != IGNORE_SLOT])
    if not examples or not words:
        raise ValueError("Evaluation needs examples with supervised slot positions")
    return Metrics(intent_sum / examples + slot_sum / words, intent_correct / examples,
                   float(f1_score(references, guesses, zero_division=0)), frame_correct / examples)


def train(steps: int | None = None, batch_size: int = 8, eval_every: int = 100,
          learning_rate: float = 5e-5, seed: int = 1234, test: bool = False) -> Path:
    if batch_size < 1 or eval_every < 1 or (steps is not None and steps < 1) or learning_rate <= 0:
        raise ValueError("Batch size, evaluation interval, step limit, and learning rate must be positive")
    started = time.perf_counter()
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = PROJECT / ".artifacts" / "m3_training" / datetime.now().strftime("fresh-%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)

    print("[1/5] Read full SNIPS splits and create batches", flush=True)
    labels = read_labels(DATA)
    training = read_examples(DATA / "train", labels)
    development = read_examples(DATA / "dev", labels)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", revision=REVISION,
                                                  cache_dir=CACHE, local_files_only=True)
    collate = partial(tokenize_batch, tokenizer=tokenizer, max_length=50)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    dev_loader = DataLoader(development, batch_size=32, collate_fn=collate)
    total_steps = len(train_loader) if steps is None else steps
    if total_steps > len(train_loader):
        raise ValueError(f"One epoch has {len(train_loader)} batches; choose a smaller --steps")
    print(f"      {len(training):,} train / {len(development):,} dev; batch {batch_size}; "
          f"{len(train_loader)} batches available; executing {total_steps} updates")

    print("\n[2/5] Construct cached BERT + two new heads", flush=True)
    encoder = BertModel.from_pretrained("bert-base-uncased", revision=REVISION, cache_dir=CACHE, local_files_only=True)
    model = JointModel(encoder, labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    print(f"      Device: {device}; {len(labels.intent_names)} intent classes / {len(labels.slot_names)} slot classes")
    initial = evaluate(model, dev_loader, device)
    best_loss, best_step = initial.loss, 0
    save_checkpoint(model, run_dir / "best.pt")
    history = []
    evaluations = [{"step": 0, **asdict(initial)}]
    used_lines: list[int] = []

    print("\n[3/5] Forward -> losses -> backward -> clip -> optimizer -> scheduler", flush=True)
    for step, batch in enumerate(train_loader, start=1):
        model.train()  # Mode flag: enable dropout. This does not execute a training update.
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        losses = compute_loss(predictions, intent_ids=batch.intent_ids, slot_ids=batch.slot_ids)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        used_lines.extend(batch.line_ids)
        history.append({"step": step, "intent": losses.intent.item(), "slot": losses.slot.item(),
                        "total": losses.total.item(), "learning_rate": rate})
        if total_steps < 30 or step % eval_every == 0 or step == total_steps:
            print(f"      Update {step}/{total_steps}: intent={losses.intent.item():.4f}, "
                  f"slot={losses.slot.item():.4f}, total={losses.total.item():.4f}, lr={rate:.7f}")
        if step % eval_every == 0 or step == total_steps:
            metrics = evaluate(model, dev_loader, device)
            evaluations.append({"step": step, **asdict(metrics)})
            print(f"      Dev: loss={metrics.loss:.4f}, intent={metrics.intent_accuracy:.3f}, slot F1={metrics.slot_f1:.3f}")
            if metrics.loss < best_loss:
                best_loss, best_step = metrics.loss, step
                save_checkpoint(model, run_dir / "best.pt")
        if step == total_steps:
            break

    print("\n[4/5] Save final weights and tokenizer", flush=True)
    save_checkpoint(model, run_dir / "last.pt")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    test_metrics = None
    if test:
        selected = load_checkpoint(run_dir / "best.pt", device)
        test_loader = DataLoader(read_examples(DATA / "test", labels), batch_size=32, collate_fn=collate)
        test_metrics = asdict(evaluate(selected, test_loader, device))
        print(f"      Test (dev-selected step {best_step}): {test_metrics}")

    print("\n[5/5] Write the plot and auditable run files", flush=True)
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.plot([r["step"] for r in history], [r["total"] for r in history], label="Train batch (dropout on)")
    axes.plot([r["step"] for r in evaluations], [r["loss"] for r in evaluations], marker="o", label="Full dev (dropout off)")
    axes.set(xlabel="Optimizer updates", ylabel="Joint cross-entropy loss", title="M3 · standalone joint model")
    axes.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "loss.png", dpi=140)
    plt.close(figure)
    sources = [Path(__file__).with_name(name) for name in ("train.py", "snips_data.py", "joint_model.py")]
    (run_dir / "source").mkdir()
    for source in [*sources, PROJECT / "uv.lock"]:
        shutil.copy2(source, run_dir / "source" / source.name)
    data_files = [DATA / split / name for split in ("train", "dev", *(["test"] if test else []))
                  for name in ("seq.in", "seq.out", "label")]
    data_files += [DATA / "intent_label.txt", DATA / "slot_label.txt"]
    report = {"seed": seed, "batch_size": batch_size, "max_length": 50, "learning_rate": learning_rate,
              "step_limit": steps, "eval_every": eval_every, "slot_weight": 1.0,
              "model_revision": REVISION, "device": str(device), "elapsed_seconds": time.perf_counter() - started,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "cuda_runtime": torch.version.cuda,
              "train_examples": len(training), "dev_examples": len(development), "available_batches": len(train_loader),
              "used_train_lines": used_lines, "labels": {"intents": labels.intent_names, "slots": labels.slot_names},
              "history": history, "evaluations": evaluations,
              "best_step": best_step, "test": test_metrics,
              "python": sys.version, "packages": {name: version(name) for name in ("torch", "transformers", "seqeval")},
              "data_sha256": {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}}
    (run_dir / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"      Used {len(used_lines)} / {len(training)} training examples.\n      Results: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="Limit updates; default is one complete training epoch")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test", action="store_true", help="Evaluate the dev-selected checkpoint on test once")
    args = parser.parse_args()
    train(args.steps, args.batch_size, args.eval_every, args.learning_rate, args.seed, args.test)
