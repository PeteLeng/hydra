"""Saved training metrics and on-demand comparisons of best/final checkpoints."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="M3 · Standalone training results")

with app.setup:
    from functools import lru_cache
    from pathlib import Path
    from types import ModuleType
    from typing import Any

    import torch
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
    from joint_model import JointModel, load_checkpoint
    from snips_data import Example, RunDataset, read_examples, read_run_dataset, text_units

    @lru_cache(maxsize=2)
    def checkpoint_assets(path: Path, modified: int) -> tuple[JointModel, BertTokenizerFast]:
        """Cache by path and modification time so refreshed weights reload."""
        return load_checkpoint(path), BertTokenizerFast.from_pretrained(path.parent / "tokenizer", local_files_only=True)


@app.cell(hide_code=True)
def _() -> tuple[ModuleType, ModuleType, Path]:
    import json
    import marimo as mo
    runs = mo.notebook_dir().parent / ".artifacts" / "m3_training"
    return json, mo, runs


@app.cell(hide_code=True)
def _(mo: ModuleType) -> tuple[marimo.ui.run_button]:
    refresh = mo.ui.run_button(label="Refresh saved runs")
    mo.vstack([mo.md("""
    # M3 · Standalone training results
    **Start in `m3_training/train.py` in VS Code.** From your `notebooks/` terminal:
    ```bash
    uv run --locked --offline --env-file local.env python m3_training/train.py --steps 30 --eval-every 10
    ```
    `snips_data.py` supplies batches. `joint_model.py` defines the model and losses.
    `train.py` owns the training loop. This notebook reads saved metrics and runs live
    predictions when you click **Compare**. CPU inference caches up to two checkpoint models.
    """), refresh])
    return (refresh,)


@app.cell(hide_code=True)
def _(json: ModuleType, mo: ModuleType, refresh: marimo.ui.run_button, runs: Path) -> tuple[dict[str, Any], marimo.ui.dropdown]:
    refresh.value
    _files = sorted(runs.glob("fresh-*/run.json"), reverse=True)
    mo.stop(not _files, mo.md("Run the command above first; then refresh saved runs."))
    reports = {str(p.parent): json.loads(p.read_text(encoding="utf-8")) for p in _files}
    picker = mo.ui.dropdown({Path(p).name: p for p in reports},
                            value=_files[0].parent.name, allow_select_none=False, label="Run for loss graph")
    picker
    return reports, picker


@app.cell(hide_code=True)
def _(mo: ModuleType, picker: marimo.ui.dropdown, reports: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    directory = Path(picker.value)
    report = reports[picker.value]
    mo.md(f"""
    **Saved run:** `{directory}` · **Device:** `{report['device']}` · **Runtime:** {report['elapsed_seconds']:.1f} s
    Dataset: `{report.get('dataset', {}).get('path', 'legacy run; inferred from saved labels')}`.

    Original train/dev: **{report['train_examples']:,} / {report['dev_examples']:,}**.
    Batch size: **{report['batch_size']}**. Batches per epoch: **{report['available_batches']}**.
    Epochs requested: **{report.get('epochs', 1)}**. Completed updates: **{len(report['history'])}**.
    Examples processed: **{len(report['used_train_lines'])}**, including repeats across epochs.
    The loader reshuffles the full split each epoch; the step limit caps updates across the whole run.
    """)
    return report, directory


@app.cell(hide_code=True)
def _(directory: Path, mo: ModuleType, report: dict[str, Any]) -> None:
    mo.vstack([
        mo.Html('<h2 id="m3-loss">1 · Loss graph</h2>'),
        mo.image(str(directory / "loss.png"), alt="Training batch and full dev losses", width=900),
        mo.md("""
        **x:** completed updates. **y:** joint cross-entropy loss.
        Blue = training batches, with dropout on. Orange = the full dev split, with dropout off.
        They use different examples and modes, so their losses need not match.
        Dev loss selects `best.pt`; `last.pt` is the final update's model.
        """),
        mo.ui.table(report["evaluations"], selection=None, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo: ModuleType, reports: dict[str, Any]) -> tuple[marimo.ui.dropdown, marimo.ui.dropdown, marimo.ui.dropdown]:
    _options = {f"{Path(run).name} / {kind} (step {step})": str(Path(run) / f"{kind}.pt")
                for run, saved in reports.items()
                for kind, step in (("best", saved["best_step"]), ("last", len(saved["history"])))
                if (Path(run) / f"{kind}.pt").is_file()}
    mo.stop(not _options, mo.md("No best/final checkpoint files are available."))
    _names = list(_options)
    checkpoint_a = mo.ui.dropdown(_options, value=_names[0], allow_select_none=False, label="Checkpoint A")
    checkpoint_b = mo.ui.dropdown(_options, value=_names[min(1, len(_names) - 1)], allow_select_none=False, label="Checkpoint B")
    source = mo.ui.dropdown(["train", "dev", "test", "Custom text"], value="dev", allow_select_none=False, label="Sample source")
    mo.vstack([mo.md("## 2 · Compare live predictions"), mo.hstack([checkpoint_a, checkpoint_b]), source])
    return checkpoint_a, checkpoint_b, source


@app.cell(hide_code=True)
def _(checkpoint_a: marimo.ui.dropdown, mo: ModuleType, reports: dict[str, Any]) -> tuple[RunDataset]:
    _settings = reports[str(Path(checkpoint_a.value).parent)]
    try:
        run_dataset = read_run_dataset(_settings, mo.notebook_dir().parents[1])
    except (OSError, ValueError) as _error:
        mo.stop(True, mo.md(str(_error)).callout(kind="warn"))
    mo.md(f"Reference dataset for **checkpoint A**: `{run_dataset.directory}`")
    return (run_dataset,)


@app.cell(hide_code=True)
def _(run_dataset: RunDataset, mo: ModuleType, source: marimo.ui.dropdown) -> tuple[
    list[Example], marimo.ui.number | marimo.ui.text_area, marimo.ui.run_button,
]:
    if source.value == "Custom text":
        examples = []
        _default = "show me family features inside" if run_dataset.tokenization == "showroom" else "book a spot for 3 in mt"
        sample = mo.ui.text_area(value=_default, label="Text")
    else:
        examples = read_examples(run_dataset.directory / source.value, run_dataset.labels)
        sample = mo.ui.number(start=1, stop=len(examples), step=1, value=1, label="Dataset line (1-based)")
    compare = mo.ui.run_button(label="Compare")
    mo.hstack([sample, compare])
    return examples, sample, compare


@app.cell(hide_code=True)
def _(checkpoint_a: marimo.ui.dropdown, checkpoint_b: marimo.ui.dropdown, compare: marimo.ui.run_button,
      examples: list[Example], run_dataset: RunDataset, mo: ModuleType, reports: dict[str, Any],
      sample: marimo.ui.number | marimo.ui.text_area, source: marimo.ui.dropdown) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mo.stop(not compare.value, mo.md("Choose a sample and checkpoints, then click **Compare**."))
    _other = read_run_dataset(reports[str(Path(checkpoint_b.value).parent)], mo.notebook_dir().parents[1])
    mo.stop(_other != run_dataset, mo.md("Choose checkpoints from the same dataset and input convention to compare.").callout(kind="warn"))
    if source.value == "Custom text":
        _words = [word for word, _, _ in text_units(str(sample.value or ""), run_dataset.tokenization)]
        _intent, _slots = "—", ["—"] * len(_words)
    else:
        mo.stop(sample.value not in range(1, len(examples) + 1), mo.md("Choose a valid dataset line."))
        _example = examples[int(sample.value) - 1]
        _words, _intent = _example.words, run_dataset.labels.intent_names[_example.intent_id]
        _slots = [run_dataset.labels.slot_names[i] for i in _example.slot_ids]
    mo.stop(not _words, mo.md("Enter some text to compare."))
    slot_rows = [{"word": word, "reference": tag} for word, tag in zip(_words, _slots)]
    intent_rows = []
    for _side, _choice in (("A", checkpoint_a), ("B", checkpoint_b)):
        _path = Path(_choice.value)
        _settings = reports[str(_path.parent)]
        _model, _tokenizer = checkpoint_assets(_path, _path.stat().st_mtime_ns)
        _encoded = _tokenizer([_words], is_split_into_words=True, padding="max_length", truncation=True,
                              max_length=_settings["max_length"], return_tensors="pt")
        with torch.inference_mode():
            _predictions = _model(input_ids=_encoded["input_ids"], attention_mask=_encoded["attention_mask"])
            _probabilities = _predictions.intent_logits.softmax(-1)[0]
            _slot_ids = _predictions.slot_logits.argmax(-1)[0].tolist()
        _intent_id = int(_probabilities.argmax())
        intent_rows.append({"side": _side, "checkpoint": f"{_path.parent.name}/{_path.name}",
                            "step": _settings["best_step"] if _path.stem == "best" else len(_settings["history"]),
                            "intent": _model.labels.intent_names[_intent_id], "confidence": float(_probabilities[_intent_id])})
        for _word_id, _row in enumerate(slot_rows):
            _span = _encoded.word_to_tokens(_word_id)
            _row[_side] = _model.labels.slot_names[_slot_ids[_span.start]] if _span is not None else "not encoded"
    mo.vstack([
        mo.md(f"**Text:** {' '.join(_words)}\n\n**Reference intent:** `{_intent}`"),
        mo.ui.table(intent_rows, selection=None, show_column_summaries=False),
        mo.ui.table(slot_rows, selection=None, page_size=20, show_column_summaries=False),
        mo.md("Slot tags use each word's first WordPiece. **not encoded** marks words absent after tokenization/truncation."),
    ])
    return intent_rows, slot_rows


@app.cell(hide_code=True)
def _(mo: ModuleType, report: dict[str, Any]) -> None:
    mo.vstack([
        mo.md("## 3 · Recorded updates"),
        mo.ui.table(report["history"], selection=None, page_size=10, show_column_summaries=False),
        mo.md(f"""
        **Dev-selected update:** {report['best_step']}. **Test metrics:** `{report['test']}`.

        In `train.py`, follow **`predictions = model(...)` → `losses = compute_loss(...)`
        → `losses.total.backward()` → `optimizer.step()`**. `model.train()` merely enables
        training behavior such as dropout. The scheduler changes the next learning rate.

        The checkpoint stores weights, architecture, and ordered label names. It does not
        store optimizer/RNG state for exact training continuation. The run also saves the
        tokenizer, source snapshot, lockfile, seed, package versions, and data hashes.
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
