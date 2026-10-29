# Verified local run · 2026-10-28

**Historical record:** this describes the earlier repo-based implementation with
8/74 outputs. The current standalone implementation uses 7/72 real label classes,
reads full splits, and writes `fresh-*/run.json`. See `README.md` for current code
and commands; these older metrics do not describe the fresh model.

Commands are in [README.md](README.md). This is a measured first-run record for
M3, using the supplied SNIPS splits and CRF disabled.

This record predates moving the teaching loop into plain `train.py`. Its saved
artifacts remain readable in the current results-only notebook. See `README.md`
for the current commands; training is now launched separately from the viewer.

## Environment

| Component | Verified value |
| --- | --- |
| Interpreter | `notebooks/.venv/Scripts/python.exe`, Python 3.12.13 |
| PyTorch | 2.7.1+cu128 |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Transformers | 4.57.6 |
| NumPy / seqeval | 2.5.4 / 1.2.2 |
| Pretrained encoder | `bert-base-uncased` |
| Pretrained revision | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| JointBERT base | `00324f60dda4a491b2913bc3065ae93966caea3f` + local compatibility changes |

The CUDA wheel was downloaded with resumable curl requests, verified against the
SHA-256 in `uv.lock`, and installed by name from the local wheel directory.
`uv sync --locked --offline` and `uv pip check` passed. A CUDA matrix multiplication
and backward pass passed. No global package installation was used.

## Interactive training-loop experiment

Run: `notebooks/.artifacts/m3_training/20261028-013952-052144/`.

256 train / 64 dev examples, batch 8, 30 updates, seed 1234, learning rate 5e-5,
linear decay, λ=1, maximum sequence length 50. Every BERT/head parameter is trainable.

| Update | Fixed train-subset loss | Fixed dev-subset loss |
| ---: | ---: | ---: |
| 0 | 6.321254 | 6.301793 |
| 10 | 4.690740 | 4.733437 |
| 20 | 4.382598 | 4.582838 |
| 30 | 4.137042 | 4.287172 |

Both task losses reached the inspected shared BERT parameter: gradient norms
0.076899 (intent) and 0.168263 (slot) on the first batch. The head-weight slice
changed after the optimizer update. Run inputs, original dataset line numbers,
settings, environment/data/source hashes, code diff, traces, and `loss.png` are saved.

Reloaded the final checkpoint in a **fresh Python process**, using the saved fixture:

- Intent logits maximum absolute difference: **0.0**.
- Slot logits maximum absolute difference: **0.0**.
- Decoded predictions: **identical**.

The notebook also executes successfully with its run button untouched, reading the
saved report rather than training again. Initial random heads produced PAD/UNK
slot predictions, and some early batches predicted no entities; seqeval reported
those conditions while the losses and learning loop remained valid.

## Actual repository CLI workflow

Run: `notebooks/.artifacts/m3_training/cli-one-epoch/`.

One epoch on **13,084 training examples**, batch 16, **818 optimizer updates**.
Evaluated 700 dev examples at updates 200/400/600/800/818. Selected update **818**
using the lowest dev loss (**0.194506**), saved under `best/`.
Training, test evaluation, and prediction ran with `HF_HUB_OFFLINE=1` after the
initial model download, using the cached local assets.

Then evaluated that selected model on the **700 test examples**:

| Test metric | Value |
| --- | ---: |
| Intent accuracy | **97.86%** |
| Slot span F1 | **93.76%** |
| Exact-frame accuracy | **85.14%** |
| Joint loss | 0.188054 |

These are a one-epoch workflow result, not the final M5 benchmark. Slot/frame
metrics use the retained first-subtoken positions under the repo's length-50
preprocessing. Full-precision values and per-example outputs are in
`best/test_metrics.json` and `best/test_predictions.jsonl`.

`predict.py` produced this correct output for the familiar M1 example:

```text
<BookRestaurant> -> book a spot for [3:B-party_size_number] in [mt:B-state]
```

The PlayMusic example's intent was correct, but it incorrectly extended the artist
span across `alumb allergic`. Gold annotations have `alumb=O` and
`allergic=B-album`. See `predictions.txt` for both outputs: a useful first error
to inspect rather than assuming one epoch solves every example.

## Checks passed

- Both milestone notebooks: `marimo check`.
- M1: full cell execution after reorganization.
- M3: actual training plus saved-report-only cell execution.
- Three offline behavioral regressions: ignored slot gradients/shared loss paths;
  partial-accumulation short-run save/reload/prediction; Windows local-path feature caching.
- Fresh-process checkpoint parity; real CLI training, test evaluation, and prediction.
- Parent/nested Git whitespace checks and installed dependency compatibility check.

## Plain-script / results-viewer refactor verification

Training now runs from `m3_training/train.py`, independently of marimo.

- `train.py --steps 1`: completed one update on 8 examples; saved and reloaded the
  model. Run: `.artifacts/m3_training/20261028-063643-896904/` under `notebooks/`.
- `train.py`: completed 30 updates on 240 examples. Run:
  `.artifacts/m3_training/20261028-063947-913405/`. Its measured data summary records
  13,084 / 700 original examples, 256 / 64 selected examples, and 32 available batches.
- Fresh-process verification of the new 30-step checkpoint again found maximum
  absolute logit differences **0.0 / 0.0** and identical decoded predictions.
- All three offline behavioral regression checks passed after the refactor.
- `marimo check` passed. Executed the viewer against both the user's older
  `20261028-055408-741780` report and the new standalone-script report.
- Verified that the graph cell emits HTML with an embedded PNG, and that executing
  the viewer does **not** import `torch`, `helpers`, or `train`.

The old graph cell ended with `plt.close(...)`, so its marimo output was `None`.
The results-only viewer now returns its display as the final expression and reads
the saved `loss.png` directly. The original report/checkpoint directories are retained.
