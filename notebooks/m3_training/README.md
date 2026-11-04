# M3 · A small, standalone joint NLU model

**Open `train.py` in VS Code.** The only resource reused from the JointBERT repo is
its SNIPS **data files**. The Python implementation here uses PyTorch and the cached
pretrained BERT directly. There are no imports from that repo, no `Trainer` class,
and no path injection or separate type-schema module.

All commands below are **Git Bash, from your opened `notebooks/` folder**.

[Verified standalone run, offline checks, and LOC comparison](fresh-run.md).

## Start here

Select a completed timestamped dataset build with `--dataset`. From `notebooks/`,
replace `<timestamp>` with the build you want to train on:

```bash
uv run --locked --offline --env-file local.env python m3_training/train.py \
  --dataset "../data/showroom-en/build/<timestamp>" --steps 1
```

The build manifest resolves its compiled `dataset/` directory. Training records
the build path and manifest hash, and the viewer checks that same reference.
Without `--dataset`, commands use the existing English pilot at
`data/showroom-en/build/20261103T115913654808Z/` (995 train / 126 dev / 125 test).
See the [dataset workflow](../m6_showroom_data/README.md) to create another build.

```bash
# One update: follow optimization and dev evaluation.
uv run --locked --offline --env-file local.env python m3_training/train.py --steps 1

# Short learning run, with dev checks after updates 10, 20, and 30.
uv run --locked --offline --env-file local.env python m3_training/train.py --steps 30 --eval-every 10

# View saved metrics and compare live checkpoint predictions.
uv run --locked --offline --env-file local.env marimo run m3_training/notebook.py
```

Restart an older viewer server with Ctrl+C before launching this version. It lists
the new `fresh-*/run.json` runs. Previous experiments and their PNGs/checkpoints are
still on disk, in their original directories; they use the earlier report format.

**Without `--steps`, the script trains one full epoch.** At batch 8, that means
1,636 updates for 13,084 examples (the last batch has 4 examples).

## Chinese span-JSONL training

Compile a reviewed Chinese build with `--format span-jsonl`. Training selects its
reader and batch adapter from the build manifest:

```text
records.jsonl → read_span_examples() → tokenize_span_batch() → Batch → JointModel
```

The tokenizer receives each original sentence directly. Unicode code-point spans
are aligned using tokenizer offsets: first token `B-type`, remaining span tokens
`I-type`, other text tokens `O`, specials/padding `-100`. Latin continuation pieces
are supervised too. Internal spaces are preserved in the original annotations.
The batch adapter checks spans when the DataLoader requests a batch: boundaries
inside tokens, unrepresentable spans, and overlength inputs raise errors with record ID/line.
`--max-length` includes `[CLS]` and `[SEP]` and must fit the encoder's position limit.

Cache MacBERT once (online setup, from `notebooks/`):

```bash
uv run --locked --env-file local.env python -c "import os; os.environ['HF_HUB_OFFLINE']='0'; from huggingface_hub import snapshot_download; snapshot_download('hfl/chinese-macbert-base', revision='a986e004d2a7f2a1c2f5a3edef4e20604a974ed1', cache_dir='.artifacts/huggingface/hub', allow_patterns=['config.json','pytorch_model.bin','tokenizer_config.json','tokenizer.json','vocab.txt','special_tokens_map.json','added_tokens.json'])"
```

Then train offline:

```bash
uv run --locked --offline --env-file local.env python m3_training/train.py \
  --dataset "../data/showroom-zh/build/<timestamp>" \
  --model hfl/chinese-macbert-base \
  --revision a986e004d2a7f2a1c2f5a3edef4e20604a974ed1 \
  --max-length 128 --steps 1
```

`MODEL`, `REVISION`, and `MAX_LENGTH` at the top of `train.py` supply the defaults.
Override model and revision together for MacBERT, as above. Tokenizer and encoder
use the same revision. The run records model ID, revision,
dataset format, maximum length, and span-file hashes. Chinese support covers
training/evaluation/checkpoints and original-text span decoding in `predict.py`.
Live Chinese notebook comparisons are the next integration step.

## Command-line prediction

From `notebooks/`, use the latest completed run's `best.pt`:

```bash
uv run --locked --offline --env-file local.env python m3_training/predict.py "show me family features inside"
```

Output is JSON with the intent, softmax confidence, and decoded slot text with
start-inclusive/end-exclusive character offsets. Use `--checkpoint PATH` to select
a particular run's `best.pt` or `last.pt`. Python callers can reuse
`predict(text, model, tokenizer, max_length=50)` with already-loaded assets.
For span-trained models, pass `format="span-jsonl"`; prediction uses every text
token's offsets and returns slices of the original input. The CLI selects this
path automatically from `run.json` and uses the saved tokenizer and length limit.

## Source map: four files to follow

| File | Read these definitions |
| --- | --- |
| `snips_data.py` | `Labels`, `Example` / `SpanExample`, `Batch`, `read_examples` / `read_span_examples`, `tokenize_batch` / `tokenize_span_batch` |
| `joint_model.py` | `Predictions`, `Losses`, `JointModel.forward`, `compute_loss`, checkpoint save/load |
| `train.py` | `train`: explicit setup and optimizer loop; `evaluate`: no-gradient forward passes and metrics |
| `notebook.py` | Read saved metrics/plots; select samples and checkpoints for live predictions |

`checks.py` contains small offline behavioral checks, using a tiny random BERT.

The meaningful data objects are ordinary dataclasses. Functions/methods have typed
parameters and returns; local types are inferred unless a hint is useful.
Logging is converted to JSON at the output boundary, rather than passing dictionaries
and positional tensor tuples through the model.

Field comments document shapes: `B` = batch size, `W` = original word count for one
example, and `L` = padded/truncated token length including special tokens. `Labels`
holds `intent_names` / `slot_names`; `Example` holds scalar `line_id` / `intent_id`
and word-aligned `slot_ids [W]`. `Batch` holds `line_ids [B]`, `intent_ids [B]`,
token-aligned `slot_ids [B, L]`, and `word_ids [B, L]` mapping tokens to original
word indices (`None` for specials/padding); span batches have `word_ids=None`. Batch inputs and targets are tensors;
line IDs and word mappings are Python lists. Prediction shapes are
`intent_logits [B, num_intents]` and `slot_logits [B, L, num_slots]`.

```text
seq.in + label + seq.out
          ↓ read_examples()
    list[Example]
          ↓ DataLoader + tokenize_batch()
        Batch
          ↓ model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
     Predictions
          ↓ compute_loss(predictions, intent_ids=..., slot_ids=...)
        Losses
          ↓ losses.total.backward()
   parameter gradients
          ↓ optimizer.step()
    updated parameters
```

## Walk through one run

### [1/5] Data and batching

Read the full supplied train/dev splits, **13,084 / 700** examples. The training
loader shuffles the full training split using a seeded generator. A step limit
stops that loop early: `--steps 30 --batch-size 8` uses 240 examples, not a hidden
256-example subset. Their original 1-based line numbers are saved in `run.json`.

`tokenize_batch()` uses the fast tokenizer's `word_ids()` to align word annotations
to WordPieces. `partial(...)` binds the tokenizer and length once; the DataLoader
then calls that collate function with each list of examples. Only each word's first
subtoken receives a slot target. `[CLS]`,
`[SEP]`, continuation subtokens, and padding have target **-100**, ignored by slot CE.
Attention masking and loss masking are different: continuation subtokens still
participate in BERT's computation. Inputs truncate to 50 tokens including specials.

The supplied label inventories include reserved `UNK` and `PAD` entries. The fresh
model uses the **7 observed intent classes and 72 ordinary slot tags**. `O` is slot
class 0 and is supervised. Input token `[PAD]` ID 0 is a different vocabulary;
**-100 is a loss sentinel, not an output class**. All supplied split labels have
been checked against the filtered inventories.

### [2/5] Model and optimizer

`train.py` explicitly loads the cached tokenizer and `BertModel`, then constructs
`JointModel(encoder, labels)`. BERT's weights are pretrained; `nn.Linear` initializes
our two new heads. Every model parameter is given to AdamW.

```text
BERT last_hidden_state [B,L,768] → dropout → Linear(768,72) → slot_logits
BERT pooler_output     [B,768]   → dropout → Linear(768,7)  → intent_logits
```

The pooler is dense+tanh on the final `[CLS]` state. The heads return raw logits.
`compute_loss()` separately calculates intent CE + slot CE, with slot weight 1.
Cross-entropy includes log-softmax internally. The model's forward method takes
**no target labels** and returns a named `Predictions` object.

### [3/5] Updates and dev evaluation

Read the `for step, batch in enumerate(train_loader, start=1)` loop:

```python
model.train()  # Set mode flags, enabling dropout; this does not run training.
optimizer.zero_grad(set_to_none=True)
predictions = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
losses = compute_loss(predictions, intent_ids=batch.intent_ids, slot_ids=batch.slot_ids)
losses.total.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

Update losses and dev metrics are printed together every `eval_every` updates
and at the final update. Runs with fewer than 30 total updates also print the
training losses after every update.

`model(...)` calls the existing object's `nn.Module.__call__` machinery, which
dispatches to our `forward()` (like C++ `operator()`). There is no returned-tuple
slicing or keyword-dictionary unpacking at this call site.

AdamW updates the same parameter objects whose `.grad` fields autograd populated.
The PyTorch `LinearLR` scheduler adjusts the optimizer's next learning rate from
5e-5 down to zero across the requested updates. The final update uses a positive
rate; the scheduler sets zero afterward. Weight decay is zero, seed 1234.

`evaluate()` sets eval mode, disables gradient recording, and uses the **full dev
split**. It reports intent accuracy, exact slot-span F1, and exact-frame accuracy.
Its loss is example-weighted intent CE + supervised-token-weighted slot CE, so the
last partial batch does not distort the mean. Empty predicted entity sets receive
F1=0 via `zero_division=0`. The lowest dev loss selects `best.pt`, including the
initial step-0 model as a candidate. The next training batch restores train mode.

### [4/5] Checkpoints and inference

- `last.pt`: the final model; `best.pt`: the dev-selected model.
- Each contains the encoder configuration, complete model state dict, ordered
  label names, and head dropout setting.
- `tokenizer/`: the matching tokenizer files for text inference.

Each run retains these two checkpoints. A better dev loss overwrites `best.pt`;
`last.pt` is saved after the final update. Their step numbers appear in the notebook.

`load_checkpoint()` reconstructs BERT **from the saved config**, loads all weights,
and sets eval mode. It makes no pretrained-model request. This is an inference
checkpoint, not an exact training-resume snapshot of optimizer/RNG state.

### [5/5] Results and audit files

Each new run lives in `.artifacts/m3_training/fresh-<timestamp>/`:

- `loss.png`: training-batch loss (dropout on) and full-dev loss (dropout off).
- `run.json`: settings, consumed training line IDs, losses/metrics, selected update,
  environment versions, and data hashes.
- `source/`: copies of the three training/model/data files and `uv.lock` for audit.

The notebook embeds the saved PNG and reads metric tables from JSON. Under
**Compare live predictions**, choose checkpoints A/B from any completed standalone
run, select a train/dev/test split and 1-based line number (or **Custom text**),
then click **Compare**. Existing `fresh-*` runs work with their saved best/final files.

Predictions are computed on demand using each run's tokenizer, maximum length, and
checkpoint label names. CPU inference uses eval mode and `torch.inference_mode()`;
up to two models/tokenizers are cached by checkpoint path and modification time.
Changing controls requires another click on **Compare**. The tables show intent
confidence and word-aligned slot tags beside available reference labels. A word
with no retained tokens is marked **not encoded**. Metrics and displayed slot tags
use first-subtoken positions under each run's truncation setting.

## Full-epoch training and test evaluation

Once settings are selected using dev data:

```bash
uv run --locked --offline --env-file local.env python m3_training/train.py \
  --batch-size 16 --eval-every 200 --test
```

This runs 818 updates on all training examples, selects the best measured dev
checkpoint, then evaluates that choice once on the 700 test examples. `--test` is
optional; test is not used in optimization or checkpoint selection.

## Offline setup and editor

Use the existing Python 3.12 environment in `notebooks/.venv/`. Both pretrained
loads specify **`local_files_only=True`**, the project-local Hugging Face cache,
and the selected revision (English default: `86b5e0934494bd15c9632b12f734a8a67f723594`). Missing cache files cause an
immediate error instead of HTTP retries. `local.env` also enables `HF_HUB_OFFLINE=1`.

The required snapshot is in
`.artifacts/huggingface/hub/models--bert-base-uncased/snapshots/<revision>/`.
For a new machine, explicit initial downloads and the dependency environment are
covered in [the environment guide](../README.md). Normal training is offline.

Open **`notebooks/`** in VS Code and select `.venv/Scripts/python.exe`.
The fresh code needs no JointBERT `extraPaths` setting. See the environment guide
for manual F12/F11 configuration; no editor settings are changed automatically.
Set breakpoints on the forward call, `losses.total.backward()`, and `optimizer.step()`.
Inspect `batch`, `predictions`, `losses`, and `model.intent_head.weight.grad`.

## Checks

```bash
uv run --locked --offline --env-file local.env python m3_training/checks.py
uv run --locked --offline --env-file local.env python -m unittest discover -s m3_training -p 'test_*.py' -v
uv run --locked --offline marimo check m3_training/notebook.py
```

These cover reading/alignment, ignored slot gradients, both loss paths into BERT,
parameter updates, checkpoint reconstruction, and evaluation across unequal batches.
They use tiny local fixtures, not downloaded weights. The former repo-adapter run
record is [historical](verified-run.md); its metrics belong to the older implementation.
