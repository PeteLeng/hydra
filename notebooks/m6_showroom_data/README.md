# Showroom generation → review → compilation

Each language has a dataset root, e.g. `data/showroom-en/` or `data/showroom-zh/`.
Raw attempts are reusable. Each build selects raw attempts and owns its review,
grouping, and compiled train/dev/test dataset. `dev` means validation.

```text
data/showroom-en/
├── raw/<timestamp>-<job>/
│   ├── manifest.json
│   ├── gen.md                 # Instruction snapshot
│   ├── generation.log         # Launcher output
│   └── part-01.jsonl ...
└── build/<timestamp>/
    ├── manifest.json
    ├── review/                # arbiter.md, accepted.jsonl; any additional notes are optional
    ├── work/                  # Arbiter intermediates
    └── dataset/               # train/, dev/, test/, label inventories
```

## Record contract

- `record_id`: `<raw-attempt-name>-000001`, assigned from the launcher's prefix.
  Keep it through corrections, rejection, deduplication, and export. It is unique
  within a dataset root and independent of chunk filenames/line positions.
- `text`, `intent`, `slots`: original text and Unicode code-point spans `[start,end)`.
- `slot_hints`: optional generator hints of the form `{"slot":0,"value":"cabin_cinema"}`.
  These describe intended canonical values for review; they are not model targets.
- `group_id`: required on accepted records, local to the build. All members stay
  in the same split. There is no separate accepted-record ID.

The shared contract is the nine-intent/five-slot showroom pilot in `schema.py`.
An utterance can contain multiple different slot types, at most one of each.
Both languages annotate Unicode code-point spans in the original text.

## 1. Generate

All commands here run from the **project root in Git Bash**:

```bash
bash notebooks/m6_showroom_data/generate.sh showroom-en

# Simplified Chinese, Mandarin, conversational wording.
bash notebooks/m6_showroom_data/generate.sh showroom-zh notebooks/m6_showroom_data/gen-zh.md
```

Model choices remain in `generate.sh`. One launcher execution starts multiple
independent agents, each with its own raw directory, instruction snapshot, record
prefix, and log. Each agent is asked for 500 examples. There are no generation
seeds. The second argument selects the brief (relative to the project root, or an
absolute path); English `gen.md` is the default. Each attempt snapshots the selected
brief as `gen.md`. Changing only the dataset name does not change the language.

For a manually launched agent, allocate its directory using `workspace.py raw`,
pass it the printed path and that directory's name as its record prefix.
`workspace.py raw --help` lists the model, brief, count, and job arguments.

## 2. Create a build and arbitrate

Select the raw attempts to review, replacing the placeholders with their names:

```bash
BUILD=$(uv --directory notebooks run --locked --offline python m6_showroom_data/workspace.py build \
  --dataset ../data/showroom-zh \
  --raw "../data/showroom-zh/raw/<attempt-A>" \
        "../data/showroom-zh/raw/<attempt-B>")
```

The command prints the new absolute build path. Give an arbiter that directory
and its instruction snapshot, for example:

```bash
opencode run --file "$BUILD/review/arbiter.md" "Follow arbiter.md for build directory $BUILD."
```

The [arbiter brief](arbiter.md) requires the agent to examine every selected
example, retain/correct/discard it, resolve duplicates and conflicts, and assign
groups. Its only required output is `review/accepted.jsonl`: the final useful
records with their original `record_id` and assigned `group_id`. The arbiter
chooses how to organize working files and any additional explanation.

`review.py` is a helper module, not a review workflow or CLI:

- `validate_record(record) -> Err | None`: first annotation error, or `None`.
- `validate_records(records) -> list[dict]`: one problem per record, in the form
  `{"index": 7, "id": "example-000008", "err": Err(ErrType.INVALID_SPAN, "...")}`.
  IDs and exact duplicate text are checked across the list. Record context stays
  in the dictionary, separate from the error's type and description.
- `deduplicate(records) -> list[Record]`: retain the first identical payload,
  ignoring only `record_id`; preserve conflicting annotations, hints, or groups.

These helpers neither read/write files nor make semantic judgments. Existing
JSONL helpers in `workspace.py` are available for the arbiter's own scripts.

Builds start as `draft` and become `complete` after compilation. Revised review
decisions belong in a new build.

## 3. Compile

After the arbiter has written `review/accepted.jsonl`:

```bash
uv --directory notebooks run --locked --offline python m6_showroom_data/build.py "$BUILD" --format span-jsonl --seed 1234
```

This seed controls **group allocation only**. Compilation reads the final record
list directly, validates it, requires group IDs, and writes the dataset. It does
not require a decisions ledger or review report.

- **Chinese: `--format span-jsonl`** writes `dataset/{train,dev,test}/records.jsonl`
  with original text, character spans, intent, record/group IDs, and any slot hints.
  No tokenization or BIO conversion occurs; `tokenization` is `null` in the manifest.
- **English: `--format snips` (default)** retains the existing `seq.in`, `seq.out`,
  and `label` files. Its input-unit convention remains `showroom`.

Both formats include label inventories and record the selected `format` in the
manifest. The split algorithm and coverage counts are shared.

`StratifiedGroupKFold` balances **intents** across ten group-preserving partitions.
Eight become train, one dev, one test: approximately 80/10/10 by record count.
At least ten independent groups are required. This uses partitioning only, not
cross-validation training. Slots do not affect allocation.

The manifest reports actual sizes/fractions, intent and slot counts, missing
coverage, and row-to-record mappings, along with labels, tokenization, and hashes.
Inspect that report before deciding whether slot-aware allocation is needed.

## 4. Train from a build

For an **English SNIPS build**:

```bash
uv --directory notebooks run --locked --offline --env-file local.env \
  python m3_training/train.py --dataset ../data/showroom-en/build/20261103T115913654808Z --steps 1
```

Training resolves `dataset/` through the completed build manifest. Its run report
records the build path and manifest hash; the results viewer resolves that same
build.

Chinese span-JSONL builds also feed the trainer through original-text tokenizer
offsets. See [Chinese training and MacBERT setup](../m3_training/README.md#chinese-span-jsonl-training).
Chinese prediction decoding is supported; live viewer comparisons are a subsequent task.

## Existing English data

The original English pilot is now at
[`data/showroom-en/build/20261103T115913654808Z/`](../../data/showroom-en/build/20261103T115913654808Z/).
The existing compiled files and split membership were preserved byte-for-byte.
See its [dataset guide](../../data/showroom-en/README.md) for retained review material.

## Source map and checks

| Module | Responsibility |
| --- | --- |
| `schema.py` | Shared record types and showroom vocabulary |
| `workspace.py` | Raw/build initialization and small JSON/hash helpers |
| `review.py` | Pure record validation, typed errors, and exact deduplication |
| `build.py` | Intent-stratified group splitting, span-to-BIO conversion, export |
| `test_pipeline.py` | Unit and local workflow tests |

```bash
uv --directory notebooks run --locked --offline python m6_showroom_data/test_pipeline.py
uv --directory notebooks run --locked --offline --env-file local.env \
  python -m unittest discover -s m3_training -p 'test_*.py' -v
```
