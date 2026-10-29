# Standalone M3 verification · 2026-10-29

Run: `.artifacts/m3_training/fresh-20261029-042225-850185/`, relative to `notebooks/`.

## Measured size

Physical Python lines, including comments, blank lines, and checks; excluding
documentation and installed libraries in both counts:

| File | Lines |
| --- | ---: |
| `snips_data.py` | 91 |
| `joint_model.py` | 69 |
| `train.py` | 251 |
| `notebook.py` | 126 |
| `checks.py` | 98 |
| **Total** | **635** |

Previous M3 total: **1,067**. Reduction: **432 lines (40.5%)**. The old integration
helper and type-schema file have been replaced by the standalone data/model code.

## Execution checked

- PyTorch 2.7.1+cu128, Transformers 4.57.6, RTX 5070 Ti Laptop GPU.
- All supplied splits read successfully: 13,084 train / 700 dev / 700 test.
- Output classes: 7 intents / 72 slots. Ignored slot targets use -100.
- Actual CUDA run: 30 updates, batch 8, 240 unique training examples from the shuffled
  full split, dev evaluation every 10 updates, seed 1234, learning rate 5e-5.
- HTTP requests were blocked during the run; the request counter remained **zero**.
- Runtime module inspection found **zero imports of JointBERT Python source**.
- Fresh-process `train.py --verify <run>` passed: maximum intent/slot logit
  differences **0.0 / 0.0**, with identical argmax predictions.
- Viewer execution produced an embedded PNG and imported no training/model modules.
- Three offline behavioral checks passed: reader/alignment, gradients/update/reload,
  and evaluation mode/unequal-batch weighting. Notebook and syntax checks passed.
- Function/method annotation inspection passed (24 inspected functions, including
  dataclass-generated constructors).

## Short-run learning curve

| Update | Full-dev joint loss |
| ---: | ---: |
| 0 | 6.312244 |
| 10 | 4.488538 |
| 20 | 4.187109 |
| 30 | 4.071125 |

Dev loss selected update 30. The optional test path was exercised afterward:
intent accuracy **36.57%**, slot F1 **0.00%**, frame accuracy **0.00%**, joint loss
**4.121146**. This is a short mechanics run using 240 training examples; quality
training uses the full-epoch command in `README.md`. Raw values, source snapshots,
data hashes, and the before/after example trace are in the run directory.
