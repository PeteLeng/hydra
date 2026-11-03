# English showroom data

Reusable generation attempts are under `raw/<timestamp>-<job>/`; each dataset is
under `build/<timestamp>/`. See the [pipeline guide](../../notebooks/m6_showroom_data/README.md).

## Existing pilot

[`build/20261103T115913654808Z/`](build/20261103T115913654808Z/) contains the relocated
English pilot: **1,500 raw / 1,246 accepted records; 995 train / 126 dev / 125 test**.
All 11 compiled SNIPS files are byte-identical to the former `data/showroom/` files.
Split membership, row order, text, labels, spans, and the 315 groups were retained.
The build timestamp records relocation; the original review was on 2026-10-30.
This build retains the original allocation strategy, not the new scikit-learn split.

- `raw/*/part-*.jsonl`: original examples with stable `record_id` added and
  `alias_proposals` renamed to `slot_hints`. Original raw hashes are in each raw
  manifest. The brief snapshot comes from repository revision `9afe429`; a separate
  per-agent instruction snapshot was not recorded at generation time.
- `review/accepted.jsonl`: accepted annotations with the same stable IDs and groups.
- `dataset/`: existing train/dev/test exports and label inventories.
- `manifest.json`: file hashes, input references, and exported row-to-record mapping.

The original arbiter's `review_notes.md`, `review_groups.json`, `review_report.json`,
and `review_decisions.jsonl` are retained under the build's `review/` directory.
They are historical material, not required formats for future reviews. Old source
paths/hashes refer to the original layout/bytes; the decisions now also carry
`record_id`, connecting those references to the relocated raw and accepted records.

The English trainer's default and the viewer's old-run lookup now point to this
build. To select it explicitly, from the project root:

```bash
uv --directory notebooks run --locked --offline --env-file local.env python m3_training/train.py \
  --dataset ../data/showroom-en/build/20261103T115913654808Z --steps 1
```
