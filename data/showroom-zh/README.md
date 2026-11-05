# Chinese showroom NLU dataset

Completed build: [`build/20261103T140305458654Z/`](build/20261103T140305458654Z/).

Simplified Chinese, conversational Mandarin, with nine intents and five slot
types. This build reviews all three generation attempts together and exports
original-text **span JSONL**, as required by the Chinese annotation brief.

## Training, validation, and test files

Paths below are relative to the build directory. `dev` is the validation split.

| Split | Records | Share | Independent groups | File |
| --- | ---: | ---: | ---: | --- |
| Train | 1,065 | 77.74% | 214 | `dataset/train/records.jsonl` |
| Validation (`dev`) | 171 | 12.48% | 35 | `dataset/dev/records.jsonl` |
| Test | 134 | 9.78% | 27 | `dataset/test/records.jsonl` |
| Total | **1,370** | **100%** | **276** | `review/accepted.jsonl` |

Each record contains `record_id`, `text`, `intent`, `slots`, and `group_id`.
Offsets are zero-based **Unicode code points**, with exclusive ends:
`text[start:end]` is the annotated surface. Chinese text is not segmented or
space-padded. Tokenizer-specific tokens and BIO targets are derived during
training. The compiler records `format: "span-jsonl"` and `tokenization: null`.

## Arbitration

Every one of the 1,500 raw examples was read with its literal slot slices and
optional generator hints, against the Chinese brief snapshotted in each raw
attempt. Text, intent, naturalness, slot boundaries, aliases, duplicates, and
cross-attempt families were reviewed together.

| Raw attempt | Raw records | Retained record IDs |
| --- | ---: | ---: |
| `20261103T133326307652Z-01` | 500 | 500 |
| `20261103T133326307652Z-02` | 500 | 480 |
| `20261103T133326307652Z-03` | 500 | 390 |

- **32 records corrected before deduplication**; 31 corrected records survive in
  the final list. Changes include four viewing requests mislabeled as `inform`,
  clearer resume wording, and more natural Chinese phrasing. Changed texts have
  explicitly recalculated character spans.
- **121 redundant records removed**: 114 exact repeated texts and seven
  punctuation-only variants. Earliest retained IDs are used consistently;
  duplicate-source family links are preserved.
- **Nine records discarded** for ambiguity, unnatural wording, or a distinctive
  nonsense sentence copied verbatim from the generation brief.
- Generator hints were judged in context. Clear catalog aliases were retained;
  hints themselves are omitted from model records because they are not canonical
  value targets. No implied catalog part/theme attributes were added.
- **276 groups** keep related paraphrases and template substitutions together
  across all three attempts. Singleton examples also have a group ID.

The compiler used the prescribed `StratifiedGroupKFold` ten-part allocation and
**seed 1234**, with eight parts for training, one for validation, and one for test.
Groups were fixed before the final partition; preserving families explains the
departure from exactly 80/10/10.

## Coverage and checks

All three splits contain **all nine intents and all five slot types**. Each also
contains all four catalog features, both vehicle parts, Chinese-number choices,
and Arabic-digit choices. Training and test cover choices 1–10 and all three
theme values.

The fixed intent-stratified split has these actual coverage limitations:

- Validation contains four theme annotations (`privacy`: 3, `entertainment`: 1),
  **no `family` theme**, and no choice value 1.
- One 56-example greeting/self-introduction family falls wholly in validation.
  Consequently, greeting counts are train 45 / validation 56 / test 2, and name
  span counts are 67 / 68 / 3. Fine-grained metrics for these small or uneven
  subsets should be read together with their support counts.

Verification passed for source/review/export hashes, complete source accounting,
record contract and literal corrected spans, original-ID preservation, explicit
family links, and reproducibility of the seed-1234 partition. There is **zero
record/group overlap between splits**, and no exact or punctuation-only duplicate
text in the accepted set. The existing Chinese training reader successfully
loaded all 1,065 / 171 / 134 examples through the completed build manifest.

Build-local supporting material:

- `manifest.json`: source hashes, completed status, split/label counts, export hashes.
- `work/decisions.json`: reviewed ranges, corrections, rejections, semantic notes.
- `work/corrections.jsonl` and `work/duplicates.jsonl`: concrete before/after and ID mappings.
- `work/families.json`: explicit paraphrase links and their reasons.
- `work/families.txt`: final group membership.
- `work/verification.json`: coverage by intent, slot, canonical value, choice, and source attempt.
- `work/verify_build.py`: rerunnable integrity and isolation audit.

## Use with the Chinese trainer

Pass the **build directory**, allowing the trainer to resolve the manifest and
select its span-JSONL reader. After caching the pinned model as described in the
[Chinese training guide](../../notebooks/m3_training/README.md#chinese-span-jsonl-training),
this one-step smoke-training command runs from the project root:

```bash
uv --directory notebooks run --locked --offline --env-file local.env python m3_training/train.py \
  --dataset "../data/showroom-zh/build/20261103T140305458654Z" \
  --model hfl/chinese-macbert-base \
  --revision a986e004d2a7f2a1c2f5a3edef4e20604a974ed1 \
  --max-length 128 --steps 1
```

To rerun this build's data audit from the project root:

```bash
uv --directory notebooks run --locked --offline python \
  ../data/showroom-zh/build/20261103T140305458654Z/work/verify_build.py
```
