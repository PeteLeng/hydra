# Showroom arbiter

The caller supplies a **build directory**. Its `manifest.json` lists the selected
raw attempts and files. Read those inputs together, including each attempt's
generation brief, and apply the agreed annotation specification.

## Your job

1. **Examine every example.** Judge meaning, naturalness, intent, and every slot
   span. Mechanical validation cannot make these decisions.
2. **Retain, correct, or discard.** Recalculate offsets after changing text.
   Judge `slot_hints` in context; they describe the generator's intention, not an
   accepted canonical mapping.
3. **Resolve duplicates and conflicts across all selected attempts.** Keep one
   useful record for repeated text. Conflicting annotations require your judgment.
4. **Assign `group_id` to every retained record**, including singletons. Related
   paraphrases/template substitutions belong together. Shared intent or feature
   alone is not a grouping rule. Preserve family links when resolving duplicates.
5. **Write the final useful records**, preserving each retained `record_id`.

Use `work/` for intermediate files. Choose whatever organization and notes help
the arbitration; no decisions ledger or report format is prescribed.

## Optional Python helpers

`review.py` contains pure helpers with no file access or CLI:

- `validate_record(record)` returns the first `Err(type, description)`, or `None`.
- `validate_records(records)` returns `{"index": ..., "id": ..., "err": Err(...)}`
  entries, one problem per record. It also flags duplicate IDs and exact text.
- `deduplicate(records)` keeps the first identical record payload, ignoring only
  `record_id`. Differences in annotations, hints, or group assignments remain for
  you to resolve. It performs no fuzzy or semantic matching.

Use the existing JSONL helpers in `workspace.py` to read/write records if useful.

## Final output

Write **`review/accepted.jsonl`** inside the build directory: one JSON object per
retained example, containing `record_id`, `text`, `intent`, `slots`, and `group_id`.
Retaining `slot_hints` is optional. Group IDs are local to this build.

This final record list is the only required arbitration output.

## Compile

From the project root, replacing `<timestamp>` with the actual build name:

```bash
# Chinese: preserve original text and character spans as JSONL.
uv --directory notebooks run --locked --offline python m6_showroom_data/build.py "../data/showroom-zh/build/<timestamp>" --format span-jsonl --seed 1234

# English: keep the existing SNIPS export.
uv --directory notebooks run --locked --offline python m6_showroom_data/build.py "../data/showroom-en/build/<timestamp>" --format snips --seed 1234
```

The compiler validates the final records and uses intent-stratified group
partitioning for a fixed approximate 80/10/10 train/dev/test split. At least ten
independent groups are required. Split counts and slot coverage are recorded in
the build manifest. The seed controls only splitting; no CV training is performed.
The span-JSONL export retains the annotated records without creating model tokens
or BIO tags; those will be derived from the model tokenizer during training.
