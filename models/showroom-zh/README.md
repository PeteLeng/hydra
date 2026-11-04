# Chinese showroom model

Portable inference bundle for the joint intent/slot model. `best.pt` is stored
with Git LFS; the matching run metadata and tokenizer are ordinary Git files.

## Run on another PC

Install Git LFS, uv, and Python **3.12**. Run these commands from the repository
root after cloning (or pulling the commit containing this bundle):

```bash
git lfs install
git lfs pull
uv --directory notebooks sync --locked
uv --directory notebooks run --locked python -X utf8 m3_training/predict.py \
  "我想看车内的DLP大灯。" \
  --checkpoint "../models/showroom-zh/best.pt"
```

If Git LFS was installed before cloning, the checkpoint normally downloads during
checkout; `git lfs pull` also handles existing clones or skipped downloads.
If `best.pt` is only a few lines starting with `version https://git-lfs.github.com/spec/v1`,
it is still a pointer: run `git lfs pull` to fetch the actual weights.

The prediction CLI runs on CPU and loads all model/tokenizer assets locally.
No GPU, training dataset, Hugging Face cache download, or `.env` file is needed.
`-X utf8` ensures Chinese JSON output works with Windows terminals and redirected
output even when the system's default encoding cannot represent Chinese.
The explicit checkpoint path selects this bundle; automatic discovery searches
only `notebooks/.artifacts/m3_training/fresh-*` runs.

## Contents and provenance

- `best.pt`: encoder and intent/slot head weights, encoder configuration, labels,
  and dropout setting. An inference snapshot, without optimizer/RNG state.
- `run.json`: original training report, including span-JSONL format, input limit,
  model revision, dataset/source hashes, and evaluation results.
- `tokenizer/`: all four tokenizer files exported by the same training run.

Source: `notebooks/.artifacts/m3_training/fresh-20261103-063719-296336/`.
Encoder: `hfl/chinese-macbert-base`, revision
`a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`.
The run trained for five epochs (670 updates); lowest dev loss selected **step 200**.
Maximum input length is **128 tokens**, including special tokens.

Recorded test results for the selected checkpoint:

| Metric | Value |
| --- | --- |
| Intent accuracy | 98.51% |
| Slot F1 | 92.82% |
| Exact-frame accuracy | 92.54% |

These are results on the run's recorded showroom test split; full values and
dataset identity are in `run.json`. Its original dataset path is provenance,
not a dependency of the prediction command.

## Publish an updated model

Replace `best.pt`, `run.json`, and the entire `tokenizer/` directory together from
one completed Chinese run. Update the provenance above, run the prediction
command, and commit the bundle. `.gitattributes` routes model `.pt` files through
Git LFS. With Git LFS installed, a normal `git push` uploads the checkpoint through
the LFS pre-push hook as well as publishing the Git commit.
