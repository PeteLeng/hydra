# NLU learning environment

Open **`notebooks/`** in VS Code. Its `.venv/` is the Python 3.12 environment for
the milestones; there is no parent Python project at `nlu/`.

| Milestone | Guide | Executable code |
| --- | --- | --- |
| M3 · Training | [Standalone joint NLU model](m3_training/README.md) | `m3_training/train.py` |

M3 imports only its local data/model modules and installed libraries. The
`repos/JointBERT/data/snips/` directory supplies data; its Python trainer is not
used. M3's `notebook.py` displays saved metrics and compares checkpoint predictions
on demand for selected dataset examples or custom text.

## Set up another PC

Install [Git](https://git-scm.com/downloads),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and **Python 3.12**.
Python downloads are disabled in `pyproject.toml`, so Python 3.12 must already be
installed when you run `uv sync`.

Clone this project into your chosen parent directory:

```bash
git clone https://github.com/PeteLeng/hydra.git nlu
```

Copy your manually downloaded JointBERT folder into `nlu/repos/JointBERT/`.
For a fresh download, use the
[JointBERT snapshot used by this project](https://github.com/monologg/JointBERT/archive/00324f60dda4a491b2913bc3065ae93966caea3f.zip)
and rename the extracted folder to `JointBERT`. No submodule setup is required.
The resulting layout must be:

```text
nlu/
  notebooks/
    pyproject.toml
    uv.lock
  repos/
    JointBERT/
      data/
        snips/
          intent_label.txt
          slot_label.txt
          train/              # seq.in, seq.out, label
          dev/                # seq.in, seq.out, label
          test/               # seq.in, seq.out, label
```

All remaining commands are **Git Bash, from `nlu/notebooks/`**. Install the locked
dependencies, including the marimo viewer:

```bash
uv sync --locked
```

If your terminal is instead at `nlu/`, use `uv --directory notebooks ...`.

Packages live in `.venv/`, package downloads/builds in `.uv-cache/`, and model/plot
assets in `.artifacts/`. These generated directories are Git-ignored. `uv.lock`
pins the environment; `.python-version` selects Python 3.12. `local.env` contains
relative cache paths and enables Hugging Face offline mode.

On Windows, the lock selects the CUDA 12.8 PyTorch wheel (about 3 GB). GPU use needs
a compatible NVIDIA driver; training selects CPU when CUDA is unavailable.

### Initial BERT download on a new machine

After `uv sync`, download the pinned BERT snapshot once. This is an **online setup
step** for an empty cache, writing under `.artifacts/`:

```bash
uv run --locked --env-file local.env python -c "import os; os.environ['HF_HUB_OFFLINE']='0'; from huggingface_hub import snapshot_download; snapshot_download('bert-base-uncased', revision='86b5e0934494bd15c9632b12f734a8a67f723594', cache_dir='.artifacts/huggingface/hub', allow_patterns=['config.json','model.safetensors','tokenizer_config.json','tokenizer.json','vocab.txt'])"
```

Normal M3 code uses `local_files_only=True`; it never refreshes remote model metadata.

### Verify and run

```bash
# Small offline checks using a tiny model and local fixtures.
uv run --locked --offline --env-file local.env python m3_training/checks.py

# One training update using the SNIPS data and cached pretrained BERT.
uv run --locked --offline --env-file local.env python m3_training/train.py --steps 1

# View the saved run.
uv run --locked --offline --env-file local.env marimo run m3_training/notebook.py
```

Stop a viewer with Ctrl+C before restarting its updated source. Use `marimo edit
--watch` to edit UI cells; study/debug M3 training directly in `train.py`.

### Carry over existing results

Copy `notebooks/.artifacts/m3_training/` to preserve your saved runs, checkpoints,
and tokenizers. You can also copy the complete `notebooks/.artifacts/huggingface/`
cache to reuse the downloaded BERT snapshot. Recreate `.venv/` with `uv sync --locked`
on the new PC; virtual environments contain machine-specific paths.

### Recovery for an interrupted CUDA PyTorch download

The locked Windows wheel is about 3 GB. If uv repeatedly restarts it, stop uv first.
This curl command can be rerun to resume from the saved file size:

```bash
mkdir -p .uv-cache/downloads
curl --fail --location --continue-at - --progress-bar \
  --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
  --output ".uv-cache/downloads/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl" \
  "https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl"
sha256sum ".uv-cache/downloads/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl"
```

Expected SHA-256 (also in `uv.lock`):
`2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30`.
After it matches, install by **name/version from the local wheel directory**:

```bash
uv pip install --python .venv/Scripts/python.exe --no-deps --no-index \
  --find-links .uv-cache/downloads --reinstall "torch==2.7.1+cu128"
uv sync --locked
```

A direct file-path installation records a different source and can make sync try
the registry again. The name/version form works with the existing lockfile.

## VS Code: manual setup

These are instructions for you to apply; no settings or extensions are installed
automatically. For the **`notebooks/` folder**:

1. Ensure Python and Pylance extensions are enabled.
2. **Ctrl+Shift+P → Python: Select Interpreter → Enter interpreter path**:
   select `.venv/Scripts/python.exe` under your local `notebooks/` directory.
3. **Developer: Reload Window** to refresh indexing. Explicit selection overrides
   an older remembered interpreter; `uv run` does not select Pylance's interpreter.

Optional `.vscode/settings.json`, written manually inside `notebooks/`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.analysis.useLibraryCodeForTypes": true
}
```

Try F12 on `BertModel` in `m3_training/joint_model.py`. It imports the concrete
module directly. Its source is `.venv/Lib/site-packages/transformers/models/bert/modeling_bert.py`;
PyTorch's Linear is in `.venv/Lib/site-packages/torch/nn/modules/linear.py`.

### F11 debugger stepping

Enable Python Debugger. You can manually create `notebooks/.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "M3: inspect one update",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/m3_training/train.py",
      "python": "${workspaceFolder}/.venv/Scripts/python.exe",
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/local.env",
      "args": ["--steps", "1"],
      "console": "integratedTerminal",
      "justMyCode": false,
      "stopOnEntry": true
    }
  ]
}
```

F12 is static navigation; `justMyCode: false` controls F11 debugger stepping into
libraries. Put breakpoints at `JointModel.forward`, `compute_loss`, and
`optimizer.step()`. F5 continues, F10 steps over, F11 steps into.

If you separately open the reference JointBERT folder, explicitly select the same
notebook interpreter for that folder/window too. Multi-root workspaces remember
interpreter selections per folder. `extraPaths` for that legacy repo is useful only
when studying its own unpackaged modules, not for the new M3 code.
