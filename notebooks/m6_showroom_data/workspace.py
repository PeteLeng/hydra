"""Initialize raw/build directories, with small shared JSON and hashing helpers."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def hashes(root: Path, files: list[Path]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def verify_hashes(root: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed input: {root / name}")


def create_raw(dataset: Path, model: str, variant: str = "", count: int = 500,
               brief: Path = HERE / "gen.md", stamp: str | None = None, job: int = 1) -> Path:
    """One directory per generator agent run; the launcher supplies its ID prefix."""
    name = f"{stamp or _timestamp()}-{job:02d}"
    if count < 1 or job < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Need a positive count/job and a filename-safe timestamp")
    instructions = brief.read_bytes()
    directory = dataset.resolve() / "raw" / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "gen.md").write_bytes(instructions)
    write_json(directory / "manifest.json", {"record_prefix": name, "model": model,
                                            "variant": variant, "expected_records": count})
    return directory


def create_build(dataset: Path, inputs: list[Path], brief: Path = HERE / "arbiter.md",
                 stamp: str | None = None) -> Path:
    dataset = dataset.resolve()
    if not inputs or len({path.resolve() for path in inputs}) != len(inputs):
        raise ValueError("Select distinct raw attempt directories")
    selected = []
    for path in inputs:
        path = path.resolve()
        if path.parent != dataset / "raw":
            raise ValueError(f"Raw attempt must belong to {dataset / 'raw'}: {path}")
        chunks = sorted(path.glob("part-*.jsonl"))
        if not chunks:
            raise ValueError(f"No generated chunks: {path}")
        selected.append({"path": path.relative_to(dataset).as_posix(),
                         "files": hashes(path, [path / "manifest.json", path / "gen.md", *chunks])})
    instructions = brief.read_bytes()
    name = stamp or _timestamp()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Need a filename-safe build timestamp")
    directory = dataset / "build" / name
    directory.mkdir(parents=True, exist_ok=False)
    for folder in ("review", "work", "dataset"):
        (directory / folder).mkdir()
    (directory / "review" / "arbiter.md").write_bytes(instructions)
    write_json(directory / "manifest.json", {"version": 1, "kind": "build", "status": "draft",
               "inputs": selected, "dataset": "dataset"})
    return directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("raw", help="allocate one generator attempt")
    raw.add_argument("--dataset", type=Path, required=True)
    raw.add_argument("--model", required=True)
    raw.add_argument("--variant", default="")
    raw.add_argument("--count", type=int, default=500)
    raw.add_argument("--brief", type=Path, default=HERE / "gen.md")
    raw.add_argument("--timestamp")
    raw.add_argument("--job", type=int, default=1)
    build = commands.add_parser("build", help="start a review of selected raw attempts")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--raw", type=Path, nargs="+", required=True)
    build.add_argument("--brief", type=Path, default=HERE / "arbiter.md")
    args = parser.parse_args(argv)
    try:
        if args.command == "raw":
            print(create_raw(args.dataset, args.model, args.variant, args.count, args.brief,
                             args.timestamp, args.job).as_posix())
        else:
            print(create_build(args.dataset, args.raw, args.brief).as_posix())
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
