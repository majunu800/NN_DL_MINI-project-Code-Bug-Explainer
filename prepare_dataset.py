"""Download and normalize a Hugging Face debugging dataset into JSONL."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def pick(record: dict, names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize(record: dict) -> dict:
    instruction = pick(record, ("instruction", "prompt", "question"), "Fix the bug in this Python code.")
    code = pick(record, ("input", "code", "buggy_code", "source"))
    answer = pick(record, ("output", "response", "answer", "completion"))
    if not code or not answer:
        raise ValueError("Record does not contain recognizable code and answer fields")
    return {"instruction": instruction, "input": code, "output": answer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="m-a-p/CodeFeedback-Filtered-Instruction")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("data/debugging.jsonl"))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in dataset:
            if written >= args.limit:
                break
            try:
                normalized = normalize(record)
            except ValueError:
                skipped += 1
                continue
            handle.write(json.dumps(normalized, ensure_ascii=True) + "\n")
            written += 1
    print(f"Wrote {written} records to {args.output}; skipped {skipped} records.")


if __name__ == "__main__":
    main()
