from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


GENERATIONS_DIR = Path("/home/jonas/models/sft/checkpoints/generations")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_training_samples() -> None:
    rows = read_jsonl(GENERATIONS_DIR / "train_samples_by_step.jsonl")
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["optimizer_step"])].append(row)

    output = GENERATIONS_DIR / "train_samples_by_step_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "optimizer_step",
                "samples",
                "mean_line_repetition",
                "max_line_repetition",
                "max_identical_normalized_line_run",
                "max_active_targets",
                "most_repetitive_code_path",
            ],
        )
        writer.writeheader()

        for step in sorted(grouped):
            samples = grouped[step]
            most_repetitive = max(
                samples,
                key=lambda row: row.get("normalized_line_repetition_ratio", 0.0),
            )
            reps = [row.get("normalized_line_repetition_ratio", 0.0) for row in samples]
            writer.writerow({
                "optimizer_step": step,
                "samples": len(samples),
                "mean_line_repetition": sum(reps) / len(reps),
                "max_line_repetition": max(reps),
                "max_identical_normalized_line_run": max(
                    row.get("max_identical_normalized_line_run", 0) for row in samples
                ),
                "max_active_targets": max(
                    row.get("active_targets", 0) for row in samples
                ),
                "most_repetitive_code_path": most_repetitive.get("code_path", ""),
            })

    print(f"Saved: {output}")


def summarize_teacher_probe() -> None:
    rows = read_jsonl(GENERATIONS_DIR / "teacher_forced_end_probe.jsonl")
    output = GENERATIONS_DIR / "teacher_forced_end_probe.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "found",
                "p_target_stop",
                "target_stop_rank",
                "p_any_stop",
                "best_stop_rank",
                "target_stop_id",
                "target_stop_token",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print(f"Saved: {output}")


if __name__ == "__main__":
    summarize_training_samples()
    summarize_teacher_probe()
