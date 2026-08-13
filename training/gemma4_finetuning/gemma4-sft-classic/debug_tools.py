from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_tikz_line(line: str) -> str:
    return _NUMBER_RE.sub("<NUM>", line.strip())


def code_structure_stats(code: str) -> dict:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    normalized = [normalize_tikz_line(line) for line in lines]

    if normalized:
        counts = Counter(normalized)
        normalized_repeat_ratio = 1.0 - len(counts) / len(normalized)
        dominant_line_fraction = max(counts.values()) / len(normalized)

        max_run = 1
        current_run = 1
        for previous, current in zip(normalized, normalized[1:]):
            if current == previous:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
    else:
        normalized_repeat_ratio = 0.0
        dominant_line_fraction = 0.0
        max_run = 0

    return {
        "code_chars": len(code),
        "code_lines": len(lines),
        "draw_count": code.count("\\draw"),
        "fill_count": code.count("\\fill"),
        "node_count": code.count("\\node"),
        "normalized_line_repetition_ratio": normalized_repeat_ratio,
        "dominant_normalized_line_fraction": dominant_line_fraction,
        "max_identical_normalized_line_run": max_run,
    }


def get_stop_ids(model, tokenizer) -> list[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(int(x) for x in value))
    return [int(value)]


def repetition_ratio(token_ids, n: int = 16) -> float:
    ids = [int(x) for x in token_ids]
    if len(ids) < n:
        return 0.0
    ngrams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def normalized_line_repetition_ratio(text: str) -> float:
    lines = [normalize_tikz_line(line) for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def audit_labels(base_collator, dataset, model, processor, output_path: Path, n: int = 100) -> dict:
    tokenizer = processor.tokenizer
    stop_ids = set(get_stop_ids(model, tokenizer))
    pad_id = tokenizer.pad_token_id

    summary = {
        "samples_checked": 0,
        "stop_ids": sorted(stop_ids),
        "missing_stop_samples": 0,
        "active_padding_samples": 0,
        "samples": [],
    }

    for index in range(min(n, len(dataset))):
        feature = dataset[index]
        clean_feature = {k: v for k, v in feature.items() if k != "_debug_meta"}
        batch = base_collator([clean_feature])

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        active = labels != -100
        active_labels = labels[active]

        stop_count = 0
        for stop_id in stop_ids:
            stop_count += int((active_labels == stop_id).sum().item())

        active_pad = 0
        if pad_id is not None:
            active_pad = int(((input_ids == pad_id) & active).sum().item())

        row = {
            "dataset_index": index,
            "active_targets": int(active.sum().item()),
            "stop_target_count": stop_count,
            "active_padding_count": active_pad,
            "target_tail": tokenizer.decode(
                active_labels[-32:].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }

        meta = feature.get("_debug_meta")
        if meta:
            row.update(meta)

        summary["samples_checked"] += 1
        summary["missing_stop_samples"] += int(stop_count == 0)
        summary["active_padding_samples"] += int(active_pad > 0)
        summary["samples"].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== LABEL AUDIT =====", flush=True)
    print(f"samples_checked={summary['samples_checked']}", flush=True)
    print(f"stop_ids={summary['stop_ids']}", flush=True)
    print(f"missing_stop_samples={summary['missing_stop_samples']}", flush=True)
    print(f"active_padding_samples={summary['active_padding_samples']}", flush=True)
    print(f"saved={output_path}", flush=True)
    print("=======================\n", flush=True)

    return summary


def _slice_prefix_inputs(batch: dict, prefix_length: int) -> dict:
    sequence_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "position_ids",
        "cache_position",
    }

    result = {}
    for key, value in batch.items():
        if key == "labels":
            continue

        if isinstance(value, torch.Tensor):
            value = value.detach()
            if key in sequence_keys and value.ndim >= 1:
                # Qwen/Gemma position tensors may have additional leading dimensions;
                # their sequence dimension is the last dimension.
                value = value[..., :prefix_length]
        result[key] = value

    return result


@torch.inference_mode()
def teacher_forced_stop_probe(model, processor, batch: dict) -> dict:
    """Measure the configured stop token on a fixed ground-truth prefix.

    The batch is produced by the real training collator. We locate the final
    active stop-token label, truncate immediately before it, and ask the model
    for only the final next-token logits.
    """
    tokenizer = processor.tokenizer
    stop_ids = get_stop_ids(model, tokenizer)
    labels = batch["labels"][0]

    candidates = []
    for stop_id in stop_ids:
        positions = torch.nonzero(labels == stop_id, as_tuple=False).flatten()
        for position in positions.tolist():
            candidates.append((int(position), int(stop_id)))

    if not candidates:
        return {
            "found": False,
            "reason": "No configured stop token is present in active labels.",
            "stop_ids": stop_ids,
        }

    target_position, target_stop_id = max(candidates, key=lambda item: item[0])
    if target_position <= 0:
        return {
            "found": False,
            "reason": "Stop token is at position 0; no causal prefix exists.",
            "stop_ids": stop_ids,
        }

    prefix_inputs = _slice_prefix_inputs(batch, target_position)
    for key, value in list(prefix_inputs.items()):
        if isinstance(value, torch.Tensor):
            prefix_inputs[key] = value.to(model.device)

    forward_kwargs = dict(prefix_inputs)
    forward_kwargs["use_cache"] = False

    try:
        outputs = model(**forward_kwargs, logits_to_keep=1)
        logits_mode = "logits_to_keep"
    except TypeError as first_error:
        try:
            outputs = model(**forward_kwargs, num_logits_to_keep=1)
            logits_mode = "num_logits_to_keep"
        except TypeError as second_error:
            return {
                "found": False,
                "reason": (
                    "Model forward supports neither logits_to_keep=1 nor "
                    f"num_logits_to_keep=1. First error: {first_error}; "
                    f"second error: {second_error}"
                ),
                "stop_ids": stop_ids,
            }

    logits = outputs.logits[0, -1].float()
    log_z = torch.logsumexp(logits, dim=-1)

    target_logit = logits[target_stop_id]
    target_probability = float(torch.exp(target_logit - log_z).item())
    target_rank = int((logits > target_logit).sum().item() + 1)

    valid_stop_ids = [stop_id for stop_id in stop_ids if 0 <= stop_id < logits.numel()]
    stop_logits = logits[valid_stop_ids]
    any_stop_probability = float(
        torch.exp(torch.logsumexp(stop_logits, dim=-1) - log_z).item()
    ) if valid_stop_ids else 0.0

    if valid_stop_ids:
        best_local = int(torch.argmax(stop_logits).item())
        best_stop_id = int(valid_stop_ids[best_local])
        best_stop_logit = logits[best_stop_id]
        best_stop_rank = int((logits > best_stop_logit).sum().item() + 1)
    else:
        best_stop_id = None
        best_stop_rank = None

    top_values, top_ids = torch.topk(logits, k=min(5, logits.numel()))
    top5 = [
        {
            "id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "logit": float(value),
        }
        for value, token_id in zip(top_values.tolist(), top_ids.tolist())
    ]

    return {
        "found": True,
        "logits_mode": logits_mode,
        "target_position": target_position,
        "target_stop_id": target_stop_id,
        "target_stop_token": tokenizer.decode([target_stop_id], skip_special_tokens=False),
        "p_target_stop": target_probability,
        "target_stop_rank": target_rank,
        "p_any_stop": any_stop_probability,
        "best_stop_id": best_stop_id,
        "best_stop_rank": best_stop_rank,
        "stop_ids": stop_ids,
        "top5": top5,
    }


class DebugVisionDataCollator:
    """Wrap a vision collator and log which train samples feed each update.

    For exact optimizer-step grouping this should be used with
    dataloader_num_workers=0; otherwise worker prefetching changes call order.
    """

    def __init__(
        self,
        base_collator,
        output_path: Path,
        gradient_accumulation_steps: int,
        resume: bool = False,
    ):
        self.base_collator = base_collator
        self.output_path = Path(output_path)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.microbatch_index = 0

        if resume and self.output_path.exists():
            with self.output_path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        row = json.loads(line)
                        self.microbatch_index = max(
                            self.microbatch_index,
                            int(row.get("microbatch", 0)),
                        )
                    except Exception:
                        pass
        else:
            self.output_path.write_text("", encoding="utf-8")

    def __call__(self, features):
        clean_features = []
        metas = []

        for feature in features:
            feature = dict(feature)
            metas.append(feature.pop("_debug_meta", None))
            clean_features.append(feature)

        batch = self.base_collator(clean_features)

        # Evaluation batches carry split='val' and are intentionally ignored.
        is_train_batch = any(meta and meta.get("split") == "train" for meta in metas)
        if not is_train_batch:
            return batch

        self.microbatch_index += 1
        optimizer_step = (
            (self.microbatch_index - 1) // self.gradient_accumulation_steps
        ) + 1
        accumulation_slot = (
            (self.microbatch_index - 1) % self.gradient_accumulation_steps
        ) + 1

        labels = batch.get("labels")
        rows = []
        for batch_index, meta in enumerate(metas):
            if not meta or meta.get("split") != "train":
                continue

            row = dict(meta)
            row.update({
                "microbatch": self.microbatch_index,
                "optimizer_step": optimizer_step,
                "accumulation_slot": accumulation_slot,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            })

            if isinstance(labels, torch.Tensor) and batch_index < labels.shape[0]:
                row["active_targets"] = int(
                    (labels[batch_index] != -100).sum().item()
                )

            rows.append(row)

        if rows:
            with self.output_path.open("a", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

            # One concise terminal marker at the beginning of each update.
            if accumulation_slot == 1:
                print(
                    f"[TRAIN-SAMPLES] optimizer_step={optimizer_step} "
                    f"microbatch={self.microbatch_index} "
                    f"log={self.output_path}",
                    flush=True,
                )

        return batch
