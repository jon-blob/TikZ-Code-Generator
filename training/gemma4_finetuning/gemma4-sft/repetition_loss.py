from __future__ import annotations

import json
from pathlib import Path

import torch
from trl import SFTTrainer


IGNORE_INDEX = -100


def _active_token_positions(labels_1d: torch.Tensor) -> tuple[list[int], list[int]]:
    positions = torch.nonzero(labels_1d != IGNORE_INDEX, as_tuple=False).flatten()
    if positions.numel() == 0:
        return [], []
    return positions.tolist(), labels_1d[positions].tolist()


def build_repetition_mask(
    labels: torch.Tensor,
    ngram_orders: tuple[int, ...] = (16,),
    free_occurrences: int = 2,
    keep_every: int = 4,
    protect_first_tokens: int = 32,
    protect_last_tokens: int = 64,
    max_mask_fraction: float = 0.35,
) -> tuple[torch.Tensor, dict]:
    """Build a loss mask for repeated target n-grams.

    The first ``free_occurrences`` of every n-gram are trained normally.
    Later occurrences become repetition candidates. To avoid removing all
    learning signal, one out of every ``keep_every`` candidates stays active.

    Only the *next-token position* completing the repeated n-gram is masked.
    With a sustained loop, this naturally masks most of the repeated suffix.

    The beginning and end of every completion are protected. In particular,
    protecting the tail keeps LaTeX closing tokens / EOS fully supervised.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected labels [batch, seq], got shape={tuple(labels.shape)}")

    mask = torch.zeros_like(labels, dtype=torch.bool)

    total_active = 0
    total_candidates = 0
    total_masked = 0
    per_sample = []

    orders = tuple(sorted({int(n) for n in ngram_orders if int(n) >= 2}))
    if not orders:
        raise ValueError("ngram_orders must contain at least one integer >= 2")

    for batch_index in range(labels.shape[0]):
        positions, token_ids = _active_token_positions(labels[batch_index])
        active_count = len(token_ids)
        total_active += active_count

        if active_count == 0:
            per_sample.append({
                "active_tokens": 0,
                "candidate_tokens": 0,
                "masked_tokens": 0,
                "masked_fraction": 0.0,
            })
            continue

        protected_left = max(0, int(protect_first_tokens))
        protected_right_start = max(
            protected_left,
            active_count - max(0, int(protect_last_tokens)),
        )

        candidate_indices: set[int] = set()

        for n in orders:
            if active_count < n:
                continue

            seen: dict[tuple[int, ...], int] = {}
            for end_index in range(n - 1, active_count):
                gram = tuple(token_ids[end_index - n + 1:end_index + 1])
                occurrence = seen.get(gram, 0) + 1
                seen[gram] = occurrence

                if occurrence <= free_occurrences:
                    continue

                if end_index < protected_left or end_index >= protected_right_start:
                    continue

                candidate_indices.add(end_index)

        sorted_candidates = sorted(candidate_indices)
        total_candidates += len(sorted_candidates)

        # Approximate a fractional repetition weight without materializing
        # per-token vocabulary logits: keep 1 of every N repetition positions.
        to_mask = []
        for local_candidate_index, active_index in enumerate(sorted_candidates, start=1):
            if keep_every > 1 and local_candidate_index % keep_every == 0:
                continue
            to_mask.append(active_index)

        max_allowed = int(active_count * max(0.0, min(1.0, float(max_mask_fraction))))
        if max_allowed >= 0 and len(to_mask) > max_allowed:
            # Keep the earliest candidates masked and stop before the cap.
            # Tail protection already prevents masking the completion ending.
            to_mask = to_mask[:max_allowed]

        for active_index in to_mask:
            sequence_position = positions[active_index]
            mask[batch_index, sequence_position] = True

        masked_count = len(to_mask)
        total_masked += masked_count

        per_sample.append({
            "active_tokens": active_count,
            "candidate_tokens": len(sorted_candidates),
            "masked_tokens": masked_count,
            "masked_fraction": masked_count / active_count if active_count else 0.0,
        })

    stats = {
        "active_tokens": total_active,
        "candidate_tokens": total_candidates,
        "masked_tokens": total_masked,
        "masked_fraction": total_masked / total_active if total_active else 0.0,
        "per_sample": per_sample,
    }
    return mask, stats


class RepetitionAwareSFTTrainer(SFTTrainer):
    """SFTTrainer with memory-efficient repetition-aware CE regularization.

    This intentionally does NOT request full [sequence, vocabulary] logits.
    Instead, it modifies labels before the native model loss is computed,
    allowing Unsloth's normal fused/chunked CE path to remain in use.

    Repeated target continuations therefore receive less positive CE pressure,
    while normal targets and the protected completion ending stay unchanged.
    """

    def __init__(
        self,
        *args,
        repetition_loss_enabled: bool = True,
        repetition_ngram_orders: tuple[int, ...] = (16,),
        repetition_free_occurrences: int = 2,
        repetition_keep_every: int = 4,
        repetition_protect_first_tokens: int = 32,
        repetition_protect_last_tokens: int = 64,
        repetition_max_mask_fraction: float = 0.35,
        repetition_log_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.repetition_loss_enabled = bool(repetition_loss_enabled)
        self.repetition_ngram_orders = tuple(int(x) for x in repetition_ngram_orders)
        self.repetition_free_occurrences = int(repetition_free_occurrences)
        self.repetition_keep_every = int(repetition_keep_every)
        self.repetition_protect_first_tokens = int(repetition_protect_first_tokens)
        self.repetition_protect_last_tokens = int(repetition_protect_last_tokens)
        self.repetition_max_mask_fraction = float(repetition_max_mask_fraction)
        self.repetition_log_path = (
            Path(repetition_log_path) if repetition_log_path is not None else None
        )
        self._repetition_microbatch = 0

        if self.repetition_log_path is not None:
            self.repetition_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.repetition_log_path.write_text("", encoding="utf-8")

        print(
            "\n===== REPETITION-AWARE LOSS =====\n"
            f"enabled={self.repetition_loss_enabled}\n"
            f"ngram_orders={self.repetition_ngram_orders}\n"
            f"free_occurrences={self.repetition_free_occurrences}\n"
            f"keep_every={self.repetition_keep_every}\n"
            f"protect_first_tokens={self.repetition_protect_first_tokens}\n"
            f"protect_last_tokens={self.repetition_protect_last_tokens}\n"
            f"max_mask_fraction={self.repetition_max_mask_fraction}\n"
            "=================================\n",
            flush=True,
        )

    def _log_repetition_stats(self, stats: dict) -> None:
        if self.repetition_log_path is None:
            return

        row = {
            "global_step": int(self.state.global_step),
            "microbatch": int(self._repetition_microbatch),
            **stats,
        }
        with self.repetition_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
        **kwargs,
    ):
        # Keep eval_loss comparable to the baseline. Regularization is applied
        # only during training, never during trainer.evaluate().
        apply_regularization = (
            self.repetition_loss_enabled
            and model.training
            and isinstance(inputs.get("labels"), torch.Tensor)
        )

        if apply_regularization:
            self._repetition_microbatch += 1
            modified_inputs = dict(inputs)
            original_labels = inputs["labels"]

            repetition_mask, stats = build_repetition_mask(
                original_labels,
                ngram_orders=self.repetition_ngram_orders,
                free_occurrences=self.repetition_free_occurrences,
                keep_every=self.repetition_keep_every,
                protect_first_tokens=self.repetition_protect_first_tokens,
                protect_last_tokens=self.repetition_protect_last_tokens,
                max_mask_fraction=self.repetition_max_mask_fraction,
            )

            if repetition_mask.any():
                labels = original_labels.clone()
                labels[repetition_mask] = IGNORE_INDEX
                modified_inputs["labels"] = labels

            self._log_repetition_stats(stats)
            inputs = modified_inputs

        # TRL / Transformers versions differ slightly in the compute_loss
        # signature. Prefer the current API, but retain compatibility.
        try:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )
        except TypeError as error:
            if "num_items_in_batch" not in str(error):
                raise
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                **kwargs,
            )
