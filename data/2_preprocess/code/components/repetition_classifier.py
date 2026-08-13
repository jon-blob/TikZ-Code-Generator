"""Local TikZ repetition classification and model token lengths."""

from __future__ import annotations

import re
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer

import config
from output import iter_batches


class RepetitionClassifier:
    NUMBER_RE = re.compile(
        r"""
        (?<![A-Za-z])
        [-+]?
        (?:\d+\.\d*|\.\d+|\d+)
        (?:[eE][-+]?\d+)?
        """,
        re.VERBOSE,
    )
    TOKEN_RE = re.compile(
        r"""
        \\[A-Za-z@]+
        | NUM
        | [A-Za-z_][A-Za-z0-9_]*
        | --|->|<-|<->
        | [^\s]
        """,
        re.VERBOSE,
    )

    def __init__(self) -> None:
        self.model_tokenizer = AutoTokenizer.from_pretrained(
            str(config.TOKENIZER_PATH),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        orders = tuple(config.REPETITION_NGRAM_ORDERS)
        if not orders:
            raise ValueError("REPETITION_NGRAM_ORDERS must not be empty")
        missing = set(orders).difference(config.REPETITION_NGRAM_WEIGHTS)
        if missing:
            raise ValueError(f"Missing REPETITION_NGRAM_WEIGHTS for {sorted(missing)}")

        thresholds = (
            config.REPETITION_MEDIUM_THRESHOLD,
            config.REPETITION_HIGH_THRESHOLD,
            config.REPETITION_VERY_HIGH_THRESHOLD,
            config.REPETITION_CRITICAL_THRESHOLD,
        )
        if not (0 <= thresholds[0] <= thresholds[1] <= thresholds[2] <= thresholds[3] <= 1):
            raise ValueError("Repetition thresholds must be ordered between 0 and 1")

    def run(self) -> tuple[dict[str, tuple[str, int]], dict]:
        metadata: dict[str, tuple[str, int]] = {}
        classes_per_dataset: dict[str, dict[str, int]] = {}
        token_stats: dict[str, dict[str, float | int]] = {}
        score_stats: dict[str, float] = {}

        for dataset_name in config.CLUSTER_DATASETS:
            counts: Counter[str] = Counter()
            token_total = 0
            token_min: int | None = None
            token_max = 0
            score_total = 0.0
            seen = 0

            batches = iter_batches(
                dataset_name,
                ["sample_id", "reference_code"],
                config.REPETITION_BATCH_SIZE,
            )
            for batch in tqdm(batches, desc=f"Repetition: {dataset_name}"):
                codes = [str(code or "") for code in batch["reference_code"]]
                model_tokens = self.model_tokenizer(
                    codes,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )["input_ids"]

                for sample_id, code, token_ids in zip(batch["sample_id"], codes, model_tokens):
                    repetition_class, score = self._classify(code)
                    token_len = len(token_ids)
                    metadata[sample_id] = (repetition_class, token_len)
                    counts[repetition_class] += 1
                    token_total += token_len
                    token_min = token_len if token_min is None else min(token_min, token_len)
                    token_max = max(token_max, token_len)
                    score_total += score
                    seen += 1

            classes_per_dataset[dataset_name] = dict(counts)
            token_stats[dataset_name] = {
                "minimum": token_min or 0,
                "maximum": token_max,
                "mean": token_total / seen if seen else 0.0,
            }
            score_stats[dataset_name] = score_total / seen if seen else 0.0

        return metadata, {
            "classes_per_dataset": classes_per_dataset,
            "mean_score_per_dataset": score_stats,
            "token_len_per_dataset": token_stats,
            "ngram_orders": list(config.REPETITION_NGRAM_ORDERS),
            "ngram_weights": dict(config.REPETITION_NGRAM_WEIGHTS),
            "min_repeat_count": config.REPETITION_MIN_REPEAT_COUNT,
            "thresholds": {
                "medium": config.REPETITION_MEDIUM_THRESHOLD,
                "high": config.REPETITION_HIGH_THRESHOLD,
                "very_high": config.REPETITION_VERY_HIGH_THRESHOLD,
                "critical": config.REPETITION_CRITICAL_THRESHOLD,
            },
        }

    @classmethod
    def _strip_comments(cls, code: str) -> str:
        return "\n".join(
            re.sub(r"(?<!\\)%.*$", "", line)
            for line in str(code or "").splitlines()
        )

    @classmethod
    def _tokens(cls, code: str) -> list[str]:
        normalized = cls.NUMBER_RE.sub("NUM", cls._strip_comments(code))
        return cls.TOKEN_RE.findall(normalized)

    @staticmethod
    def _ngrams(tokens: list[str], n: int):
        for index in range(len(tokens) - n + 1):
            yield tuple(tokens[index:index + n])

    @classmethod
    def _coverage(cls, tokens: list[str], n: int) -> float:
        counts = Counter(cls._ngrams(tokens, n))
        total = sum(counts.values())
        if not total:
            return 0.0
        repeated = sum(
            count for count in counts.values()
            if count >= config.REPETITION_MIN_REPEAT_COUNT
        )
        return repeated / total

    @classmethod
    def _classify(cls, code: str) -> tuple[str, float]:
        tokens = cls._tokens(code)
        if not tokens:
            return "low", 0.0

        weight_sum = sum(
            float(config.REPETITION_NGRAM_WEIGHTS[n])
            for n in config.REPETITION_NGRAM_ORDERS
        )
        score = sum(
            float(config.REPETITION_NGRAM_WEIGHTS[n]) * cls._coverage(tokens, n)
            for n in config.REPETITION_NGRAM_ORDERS
        ) / weight_sum

        if score < config.REPETITION_MEDIUM_THRESHOLD:
            return "low", score
        if score < config.REPETITION_HIGH_THRESHOLD:
            return "medium", score
        if score < config.REPETITION_VERY_HIGH_THRESHOLD:
            return "high", score
        if score < config.REPETITION_CRITICAL_THRESHOLD:
            return "very_high", score
        return "critical", score
