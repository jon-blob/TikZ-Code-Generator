"""Independent mode orchestration for the TikZ dataset pipeline."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import gcd
import gc
import json
import random

from tqdm import tqdm

import config
from cleaning import DeterministicCleaner
from ollama_client import OllamaClient
from processing import (
    ProcessingMode,
    SampleProcessingError,
    SampleProcessor,
)
from rendering import ValidatedRenderer
from storage import ParquetShardWriter, SplitLoader
from token_counting import LatexTokenCounter


@dataclass(frozen=True)
class ModePlan:
    mode: ProcessingMode
    num: int
    absolute_num: int
    train: bool = False

    @classmethod
    def from_config(cls, raw: dict) -> "ModePlan":
        train = raw.get("train", False)
        if not isinstance(train, bool):
            raise TypeError("train must be true or false.")

        plan = cls(
            mode=ProcessingMode(raw["type"]),
            num=int(raw["num"]),
            absolute_num=int(raw["absolute_num"]),
            train=train,
        )
        if plan.num < 0 or plan.absolute_num < 0:
            raise ValueError("num and absolute_num must be non-negative.")
        if plan.num > plan.absolute_num:
            raise ValueError("num cannot exceed absolute_num.")
        return plan

    @property
    def submode(self) -> str:
        return "train" if self.train else "benchmark"


@dataclass(frozen=True)
class PreparedSample:
    index: int
    original_code: str
    token_count: int


@dataclass
class SelectionStats:
    checked: int = 0
    empty: int = 0
    too_long: int = 0


@dataclass
class ModeStats:
    requested: int = 0
    processed: int = 0
    mode_attempted: int = 0
    mode_saved: int = 0
    base_validation_attempted: int = 0
    base_only_saved: int = 0
    rejected: int = 0

    @property
    def saved(self) -> int:
        return self.mode_saved + self.base_only_saved


class TikzDatasetPipeline:
    def __init__(self) -> None:
        self.cleaner = DeterministicCleaner()
        self.ollama = OllamaClient(self.cleaner)
        self.renderer = ValidatedRenderer()
        self.token_counter = LatexTokenCounter()
        self.processor = SampleProcessor(self.cleaner, self.ollama, self.renderer)
        self.loader = SplitLoader()
        self.failure_dir = config.OUTPUT_DIR / "failures"
        self.failure_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> None:
        plans = self._all_plans()

        if any(plan.num > 0 for plan in plans):
            self.ollama.check_connection()

        if any(plan.num > 0 or plan.train for plan in plans):
            self.renderer.check_dependencies()

    def run_split(self, split_name: str, raw_plans: list[dict]) -> None:
        plans = [ModePlan.from_config(raw) for raw in raw_plans]
        self._validate_unique_modes(split_name, plans)

        print(f"\n{'=' * 72}\nSplit: {split_name}\n{'=' * 72}")
        self.loader.clear_cache()
        dataset = self.loader.load(split_name)

        try:
            for plan in plans:
                if plan.absolute_num > len(dataset):
                    raise ValueError(
                        f"absolute_num={plan.absolute_num} exceeds "
                        f"split size={len(dataset)} for {plan.mode.value}."
                    )

            benchmark_plans = [plan for plan in plans if not plan.train]
            prepared: list[PreparedSample] = []

            if benchmark_plans:
                max_absolute = max(
                    (plan.absolute_num for plan in benchmark_plans),
                    default=0,
                )
                prepared, selection_stats = self._prepare_shared_samples(
                    dataset=dataset,
                    split_name=split_name,
                    target=max_absolute,
                )

                print(
                    f"Shared benchmark selection: "
                    f"{len(prepared):,}/{max_absolute:,} valid; "
                    f"{selection_stats.checked:,} checked, "
                    f"{selection_stats.too_long:,} over token limit, "
                    f"{selection_stats.empty:,} empty"
                )

            for plan in plans:
                if plan.train:
                    self._run_train_mode(
                        split_name=split_name,
                        dataset=dataset,
                        plan=plan,
                    )
                else:
                    self._run_benchmark_mode(
                        split_name=split_name,
                        prepared=prepared,
                        plan=plan,
                    )
        finally:
            del dataset
            gc.collect()
            self.loader.clear_cache()

    def _iter_prepared_samples(
        self,
        dataset,
        split_name: str,
        stats: SelectionStats,
    ):
        """Yield token-valid rows in the deterministic shared random order."""

        for index in self._random_index_order(len(dataset), split_name):
            stats.checked += 1
            sample = dataset[index]
            original_code = str(
                sample.get(config.CODE_WITH_TEXT_COL) or ""
            ).strip()

            if not original_code:
                stats.empty += 1
                continue

            fits, token_count = self.token_counter.fits(original_code)
            if not fits:
                stats.too_long += 1
                continue

            yield PreparedSample(
                index=index,
                original_code=original_code,
                token_count=token_count,
            )

    def _prepare_shared_samples(
        self,
        dataset,
        split_name: str,
        target: int,
    ) -> tuple[list[PreparedSample], SelectionStats]:
        """Create one seeded sample order shared by benchmark modes."""

        prepared: list[PreparedSample] = []
        stats = SelectionStats()

        if target <= 0:
            return prepared, stats

        for item in self._iter_prepared_samples(dataset, split_name, stats):
            prepared.append(item)
            if len(prepared) >= target:
                break

        return prepared, stats

    def _run_benchmark_mode(
        self,
        split_name: str,
        prepared: list[PreparedSample],
        plan: ModePlan,
    ) -> ModeStats:
        """Run the previous behavior unchanged for train=False."""

        selected = prepared[: plan.absolute_num]
        active_count = min(plan.num, len(selected))
        stats = ModeStats(requested=plan.absolute_num)
        writer = ParquetShardWriter(split_name, plan.mode.value)

        print(
            f"\nMode: {plan.mode.value} | submode=benchmark | "
            f"active={plan.num:,} | absolute={plan.absolute_num:,}"
        )

        try:
            for position, item in enumerate(
                tqdm(selected, desc=f"{split_name}:{plan.mode.value}:benchmark")
            ):
                stats.processed += 1

                if position >= active_count:
                    writer.add(
                        self.processor.base_only(
                            output_mode=plan.mode,
                            original_code=item.original_code,
                        )
                    )
                    stats.base_only_saved += 1
                    continue

                stats.mode_attempted += 1
                try:
                    row = self.processor.process(
                        mode=plan.mode,
                        original_code=item.original_code,
                    )
                except Exception as error:
                    stats.rejected += 1
                    self._log_failure(split_name, item, plan.mode, error)
                    continue

                writer.add(row)
                stats.mode_saved += 1
        finally:
            writer.close()

        if len(selected) < plan.absolute_num:
            print(
                f"Warning: only {len(selected):,}/{plan.absolute_num:,} shared "
                "valid samples were available."
            )

        print(
            f"{split_name}/{plan.mode.value}/benchmark: "
            f"{stats.saved:,} saved; "
            f"{stats.mode_saved:,}/{stats.mode_attempted:,} active-mode rows, "
            f"{stats.base_only_saved:,} base-only rows, "
            f"{stats.rejected:,} rejected"
        )
        return stats

    def _run_train_mode(
        self,
        split_name: str,
        dataset,
        plan: ModePlan,
    ) -> ModeStats:
        """Create an exact-size training output with validated base rows.

        Active mode rows remain sequential because they may use Ollama. After
        exactly ``num`` active rows have been saved, the remaining original
        LaTeX rows are rendered and validated concurrently. Their rendered
        images are stored in ``image_with_text``. Failed rows are discarded and
        replaced until ``absolute_num`` rows have been written.
        """

        stats = ModeStats(requested=plan.absolute_num)
        selection_stats = SelectionStats()
        writer = ParquetShardWriter(
            split_name,
            plan.mode.value,
            include_original_image=True,
        )
        prepared_iter = iter(
            self._iter_prepared_samples(
                dataset,
                split_name,
                selection_stats,
            )
        )

        print(
            f"\nMode: {plan.mode.value} | submode=train | "
            f"active={plan.num:,} | absolute={plan.absolute_num:,} | "
            f"base_render_workers={config.TRAIN_RENDER_WORKERS:,}"
        )

        try:
            with tqdm(
                total=plan.absolute_num,
                desc=f"{split_name}:{plan.mode.value}:train",
            ) as progress:
                self._run_train_active_rows(
                    split_name=split_name,
                    plan=plan,
                    prepared_iter=prepared_iter,
                    writer=writer,
                    stats=stats,
                    progress=progress,
                )

                if stats.mode_saved == plan.num:
                    self._run_parallel_train_base_rows(
                        split_name=split_name,
                        plan=plan,
                        prepared_iter=prepared_iter,
                        writer=writer,
                        stats=stats,
                        progress=progress,
                    )
        finally:
            writer.close()

        print(
            f"Train selection: {selection_stats.checked:,} checked, "
            f"{selection_stats.too_long:,} over token limit, "
            f"{selection_stats.empty:,} empty"
        )
        print(
            f"{split_name}/{plan.mode.value}/train: "
            f"{stats.saved:,}/{plan.absolute_num:,} saved; "
            f"{stats.mode_saved:,}/{stats.mode_attempted:,} active-mode rows, "
            f"{stats.base_only_saved:,}/{stats.base_validation_attempted:,} "
            f"validated base rows, {stats.rejected:,} rejected"
        )

        if stats.saved != plan.absolute_num:
            raise RuntimeError(
                f"Could only save {stats.saved:,}/{plan.absolute_num:,} rows "
                f"for {split_name}/{plan.mode.value}/train after exhausting "
                "the source split."
            )

        if stats.mode_saved != plan.num:
            raise RuntimeError(
                f"Could only save {stats.mode_saved:,}/{plan.num:,} active "
                f"rows for {split_name}/{plan.mode.value}/train."
            )

        return stats

    def _run_train_active_rows(
        self,
        split_name: str,
        plan: ModePlan,
        prepared_iter,
        writer: ParquetShardWriter,
        stats: ModeStats,
        progress,
    ) -> None:
        """Save the active, potentially Ollama-backed rows sequentially."""

        while stats.mode_saved < plan.num:
            try:
                item = next(prepared_iter)
            except StopIteration:
                return

            stats.processed += 1
            stats.mode_attempted += 1

            try:
                row = self.processor.process(
                    mode=plan.mode,
                    original_code=item.original_code,
                )
            except Exception as error:
                stats.rejected += 1
                self._log_failure(
                    split_name,
                    item,
                    plan.mode,
                    error,
                )
                progress.set_postfix(rejected=stats.rejected)
                continue

            writer.add(row)
            stats.mode_saved += 1
            progress.update(1)
            progress.set_postfix(rejected=stats.rejected)

    def _run_parallel_train_base_rows(
        self,
        split_name: str,
        plan: ModePlan,
        prepared_iter,
        writer: ParquetShardWriter,
        stats: ModeStats,
        progress,
    ) -> None:
        """Render and validate train base rows with a bounded worker pool.

        Dataset access, failure logging, progress updates, and Parquet writes
        stay in the main thread. Worker threads independently render and
        validate the original LaTeX and return the PNG bytes for storage.
        """

        target = plan.absolute_num - plan.num
        if target <= 0:
            return

        workers = max(1, min(int(config.TRAIN_RENDER_WORKERS), target))
        max_in_flight = max(
            workers,
            int(config.TRAIN_RENDER_MAX_IN_FLIGHT),
        )
        futures = {}
        source_exhausted = False

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tikz-train-render",
        ) as executor:
            while stats.base_only_saved < target:
                remaining = target - stats.base_only_saved
                in_flight_limit = min(max_in_flight, remaining)

                while (
                    not source_exhausted
                    and len(futures) < in_flight_limit
                ):
                    try:
                        item = next(prepared_iter)
                    except StopIteration:
                        source_exhausted = True
                        break

                    stats.processed += 1
                    stats.base_validation_attempted += 1
                    future = executor.submit(
                        self.processor.validated_base_only,
                        output_mode=plan.mode,
                        original_code=item.original_code,
                    )
                    futures[future] = item

                if not futures:
                    break

                completed, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in completed:
                    item = futures.pop(future)

                    try:
                        row = future.result()
                    except Exception as error:
                        stats.rejected += 1
                        self._log_failure(
                            split_name,
                            item,
                            plan.mode,
                            error,
                        )
                    else:
                        writer.add(row)
                        stats.base_only_saved += 1
                        progress.update(1)

                    progress.set_postfix(
                        rejected=stats.rejected,
                        workers=workers,
                    )

    def _all_plans(self) -> list[ModePlan]:
        return [
            ModePlan.from_config(raw)
            for raw_plans in config.SPLITS.values()
            for raw in raw_plans
        ]

    @staticmethod
    def _validate_unique_modes(split_name: str, plans: list[ModePlan]) -> None:
        modes = [plan.mode for plan in plans]
        if len(modes) != len(set(modes)):
            raise ValueError(
                f"Each mode may appear only once per split: {split_name}."
            )

    @staticmethod
    def _random_index_order(size: int, split_name: str):
        """Visit rows once in a deterministic random order shared by all modes."""

        if size <= 0:
            return

        rng = random.Random(f"{config.RANDOM_SEED}:rows:{split_name}")
        start = rng.randrange(size)

        if size == 1:
            yield 0
            return

        step = rng.randrange(1, size)
        while gcd(step, size) != 1:
            step = rng.randrange(1, size)

        for offset in range(size):
            yield (start + offset * step) % size

    def _log_failure(
        self,
        split_name: str,
        item: PreparedSample,
        mode: ProcessingMode,
        error: Exception,
    ) -> None:
        if isinstance(error, SampleProcessingError):
            stage = error.stage
            latest_code = error.latest_code
            cause = error.cause
        else:
            stage = "unknown"
            latest_code = item.original_code
            cause = error

        text_dir = self.failure_dir / split_name / mode.value
        text_dir.mkdir(parents=True, exist_ok=True)
        text_path = text_dir / f"sample-{item.index:09d}.txt"

        text_path.write_text(
            self._failure_text(
                split_name=split_name,
                index=item.index,
                mode=mode,
                token_count=item.token_count,
                stage=stage,
                error=cause,
                original_code=item.original_code,
                latest_code=latest_code,
            ),
            encoding="utf-8",
        )

        jsonl_path = self.failure_dir / f"{split_name}-{mode.value}.jsonl"
        record = {
            "split": split_name,
            "index": item.index,
            "mode": mode.value,
            "stage": stage,
            "token_count": item.token_count,
            "error_type": type(cause).__name__,
            "error": str(cause),
            "details_file": str(text_path),
        }
        with jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _failure_text(
        split_name: str,
        index: int,
        mode: ProcessingMode,
        token_count: int,
        stage: str,
        error: Exception,
        original_code: str,
        latest_code: str,
    ) -> str:
        divider = "=" * 80
        return (
            f"Split: {split_name}\n"
            f"Index: {index}\n"
            f"Mode: {mode.value}\n"
            f"Stage: {stage}\n"
            f"Token count: {token_count}\n"
            f"Error type: {type(error).__name__}\n"
            f"Error: {error}\n\n"
            f"{divider}\nORIGINAL CODE\n{divider}\n"
            f"{original_code}\n\n"
            f"{divider}\nLATEST NEW CODE\n{divider}\n"
            f"{latest_code}\n"
        )
