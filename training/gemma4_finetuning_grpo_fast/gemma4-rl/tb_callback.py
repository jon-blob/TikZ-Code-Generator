from __future__ import annotations

import html

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback


def pil_to_hwc(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def make_side_by_side(left: Image.Image, right: Image.Image | None) -> Image.Image:
    left = left.convert("RGB")
    if right is None:
        right = Image.new("RGB", left.size, "white")
        ImageDraw.Draw(right).text((10, 10), "not renderable", fill=(0, 0, 0))
    else:
        right = right.convert("RGB").resize(left.size)

    canvas = Image.new("RGB", (left.width + right.width, left.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def truncate_text(text, max_chars: int = 8000) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars] + "\n\n...[truncated]..."


class TensorBoardRewardCallback(TrainerCallback):
    def __init__(self, log_dir: str, reward_fn=None) -> None:
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None
        self.reward_fn = reward_fn

    def _writer_for(self, state) -> SummaryWriter | None:
        if not getattr(state, "is_world_process_zero", True):
            return None
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        return self.writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        writer = self._writer_for(state)
        if writer is None:
            return

        step = state.global_step
        for key, value in (logs or {}).items():
            try:
                writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass

        if self.reward_fn is not None:
            self._log_reward_metrics(writer, step)
            self._log_reward_examples(writer, step)
        writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _log_reward_metrics(self, writer: SummaryWriter, step: int) -> None:
        for key, value in getattr(self.reward_fn, "last_metrics", {}).items():
            try:
                writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass

    def _log_reward_examples(self, writer: SummaryWriter, step: int) -> None:
        cfg = getattr(self.reward_fn, "cfg", None)
        if cfg is None or cfg.log_examples_every <= 0:
            return
        if step % cfg.log_examples_every != 0:
            return

        for idx, example in enumerate(getattr(self.reward_fn, "last_examples", [])):
            input_image = example.get("input_image")
            rendered_image = example.get("rendered_image")

            if isinstance(input_image, Image.Image):
                writer.add_image(
                    f"samples/{idx}/original",
                    pil_to_hwc(input_image),
                    step,
                    dataformats="HWC",
                )
                writer.add_image(
                    f"samples/{idx}/original_vs_generated",
                    pil_to_hwc(make_side_by_side(input_image, rendered_image)),
                    step,
                    dataformats="HWC",
                )

            if isinstance(rendered_image, Image.Image):
                writer.add_image(
                    f"samples/{idx}/generated_render",
                    pil_to_hwc(rendered_image),
                    step,
                    dataformats="HWC",
                )

            reason = html.escape(truncate_text(example.get("render_reason"), 2000))
            generated = html.escape(
                truncate_text(example.get("generated_code"), cfg.max_logged_code_chars)
            )
            reference = html.escape(
                truncate_text(example.get("reference_code"), cfg.max_logged_code_chars)
            )
            text = (
                f"<b>Score:</b> {example.get('score')}<br>"
                f"<b>Render OK:</b> {example.get('render_ok')}<br><br>"
                f"<b>Render reason</b><pre>{reason}</pre>"
                f"<b>Generated code</b><pre>{generated}</pre>"
                f"<b>Reference code</b><pre>{reference}</pre>"
            )
            writer.add_text(f"samples/{idx}/code", text, step)
