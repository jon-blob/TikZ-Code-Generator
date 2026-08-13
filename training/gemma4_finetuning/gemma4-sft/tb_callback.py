from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from transformers import TrainerCallback
from unsloth import FastVisionModel

from debug_tools import (
    get_stop_ids,
    normalized_line_repetition_ratio,
    repetition_ratio,
    teacher_forced_stop_probe,
)


class LatexEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        image_paths,
        descripion_paths,
        prompt,
        output_dir,
        max_new_tokens=512,
        teacher_probe_batch=None,
    ):
        self.processor = processor
        self.image_paths = [Path(path) for path in image_paths]
        self.description_paths = [
            None if path is None else Path(path)
            for path in descripion_paths
        ]
        self.prompt = prompt
        self.output_dir = Path(output_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.teacher_probe_batch = teacher_probe_batch

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_log_path = self.output_dir / "generation_debug.jsonl"
        self.teacher_probe_log_path = self.output_dir / "teacher_forced_end_probe.jsonl"

    @staticmethod
    def _append_jsonl(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_evaluate(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return control

        tokenizer = self.processor.tokenizer
        stop_ids = get_stop_ids(model, tokenizer)
        output_path = self.output_dir / f"eval_step_{state.global_step}.txt"

        print(
            f"\n[CALLBACK] Step {state.global_step} gestartet | "
            f"max_new_tokens={self.max_new_tokens} | stop_ids={stop_ids}",
            flush=True,
        )

        FastVisionModel.for_inference(model)
        model.eval()

        try:
            # Fixed teacher-forced probe: always ask the same ground-truth prefix
            # how likely its real final stop token is.
            teacher_probe = None
            if self.teacher_probe_batch is not None:
                teacher_probe = teacher_forced_stop_probe(
                    model,
                    self.processor,
                    self.teacher_probe_batch,
                )
                teacher_row = {
                    "step": int(state.global_step),
                    **teacher_probe,
                }
                self._append_jsonl(self.teacher_probe_log_path, teacher_row)
                print(f"[TEACHER-END] {teacher_row}", flush=True)

            with output_path.open("w", encoding="utf-8") as file:
                if teacher_probe is not None:
                    file.write("TEACHER-FORCED END PROBE\n")
                    file.write(json.dumps(teacher_probe, indent=2, ensure_ascii=False))
                    file.write("\n\n" + "=" * 80 + "\n\n")

                for index, image_path in enumerate(self.image_paths, start=1):
                    print(f"[CALLBACK] Bild {index} vorbereiten", flush=True)

                    final_prompt = self.prompt

                    try:
                        if self.description_paths[index - 1] is not None:
                            description_path = self.description_paths[index-1]
                            with open(description_path, "r") as descr_file:
                                description = descr_file.read()

                            final_prompt += (
                                        "\n\nAdditionally, here is a description of the image "
                                        "with some creation hints:\n"
                                        f"{description.strip()}"
                                    )
                        else:
                            final_prompt = self.prompt

                        print(final_prompt)
                    except Exception:
                        print("no description file found.")

                    try:
                        with Image.open(image_path) as image:
                            image = image.convert("RGB")

                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": final_prompt},
                                ],
                            }]

                            inputs = self.processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_dict=True,
                                return_tensors="pt",
                            ).to(model.device)

                            prompt_length = int(inputs["input_ids"].shape[1])

                            print(
                                f"[CALLBACK] Bild {index}: generate() START | "
                                f"max_new_tokens={self.max_new_tokens}",
                                flush=True,
                            )

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            start = perf_counter()

                            with torch.inference_mode():
                                output = model.generate(
                                    **inputs,
                                    max_new_tokens=self.max_new_tokens,
                                    do_sample=False,
                                    use_cache=True,
                                    return_dict_in_generate=True,
                                )

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            duration = perf_counter() - start

                            sequence = output.sequences[0]
                            generated_ids = sequence[prompt_length:]
                            generated_list = generated_ids.tolist()
                            text = tokenizer.decode(
                                generated_list,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )

                            terminated = bool(
                                generated_list and generated_list[-1] in stop_ids
                            )
                            hit_max = len(generated_list) >= self.max_new_tokens

                            row = {
                                "step": int(state.global_step),
                                "image_index": index,
                                "image_path": str(image_path),
                                "generated_tokens": len(generated_list),
                                "generation_time_seconds": duration,
                                "terminated_by_stop_token": terminated,
                                "hit_max_new_tokens": hit_max,
                                "last_token_id": generated_list[-1] if generated_list else None,
                                "contains_end_document": "\\end{document}" in text,
                                "draw_count": text.count("\\draw"),
                                "fill_count": text.count("\\fill"),
                                "node_count": text.count("\\node"),
                                "repeat_16gram_ratio": repetition_ratio(generated_list, n=16),
                                "normalized_line_repetition_ratio": normalized_line_repetition_ratio(text),
                                "last_20_ids": generated_list[-20:],
                            }
                            self._append_jsonl(self.generation_log_path, row)

                            print(
                                f"[CALLBACK] Bild {index}: ENDE nach {duration:.1f}s | "
                                f"tokens={len(generated_list)} | stopped={terminated} | "
                                f"hit_max={hit_max} | line_repeat="
                                f"{row['normalized_line_repetition_ratio']:.3f}",
                                flush=True,
                            )

                            file.write(f"IMAGE {index}: {image_path}\n\n")
                            file.write(text)
                            file.write("\n\n--- DEBUG ---\n")
                            file.write(json.dumps(row, indent=2, ensure_ascii=False))
                            file.write("\n--- END DEBUG ---\n")

                    except Exception as error:
                        error_row = {
                            "step": int(state.global_step),
                            "image_index": index,
                            "image_path": str(image_path),
                            "error": f"{type(error).__name__}: {error}",
                        }
                        self._append_jsonl(self.generation_log_path, error_row)
                        file.write(
                            "\n"
                            f"[ERROR: {type(error).__name__}: {error}]\n"
                        )

                    file.write("\n" + "=" * 80 + "\n\n")
                    file.flush()

        finally:
            print("[CALLBACK] Wechsel zurück zu training mode ...", flush=True)
            FastVisionModel.for_training(model)
            model.train()
            print("[CALLBACK] training mode wieder aktiv", flush=True)

        print(f"[CALLBACK] KOMPLETT BEENDET, Datei: {output_path}", flush=True)
        return control
