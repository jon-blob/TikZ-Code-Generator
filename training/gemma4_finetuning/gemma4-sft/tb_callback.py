from pathlib import Path

import torch
from PIL import Image
from transformers import TextStreamer, TrainerCallback
from unsloth import FastVisionModel


class FileStreamer(TextStreamer):
    def __init__(self, tokenizer, file):
        self.file = file
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.file.write(text)
        self.file.flush()

        if stream_end:
            self.file.write("\n")
            self.file.flush()


class LatexEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        image_paths,
        prompt,
        output_dir,
        max_new_tokens=8192,
    ):
        self.processor = processor
        self.image_paths = [Path(path) for path in image_paths]
        self.prompt = prompt
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens

        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        output_path = (
            self.output_dir
            / f"eval_step_{state.global_step}.txt"
        )

        tokenizer = self.processor.tokenizer
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        if pad_token_id is None:
            if isinstance(eos_token_id, (list, tuple)):
                pad_token_id = eos_token_id[0]
            else:
                pad_token_id = eos_token_id

        FastVisionModel.for_inference(model)
        model.eval()

        try:
            with output_path.open("w", encoding="utf-8") as file:
                for index, image_path in enumerate(
                    self.image_paths,
                    start=1,
                ):
                    file.write(
                        f"IMAGE {index}: {image_path}\n\n"
                    )
                    file.flush()

                    try:
                        with Image.open(image_path) as image:
                            image = image.convert("RGB")

                            messages = [{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image,
                                    },
                                    {
                                        "type": "text",
                                        "text": self.prompt,
                                    },
                                ],
                            }]

                            inputs = self.processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_dict=True,
                                return_tensors="pt",
                            ).to(model.device)

                            streamer = FileStreamer(
                                tokenizer,
                                file,
                            )

                            with torch.inference_mode():
                                model.generate(
                                    **inputs,
                                    streamer=streamer,
                                    max_new_tokens=self.max_new_tokens,
                                    do_sample=False,
                                    use_cache=True,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id,
                                )

                    except Exception as error:
                        file.write(
                            "\n"
                            f"[ERROR: {type(error).__name__}: "
                            f"{error}]\n"
                        )
                        file.flush()

                    file.write(
                        "\n" + "=" * 80 + "\n\n"
                    )
                    file.flush()

        finally:
            FastVisionModel.for_training(model)
            model.train()

        print(f"Gespeichert: {output_path}")

        return control