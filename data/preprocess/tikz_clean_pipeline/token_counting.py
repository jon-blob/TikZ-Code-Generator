"""Token-length validation without loading model weights."""

from transformers import AutoTokenizer

import config


class LatexTokenCounter:
    def __init__(self) -> None:
        config.TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.TOKENIZER_MODEL,
            cache_dir=str(config.TOKENIZER_CACHE_DIR),
            use_fast=True,
        )

    def count_up_to_limit(self, tex: str) -> int:
        """Return the exact count up to MAX+1, enough for limit validation."""

        token_ids = self.tokenizer.encode(
            tex,
            add_special_tokens=False,
            truncation=True,
            max_length=config.MAX_LATEX_TOKENS + 1,
        )
        return len(token_ids)

    def fits(self, tex: str) -> tuple[bool, int]:
        count = self.count_up_to_limit(tex)
        return count <= config.MAX_LATEX_TOKENS, count
