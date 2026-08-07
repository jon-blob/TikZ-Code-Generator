"""Deterministic and post-LLM LaTeX cleaning."""

import re

from clean_text_in_tex import process_latex


TEXT_GENERATOR_PATTERN = re.compile(
    r"""
    \\(?:
        lipsum|unpacklipsum|blindtext|Blindtext|blinddocument|Blinddocument|
        blindmathpaper|blinditemize|blindenumerate|blinddescription|kant|kantlipsum
    )
    \*?
    (?:\s*\[[^\]]*\]){0,3}
    (?:\s*\{[^{}]*\})?
    """,
    flags=re.VERBOSE,
)


class DeterministicCleaner:
    def remove_standard_text(self, tex: str) -> str:
        """Remove known text generators and TikZ node contents."""

        cleaned = TEXT_GENERATOR_PATTERN.sub("", tex or "")
        return process_latex(
            tex=cleaned,
            mode="replace_all",
            replacement="",
        ).strip()

    def normalize_llm_latex(self, text: str, original: str) -> str:
        """Keep only the complete LaTeX document from an LLM response."""

        text = self._strip_code_fences(text)

        if r"\documentclass" in text:
            text = text[text.index(r"\documentclass") :]

        if r"\end{document}" in text:
            end = text.index(r"\end{document}") + len(r"\end{document}")
            text = text[:end]

        text = text.strip()

        if not text:
            raise ValueError("The LLM returned empty LaTeX.")
        if r"\documentclass" in original and r"\documentclass" not in text:
            raise ValueError("The LLM removed the document preamble.")
        if r"\begin{document}" in original and r"\begin{document}" not in text:
            raise ValueError("The LLM removed the document environment.")

        return text

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^\s*```(?:latex|tex|tikz)?\s*\n?", "", text, flags=re.I)
        text = re.sub(r"\n?\s*```\s*$", "", text)
        return text.strip()
