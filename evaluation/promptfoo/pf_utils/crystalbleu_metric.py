from collections import Counter
from hashlib import sha256
from itertools import chain, tee
from pathlib import Path
import pickle

from crystalbleu import SmoothingFunction, corpus_bleu
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer

from config import PATHS

LEXER = TexLexer()
TOKENIZER = MosesTokenizer()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token_type, value in LEXER.get_tokens(text):
        value = value.strip()
        if not value or token_type in Comment:
            continue
        if (token_type in Text or token_type in Name.Attribute or token_type in Name.Builtin):
            tokens.extend(TOKENIZER.tokenize(value))
        else:
            tokens.append(value)
    return tokens


def ngrams(sequence, n: int):
    iterables = tee(iter(sequence), n)
    for index, iterable in enumerate(iterables):
        for _ in range(index):
            next(iterable, None)
    return zip(*iterables)


def load_corpus(directory: str | Path) -> list[str]:
    files = sorted(Path(directory).glob("*.txt"))
    corpus = [path.read_text(encoding="utf-8").strip() for path in files]
    return [text for text in corpus if text]


def shared_ngrams(corpus: list[str], k: int, n: int, use_cache: bool) -> dict:
    digest = sha256((repr(corpus) + f"|{k}|{n}").encode("utf-8")).hexdigest()
    cache_file = PATHS.metric_cache / f"crystalbleu-{digest}.pkl"
    if use_cache and cache_file.is_file():
        return pickle.loads(cache_file.read_bytes())

    values = chain.from_iterable(
        ngrams(tokenize(text), order)
        for order in range(1, n + 1)
        for text in corpus
    )
    result = dict(Counter(values).most_common(k))
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(pickle.dumps(result))
    return result


def compute_crystalbleu_score(
    reference_code: str,
    generated_code: str,
    corpus_dir: str | Path,
    k: int = 500,
    n: int = 4,
    smoothing: bool = True,
    use_cache: bool = True,
) -> float:
    corpus = load_corpus(corpus_dir)

    if not corpus:
        raise ValueError(f"CrystalBLEU corpus is empty: {corpus_dir}")

    if not reference_code.strip() or not generated_code.strip():
        raise ValueError("Reference or generated code is empty")

    if n < 1:
        raise ValueError("n must be at least 1")

    reference_tokens = tokenize(reference_code)
    generated_tokens = tokenize(generated_code)

    smoothing_function = (
        SmoothingFunction().method1
        if smoothing
        else None
    )

    return float(
        corpus_bleu(
            list_of_references=[[reference_tokens]],
            hypotheses=[generated_tokens],
            weights=tuple(1 / n for _ in range(n)),
            ignoring=shared_ngrams(
                corpus=corpus,
                k=k,
                n=n,
                use_cache=use_cache,
            ),
            smoothing_function=smoothing_function,
        )
    )
