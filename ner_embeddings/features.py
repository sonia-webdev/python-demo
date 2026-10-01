import re

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from .embeddings import CooccurrenceEmbeddings


def _shape(word: str) -> str:
    return re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "0", word)))


def token_features(words: list[str], index: int) -> dict[str, str]:
    word = words[index]
    previous = words[index - 1] if index else "<BOS>"
    following = words[index + 1] if index + 1 < len(words) else "<EOS>"
    return {
        "word": word.lower(),
        "word.isupper": str(word.isupper()),
        "word.istitle": str(word.istitle()),
        "word.hasdigit": str(any(char.isdigit() for char in word)),
        "word.shape": _shape(word),
        "prefix1": word[:1].lower(),
        "prefix2": word[:2].lower(),
        "suffix2": word[-2:].lower(),
        "suffix3": word[-3:].lower(),
        "prev": previous.lower(),
        "prev.shape": _shape(previous),
        "next": following.lower(),
        "next.shape": _shape(following),
        "prev2": words[index - 2].lower() if index > 1 else "<BOS>",
        "next2": words[index + 2].lower() if index + 2 < len(words) else "<EOS>",
    }


def sparse_features(
    sentences: list[list[str]], vectorizer: DictVectorizer | None = None
) -> tuple[np.ndarray, DictVectorizer]:
    rows = [token_features(words, index) for words in sentences for index in range(len(words))]
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(rows)
    else:
        matrix = vectorizer.transform(rows)
    return matrix, vectorizer


def embedding_features(
    sentences: list[list[str]], embeddings: CooccurrenceEmbeddings
) -> np.ndarray:
    feature_rows: list[np.ndarray] = []
    for words in sentences:
        for index, word in enumerate(words):
            neighbors = [words[position] for position in (index - 1, index + 1) if 0 <= position < len(words)]
            current = embeddings.vector(word)
            context = (
                np.mean([embeddings.vector(neighbor) for neighbor in neighbors], axis=0)
                if neighbors
                else np.zeros_like(current)
            )
            flags = np.array(
                [
                    word.isupper(),
                    word.istitle(),
                    any(char.isdigit() for char in word),
                    word.isalpha(),
                ],
                dtype=np.float32,
            )
            feature_rows.append(np.concatenate((current, context, flags)))
    return np.vstack(feature_rows)
