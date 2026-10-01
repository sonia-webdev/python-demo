from collections import Counter, defaultdict
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


class CooccurrenceEmbeddings:
    """Small corpus-trained PPMI/SVD embeddings; no external model download needed."""

    def __init__(self, dimensions: int = 32, window: int = 2):
        self.dimensions = dimensions
        self.window = window
        self.vectors: dict[str, np.ndarray] = {}

    def fit(self, sentences: list[list[str]]) -> "CooccurrenceEmbeddings":
        counts: Counter[str] = Counter()
        cooccurrences: dict[tuple[str, str], float] = defaultdict(float)
        for sentence in sentences:
            words = [word.lower() for word in sentence]
            counts.update(words)
            for index, word in enumerate(words):
                for other_index in range(max(0, index - self.window), min(len(words), index + self.window + 1)):
                    if other_index != index:
                        cooccurrences[(word, words[other_index])] += 1.0 / abs(index - other_index)

        vocabulary = sorted(counts)
        if len(vocabulary) < 2:
            self.vectors = {}
            return self

        word_index = {word: index for index, word in enumerate(vocabulary)}
        row_sums = np.zeros(len(vocabulary), dtype=np.float64)
        column_sums = np.zeros(len(vocabulary), dtype=np.float64)
        total = sum(cooccurrences.values())
        for (word, context), value in cooccurrences.items():
            row_sums[word_index[word]] += value
            column_sums[word_index[context]] += value

        matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.float32)
        for (word, context), value in cooccurrences.items():
            row = word_index[word]
            column = word_index[context]
            pmi = np.log((value * total) / (row_sums[row] * column_sums[column]))
            matrix[row, column] = max(0.0, pmi)

        components = min(self.dimensions, len(vocabulary) - 1)
        reduced = TruncatedSVD(n_components=components, random_state=42).fit_transform(matrix)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced / np.maximum(norms, 1e-12)
        if components < self.dimensions:
            reduced = np.pad(reduced, ((0, 0), (0, self.dimensions - components)))
        self.vectors = {word: reduced[index].astype(np.float32) for word, index in word_index.items()}
        return self

    def vector(self, word: str) -> np.ndarray:
        return self.vectors.get(word.lower(), np.zeros(self.dimensions, dtype=np.float32))
