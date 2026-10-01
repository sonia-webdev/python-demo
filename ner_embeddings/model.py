from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class TokenClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    def fit(self, features: np.ndarray, labels: list[str]) -> "TokenClassifier":
        encoded = self.label_encoder.fit_transform(labels)
        self.classifier.fit(features, encoded)
        return self

    def predict(self, features: np.ndarray) -> list[str]:
        encoded = self.classifier.predict(features)
        return self.label_encoder.inverse_transform(encoded).tolist()


def predict_sentences(
    sentences: list[list[str]],
    features_for_sentences: Callable[[list[list[str]]], np.ndarray],
    model: TokenClassifier,
) -> list[list[str]]:
    flat_predictions = model.predict(features_for_sentences(sentences))
    nested_predictions: list[list[str]] = []
    offset = 0
    for sentence in sentences:
        nested_predictions.append(flat_predictions[offset : offset + len(sentence)])
        offset += len(sentence)
    return nested_predictions
