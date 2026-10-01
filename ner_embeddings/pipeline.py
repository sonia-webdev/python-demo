from pathlib import Path

from .dataset import read_conll
from .embeddings import CooccurrenceEmbeddings, tokenize
from .features import embedding_features, sparse_features
from .metrics import normalize_iob2, spans
from .model import TokenClassifier


ROOT = Path(__file__).resolve().parent.parent


def load_models() -> tuple:
    train_rows = read_conll(ROOT / "data" / "train.conll")
    train_words = [[word for word, _ in sentence] for sentence in train_rows]
    train_labels = [tag for sentence in train_rows for _, tag in sentence]

    train_sparse, vectorizer = sparse_features(train_words)
    baseline = TokenClassifier().fit(train_sparse, train_labels)

    embeddings = CooccurrenceEmbeddings(dimensions=32).fit(train_words)
    embedded = TokenClassifier().fit(embedding_features(train_words, embeddings), train_labels)
    return vectorizer, baseline, embeddings, embedded


def annotate_entities(words: list[str], tags: list[str]) -> list[dict[str, str]]:
    entity_starts = {
        start: (entity_type, end)
        for entity_type, start, end in spans(tags)
    }
    result: list[dict[str, str]] = []
    index = 0
    while index < len(words):
        if index in entity_starts:
            entity_type, end = entity_starts[index]
            result.append({"token": " ".join(words[index:end]), "tag": f"B-{entity_type}"})
            index = end
        else:
            result.append({"token": words[index], "tag": "O"})
            index += 1
    return result


def predict_text(text: str, models: tuple) -> dict[str, list[dict[str, str]]]:
    if not isinstance(text, str) or not text.strip() or len(text) > 5_000:
        raise ValueError("Enter text between 1 and 5,000 characters.")
    words = tokenize(text)
    if not words:
        raise ValueError("No words found in the provided text.")

    vectorizer, baseline, embeddings, embedded = models
    sparse, _ = sparse_features([words], vectorizer)
    baseline_tags = normalize_iob2(baseline.predict(sparse))
    embedded_features = embedding_features([words], embeddings)
    embedding_tags = normalize_iob2(embedded.predict(embedded_features))
    return {
        "baseline": annotate_entities(words, baseline_tags),
        "word_embeddings": annotate_entities(words, embedding_tags),
    }
