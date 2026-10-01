import json
from pathlib import Path

from .dataset import read_conll
from .embeddings import CooccurrenceEmbeddings
from .features import embedding_features, sparse_features
from .metrics import entity_report, normalize_iob2
from .model import TokenClassifier, predict_sentences


ROOT = Path(__file__).resolve().parent.parent


def _flatten_tags(sentences: list[list[tuple[str, str]]]) -> list[str]:
    return [tag for sentence in sentences for _, tag in sentence]


def _format_example(
    words: list[str], gold: list[str], baseline: list[str], embedded: list[str]
) -> dict[str, list[dict[str, str]]]:
    return {
        "tokens": [
            {
                "token": word,
                "gold": gold_tag,
                "baseline": baseline_tag,
                "embedding_model": embedding_tag,
            }
            for word, gold_tag, baseline_tag, embedding_tag in zip(words, gold, baseline, embedded)
        ]
    }


def run(train_path: Path, test_path: Path) -> dict:
    train_rows = read_conll(train_path)
    test_rows = read_conll(test_path)
    train_words = [[word for word, _ in sentence] for sentence in train_rows]
    test_words = [[word for word, _ in sentence] for sentence in test_rows]
    train_labels = _flatten_tags(train_rows)
    test_labels = [ [tag for _, tag in sentence] for sentence in test_rows]

    # Baseline: lexical token/context indicators (a sparse bag-of-features model).
    train_sparse, vectorizer = sparse_features(train_words)
    test_sparse, _ = sparse_features(test_words, vectorizer)
    baseline_model = TokenClassifier().fit(train_sparse, train_labels)
    baseline_flat = baseline_model.predict(test_sparse)
    baseline_predictions: list[list[str]] = []
    offset = 0
    for sentence in test_words:
        baseline_predictions.append(
            normalize_iob2(baseline_flat[offset : offset + len(sentence)])
        )
        offset += len(sentence)

    # New approach: corpus-trained PPMI/SVD word vectors plus neighboring vectors.
    embeddings = CooccurrenceEmbeddings(dimensions=32).fit(train_words)
    train_embedding_features = embedding_features(train_words, embeddings)
    test_embedding_features = embedding_features(test_words, embeddings)
    embedding_model = TokenClassifier().fit(train_embedding_features, train_labels)
    embedded_predictions: list[list[str]] = []
    embedded_flat = embedding_model.predict(test_embedding_features)
    offset = 0
    for sentence in test_words:
        embedded_predictions.append(
            normalize_iob2(embedded_flat[offset : offset + len(sentence)])
        )
        offset += len(sentence)

    report = {
        "dataset": {
            "train_sentences": len(train_rows),
            "test_sentences": len(test_rows),
            "entity_types": ["PER", "ORG", "LOC", "DATE"],
            "embedding_vocabulary_size": len(embeddings.vectors),
        },
        "evaluation": "Exact entity-span precision, recall, and F1; micro average and per entity type.",
        "baseline": entity_report(test_labels, baseline_predictions),
        "word_embeddings": entity_report(test_labels, embedded_predictions),
        "examples": [
            _format_example(words, gold, baseline, embedded)
            for words, gold, baseline, embedded in zip(
                test_words, test_labels, baseline_predictions, embedded_predictions
            )
        ],
    }
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate the NER word-embedding project.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.conll")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.conll")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "results.json")
    args = parser.parse_args()

    report = run(args.train, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved results to {args.output}")
