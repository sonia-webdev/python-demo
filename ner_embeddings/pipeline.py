from pathlib import Path

from .dataset import read_conll
from .embeddings import CooccurrenceEmbeddings, tokenize
from .features import embedding_features, sparse_features
from .metrics import normalize_iob2, spans
from .model import TokenClassifier


ROOT = Path(__file__).resolve().parent.parent
ENTITY_STOPWORDS = {"a", "an", "the", "this", "that", "these", "those"}
MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
ORGANIZATION_SUFFIXES = {
    "inc", "incorporated", "ltd", "limited", "llc", "corp", "corporation",
    "company", "university", "institute", "foundation", "hospital", "bank",
    "agency", "commission", "organization", "organisation", "association",
}
LOCATION_CUES = {"in", "to", "from", "near", "toward", "towards", "across"}
ORGANIZATION_CUES = {"at", "for", "joined", "hired", "founded", "contacted"}
PERSON_SUBJECT_VERBS = {
    "met", "said", "visited", "moved", "arrived", "started", "spoke", "joined",
    "works", "worked", "announced", "travelled", "traveled", "published",
}
PERSON_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof"}


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


def _capitalized_span(words: list[str], start: int) -> tuple[int, int] | None:
    while start < len(words) and words[start].lower() in ENTITY_STOPWORDS:
        start += 1
    if start >= len(words) or not words[start].isalpha() or not words[start][0].isupper():
        return None
    end = start + 1
    while end < len(words) and words[end].isalpha() and words[end][0].isupper():
        end += 1
    return start, end


def contextual_fallback(words: list[str], model_tags: list[str]) -> list[str]:
    """Add high-confidence date and context-based labels for unseen names."""
    tags = model_tags.copy()

    def mark(start: int, end: int, entity_type: str, *, replace: bool = False) -> None:
        if start >= end or (not replace and any(tag != "O" for tag in tags[start:end])):
            return
        tags[start] = f"B-{entity_type}"
        for position in range(start + 1, end):
            tags[position] = f"I-{entity_type}"

    lower = [word.casefold().rstrip(".") for word in words]
    for index, word in enumerate(words):
        if word.isdigit() and len(word) == 4 and 1800 <= int(word) <= 2099:
            mark(index, index + 1, "DATE", replace=True)
            if index and lower[index - 1] in MONTHS:
                mark(index - 1, index, "DATE", replace=True)
            continue
        if lower[index] in MONTHS:
            has_date_context = (
                (index > 0 and lower[index - 1] in {"on", "in", "during", "since"})
                or (index + 1 < len(words) and words[index + 1].isdigit())
            )
            if has_date_context:
                end = index + 1
                if end < len(words) and words[end].isdigit() and len(words[end]) == 4:
                    end += 1
                mark(index, end, "DATE", replace=True)
        elif lower[index] in WEEKDAYS and index > 0 and lower[index - 1] in {"on", "next", "last", "this"}:
            mark(index, index + 1, "DATE", replace=True)

    for index, word in enumerate(words):
        if lower[index] not in ORGANIZATION_SUFFIXES:
            continue
        start = index
        while start > 0 and words[start - 1].isalpha() and words[start - 1][0].isupper():
            start -= 1
        mark(start, index + 1, "ORG", replace=True)

    for index, cue in enumerate(lower):
        if cue not in LOCATION_CUES | ORGANIZATION_CUES or index + 1 >= len(words):
            continue
        candidate = _capitalized_span(words, index + 1)
        if candidate is None:
            continue
        start, end = candidate
        entity_type = "LOC" if cue in LOCATION_CUES else "ORG"
        mark(start, end, entity_type, replace=True)

    for index, verb in enumerate(lower):
        if verb not in PERSON_SUBJECT_VERBS or index == 0:
            continue
        end = index
        start = end - 1
        while start > 0 and words[start - 1].isalpha() and words[start - 1][0].isupper():
            start -= 1
        if all(word.isalpha() and word[0].isupper() for word in words[start:end]):
            mark(start, end, "PER", replace=True)

    for index, title in enumerate(lower):
        if title not in PERSON_TITLES:
            continue
        start = index + 1
        if start < len(words) and words[start] == ".":
            start += 1
        candidate = _capitalized_span(words, start)
        if candidate is not None:
            mark(*candidate, "PER", replace=True)

    return normalize_iob2(tags)


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
    assisted_tags = contextual_fallback(words, embedding_tags)
    return {
        "baseline": annotate_entities(words, baseline_tags),
        "word_embeddings": annotate_entities(words, embedding_tags),
        "context_assisted": annotate_entities(words, assisted_tags),
    }
