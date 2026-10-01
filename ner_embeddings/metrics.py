from collections import defaultdict

def spans(tags: list[str]) -> set[tuple[str, int, int]]:
    result: set[tuple[str, int, int]] = set()
    active_type: str | None = None
    start = 0
    for index in range(len(tags) + 1):
        tag = tags[index] if index < len(tags) else "O"
        prefix, _, entity_type = tag.partition("-")
        if active_type is not None and (prefix != "I" or entity_type != active_type):
            result.add((active_type, start, index))
            active_type = None
        if prefix == "B" or (prefix == "I" and active_type is None):
            active_type = entity_type
            start = index
    return result


def normalize_iob2(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    previous_type: str | None = None
    for tag in tags:
        prefix, _, entity_type = tag.partition("-")
        if prefix == "I" and entity_type != previous_type:
            prefix = "B"
        normalized_tag = f"{prefix}-{entity_type}" if prefix in {"B", "I"} else "O"
        normalized.append(normalized_tag)
        previous_type = entity_type if prefix in {"B", "I"} else None
    return normalized


def entity_report(
    expected: list[list[str]], predicted: list[list[str]]
) -> dict[str, dict[str, float | int]]:
    labels = sorted({entity_type for tags in expected for entity_type, _, _ in spans(tags)})
    true_by_type: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    pred_by_type: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for sentence_id, (gold_tags, predicted_tags) in enumerate(zip(expected, predicted)):
        for entity_type, start, end in spans(gold_tags):
            true_by_type[entity_type].append((str(sentence_id), start, end))
        for entity_type, start, end in spans(predicted_tags):
            pred_by_type[entity_type].append((str(sentence_id), start, end))

    report: dict[str, dict[str, float | int]] = {}
    for entity_type in labels:
        gold = set(true_by_type[entity_type])
        guess = set(pred_by_type[entity_type])
        true_positive = len(gold & guess)
        false_positive = len(guess - gold)
        false_negative = len(gold - guess)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(gold),
        }

    gold_all = {
        (sentence_id, entity_type, start, end)
        for sentence_id, tags in enumerate(expected)
        for entity_type, start, end in spans(tags)
    }
    predicted_all = {
        (sentence_id, entity_type, start, end)
        for sentence_id, tags in enumerate(predicted)
        for entity_type, start, end in spans(tags)
    }
    correct = len(gold_all & predicted_all)
    precision = correct / len(predicted_all) if predicted_all else 0.0
    recall = correct / len(gold_all) if gold_all else 0.0
    report["micro_avg"] = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "support": len(gold_all),
    }
    return report
