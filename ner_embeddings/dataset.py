from pathlib import Path


def read_conll(path: Path) -> list[list[tuple[str, str]]]:
    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            fields = line.split()
            if len(fields) != 2 or not fields[1].startswith(("O", "B-", "I-")):
                raise ValueError(f"Invalid CoNLL row at {path}:{line_number}: {line!r}")
            current.append((fields[0], fields[1]))

    if current:
        sentences.append(current)
    if not sentences:
        raise ValueError(f"No sentences found in {path}")
    return sentences
