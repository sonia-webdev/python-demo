import tempfile
import unittest
from pathlib import Path

import numpy as np

from ner_embeddings.dataset import read_conll
from ner_embeddings.embeddings import CooccurrenceEmbeddings
from ner_embeddings.metrics import entity_report, normalize_iob2, spans


class DatasetTests(unittest.TestCase):
    def test_reads_conll_sentences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.conll"
            path.write_text("Ana B-PER\nworks O\n\nParis B-LOC\n", encoding="utf-8")
            self.assertEqual(
                read_conll(path),
                [[("Ana", "B-PER"), ("works", "O")], [("Paris", "B-LOC")]],
            )


class EmbeddingTests(unittest.TestCase):
    def test_embeddings_have_configured_dimension_and_unknown_is_zero(self):
        embeddings = CooccurrenceEmbeddings(dimensions=8).fit([["one", "two"], ["two", "three"]])
        self.assertEqual(embeddings.vector("one").shape, (8,))
        np.testing.assert_array_equal(embeddings.vector("missing"), np.zeros(8, dtype=np.float32))


class MetricTests(unittest.TestCase):
    def test_repairs_invalid_iob_and_scores_exact_spans(self):
        self.assertEqual(normalize_iob2(["I-PER", "I-PER", "O"]), ["B-PER", "I-PER", "O"])
        self.assertEqual(spans(["B-PER", "I-PER", "O"]), {("PER", 0, 2)})
        report = entity_report([["B-PER", "O"]], [["B-PER", "O"]])
        self.assertEqual(report["PER"]["f1"], 1.0)
        self.assertEqual(report["micro_avg"]["support"], 1)
        wrong_type = entity_report([["B-PER"]], [["B-ORG"]])
        self.assertEqual(wrong_type["micro_avg"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
