import unittest

from ner_embeddings.pipeline import contextual_fallback, load_models, predict_text


class ContextFallbackTests(unittest.TestCase):
    def test_finds_unseen_names_locations_and_dates_from_context(self):
        words = [
            "Avery", "Chen", "joined", "Northstar", "Robotics",
            "in", "Helsinki", "in", "2031", ".",
        ]
        tags = contextual_fallback(words, ["O"] * len(words))
        self.assertEqual(tags, [
            "B-PER", "I-PER", "O", "B-ORG", "I-ORG",
            "O", "B-LOC", "O", "B-DATE", "O",
        ])

    def test_accepts_arbitrary_user_text_and_returns_assisted_entities(self):
        models = load_models()
        result = predict_text(
            "Avery Chen joined Northstar Robotics in Helsinki in 2031.",
            models,
        )
        self.assertIn(
            {"token": "Avery Chen", "tag": "B-PER"},
            result["context_assisted"],
        )
        self.assertIn(
            {"token": "Northstar Robotics", "tag": "B-ORG"},
            result["context_assisted"],
        )
        self.assertIn(
            {"token": "Helsinki", "tag": "B-LOC"},
            result["context_assisted"],
        )
        self.assertIn(
            {"token": "2031", "tag": "B-DATE"},
            result["context_assisted"],
        )

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            predict_text("  ", load_models())


if __name__ == "__main__":
    unittest.main()
