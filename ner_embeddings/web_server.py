import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dataset import read_conll
from .embeddings import CooccurrenceEmbeddings, tokenize
from .features import embedding_features, sparse_features
from .metrics import normalize_iob2, spans
from .model import TokenClassifier


ROOT = Path(__file__).resolve().parent.parent
MAX_REQUEST_SIZE = 20_000


def load_models() -> tuple:
    train_rows = read_conll(ROOT / "data" / "train.conll")
    train_words = [[word for word, _ in sentence] for sentence in train_rows]
    train_labels = [tag for sentence in train_rows for _, tag in sentence]

    train_sparse, vectorizer = sparse_features(train_words)
    baseline = TokenClassifier().fit(train_sparse, train_labels)

    embeddings = CooccurrenceEmbeddings(dimensions=32).fit(train_words)
    embedded = TokenClassifier().fit(embedding_features(train_words, embeddings), train_labels)
    return vectorizer, baseline, embeddings, embedded


class NERHandler(BaseHTTPRequestHandler):
    vectorizer, baseline, embeddings, embedded = load_models()

    def _send(self, status: int, payload: dict | bytes, content_type: str) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            try:
                page = (ROOT / "web" / "index.html").read_bytes()
            except OSError:
                self._send(500, {"error": "Could not load the web interface."}, "application/json")
                return
            self._send(200, page, "text/html; charset=utf-8")
            return
        if self.path == "/results.json":
            try:
                report = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                self._send(500, {"error": f"Could not load evaluation report: {error}"}, "application/json")
                return
            self._send(200, report, "application/json; charset=utf-8")
            return
        self._send(404, {"error": "Not found."}, "application/json; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send(404, {"error": "Not found."}, "application/json; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_SIZE:
                self._send(400, {"error": "Text must be between 1 and 5,000 characters."}, "application/json; charset=utf-8")
                return
            request = json.loads(self.rfile.read(length))
            text = request.get("text") if isinstance(request, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > 5_000:
                self._send(400, {"error": "Text must be between 1 and 5,000 characters."}, "application/json; charset=utf-8")
                return

            words = tokenize(text)
            if not words:
                self._send(400, {"error": "No words found in the provided text."}, "application/json; charset=utf-8")
                return
            sparse, _ = sparse_features([words], self.vectorizer)
            baseline_tags = normalize_iob2(self.baseline.predict(sparse))
            embedded_features = embedding_features([words], self.embeddings)
            embedding_tags = normalize_iob2(self.embedded.predict(embedded_features))
            self._send(200, {
                "baseline": self._annotate(words, baseline_tags),
                "word_embeddings": self._annotate(words, embedding_tags),
            }, "application/json; charset=utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send(400, {"error": f"Invalid prediction request: {error}"}, "application/json; charset=utf-8")

    @staticmethod
    def _annotate(words: list[str], tags: list[str]) -> list[dict[str, str]]:
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

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), NERHandler)
    print("NER demo ready at http://127.0.0.1:8765 (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NER demo.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
