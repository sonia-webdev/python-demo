import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .pipeline import annotate_entities, load_models, predict_text


ROOT = Path(__file__).resolve().parent.parent
MAX_REQUEST_SIZE = 20_000


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

            predictions = predict_text(text, (
                self.vectorizer, self.baseline, self.embeddings, self.embedded,
            ))
            self._send(200, predictions, "application/json; charset=utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send(400, {"error": f"Invalid prediction request: {error}"}, "application/json; charset=utf-8")

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
