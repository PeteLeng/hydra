"""HTTP serving of a trained checkpoint: text in, prediction out. No act knowledge."""
import argparse
from dataclasses import asdict
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from joint_model import load_checkpoint
from predict import predict

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "showroom-zh" / "best.pt"


class ServedModel:
    def __init__(self, checkpoint):
        settings = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
        self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint.parent / "tokenizer", local_files_only=True)
        self.model = load_checkpoint(checkpoint)
        dataset = settings.get("dataset", {})
        self.max_length = settings["max_length"]
        # Legacy showroom runs retain UNK; the standalone SNIPS trainer excluded it.
        self.tokenization = dataset.get("tokenization",
                     "showroom" if "UNK" in self.model.labels.intent_names else "whitespace")
        self.format = dataset.get("format", "snips")
        self.lock = threading.Lock()

    def parse(self, text):
        with self.lock:
            return predict(text, self.model, self.tokenizer, self.max_length,
                           tokenization=self.tokenization, format=self.format)


def handler(model):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/parse":
                self.send_error(404)
                return
            status, body = self.parse_body()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def parse_body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(request, dict) or not isinstance(request.get("text"), str):
                    raise ValueError("Expected a JSON body with a string 'text' field")
                prediction = model.parse(request["text"])
                return 200, json.dumps(asdict(prediction), ensure_ascii=False).encode("utf-8")
            except ValueError as error:
                return 400, json.dumps({"error": str(error)}).encode("utf-8")
            except Exception as error:  # keep serving; report what broke
                return 500, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode("utf-8")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"Defaults to {DEFAULT_CHECKPOINT}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()
    if args.text:
        parser.error("This is a server; predict.py classifies single sentences")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}; supply --checkpoint")
    model = ServedModel(args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), handler(model))
    print(f"serving {args.checkpoint} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
