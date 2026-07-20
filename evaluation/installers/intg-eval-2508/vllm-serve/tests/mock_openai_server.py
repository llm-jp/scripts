"""Minimal OpenAI-completions-compatible mock server for plumbing tests.

Implements just enough of vLLM's /v1/completions behavior to exercise the
serve-mode clients on a CPU-only node:

- accepts `prompt` as str / list[str] / list[int] / list[list[int]]
- echo=True + logprobs=N + max_tokens=0 -> prompt logprobs (loglikelihood path)
- max_tokens>0 -> canned generated text (generation path)
- GET /health -> 200 (matches vLLM's health endpoint)

Scores obtained through this server are meaningless; only request/response
compatibility is being tested.

Usage: python mock_openai_server.py [--port 8000]
"""

import argparse
import json
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def normalize_prompts(prompt):
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            return []
        if isinstance(prompt[0], int):
            return [prompt]  # single token-array prompt
        return prompt  # list of strings or list of token arrays
    raise ValueError(f"unsupported prompt type: {type(prompt)}")


def prompt_tokens(p):
    # Token-array prompts keep their ids; string prompts are pseudo-tokenized.
    if isinstance(p, list):
        return [str(t) for t in p]
    return p.split() or ["<empty>"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(length))

        echo = req.get("echo", False)
        want_logprobs = req.get("logprobs")
        max_tokens = req.get("max_tokens", 16)

        choices = []
        for i, p in enumerate(normalize_prompts(req.get("prompt", ""))):
            text = ""
            logprobs = None
            if echo and want_logprobs is not None:
                toks = prompt_tokens(p)
                logprobs = {
                    "tokens": toks,
                    "token_logprobs": [None] + [-1.0] * (len(toks) - 1),
                    # each token is also the argmax -> is_greedy True
                    "top_logprobs": [None] + [{t: -1.0} for t in toks[1:]],
                    "text_offset": list(range(len(toks))),
                }
            if max_tokens and max_tokens > 0:
                text = " 42"
            choices.append(
                {
                    "index": i,
                    "text": text,
                    "finish_reason": "stop",
                    "logprobs": logprobs,
                }
            )

        body = json.dumps(
            {
                "id": "cmpl-mock",
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.get("model", "mock"),
                "choices": choices,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock server listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
