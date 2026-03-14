#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""HTTP server for Clean Rooms ML inference. Implements /ping and /invocations on port 8080."""

import os, sys, logging, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
PORT = 8080

_model_error = None
try:
    from inference_handler import predict, load_model
    load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    _model_error = str(e)
    logger.error(f"Failed to load model: {e}")


class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/invocations":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                content_type = self.headers.get("Content-Type", "text/csv")
                body = self.rfile.read(content_length).decode("utf-8")
                if _model_error:
                    raise RuntimeError(f"Model not loaded: {_model_error}")
                result = predict(body, content_type)
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.end_headers()
                self.wfile.write(result.encode("utf-8"))
            except Exception as e:
                logger.error(f"Invocation error: {e}")
                traceback.print_exc()
                self.send_response(500); self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


if __name__ == "__main__":
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), InferenceHandler).serve_forever()
