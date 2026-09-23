"""Development HTTP API. Run: python -m demo_api.app"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .jobs import (
    MAX_BYTES,
    SOCKET_TIMEOUT_SEC,
    CapacityError,
    JobStore,
    UploadError,
    UploadTimeout,
)


def make_handler(store: JobStore):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(SOCKET_TIMEOUT_SEC)
            super().setup()

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/health":
                self._json(200, {"ok": True, "model_connected": store.adapter is not None})
                return
            pieces = path.strip("/").split("/")
            if len(pieces) not in (3, 4) or pieces[:2] != ["api", "jobs"]:
                self._json(404, {"error": "Not found."})
                return
            job_id = pieces[2]
            if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
                self._json(404, {"error": "Job not found."})
                return
            job = store.get(job_id)
            if job is None:
                self._json(404, {"error": "Job not found."})
            elif len(pieces) == 3:
                self._json(200, job)
            elif pieces[3] == "result":
                result = store.result(job_id)
                if result is None:
                    self._json(409, {"error": "Result is not available for this job."})
                else:
                    self._json(200, result)
            else:
                self._json(404, {"error": "Not found."})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/jobs":
                self._json(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0:
                self._json(411, {"error": "A valid Content-Length is required."})
                return
            if length == 0:
                self._json(400, {"error": "Upload must not be empty."})
                return
            if length > MAX_BYTES:
                self._json(413, {"error": "Upload exceeds the 100 MiB limit."})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "video/mp4":
                self._json(415, {"error": "Content-Type must be video/mp4."})
                return
            filename = unquote(self.headers.get("X-File-Name", ""))
            try:
                job = store.create(filename, self.rfile, length, set_read_timeout=self.connection.settimeout)
            except CapacityError as error:
                self._json(429, {"error": str(error)})
                return
            except UploadTimeout as error:
                self._json(408, {"error": str(error)})
                return
            except UploadError as error:
                self._json(400, {"error": str(error)})
                return
            except Exception:
                # Disk and transport exceptions may contain private paths.
                self._json(500, {"error": "Upload could not be processed."})
                return
            self._json(202, job)

    return Handler


def main():
    store = JobStore()
    server = ThreadingHTTPServer(("127.0.0.1", 8765), make_handler(store))
    print("WIUT demo API at http://127.0.0.1:8765 (model adapter disconnected)")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
