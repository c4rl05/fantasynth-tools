"""Static server for viewer.html with HTTP Range support (the stdlib one has none, and
Chrome's <audio> stalls at readyState 0 without it). Usage: python serve.py [port]

The page is code and the data is not, so they come from two roots (see workspace.py):
  /out/...  and  /audio/...   ->  <workspace>/out/..., <workspace>/audio/...
  everything else             ->  music-events/ (viewer.html)
viewer.html fetches `out/<slug>/events.json` and `audio/<slug>.wav` by relative URL, so it
needs no change. /slugs.json lists the workspace's slugs (every out/<slug>/ holding an
events.json) for the viewer's track menu."""
import json
import os
import posixpath
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from workspace import OUT, TOOL_DIR, WORKSPACE

DATA_DIRS = {"out", "audio"}  # first URL segments served from the workspace


def slugs():
    """Every slug in the workspace that has an events.json, sorted."""
    return sorted(p.parent.name for p in OUT.glob("*/events.json"))


def root_for(url_path):
    """The directory a request path is served from: the workspace for /out and /audio.
    The first segment is found the way the stdlib's translate_path will read the path
    (query/fragment cut, unquoted, normalised, empty segments skipped), so the two agree."""
    path = posixpath.normpath(unquote(url_path.split("?", 1)[0].split("#", 1)[0]))
    first = next((w for w in path.split("/") if w), "")
    return WORKSPACE if first in DATA_DIRS else TOOL_DIR


class RangeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(TOOL_DIR), **kw)

    def translate_path(self, path):
        # Pick the root per request, then let the stdlib do the safe URL -> file mapping
        # (it drops "..", drive and backslash components). A handler instance serves one
        # connection on one thread, so setting self.directory here is not shared state.
        self.directory = str(root_for(path))
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/slugs.json":
            body = json.dumps(slugs()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError):
                    return
                remaining -= len(chunk)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://127.0.0.1:{port}/viewer.html")
    print(f"  page from {TOOL_DIR}")
    print(f"  /out, /audio from {WORKSPACE}")
    ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
