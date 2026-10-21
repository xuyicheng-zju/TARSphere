#!/usr/bin/env python3
"""Serve the local demo website with single-byte-range MP4 support."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".vtt": "text/vtt"}

    def send_head(self):
        self._byte_range = None
        range_header = self.headers.get("Range")
        # Preserve the standard handler's directory redirects/listing/index,
        # conditional requests and ordinary file responses.
        path = Path(self.translate_path(self.path))
        if not range_header or not path.is_file():
            return super().send_head()
        # If-Range requires validator matching. Fall back to a complete response
        # when supplied rather than serving a potentially stale partial object.
        if self.headers.get("If-Range"):
            return super().send_head()
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        start = end = None
        if match:
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last and int(last) > 0:
                start = max(0, size - int(last))
                end = size - 1
        if start is None or end is None or start >= size or start > end:
            stream.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self._byte_range = (start, end)
        stream.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self._byte_range is None:
            return super().copyfile(source, outputfile)
        remaining = self._byte_range[1] - self._byte_range[0] + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Path to the separately distributed project website.",
    )
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error("--directory must be a directory")
    handler = partial(RangeHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(
            f"Serving {directory} at http://{args.bind}:{server.server_port}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
