#!/usr/bin/env python3
"""Static gallery server with single byte-range support for HTML video seeking."""
import argparse
import functools
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class VideoHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def send_head(self):
        self.remaining = None
        path = self.translate_path(self.path)
        request_range = self.headers.get('Range')
        if not request_range or not os.path.isfile(path):
            return super().send_head()
        try:
            stream = open(path, 'rb')
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(stream.fileno()).st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range.strip())
        try:
            if not match or not any(match.groups()):
                raise ValueError()
            left, right = match.groups()
            start = int(left) if left else max(0, size - int(right))
            end = min(int(right), size - 1) if left and right else size - 1
            if start > end or start >= size:
                raise ValueError()
        except ValueError:
            stream.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining:
                data = source.read(min(self.remaining, 1024 * 1024))
                if not data:
                    break
                outputfile.write(data)
                self.remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel in-flight downloads when scrubbing.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8878)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    handler = functools.partial(VideoHandler, directory=args.directory)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print(f'Gallery server with byte ranges: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
