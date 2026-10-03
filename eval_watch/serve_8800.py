#!/usr/bin/env python3
"""Static server for the :8800 K-Bot dashboard + an ON-DEMAND render trigger.

Replaces `python -m http.server 8800`. Same static serving of eval_watch/, plus:
  GET /request_render[?exp=auto|rough|amp|track|clip]
                       -> write the choice into render_request.flag; the watcher renders
                          on its next ~10 s poll, then returns to idle. "auto" (default)
                          = whatever is training right now (see watcher_isaac.sh).
This makes rendering cost ZERO GPU during training unless the user clicks Render.
"""
import http.server
import os
import socketserver
import threading

DIR = "/home/faisal/IsaacLab/eval_watch"
FLAG = os.path.join(DIR, "render_request.flag")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIR, **k)

    def do_GET(self):
        if self.path.split("?")[0] == "/request_render":
            from urllib.parse import parse_qs, urlparse
            exp = parse_qs(urlparse(self.path).query).get("exp", ["auto"])[0]
            if exp not in ("auto", "rough", "amp", "track", "clip", "multi"):
                exp = "auto"
            with open(FLAG, "w") as f:
                f.write(exp)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"render requested")
            return
        super().do_GET()

    def log_message(self, *a):  # quiet; the 404-noise polls flooded httpd.log
        pass


class Server(socketserver.ThreadingTCPServer):
    """THREADED (2026-08-03 fix). The original single-threaded TCPServer HUNG the
    whole dashboard: one stalled client (browser keep-alive / half-open tab)
    blocked the single accept loop forever — `ss` showed 6 connections queued
    behind a backlog of 5 while every request timed out, and the watcher's
    /request_render call hung for 120 s. Each request now gets its own daemon
    thread, so a stuck client can't take the server down.
    """

    allow_reuse_address = True
    daemon_threads = True          # don't let stuck threads block shutdown
    request_queue_size = 64        # deeper backlog than the default 5

    def handle_error(self, request, client_address):
        pass                       # a dropped client is normal; don't spam


if __name__ == "__main__":
    with Server(("0.0.0.0", 8800), Handler) as httpd:
        httpd.serve_forever()
