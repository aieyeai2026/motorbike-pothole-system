"""Tests for async video evaluation job endpoints and non-blocking proxy timeout avoidance."""

import http.client
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from video_evaluator.server import VideoEvalRequestHandler  # noqa: E402


def _make_tiny_mp4(path: str, n_frames: int = 4) -> bytes:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    frame = np.full((180, 320, 3), (110, 110, 115), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        return f.read()


def test_async_process_video_returns_202_immediately_and_completes_via_poll(tmp_path):
    payload = _make_tiny_mp4(str(tmp_path / "tiny.mp4"), n_frames=4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), VideoEvalRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/process-video-async",
            body=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload)),
                "X-Max-Frames": "4",
            },
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 202
        assert body["status"] == "running"
        job_id = body["job_id"]
        conn.close()

        # Unknown job returns 404 JSON
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/job/doesnotexist")
        miss = conn.getresponse()
        assert miss.status == 404
        miss.read()
        conn.close()

        # Poll until completion
        final = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", f"/api/job/{job_id}")
            poll_resp = conn.getresponse()
            poll_data = json.loads(poll_resp.read().decode("utf-8"))
            conn.close()
            assert poll_resp.status == 200
            if poll_data["status"] == "done":
                final = poll_data
                break
            assert poll_data["status"] == "running"
            time.sleep(0.2)

        assert final is not None, "Async evaluation job did not finish before deadline"
        assert final["frames_processed"] == 4
        assert len(final["telemetry"]) == 4
        assert final["hud_url"].endswith(".mp4")
    finally:
        server.shutdown()
        server.server_close()


def test_app_js_uses_async_polling_and_safe_json_error_parsing():
    app_js_path = os.path.join(os.path.dirname(current_dir), "video_evaluator", "app.js")
    with open(app_js_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "/api/process-video-async" in src
    assert "/api/job/" in src
    assert "resp.text()" in src
