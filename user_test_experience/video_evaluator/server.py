#!/usr/bin/env python3
"""
Local HTTP Server & Real Rust C-ABI Video Pipeline Processor for the
Dual-Video Computer Vision Evaluator.

- Accepts any uploaded raw motorcycle video via POST /api/process-video
- Runs every frame through OpenCV pixel contour/edge extraction + the compiled
  Rust C-ABI library (libpothole_core.dylib: pothole_core_run_postprocess &
  pothole_core_compute_lane_centering)
- Outputs the processed H.264 HUD video and real per-frame telemetry JSON
- Supports HTTP Range requests (206 Partial Content) for millisecond scrubbing
"""

import argparse
import ctypes
import json
import mimetypes
import os
import re
import subprocess
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", ".."))
FIXTURES_DIR = os.path.join(REPO_ROOT, "benchmarks", "fixtures")
UPLOADS_DIR = os.path.join(FIXTURES_DIR, "uploaded_evals")
LIB_PATH = os.path.join(REPO_ROOT, "pothole_core", "target", "release", "libpothole_core.dylib")


class CDetectedPothole(ctypes.Structure):
    _fields_ = [
        ("x_min", ctypes.c_float),
        ("y_min", ctypes.c_float),
        ("x_max", ctypes.c_float),
        ("y_max", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("distance_meters", ctypes.c_float),
        ("severity_score", ctypes.c_float),
        ("lane_code", ctypes.c_int32),
        ("class_id", ctypes.c_int32),
    ]


class CLaneCentering(ctypes.Structure):
    _fields_ = [
        ("offset_meters", ctypes.c_float),
        ("dist_left_m", ctypes.c_float),
        ("dist_right_m", ctypes.c_float),
        ("status_code", ctypes.c_int32),
    ]


_RUST_LIB = None


def get_rust_lib():
    global _RUST_LIB
    if _RUST_LIB is not None:
        return _RUST_LIB
    if not os.path.exists(LIB_PATH):
        raise FileNotFoundError(f"Rust C-ABI library not found at {LIB_PATH}")
    lib = ctypes.CDLL(LIB_PATH)
    lib.pothole_core_run_postprocess.restype = ctypes.c_size_t
    lib.pothole_core_run_postprocess.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(CDetectedPothole),
        ctypes.c_size_t,
    ]
    lib.pothole_core_compute_lane_centering.restype = ctypes.c_int32
    lib.pothole_core_compute_lane_centering.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(CLaneCentering),
    ]
    _RUST_LIB = lib
    return lib


def extract_road_edges_from_frame(gray: np.ndarray, horizon_y: int) -> Tuple[float, float]:
    """Detect left and right road boundary normalized x-intercepts from actual frame pixels."""
    h, w = gray.shape
    roi = gray[horizon_y:int(h * 0.88), :]
    edges = cv2.Canny(roi, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=30, maxLineGap=20)

    x_left_candidates = []
    x_right_candidates = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 5 or abs(dy) < 10:
                continue
            slope = dy / float(dx)
            if -2.5 < slope < -0.25 and x1 < w * 0.55:
                # Project to bottom of road ROI
                x_bottom = x1 + (roi.shape[0] - y1) / slope
                x_left_candidates.append(np.clip(x_bottom / w, 0.02, 0.45))
            elif 0.25 < slope < 2.5 and x2 > w * 0.45:
                x_bottom = x1 + (roi.shape[0] - y1) / slope
                x_right_candidates.append(np.clip(x_bottom / w, 0.55, 0.98))

    x_left = float(np.median(x_left_candidates)) if x_left_candidates else 0.12
    x_right = float(np.median(x_right_candidates)) if x_right_candidates else 0.88
    return x_left, x_right


def extract_candidates_tensor_from_frame(frame: np.ndarray, horizon_y: int) -> Tuple[List[float], int]:
    """
    Extracts real visual anomaly & obstacle candidates from the frame pixels
    and packs them into a [1, 7, N] multi-class tensor for Rust C-ABI postprocessing.
    Classes: 0 = pothole, 1 = car/vehicle, 2 = debris.
    """
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y_start = horizon_y
    y_end = int(h * 0.86)  # exclude motorcycle hood/helmet at bottom
    road_gray = gray[y_start:y_end, :]

    # 1. Local background contrast & gradient analysis for dark road depressions (potholes) and debris
    local_bg = cv2.GaussianBlur(road_gray, (61, 61), 0)
    blackhat = cv2.subtract(local_bg, road_gray)
    grad_x = cv2.Sobel(road_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(road_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    _, thresh = cv2.threshold(blackhat, 22, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anchors: List[Tuple[float, float, float, float, float, float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 90 or area > (w * h * 0.15):
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(max(bh, 1))
        # Phase 5 false-positive suppression: reject wide crosswalk paint stripes
        if aspect > 5.5 or aspect < 0.25:
            continue

        # Check interior edge gradient to reject smooth canopy shadows
        patch_grad = float(np.mean(grad_mag[by:by + bh, bx:bx + bw]))
        patch_dark = float(np.mean(blackhat[by:by + bh, bx:bx + bw]))
        if patch_grad < 14.0:
            continue

        cx_640 = ((bx + bw * 0.5) / w) * 640.0
        cy_640 = (((by + y_start) + bh * 0.5) / h) * 640.0
        w_640 = (bw / w) * 640.0
        h_640 = (bh / h) * 640.0

        score = float(np.clip(0.52 + (patch_dark / 120.0) + (patch_grad / 180.0), 0.51, 0.96))
        if area >= 220 and 0.6 <= aspect <= 4.2:
            # Class 0: pothole
            anchors.append((cx_640, cy_640, w_640, h_640, score, 0.10, 0.10))
        else:
            # Class 2: debris
            anchors.append((cx_640, cy_640, w_640, h_640, 0.10, 0.10, score * 0.92))

    # 2. Vertical obstacle / vehicle detection via high vertical edge density boxes
    edges_full = cv2.Canny(gray[int(h * 0.25):y_end, :], 80, 180)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges_full, cv2.MORPH_CLOSE, v_kernel)
    v_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_v_offset = int(h * 0.25)

    for cnt in v_contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < w * 0.11 or bh < h * 0.14 or bw > w * 0.75:
            continue
        aspect = bw / float(max(bh, 1))
        if not (0.45 <= aspect <= 2.4):
            continue
        y_bottom = by + y_v_offset + bh
        if y_bottom < horizon_y + 15:
            continue
        cx_640 = ((bx + bw * 0.5) / w) * 640.0
        cy_640 = ((by + y_v_offset + bh * 0.5) / h) * 640.0
        w_640 = (bw / w) * 640.0
        h_640 = (bh / h) * 640.0
        edge_density = float(np.mean(edges_full[by:by + bh, bx:bx + bw])) / 255.0
        if edge_density > 0.08:
            conf = float(np.clip(0.55 + edge_density * 1.5, 0.55, 0.94))
            anchors.append((cx_640, cy_640, w_640, h_640, 0.05, conf, 0.05))

    if not anchors:
        return [], 0

    # Keep top 40 candidate anchors before passing to Rust C-ABI NMS
    anchors = anchors[:40]
    n = len(anchors)
    tensor: List[float] = []
    for ch in range(7):
        for a in anchors:
            tensor.append(float(a[ch]))
    return tensor, n


def process_video_through_rust_pipeline(
    input_path: str,
    output_path: str,
    max_frames: int = 900,
    target_fps: float = 30.0,
    convert_h264: bool = True,
) -> Dict[str, Any]:
    """
    Runs an uploaded video file through OpenCV pixel extraction + the compiled
    Rust C-ABI library (libpothole_core.dylib), writing the annotated HUD video
    and returning real per-frame telemetry.
    """
    lib = get_rust_lib()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames

    out_fps = min(fps_in, target_fps) if fps_in > 0 else 30.0
    step = max(1, int(round(fps_in / out_fps)))
    horizon_y = int(h * 0.40)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_raw_out = output_path + ".raw.mp4" if convert_h264 else output_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw_out, fourcc, out_fps, (w, h))

    telemetry: List[Dict[str, Any]] = []
    frame_idx = 0
    processed_count = 0

    while processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter_ns()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x_left, x_right = extract_road_edges_from_frame(gray, horizon_y)

        # Call Rust C-ABI lane centering
        c_center = CLaneCentering()
        lib.pothole_core_compute_lane_centering(
            ctypes.c_float(0.50),
            ctypes.c_float(x_left),
            ctypes.c_float(x_right),
            ctypes.c_float(3.6),
            ctypes.byref(c_center),
        )
        center_status = {0: "CENTERED", 1: "DRIFTING_LEFT", 2: "DRIFTING_RIGHT"}.get(
            c_center.status_code, "CENTERED"
        )

        # Extract pixel candidates & call Rust C-ABI multi-class postprocess
        tensor, n_anchors = extract_candidates_tensor_from_frame(frame, horizon_y)
        detections = []
        if n_anchors > 0:
            tensor_arr = (ctypes.c_float * len(tensor))(*tensor)
            max_out = 20
            out_buf = (CDetectedPothole * max_out)()
            count = lib.pothole_core_run_postprocess(
                tensor_arr,
                len(tensor),
                n_anchors,
                ctypes.c_float(640.0),
                ctypes.c_float(0.50),
                ctypes.c_float(0.45),
                ctypes.c_float(0.40),
                ctypes.c_float(1.25),
                out_buf,
                max_out,
            )
            for i in range(count):
                d = out_buf[i]
                lane_str = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}.get(d.lane_code, "CENTER")
                cls_str = {0: "POTHOLE", 1: "CAR", 2: "DEBRIS"}.get(d.class_id, "POTHOLE")
                detections.append({
                    "x_min": round(float(d.x_min), 4),
                    "y_min": round(float(d.y_min), 4),
                    "x_max": round(float(d.x_max), 4),
                    "y_max": round(float(d.y_max), 4),
                    "confidence": round(float(d.confidence), 3),
                    "distance_m": round(float(d.distance_meters), 2),
                    "severity": round(float(d.severity_score), 1),
                    "lane": lane_str,
                    "class_name": cls_str,
                    "class_id": int(d.class_id),
                })

        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # Draw HUD on output frame
        overlay = frame.copy()
        cv2.line(overlay, (0, horizon_y), (w, horizon_y), (255, 229, 0), 1)
        cv2.line(overlay, (int(w * 0.42), horizon_y), (int(x_left * w), h), (118, 230, 0), 2)
        cv2.line(overlay, (int(w * 0.58), horizon_y), (int(x_right * w), h), (118, 230, 0), 2)

        potholes = [d for d in detections if d["class_id"] == 0]
        closest_pothole_m = min((d["distance_m"] for d in potholes), default=None)
        closest_any = min(detections, key=lambda d: d["distance_m"], default=None)

        crit_pothole = None
        for det in detections:
            bx1, by1 = int(det["x_min"] * w), int(det["y_min"] * h)
            bx2, by2 = int(det["x_max"] * w), int(det["y_max"] * h)
            is_crit = det["class_id"] == 0 and det["distance_m"] < 8.0 and det["confidence"] >= 0.75
            if is_crit and crit_pothole is None:
                crit_pothole = det

            if det["class_id"] == 0:
                color = (68, 23, 255) if is_crit else (255, 229, 0)
            elif det["class_id"] == 1:
                color = (0, 170, 255)
            else:
                color = (0, 230, 118)

            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 2)
            label = f"{det['class_name']} {det['distance_m']:.1f}m [{det['lane']}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(overlay, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), color, -1)
            cv2.putText(
                overlay, label, (bx1 + 3, max(10, by1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA
            )

        cv2.rectangle(overlay, (0, 0), (w, 30), (15, 15, 22), -1)
        hdr = (
            f"RUST C-ABI | Frame {processed_count} | Hazards: {len(detections)} | "
            f"Ego: {center_status} ({c_center.offset_meters:+.2f}m) | {elapsed_ms:.2f}ms"
        )
        cv2.putText(overlay, hdr, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 229, 0), 1, cv2.LINE_AA)

        if crit_pothole is not None:
            cv2.rectangle(overlay, (w // 6, 34), (w * 5 // 6, 64), (40, 15, 230), -1)
            msg = f"CRITICAL POTHOLE < 8M: {crit_pothole['distance_m']:.1f}m [{crit_pothole['lane']}]"
            cv2.putText(overlay, msg, (w // 6 + 14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(cv2.addWeighted(overlay, 0.88, frame, 0.12, 0))

        telemetry.append({
            "frame": processed_count,
            "time_s": round(processed_count / out_fps, 3),
            "hazard_count": len(detections),
            "pothole_count": len(potholes),
            "closest_distance_m": closest_pothole_m if closest_pothole_m is not None else (
                closest_any["distance_m"] if closest_any else None
            ),
            "closest_is_critical": bool(crit_pothole is not None),
            "assigned_lane": closest_any["lane"] if closest_any else "CENTER",
            "centering_status": center_status,
            "offset_m": round(float(c_center.offset_meters), 2),
            "postprocess_ms": round(elapsed_ms, 2),
            "fps": round(1000.0 / max(elapsed_ms, 0.1), 1),
        })

        processed_count += 1
        frame_idx += 1

    cap.release()
    writer.release()

    if convert_h264 and os.path.exists("/usr/bin/avconvert"):
        res = subprocess.run(
            [
                "/usr/bin/avconvert",
                "--source", tmp_raw_out,
                "--preset", "PresetHighestQuality",
                "--output", output_path,
                "--replace",
            ],
            capture_output=True,
        )
        if res.returncode == 0 and os.path.exists(output_path):
            os.remove(tmp_raw_out)
        else:
            os.replace(tmp_raw_out, output_path)

    return {
        "output_path": output_path,
        "frames_processed": processed_count,
        "fps": out_fps,
        "telemetry": telemetry,
    }


def parse_byte_range(range_header: Optional[str], file_size: int) -> Optional[Tuple[int, int]]:
    """Parse HTTP Range header (e.g. 'bytes=0-1023') into (start, end) inclusive."""
    if not range_header or file_size <= 0:
        return None
    match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        return None
    start_str, end_str = match.groups()
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else file_size - 1
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    return (start, end)


def get_available_presets() -> List[Dict[str, str]]:
    """Return available benchmark video pairs from benchmarks/fixtures/."""
    return [
        {
            "id": "ride_test",
            "name": "Benchmark Footage (ride_test.mp4 — Run Live Rust Pipeline)",
            "raw_url": "/fixtures/ride_test.mp4",
            "hud_url": "/fixtures/pipeline_output/pipeline_output.mp4",
            "fps": "30.0",
        }
    ]


class VideoEvalRequestHandler(SimpleHTTPRequestHandler):
    """Serves evaluator frontend, benchmark fixtures with Range support, and live video processing API."""

    def translate_path(self, path: str) -> str:
        parsed = unquote(urlparse(path).path)
        if parsed.startswith("/fixtures/"):
            rel = parsed[len("/fixtures/"):]
            return os.path.abspath(os.path.join(FIXTURES_DIR, rel))
        rel = parsed.lstrip("/") or "index.html"
        return os.path.abspath(os.path.join(EVAL_DIR, rel))

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/process-video":
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self._send_json(400, {"error": "Empty video upload payload"})
                return

            os.makedirs(UPLOADS_DIR, exist_ok=True)
            job_id = uuid.uuid4().hex[:10]
            raw_filename = f"raw_{job_id}.mp4"
            hud_filename = f"hud_{job_id}.mp4"
            raw_path = os.path.join(UPLOADS_DIR, raw_filename)
            hud_path = os.path.join(UPLOADS_DIR, hud_filename)

            remaining = content_length
            with open(raw_path, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)

            try:
                max_frames = int(self.headers.get("X-Max-Frames", "600"))
                result = process_video_through_rust_pipeline(
                    raw_path, hud_path, max_frames=max_frames, convert_h264=True
                )
                self._send_json(200, {
                    "status": "success",
                    "job_id": job_id,
                    "raw_url": f"/fixtures/uploaded_evals/{raw_filename}",
                    "hud_url": f"/fixtures/uploaded_evals/{hud_filename}",
                    "frames_processed": result["frames_processed"],
                    "fps": result["fps"],
                    "telemetry": result["telemetry"],
                })
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if parsed_path == "/api/process-fixture":
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            raw_path = os.path.join(FIXTURES_DIR, "ride_test.mp4")
            hud_filename = "hud_ride_test_live.mp4"
            hud_path = os.path.join(UPLOADS_DIR, hud_filename)
            try:
                result = process_video_through_rust_pipeline(
                    raw_path, hud_path, max_frames=450, convert_h264=True
                )
                self._send_json(200, {
                    "status": "success",
                    "raw_url": "/fixtures/ride_test.mp4",
                    "hud_url": f"/fixtures/uploaded_evals/{hud_filename}",
                    "frames_processed": result["frames_processed"],
                    "fps": result["fps"],
                    "telemetry": result["telemetry"],
                })
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self.send_error(404, "Endpoint not found")

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/presets":
            self._send_json(200, {"presets": get_available_presets()})
            return

        file_path = self.translate_path(self.path)
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "index.html")

        if not os.path.isfile(file_path):
            self.send_error(404, "File not found")
            return

        file_size = os.path.getsize(file_path)
        ctype, _ = mimetypes.guess_type(file_path)
        ctype = ctype or "application/octet-stream"

        byte_range = parse_byte_range(self.headers.get("Range"), file_size)
        if byte_range is not None:
            start, end = byte_range
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(file_size))
            self.end_headers()
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)


def run_server(host: str = "127.0.0.1", port: int = 8085) -> None:
    server = ThreadingHTTPServer((host, port), VideoEvalRequestHandler)
    print("============================================================")
    print("Motorbike Vision Model Evaluator (Real Rust C-ABI Pipeline):")
    print(f"  http://{host}:{port}")
    print("============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local Video Evaluator HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8085, help="Bind port (default: 8085)")
    args = parser.parse_args()
    run_server(args.host, args.port)
