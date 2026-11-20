#!/usr/bin/env python3
"""
Phase 6 Automotive-Grade AR HUD & Real Rust C-ABI Video Pipeline Processor
for the Dual-Video Computer Vision Evaluator.

1. Calibrates road-level horizon strictly to the physical vanishing horizon (50% - 58%).
2. Fits dynamic 2nd-degree polynomial curved road boundaries: x(y) = a*y^2 + b*y + c.
3. Hard-filters all candidates against the curved drivable asphalt polygon via
   compiled Rust FFI (pothole_core_classify_hazard_corridor) + HSV asphalt mask
   (zero off-road boxes on trees, fences, buildings, or parked cars).
4. Projects car backup-camera style dynamic curved motorcycle trajectory tracks
   with 5m, 10m, and 15m horizontal distance ticks.
5. Highlights trajectory in crimson and triggers '⚠️ POTHOLE IN PATH: {d}m' only
   when a road pothole (<10m) intersects the motorcycle's actual trajectory corridor.
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
    lib.pothole_core_classify_hazard_corridor.restype = ctypes.c_int32
    lib.pothole_core_classify_hazard_corridor.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _RUST_LIB = lib
    return lib


def compute_asphalt_mask(frame: np.ndarray) -> np.ndarray:
    """
    Segments low-saturation gray asphalt road surface pixels, rejecting green
    foliage, high-saturation painted cars (red/yellow/blue), bright sky, and dark shadows.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Gray asphalt has low saturation (S <= 58) and mid brightness (45 <= V <= 205)
    b, g, r = cv2.split(frame)
    max_c = np.maximum(np.maximum(b, g), r)
    min_c = np.minimum(np.minimum(b, g), r)
    chroma = max_c - min_c

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat <= 60) & (val >= 45) & (val <= 205) & (chroma <= 32)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def calibrate_road_horizon_norm(frame: np.ndarray) -> float:
    """
    Calibrates the road-level horizon vanishing point strictly within [0.50, 0.58]
    by locating the highest vertical row in the central road corridor where continuous
    gray asphalt road surface begins.
    """
    h, w, _ = frame.shape
    asphalt = compute_asphalt_mask(frame)
    x0, x1 = int(w * 0.35), int(w * 0.65)
    y_min = int(h * 0.50)
    y_max = int(h * 0.58)

    for y in range(y_min, y_max + 1):
        row_ratio = float(np.mean(asphalt[y:min(h, y + 6), x0:x1] > 0))
        if row_ratio >= 0.48:
            return round(float(np.clip(y / float(h), 0.50, 0.58)), 3)
    return 0.54


def fit_curved_road_boundaries(
    frame: np.ndarray,
    horizon_y: int,
    prev_left: Optional[np.ndarray] = None,
    prev_right: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float], float]:
    """
    Extracts curving left and right asphalt road boundaries using sliding horizontal
    scanlines from y_bottom (0.83) to y_horizon and fits 2nd-degree polynomials:
      x_left(y)  = a1 * y^2 + b1 * y + c1
      x_right(y) = a2 * y^2 + b2 * y + c2
    Also returns the estimated road steering curvature kappa in [-0.24, 0.24].
    """
    h, w, _ = frame.shape
    asphalt = compute_asphalt_mask(frame)
    horizon_norm = horizon_y / float(h)
    bottom_norm = 0.83

    y_samples = np.linspace(horizon_norm + 0.01, bottom_norm, 10)
    left_xs = []
    right_xs = []
    valid_ys = []

    center_x = 0.50
    for yn in y_samples:
        py = int(yn * h)
        prog = (yn - horizon_norm) / max(0.05, bottom_norm - horizon_norm)
        # Expected road corridor bounds widen from [0.36, 0.64] at horizon to [0.08, 0.92] at bottom
        default_l = 0.37 - 0.29 * prog
        default_r = 0.63 + 0.29 * prog

        strip = asphalt[max(0, py - 3):min(h, py + 4), :]
        col_score = np.mean(strip > 0, axis=0)
        cx_px = int(center_x * w)

        # Scan left from center for non-asphalt boundary transition
        found_l = default_l
        non_asphalt_run = 0
        for px in range(cx_px, int(w * 0.04), -2):
            if col_score[px] < 0.35:
                non_asphalt_run += 1
                if non_asphalt_run >= 4:
                    found_l = (px + 6) / float(w)
                    break
            else:
                non_asphalt_run = 0
        found_l = float(np.clip(found_l, default_l - 0.08, 0.44))

        # Scan right from center for non-asphalt boundary transition
        found_r = default_r
        non_asphalt_run = 0
        for px in range(cx_px, int(w * 0.96), 2):
            if col_score[px] < 0.35:
                non_asphalt_run += 1
                if non_asphalt_run >= 4:
                    found_r = (px - 6) / float(w)
                    break
            else:
                non_asphalt_run = 0
        found_r = float(np.clip(found_r, 0.56, default_r + 0.08))

        if found_r - found_l >= 0.16:
            valid_ys.append(yn)
            left_xs.append(found_l)
            right_xs.append(found_r)
            center_x = 0.5 * (found_l + found_r)

    if len(valid_ys) >= 4:
        l_poly = np.polyfit(valid_ys, left_xs, 2)
        r_poly = np.polyfit(valid_ys, right_xs, 2)
        # Limit quadratic curvature coefficient to prevent wild oscillation
        l_poly[0] = np.clip(l_poly[0], -1.2, 1.2)
        r_poly[0] = np.clip(r_poly[0], -1.2, 1.2)
    else:
        l_poly = np.array([0.0, -0.95, 0.89])
        r_poly = np.array([0.0, 0.95, 0.11])

    if prev_left is not None and prev_right is not None:
        l_poly = 0.72 * prev_left + 0.28 * l_poly
        r_poly = 0.72 * prev_right + 0.28 * r_poly

    # Evaluate road centerline at horizon vs bottom to compute steering curvature
    l_top = float(np.polyval(l_poly, horizon_norm + 0.02))
    r_top = float(np.polyval(r_poly, horizon_norm + 0.02))
    l_bot = float(np.polyval(l_poly, bottom_norm))
    r_bot = float(np.polyval(r_poly, bottom_norm))
    center_top = 0.5 * (l_top + r_top)
    center_bot = 0.5 * (l_bot + r_bot)
    steering_curv = float(np.clip((center_top - center_bot) * 0.95, -0.22, 0.22))

    return [float(c) for c in l_poly], [float(c) for c in r_poly], steering_curv


def classify_hazard_in_corridor_ffi(
    x_min: float,
    x_max: float,
    y_bottom: float,
    y_horizon: float,
    frame_bottom: float,
    left_poly: List[float],
    right_poly: List[float],
    ego_x: float = 0.50,
    steering_curv: float = 0.0,
    base_half_width: float = 0.11,
) -> int:
    """
    Calls compiled Rust pothole_core_classify_hazard_corridor:
      0 = off-road (outside curved drivable polygon)
      1 = on drivable road, outside motorcycle trajectory corridor
      2 = on drivable road AND inside motorcycle trajectory corridor
    """
    lib = get_rust_lib()
    l_arr = (ctypes.c_float * 3)(*left_poly)
    r_arr = (ctypes.c_float * 3)(*right_poly)
    return int(
        lib.pothole_core_classify_hazard_corridor(
            ctypes.c_float(x_min),
            ctypes.c_float(x_max),
            ctypes.c_float(y_bottom),
            ctypes.c_float(y_horizon),
            ctypes.c_float(frame_bottom),
            l_arr,
            r_arr,
            ctypes.c_float(ego_x),
            ctypes.c_float(steering_curv),
            ctypes.c_float(base_half_width),
        )
    )


def build_drivable_polygon_mask(
    h: int,
    w: int,
    horizon_norm: float,
    bottom_norm: float,
    left_poly: List[float],
    right_poly: List[float],
    inner_margin: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the binary mask and boundary polyline points of the curved drivable road surface."""
    ys = np.linspace(horizon_norm, bottom_norm, 24)
    left_pts = []
    right_pts = []
    for yn in ys:
        py = int(yn * h)
        lx = int(np.clip(np.polyval(left_poly, yn) + inner_margin, 0.02, 0.48) * w)
        rx = int(np.clip(np.polyval(right_poly, yn) - inner_margin, 0.52, 0.98) * w)
        left_pts.append([lx, py])
        right_pts.append([rx, py])

    poly_pts = np.array(left_pts + right_pts[::-1], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts], 255)

    # Also return the un-margined visual boundary curves for drawing
    vis_left = []
    vis_right = []
    for yn in ys:
        py = int(yn * h)
        vis_left.append([int(np.clip(np.polyval(left_poly, yn), 0.01, 0.49) * w), py])
        vis_right.append([int(np.clip(np.polyval(right_poly, yn), 0.51, 0.99) * w), py])

    return mask, np.array(vis_left, dtype=np.int32), np.array(vis_right, dtype=np.int32)


def extract_pothole_candidates_on_drivable_road(
    frame: np.ndarray,
    horizon_norm: float,
    bottom_norm: float,
    left_poly: List[float],
    right_poly: List[float],
) -> Tuple[List[float], int]:
    """
    Phase 6 Strict Road-Surface Pothole Extractor:
    - Searches ONLY inside the curved drivable road polygon AND low-saturation gray asphalt mask.
    - Verifies that the surrounding ring around each dark depression is homogeneous gray asphalt
      (eliminating cars, motorcycles, roadside fences, trees, and shadows).
    - Packs genuine road-surface potholes (class 0) into a [1, 7, N] tensor for Rust C-ABI NMS.
    """
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    asphalt_mask = compute_asphalt_mask(frame)
    drivable_mask, _, _ = build_drivable_polygon_mask(
        h, w, horizon_norm, bottom_norm, left_poly, right_poly, inner_margin=0.035
    )
    road_only_mask = cv2.bitwise_and(drivable_mask, asphalt_mask)

    local_bg = cv2.GaussianBlur(gray, (55, 55), 0)
    depression = cv2.subtract(local_bg, gray)
    depression_masked = cv2.bitwise_and(depression, depression, mask=drivable_mask)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    _, thresh = cv2.threshold(depression_masked, 24, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anchors: List[Tuple[float, float, float, float, float, float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Potholes on the road surface are compact localized defects, never huge vehicle blobs
        if area < 95 or area > (w * h * 0.035):
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < 12 or bh < 8 or bw > w * 0.24 or bh > h * 0.16:
            continue
        aspect = bw / float(max(bh, 1))
        if aspect > 4.5 or aspect < 0.45:
            continue

        # Verify surrounding context ring is genuine gray asphalt (not a car tire/shadow)
        pad = 10
        rx0, ry0 = max(0, bx - pad), max(0, by - pad)
        rx1, ry1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        surround_asphalt_ratio = float(np.mean(road_only_mask[ry0:ry1, rx0:rx1] > 0))
        if surround_asphalt_ratio < 0.42:
            continue

        patch_grad = float(np.mean(grad_mag[by:by + bh, bx:bx + bw]))
        patch_dark = float(np.mean(depression[by:by + bh, bx:bx + bw]))
        if patch_grad < 16.0 or patch_dark < 22.0:
            continue

        cx_640 = ((bx + bw * 0.5) / w) * 640.0
        cy_640 = ((by + bh * 0.5) / h) * 640.0
        w_640 = (bw / w) * 640.0
        h_640 = (bh / h) * 640.0
        score = float(np.clip(0.55 + (patch_dark / 110.0) + (patch_grad / 160.0), 0.55, 0.96))
        anchors.append((cx_640, cy_640, w_640, h_640, score, 0.05, 0.05))

    if not anchors:
        return [], 0

    # Sort by confidence and pass top 12 road-surface pothole candidates to Rust NMS
    anchors.sort(key=lambda a: a[4], reverse=True)
    anchors = anchors[:12]
    n = len(anchors)
    tensor: List[float] = []
    for ch in range(7):
        for a in anchors:
            tensor.append(float(a[ch]))
    return tensor, n


def compute_trajectory_point_pair(
    yn: float,
    horizon_norm: float,
    bottom_norm: float,
    ego_x: float,
    steering_curv: float,
    base_half_w: float = 0.11,
) -> Tuple[float, float]:
    """Matches pothole_core::geometry::compute_trajectory_corridor for rendering."""
    span = max(0.05, bottom_norm - horizon_norm)
    prog = float(np.clip((yn - horizon_norm) / span, 0.0, 1.0))
    dist_from_bike = 1.0 - prog
    center_x = ego_x + steering_curv * dist_from_bike * dist_from_bike
    half_w = base_half_w * (0.22 + 0.78 * prog)
    return center_x - half_w, center_x + half_w


def inverse_y_for_distance(distance_m: float, horizon_norm: float) -> float:
    """Matches pothole_core::geometry::inverse_perspective_y_for_distance (2m..30m)."""
    d = float(np.clip(distance_m, 2.0, 30.0))
    curved = (d - 2.0) / 28.0
    road_progress = float(np.clip(1.0 - curved, 0.0, 1.0)) ** (1.0 / 0.65)
    return float(np.clip(horizon_norm + road_progress * (1.0 - horizon_norm), horizon_norm, 1.0))


def process_video_through_rust_pipeline(
    input_path: str,
    output_path: str,
    max_frames: int = 600,
    target_fps: float = 30.0,
    convert_h264: bool = True,
) -> Dict[str, Any]:
    """
    Phase 6 Automotive AR HUD Pipeline:
    - Calibrates true road-level horizon (50% - 58%).
    - Fits dynamic 2nd-degree curved road boundaries.
    - Hard-filters all candidates against the curved drivable polygon in Rust C-ABI.
    - Renders car backup-camera style curved trajectory tracks with 5m/10m/15m ticks.
    - Triggers crimson path highlight + '⚠️ POTHOLE IN PATH: {d}m' only for in-path collisions.
    """
    lib = get_rust_lib()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    out_fps = min(fps_in, target_fps) if fps_in > 0 else 30.0
    step = max(1, int(round(fps_in / out_fps)))
    bottom_norm = 0.83

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_raw_out = output_path + ".raw.mp4" if convert_h264 else output_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw_out, fourcc, out_fps, (w, h))

    telemetry: List[Dict[str, Any]] = []
    frame_idx = 0
    processed_count = 0
    smooth_horizon = 0.54
    prev_left_poly: Optional[np.ndarray] = None
    prev_right_poly: Optional[np.ndarray] = None

    while processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter_ns()

        # 1. Calibrate road-level horizon (50% - 58%) with smooth temporal damping
        raw_horizon = calibrate_road_horizon_norm(frame)
        smooth_horizon = round(0.80 * smooth_horizon + 0.20 * raw_horizon, 4)
        horizon_y = int(smooth_horizon * h)

        # 2. Fit dynamic 2nd-degree polynomial curved road boundaries
        left_poly, right_poly, steering_curv = fit_curved_road_boundaries(
            frame, horizon_y, prev_left_poly, prev_right_poly
        )
        prev_left_poly = np.array(left_poly)
        prev_right_poly = np.array(right_poly)

        # 3. Compute Rust C-ABI lane centering at near-field ground contact
        x_left_bot = float(np.clip(np.polyval(left_poly, bottom_norm), 0.02, 0.48))
        x_right_bot = float(np.clip(np.polyval(right_poly, bottom_norm), 0.52, 0.98))
        c_center = CLaneCentering()
        lib.pothole_core_compute_lane_centering(
            ctypes.c_float(0.50),
            ctypes.c_float(x_left_bot),
            ctypes.c_float(x_right_bot),
            ctypes.c_float(3.6),
            ctypes.byref(c_center),
        )
        center_status = {0: "CENTERED", 1: "DRIFTING_LEFT", 2: "DRIFTING_RIGHT"}.get(
            c_center.status_code, "CENTERED"
        )

        # 4. Extract road-surface pothole candidates & run Rust C-ABI postprocess + corridor filter
        tensor, n_anchors = extract_pothole_candidates_on_drivable_road(
            frame, smooth_horizon, bottom_norm, left_poly, right_poly
        )
        detections = []
        if n_anchors > 0:
            tensor_arr = (ctypes.c_float * len(tensor))(*tensor)
            max_out = 10
            out_buf = (CDetectedPothole * max_out)()
            count = lib.pothole_core_run_postprocess(
                tensor_arr,
                len(tensor),
                n_anchors,
                ctypes.c_float(640.0),
                ctypes.c_float(0.55),
                ctypes.c_float(0.45),
                ctypes.c_float(smooth_horizon),
                ctypes.c_float(1.25),
                out_buf,
                max_out,
            )
            for i in range(count):
                d = out_buf[i]
                corridor_code = classify_hazard_in_corridor_ffi(
                    float(d.x_min),
                    float(d.x_max),
                    float(d.y_max),
                    smooth_horizon,
                    bottom_norm,
                    left_poly,
                    right_poly,
                    0.50,
                    steering_curv,
                    0.11,
                )
                # Hard filter: suppress everything outside the curved drivable road polygon
                if corridor_code == 0:
                    continue
                lane_str = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}.get(d.lane_code, "CENTER")
                detections.append({
                    "x_min": round(float(d.x_min), 4),
                    "y_min": round(float(d.y_min), 4),
                    "x_max": round(float(d.x_max), 4),
                    "y_max": round(float(d.y_max), 4),
                    "confidence": round(float(d.confidence), 3),
                    "distance_m": round(float(d.distance_meters), 2),
                    "severity": round(float(d.severity_score), 1),
                    "lane": lane_str,
                    "in_path": corridor_code == 2,
                })

        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # ── 5. Render Automotive AR HUD Overlay ──
        overlay = frame.copy()
        _, vis_left_pts, vis_right_pts = build_drivable_polygon_mask(
            h, w, smooth_horizon, bottom_norm, left_poly, right_poly, inner_margin=0.0
        )

        # Subtle road-level horizon line (calibrated at 50-58%)
        cv2.line(overlay, (0, horizon_y), (w, horizon_y), (180, 210, 0), 1, cv2.LINE_AA)

        # Smooth curved neon cyan road boundaries contouring the asphalt edges
        cv2.polylines(overlay, [vis_left_pts], False, (255, 229, 0), 2, cv2.LINE_AA)
        cv2.polylines(overlay, [vis_right_pts], False, (255, 229, 0), 2, cv2.LINE_AA)

        # Check for imminent in-path collision (< 10m directly inside trajectory corridor)
        in_path_potholes = [d for d in detections if d["in_path"] and d["distance_m"] < 10.0]
        crit_pothole = min(in_path_potholes, key=lambda d: d["distance_m"], default=None)

        # Dynamic Backup-Camera Vehicle Trajectory Tracks
        traj_ys = np.linspace(smooth_horizon + 0.02, bottom_norm, 20)
        traj_l_pts = []
        traj_r_pts = []
        for yn in traj_ys:
            tl, tr = compute_trajectory_point_pair(yn, smooth_horizon, bottom_norm, 0.50, steering_curv, 0.11)
            traj_l_pts.append([int(tl * w), int(yn * h)])
            traj_r_pts.append([int(tr * w), int(yn * h)])

        traj_color = (68, 23, 255) if crit_pothole is not None else (118, 230, 0)
        traj_thickness = 3 if crit_pothole is not None else 2
        cv2.polylines(overlay, [np.array(traj_l_pts, dtype=np.int32)], False, traj_color, traj_thickness, cv2.LINE_AA)
        cv2.polylines(overlay, [np.array(traj_r_pts, dtype=np.int32)], False, traj_color, traj_thickness, cv2.LINE_AA)

        # Horizontal distance ticks along trajectory corridor at 5m, 10m, and 15m
        for tick_m in (5.0, 10.0, 15.0):
            yn_tick = inverse_y_for_distance(tick_m, smooth_horizon)
            if smooth_horizon < yn_tick <= bottom_norm:
                tl, tr = compute_trajectory_point_pair(yn_tick, smooth_horizon, bottom_norm, 0.50, steering_curv, 0.11)
                py_t = int(yn_tick * h)
                px_l, px_r = int(tl * w), int(tr * w)
                cv2.line(overlay, (px_l, py_t), (px_r, py_t), traj_color, 1, cv2.LINE_AA)
                cv2.putText(
                    overlay, f"{int(tick_m)}m", (px_r + 5, py_t + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, traj_color, 1, cv2.LINE_AA
                )

        # Render road-surface hazards: unobtrusive dots outside path vs crimson reticle in path
        for det in detections:
            cx_px = int(((det["x_min"] + det["x_max"]) * 0.5) * w)
            cy_px = int(((det["y_min"] + det["y_max"]) * 0.5) * h)

            if not det["in_path"]:
                # Safe (outside bike trajectory path): faint, unobtrusive road dot only
                cv2.circle(overlay, (cx_px, cy_px), 4, (255, 229, 0), -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx_px, cy_px), 6, (255, 229, 0), 1, cv2.LINE_AA)
                continue

            bx1, by1 = int(det["x_min"] * w), int(det["y_min"] * h)
            bx2, by2 = int(det["x_max"] * w), int(det["y_max"] * h)
            is_imminent = det["distance_m"] < 10.0
            box_color = (68, 23, 255) if is_imminent else (118, 230, 0)

            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)
            if is_imminent:
                # Automotive impact target reticle
                cv2.circle(overlay, (cx_px, cy_px), 10, (68, 23, 255), 2, cv2.LINE_AA)
                cv2.line(overlay, (cx_px - 14, cy_px), (cx_px + 14, cy_px), (68, 23, 255), 1, cv2.LINE_AA)
                cv2.line(overlay, (cx_px, cy_px - 14), (cx_px, cy_px + 14), (68, 23, 255), 1, cv2.LINE_AA)

            label = f"POTHOLE {det['distance_m']:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(overlay, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, by1), box_color, -1)
            cv2.putText(
                overlay, label, (bx1 + 3, max(10, by1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Clean top telemetry bar
        cv2.rectangle(overlay, (0, 0), (w, 28), (12, 16, 24), -1)
        hdr = (
            f"AR HUD PHASE 6 | Horizon: {int(smooth_horizon * 100)}% | "
            f"Road Hazards: {len(detections)} (In-Path: {len(in_path_potholes)}) | "
            f"Ego: {center_status} ({c_center.offset_meters:+.2f}m) | {elapsed_ms:.2f}ms"
        )
        cv2.putText(overlay, hdr, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 229, 0), 1, cv2.LINE_AA)

        if crit_pothole is not None:
            cv2.rectangle(overlay, (w // 5, 32), (w * 4 // 5, 62), (45, 18, 235), -1)
            msg = f"WARNING: POTHOLE IN PATH: {crit_pothole['distance_m']:.1f}m"
            cv2.putText(overlay, msg, (w // 5 + 18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(cv2.addWeighted(overlay, 0.86, frame, 0.14, 0))

        closest_any = min(detections, key=lambda d: d["distance_m"], default=None)
        telemetry.append({
            "frame": processed_count,
            "time_s": round(processed_count / out_fps, 3),
            "hazard_count": len(detections),
            "pothole_count": len(detections),
            "in_path_count": len(in_path_potholes),
            "closest_distance_m": crit_pothole["distance_m"] if crit_pothole else (
                closest_any["distance_m"] if closest_any else None
            ),
            "closest_is_critical": bool(crit_pothole is not None),
            "assigned_lane": crit_pothole["lane"] if crit_pothole else (
                closest_any["lane"] if closest_any else "CENTER"
            ),
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
            "name": "Benchmark Footage (ride_test.mp4 — Run Phase 6 AR HUD Pipeline)",
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
            hud_filename = "hud_ride_test_phase6.mp4"
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
    print("Motorbike Vision Model Evaluator (Phase 6 AR HUD Pipeline):")
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
