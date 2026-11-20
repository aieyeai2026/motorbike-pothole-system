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
import math
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
    lib.pothole_core_eval_green_carpet_row.restype = ctypes.c_int32
    lib.pothole_core_eval_green_carpet_row.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
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


def compute_phase8_horizon_and_boundaries(
    frame: np.ndarray,
    prev_vp: Tuple[float, float] = (0.50, 0.54),
    prev_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Phase 8 Dynamic Vanishing Point, Tilting Horizon & Curved Cyan Road Boundaries:
    Returns (x_vp, y_vp, theta_bank, x_left_bot, x_right_bot, kappa_road).
    """
    h, w, _ = frame.shape
    asphalt = compute_asphalt_mask(frame)
    # Bridge small painted lane dashes and shadows horizontally so road width stays intact
    road_fill = cv2.morphologyEx(asphalt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    y_h_guess = calibrate_road_horizon_norm(frame)
    y_bot = 0.88

    y_samples = np.linspace(min(0.56, y_h_guess + 0.02), 0.84, 12)
    valid_ys: List[float] = []
    left_xs: List[float] = []
    right_xs: List[float] = []

    for yn in y_samples:
        py = int(yn * h)
        strip = road_fill[max(0, py - 2):min(h, py + 3), :]
        col_score = np.mean(strip > 0, axis=0)
        prog = (yn - y_h_guess) / max(0.05, y_bot - y_h_guess)
        def_l = 0.42 - 0.34 * prog
        def_r = 0.58 + 0.34 * prog

        # Find outermost contiguous asphalt spans within expected corridor so vehicles
        # inside the road do not collapse the road boundary inward.
        search_l = int(max(0.02, def_l - 0.14) * w)
        search_r = int(min(0.98, def_r + 0.14) * w)
        asphalt_cols = np.where(col_score[search_l:search_r] >= 0.40)[0] + search_l

        if len(asphalt_cols) >= max(20, int(w * 0.08)):
            fl = float(asphalt_cols[int(len(asphalt_cols) * 0.02)]) / float(w)
            fr = float(asphalt_cols[int(len(asphalt_cols) * 0.98)]) / float(w)
            fl = float(np.clip(fl, 0.02, 0.46))
            fr = float(np.clip(fr, 0.54, 0.98))
        else:
            fl, fr = def_l, def_r

        if fr - fl >= 0.16:
            valid_ys.append(float(yn))
            left_xs.append(fl)
            right_xs.append(fr)

    if len(valid_ys) >= 4:
        m_left, c_left = np.polyfit(valid_ys, left_xs, 1)
        m_right, c_right = np.polyfit(valid_ys, right_xs, 1)
        # Ensure tangents converge downward cleanly
        if m_left > -0.20:
            m_left = -0.95
            c_left = float(np.mean(left_xs) - m_left * np.mean(valid_ys))
        if m_right < 0.20:
            m_right = 0.95
            c_right = float(np.mean(right_xs) - m_right * np.mean(valid_ys))
    else:
        m_left, c_left = -0.98, 0.99
        m_right, c_right = 0.98, 0.01

    denom = float(m_left - m_right)
    if abs(denom) > 1e-4:
        y_meas = float(np.clip((c_right - c_left) / denom, 0.48, 0.58))
        x_meas = float(np.clip(0.5 * ((m_left * y_meas + c_left) + (m_right * y_meas + c_right)), 0.28, 0.72))
    else:
        x_meas, y_meas = prev_vp

    x_vp = float(np.clip(0.85 * prev_vp[0] + 0.15 * x_meas, 0.25, 0.75))
    y_vp = float(np.clip(0.85 * prev_vp[1] + 0.15 * y_meas, 0.48, 0.58))
    theta_bank = float(np.clip(math.atan((m_left + m_right) * 0.5), -0.18, 0.18))

    x_left_bot = float(np.clip(m_left * y_bot + c_left, 0.04, 0.44))
    x_right_bot = float(np.clip(m_right * y_bot + c_right, 0.56, 0.96))

    # Estimate lateral road curvature kappa_road from midpoint deviation
    kappa_road = 0.0
    if len(valid_ys) >= 5:
        l_quad = np.polyfit(valid_ys, left_xs, 2)
        r_quad = np.polyfit(valid_ys, right_xs, 2)
        y_mid = 0.5 * (y_vp + y_bot)
        l_mid_obs = float(np.polyval(l_quad, y_mid))
        r_mid_obs = float(np.polyval(r_quad, y_mid))
        l_mid_chord = x_vp + (x_left_bot - x_vp) * (0.5 ** 1.3)
        r_mid_chord = x_vp + (x_right_bot - x_vp) * (0.5 ** 1.3)
        kappa_road = float(np.clip(0.5 * ((l_mid_obs - l_mid_chord) + (r_mid_obs - r_mid_chord)), -0.14, 0.14))

    if prev_bounds is not None:
        px_lb, px_rb, p_kappa, p_theta = prev_bounds
        x_left_bot = 0.75 * px_lb + 0.25 * x_left_bot
        x_right_bot = 0.75 * px_rb + 0.25 * x_right_bot
        kappa_road = 0.80 * p_kappa + 0.20 * kappa_road
        theta_bank = 0.80 * p_theta + 0.20 * theta_bank

    return x_vp, y_vp, theta_bank, x_left_bot, x_right_bot, kappa_road


def extract_onroad_obstacles_for_carpet(
    frame: np.ndarray,
    y_vp: float,
    x_vp: float,
    x_left_bot: float,
    x_right_bot: float,
    kappa_road: float,
    y_bot: float = 0.88,
) -> List[Dict[str, Any]]:
    """
    Extracts foreground vehicles, motorbikes, and pedestrians located inside the
    drivable road boundaries so the Green Carpet cleanly weaves around or stops
    behind them without cluttering off-road edges.
    """
    lib = get_rust_lib()
    h, w, _ = frame.shape
    asphalt = compute_asphalt_mask(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 65, 165)

    # Build drivable road mask from Phase 8 cyan boundaries
    ys = np.linspace(y_vp, y_bot, 20)
    l_pts = []
    r_pts = []
    out_quad = (ctypes.c_float * 4)()
    for yn in ys:
        lib.pothole_core_eval_green_carpet_row(
            ctypes.c_float(float(yn)),
            ctypes.c_float(x_vp),
            ctypes.c_float(y_vp),
            ctypes.c_float(y_bot),
            ctypes.c_float(x_left_bot),
            ctypes.c_float(x_right_bot),
            ctypes.c_float(kappa_road),
            ctypes.c_float(0.50),
            ctypes.c_float(0.50),
            ctypes.c_float(0.0),
            ctypes.c_float(0.18),
            ctypes.c_float(0.02),
            out_quad,
        )
        py = int(yn * h)
        l_pts.append([int(np.clip(out_quad[0] + 0.02, 0.0, 1.0) * w), py])
        r_pts.append([int(np.clip(out_quad[1] - 0.02, 0.0, 1.0) * w), py])

    road_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(road_mask, [np.array(l_pts + r_pts[::-1], dtype=np.int32)], 255)

    # Non-asphalt objects inside the road corridor that have clear vertical structure
    non_asphalt = cv2.bitwise_and(cv2.bitwise_not(asphalt), road_mask)
    fg = cv2.bitwise_and(non_asphalt, cv2.dilate(edges, np.ones((7, 7), np.uint8)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles: List[Dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 700 or area > w * h * 0.32:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh < h * 0.065 or bw < w * 0.04:
            continue
        y_max = (by + bh) / float(h)
        y_min = by / float(h)
        x_min = bx / float(w)
        x_max = (bx + bw) / float(w)
        x_center = 0.5 * (x_min + x_max)
        # Skip rider's own windshield/helmet visor at bottom center
        if y_min > 0.72 and 0.25 <= x_center <= 0.75:
            continue
        if y_max < y_vp + 0.08 or y_max >= 0.96:
            continue
        # Skip flat soft tree shadows that lack sharp internal vehicle/object edges
        edge_density = float(np.mean(edges[by:by + bh, bx:bx + bw] > 0))
        if edge_density < 0.065:
            continue
        # Skip huge flat horizontal shadows with almost no height
        if bw / float(max(bh, 1)) > 4.0:
            continue
        dist_m = 2.0 + ((1.0 - ((y_max - y_vp) / max(0.05, 1.0 - y_vp))) ** 0.65) * 28.0
        obstacles.append({
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max),
            "distance_m": round(float(dist_m), 2),
            "confidence": 0.82,
            "class_id": 1,
            "in_path": (x_max >= 0.39 and x_min <= 0.61),
        })
    return obstacles


def compute_phase8_green_carpet_polygon(
    frame: np.ndarray,
    x_vp: float,
    y_vp: float,
    x_left_bot: float,
    x_right_bot: float,
    kappa_road: float,
    detections: List[Dict[str, Any]],
    x_ego: float = 0.50,
    y_bot: float = 0.88,
    w_near: float = 0.18,
    w_far: float = 0.02,
) -> Dict[str, Any]:
    """
    Computes the Phase 8 curved cyan boundaries and optimal Green Carpet trajectory polygon
    via compiled Rust C-ABI `pothole_core_eval_green_carpet_row`.
    Weaves around obstacles toward the safest open drivable gap and truncates cleanly
    before any blocking obstacle without drawing over it.
    """
    lib = get_rust_lib()
    out_quad = (ctypes.c_float * 4)()

    # 1. Full cyan road boundaries from y_vp down to y_bot
    bound_ys = [float(v) for v in np.linspace(y_vp, y_bot, 28)]
    left_bound_xs: List[float] = []
    right_bound_xs: List[float] = []
    for yn in bound_ys:
        lib.pothole_core_eval_green_carpet_row(
            ctypes.c_float(yn),
            ctypes.c_float(x_vp),
            ctypes.c_float(y_vp),
            ctypes.c_float(y_bot),
            ctypes.c_float(x_left_bot),
            ctypes.c_float(x_right_bot),
            ctypes.c_float(kappa_road),
            ctypes.c_float(x_ego),
            ctypes.c_float(x_vp),
            ctypes.c_float(0.0),
            ctypes.c_float(w_near),
            ctypes.c_float(w_far),
            out_quad,
        )
        left_bound_xs.append(float(out_quad[0]))
        right_bound_xs.append(float(out_quad[1]))

    # 2. Choose optimal target_gap_x weaving around active obstacles in road corridor
    active_obs = [d for d in detections if d.get("y_max", 0.0) > y_vp + 0.02]
    y_mid = 0.5 * (y_vp + y_bot)
    lib.pothole_core_eval_green_carpet_row(
        ctypes.c_float(y_mid),
        ctypes.c_float(x_vp),
        ctypes.c_float(y_vp),
        ctypes.c_float(y_bot),
        ctypes.c_float(x_left_bot),
        ctypes.c_float(x_right_bot),
        ctypes.c_float(kappa_road),
        ctypes.c_float(x_ego),
        ctypes.c_float(x_vp),
        ctypes.c_float(0.0),
        ctypes.c_float(w_near),
        ctypes.c_float(w_far),
        out_quad,
    )
    mid_l, mid_r = float(out_quad[0]) + 0.05, float(out_quad[1]) - 0.05
    default_target = float(np.clip(x_vp + kappa_road * 0.4, max(0.25, mid_l), min(0.75, mid_r)))

    if not active_obs or mid_r <= mid_l:
        target_gap_x = float(np.clip(default_target, x_vp - 0.04, x_vp + 0.04))
    else:
        best_score = -1e9
        best_gx = default_target
        for gx in np.linspace(mid_l, mid_r, 21):
            min_clear = 1.0
            for obs in active_obs:
                ox_min = float(obs.get("x_min", 0.5))
                ox_max = float(obs.get("x_max", 0.5))
                if gx < ox_min:
                    dist_x = ox_min - gx
                elif gx > ox_max:
                    dist_x = gx - ox_max
                else:
                    dist_x = -0.05  # inside obstacle footprint
                min_clear = min(min_clear, dist_x)
            # Prefer clearance from obstacles with slight preference for center/straight lane
            score = min_clear - 0.18 * abs(gx - default_target)
            if score > best_score:
                best_score = score
                best_gx = float(gx)
        target_gap_x = best_gx

    kappa_steer = 0.0

    # 3. Dynamic truncation at closest obstacle intersecting the woven carpet corridor
    y_target_norm = min(y_bot - 0.03, y_vp + 0.02)
    blocking_obs: Optional[Dict[str, Any]] = None
    is_critical_collision = False

    for obs in active_obs:
        oy = float(np.clip(obs.get("y_max", y_vp + 0.03), y_vp + 0.03, y_bot - 0.03))
        lib.pothole_core_eval_green_carpet_row(
            ctypes.c_float(oy),
            ctypes.c_float(x_vp),
            ctypes.c_float(y_vp),
            ctypes.c_float(y_bot),
            ctypes.c_float(x_left_bot),
            ctypes.c_float(x_right_bot),
            ctypes.c_float(kappa_road),
            ctypes.c_float(x_ego),
            ctypes.c_float(target_gap_x),
            ctypes.c_float(kappa_steer),
            ctypes.c_float(w_near),
            ctypes.c_float(w_far),
            out_quad,
        )
        cl, cr = float(out_quad[2]), float(out_quad[3])
        ox_min = float(obs.get("x_min", 0.5))
        ox_max = float(obs.get("x_max", 0.5))
        overlaps_carpet = (ox_max >= cl and ox_min <= cr)
        dist_m = float(obs.get("distance_m", 99.0))
        blocks_center = dist_m < 8.0 and (ox_max >= x_ego - 0.09 and ox_min <= x_ego + 0.09)

        if overlaps_carpet or blocks_center:
            stop_y = max(oy, y_vp + 0.03)
            if stop_y > y_target_norm:
                y_target_norm = stop_y
                blocking_obs = obs
            if dist_m < 8.0 and (overlaps_carpet or blocks_center):
                is_critical_collision = True

    if blocking_obs is None and not active_obs:
        target_gap_x = float(np.clip(target_gap_x, x_vp - 0.05, x_vp + 0.05))

    # 4. Build dense carpet polygon from y_bot up to y_target_norm via Rust FFI
    carpet_ys = [float(v) for v in np.linspace(y_bot, y_target_norm, 32)]
    carpet_left_xs: List[float] = []
    carpet_right_xs: List[float] = []
    carpet_bound_left_xs: List[float] = []
    carpet_bound_right_xs: List[float] = []

    for yn in carpet_ys:
        # Smoothly converge toward x_vp with a C1-continuous sine blend when path is open
        t_row = max(0.0, min(1.0, (yn - y_vp) / max(0.05, y_bot - y_vp)))
        if blocking_obs is None and y_target_norm <= y_vp + 0.04:
            blend = math.sin(t_row * math.pi * 0.5)
            row_target_x = x_vp + (target_gap_x - x_vp) * blend
        else:
            row_target_x = target_gap_x

        lib.pothole_core_eval_green_carpet_row(
            ctypes.c_float(yn),
            ctypes.c_float(x_vp),
            ctypes.c_float(y_vp),
            ctypes.c_float(y_bot),
            ctypes.c_float(x_left_bot),
            ctypes.c_float(x_right_bot),
            ctypes.c_float(kappa_road),
            ctypes.c_float(x_ego),
            ctypes.c_float(row_target_x),
            ctypes.c_float(kappa_steer),
            ctypes.c_float(w_near),
            ctypes.c_float(w_far),
            out_quad,
        )
        carpet_bound_left_xs.append(float(out_quad[0]))
        carpet_bound_right_xs.append(float(out_quad[1]))
        carpet_left_xs.append(float(out_quad[2]))
        carpet_right_xs.append(float(out_quad[3]))

    return {
        "y_target_norm": float(y_target_norm),
        "target_gap_x": float(target_gap_x),
        "kappa_steer": float(kappa_steer),
        "is_critical_collision": bool(is_critical_collision),
        "blocking_obstacle": blocking_obs,
        "bound_ys": bound_ys,
        "left_bound_xs": left_bound_xs,
        "right_bound_xs": right_bound_xs,
        "carpet_ys": carpet_ys,
        "carpet_left_xs": carpet_left_xs,
        "carpet_right_xs": carpet_right_xs,
        "carpet_bound_left_xs": carpet_bound_left_xs,
        "carpet_bound_right_xs": carpet_bound_right_xs,
    }


def render_phase8_ar_hud(
    frame: np.ndarray,
    x_vp: float,
    y_vp: float,
    theta_bank: float,
    carpet_data: Dict[str, Any],
    pothole_detections: List[Dict[str, Any]],
    header_text: str = "",
) -> np.ndarray:
    """
    Renders Phase 8 AR "Green Carpet" & Tilting Horizon HUD onto frame:
    - Green tilting horizon line centered at (x_vp, y_vp), rotated by theta_bank
    - Smooth cyan curved road boundaries converging at (x_vp, y_vp)
    - Translucent 38%-opacity green carpet (transitions to crimson on <8m collision)
    """
    h, w, _ = frame.shape
    out = frame.copy()

    # 1. Translucent Optimal Trajectory Carpet (38% opacity blend)
    carpet_layer = out.copy()
    c_ys = carpet_data["carpet_ys"]
    c_lxs = carpet_data["carpet_left_xs"]
    c_rxs = carpet_data["carpet_right_xs"]

    left_pts = [[int(lx * w), int(yn * h)] for lx, yn in zip(c_lxs, c_ys)]
    right_pts = [[int(rx * w), int(yn * h)] for rx, yn in zip(c_rxs, c_ys)]
    poly_pts = np.array(left_pts + right_pts[::-1], dtype=np.int32)

    is_crit = carpet_data["is_critical_collision"]
    carpet_bgr = (68, 23, 255) if is_crit else (118, 230, 0)
    cv2.fillPoly(carpet_layer, [poly_pts], carpet_bgr)
    out = cv2.addWeighted(carpet_layer, 0.38, out, 0.62, 0)

    # Crisp neon left/right carpet edge strokes
    cv2.polylines(out, [np.array(left_pts, dtype=np.int32)], False, carpet_bgr, 2, cv2.LINE_AA)
    cv2.polylines(out, [np.array(right_pts, dtype=np.int32)], False, carpet_bgr, 2, cv2.LINE_AA)

    # 2. Cyan Curved Road Boundaries (#00E5FF -> BGR 255, 229, 0)
    b_ys = carpet_data["bound_ys"]
    lb_pts = np.array([[int(lx * w), int(yn * h)] for lx, yn in zip(carpet_data["left_bound_xs"], b_ys)], dtype=np.int32)
    rb_pts = np.array([[int(rx * w), int(yn * h)] for rx, yn in zip(carpet_data["right_bound_xs"], b_ys)], dtype=np.int32)
    cv2.polylines(out, [lb_pts], False, (255, 229, 0), 2, cv2.LINE_AA)
    cv2.polylines(out, [rb_pts], False, (255, 229, 0), 2, cv2.LINE_AA)

    # 3. Tilting Green Horizon Line (#00E676 -> BGR 118, 230, 0) centered at (x_vp, y_vp)
    vp_px, vp_py = int(x_vp * w), int(y_vp * h)
    span_px = int(w * 0.65)
    dx = int(span_px * math.cos(theta_bank))
    dy = int(span_px * math.sin(theta_bank))
    cv2.line(out, (vp_px - dx, vp_py - dy), (vp_px + dx, vp_py + dy), (118, 230, 0), 2, cv2.LINE_AA)

    # 4. Pothole Markers (dots out of path, crimson reticle only when in collision path)
    for det in pothole_detections:
        cx_px = int(((det["x_min"] + det["x_max"]) * 0.5) * w)
        cy_px = int(((det["y_min"] + det["y_max"]) * 0.5) * h)
        if not det.get("in_path"):
            cv2.circle(out, (cx_px, cy_px), 4, (255, 229, 0), -1, cv2.LINE_AA)
            continue
        bx1, by1 = int(det["x_min"] * w), int(det["y_min"] * h)
        bx2, by2 = int(det["x_max"] * w), int(det["y_max"] * h)
        is_imminent = det["distance_m"] < 8.0
        col = (68, 23, 255) if is_imminent else (118, 230, 0)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
        if is_imminent:
            cv2.circle(out, (cx_px, cy_px), 10, col, 2, cv2.LINE_AA)

    # 5. Telemetry Header & Brake Warning Banner
    if header_text:
        cv2.rectangle(out, (0, 0), (w, 26), (12, 16, 24), -1)
        cv2.putText(out, header_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (118, 230, 0), 1, cv2.LINE_AA)

    if is_crit and carpet_data["blocking_obstacle"] is not None:
        dist_m = float(carpet_data["blocking_obstacle"].get("distance_m", 6.0))
        cv2.rectangle(out, (w // 5, 30), (w * 4 // 5, 58), (45, 18, 235), -1)
        msg = f"BRAKE / COLLISION HAZARD: {dist_m:.1f}m"
        cv2.putText(out, msg, (w // 5 + 22, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def process_video_through_rust_pipeline(
    input_path: str,
    output_path: str,
    max_frames: int = 600,
    target_fps: float = 30.0,
    convert_h264: bool = True,
) -> Dict[str, Any]:
    """
    Phase 8 Automotive AR "Green Carpet" & Real Rust C-ABI Pipeline:
    - Computes dynamic vanishing point & road banking tilt for green horizon line.
    - Renders curved cyan road boundaries converging at vanishing point.
    - Extracts on-road obstacles & potholes strictly inside drivable asphalt polygon.
    - Projects translucent green trajectory carpet weaving around obstacles via Rust FFI.
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
    bottom_norm = 0.88

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_raw_out = output_path + ".raw.mp4" if convert_h264 else output_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw_out, fourcc, out_fps, (w, h))

    telemetry: List[Dict[str, Any]] = []
    frame_idx = 0
    processed_count = 0
    prev_vp = (0.50, 0.54)
    prev_bounds: Optional[Tuple[float, float, float, float]] = None

    while processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter_ns()

        # 1. Phase 8 vanishing point, tilting horizon, and cyan curved road boundaries
        x_vp, y_vp, theta_bank, x_left_bot, x_right_bot, kappa_road = compute_phase8_horizon_and_boundaries(
            frame, prev_vp=prev_vp, prev_bounds=prev_bounds
        )
        prev_vp = (x_vp, y_vp)
        prev_bounds = (x_left_bot, x_right_bot, kappa_road, theta_bank)

        # 2. Compute Rust C-ABI lane centering at bottom road contact
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

        # 3. Extract road potholes (Rust NMS) + foreground on-road obstacles for carpet weaving
        left_poly = [0.0, (x_left_bot - x_vp) / max(0.05, bottom_norm - y_vp), x_vp - ((x_left_bot - x_vp) / max(0.05, bottom_norm - y_vp)) * y_vp]
        right_poly = [0.0, (x_right_bot - x_vp) / max(0.05, bottom_norm - y_vp), x_vp - ((x_right_bot - x_vp) / max(0.05, bottom_norm - y_vp)) * y_vp]
        tensor, n_anchors = extract_pothole_candidates_on_drivable_road(
            frame, y_vp, bottom_norm - 0.05, left_poly, right_poly
        )

        pothole_detections: List[Dict[str, Any]] = []
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
                ctypes.c_float(y_vp),
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
                    y_vp,
                    bottom_norm - 0.05,
                    left_poly,
                    right_poly,
                    0.50,
                    kappa_road,
                    0.11,
                )
                if corridor_code == 0:
                    continue
                lane_str = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}.get(d.lane_code, "CENTER")
                pothole_detections.append({
                    "x_min": round(float(d.x_min), 4),
                    "y_min": round(float(d.y_min), 4),
                    "x_max": round(float(d.x_max), 4),
                    "y_max": round(float(d.y_max), 4),
                    "confidence": round(float(d.confidence), 3),
                    "distance_m": round(float(d.distance_meters), 2),
                    "severity": round(float(d.severity_score), 1),
                    "lane": lane_str,
                    "class_id": 0,
                    "in_path": corridor_code == 2,
                })

        fg_obstacles = extract_onroad_obstacles_for_carpet(
            frame, y_vp, x_vp, x_left_bot, x_right_bot, kappa_road, bottom_norm
        )
        all_obstacles = pothole_detections + fg_obstacles

        # 4. Compute Phase 8 Green Carpet weaving around all on-road obstacles via Rust FFI
        carpet_data = compute_phase8_green_carpet_polygon(
            frame, x_vp, y_vp, x_left_bot, x_right_bot, kappa_road, all_obstacles
        )
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # 5. Render Phase 8 HUD
        hdr = (
            f"AR HUD PHASE 8 | VP:({x_vp:.2f},{y_vp:.2f}) Bank:{math.degrees(theta_bank):+.1f}deg | "
            f"Road Hazards:{len(pothole_detections)} | Ego:{center_status} ({c_center.offset_meters:+.2f}m) | {elapsed_ms:.2f}ms"
        )
        out_frame = render_phase8_ar_hud(
            frame, x_vp, y_vp, theta_bank, carpet_data, pothole_detections, header_text=hdr
        )
        writer.write(out_frame)

        closest_any = min(all_obstacles, key=lambda d: d["distance_m"], default=None)
        in_path_potholes = [d for d in pothole_detections if d["in_path"] and d["distance_m"] < 8.0]
        crit_obs = carpet_data["blocking_obstacle"] if carpet_data["is_critical_collision"] else None

        telemetry.append({
            "frame": processed_count,
            "time_s": round(processed_count / out_fps, 3),
            "hazard_count": len(pothole_detections),
            "pothole_count": len(pothole_detections),
            "in_path_count": len(in_path_potholes),
            "closest_distance_m": crit_obs["distance_m"] if crit_obs else (
                closest_any["distance_m"] if closest_any else None
            ),
            "closest_is_critical": bool(carpet_data["is_critical_collision"]),
            "assigned_lane": closest_any.get("lane", "CENTER") if closest_any else "CENTER",
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
