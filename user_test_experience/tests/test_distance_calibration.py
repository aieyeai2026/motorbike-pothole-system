"""Tests for road horizon/bottom calibration and consistent perspective distance estimation.

Regression:
- Helmet/chest-mounted cameras show the motorcycle dash/handlebars in the bottom ~40% of the frame,
  while the real vanishing point is near y=0.10..0.35. Clamping y_vp to [0.48, 0.56] and y_bot=0.88
  caused potholes ahead on the road to clamp to 30.0m and drew distance ticks over the handlebars.
- Distance ticks and pothole distance labels must be exact inverses over [y_vp, y_bot].
"""

import os
import sys

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from video_evaluator.server import (  # noqa: E402
    compute_perspective_distance_ticks,
    detect_ego_road_bottom,
    estimate_road_distance_meters,
    process_video_through_rust_pipeline,
)


def _make_chestcam_frame(w: int = 640, h: int = 360) -> np.ndarray:
    """Synthetic helmet/chest-cam frame: road from y=0.10..0.56, black/colored bike dash below y=0.56."""
    frame = np.full((h, w, 3), (45, 110, 50), dtype=np.uint8)  # grass shoulders
    frame[: int(0.10 * h), :] = (195, 180, 160)  # sky
    road = np.array(
        [
            [int(0.47 * w), int(0.10 * h)],
            [int(0.53 * w), int(0.10 * h)],
            [int(0.95 * w), int(0.58 * h)],
            [int(0.05 * w), int(0.58 * h)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [road], (112, 112, 116))
    # Dark windscreen & handlebars + bright phone + yellow tank bag from y=0.56 to bottom
    frame[int(0.56 * h) :, :] = (25, 25, 30)
    frame[int(0.65 * h) : int(0.88 * h), int(0.60 * w) : int(0.75 * w)] = (230, 230, 235)
    frame[int(0.82 * h) :, int(0.25 * w) : int(0.75 * w)] = (20, 190, 210)
    return frame


def _make_dashcam_frame(w: int = 640, h: int = 360, with_car: bool = False) -> np.ndarray:
    """Synthetic front-mounted dashcam frame: road extends cleanly down to bottom of frame."""
    frame = np.full((h, w, 3), (45, 110, 50), dtype=np.uint8)
    frame[: int(0.35 * h), :] = (195, 180, 160)
    road = np.array(
        [
            [int(0.46 * w), int(0.35 * h)],
            [int(0.54 * w), int(0.35 * h)],
            [int(0.96 * w), h - 1],
            [int(0.04 * w), h - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [road], (112, 112, 116))
    if with_car:
        # Vehicle driving ahead around y=0.45..0.60 with open asphalt below it
        cv2.rectangle(frame, (int(0.42 * w), int(0.45 * h)), (int(0.58 * w), int(0.60 * h)), (20, 30, 180), -1)
    return frame


def test_distance_and_ticks_are_exact_inverses_for_any_horizon_and_bottom():
    for y_vp, y_bot in ((0.10, 0.56), (0.33, 0.88), (0.50, 0.88)):
        carpet_data = {
            "y_bot": y_bot,
            "bound_ys": [y_vp, y_bot],
            "left_bound_xs": [0.50, 0.10],
            "right_bound_xs": [0.50, 0.90],
        }
        ticks = compute_perspective_distance_ticks(y_vp, carpet_data, y_bot=y_bot)
        assert [t["distance_m"] for t in ticks] == [5, 10, 15, 20]
        for tick in ticks:
            assert y_vp < tick["y_norm"] <= y_bot
            reconstructed = estimate_road_distance_meters(tick["y_norm"], y_vp, y_bot=y_bot)
            assert abs(reconstructed - tick["distance_m"]) < 0.05


def test_detect_ego_road_bottom_distinguishes_chestcam_from_dashcam():
    chest_y = detect_ego_road_bottom(_make_chestcam_frame(), y_vp=0.10)
    assert 0.50 <= chest_y <= 0.62, f"Expected chest-cam dash top around 0.56, got {chest_y}"

    dash_y = detect_ego_road_bottom(_make_dashcam_frame(with_car=False), y_vp=0.35)
    assert dash_y == 0.88

    dash_car_y = detect_ego_road_bottom(_make_dashcam_frame(with_car=True), y_vp=0.35)
    assert dash_car_y == 0.88


def test_pipeline_uses_measured_horizon_and_road_bottom_on_chestcam_video(tmp_path):
    video_path = str(tmp_path / "chestcam.mp4")
    out_path = str(tmp_path / "chestcam_hud.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    frame = _make_chestcam_frame()
    for _ in range(18):
        writer.write(frame)
    writer.release()

    result = process_video_through_rust_pipeline(video_path, out_path, max_frames=18, convert_h264=False)
    last = result["telemetry"][-1]
    assert last["vp"][1] < 0.35, f"Expected real horizon < 0.35 on chest-cam, got {last['vp']}"
    for tick in last["distance_ticks"]:
        assert tick["y_norm"] <= 0.64, f"Tick {tick} rendered over bike dash instead of road"
