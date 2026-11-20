import os
import sys
import pytest

# Add user_test_experience and pothole-sync-backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(os.path.dirname(parent_dir), "pothole-sync-backend")

sys.path.insert(0, parent_dir)
sys.path.insert(0, backend_dir)


def test_t_ux_1_files_exist():
    """Verify all required user experience assets are present."""
    assert os.path.isfile(os.path.join(parent_dir, "simulate_ride.py"))
    assert os.path.isfile(os.path.join(parent_dir, "start_test_server.py"))
    assert os.path.isfile(os.path.join(parent_dir, "README.md"))
    assert os.path.isfile(os.path.join(parent_dir, "web_console", "index.html"))
    assert os.path.isfile(os.path.join(parent_dir, "web_console", "app.js"))
    assert os.path.isfile(os.path.join(parent_dir, "web_console", "style.css"))


def test_t_ux_2_haversine_and_deduplication():
    """Verify simulation math complies with 3.0m spatial deduplication."""
    from simulate_ride import haversine_distance_meters, LocalSimulationQueue

    # Exactly same point = 0m
    d0 = haversine_distance_meters(12.9716, 77.5946, 12.9716, 77.5946)
    assert abs(d0) < 1e-4

    # Close points (< 3m)
    # 0.00001 deg lat is ~1.11 meters
    d_close = haversine_distance_meters(12.97160, 77.59460, 12.97161, 77.59460)
    assert d_close < 3.0

    # Farther points (> 3m)
    # 0.00005 deg lat is ~5.55 meters
    d_far = haversine_distance_meters(12.97160, 77.59460, 12.97165, 77.59460)
    assert d_far > 3.0

    queue = LocalSimulationQueue(dedup_radius_meters=3.0)
    # First item enqueued
    first = queue.enqueue_if_not_duplicate(12.97160, 77.59460, 45.0, 0.9, 7.5, "CENTER")
    assert first is True
    assert len(queue.pending) == 1

    # Duplicate within 3m rejected
    dup = queue.enqueue_if_not_duplicate(12.97161, 77.59460, 45.0, 0.9, 7.5, "CENTER")
    assert dup is False
    assert len(queue.pending) == 1

    # Independent detection beyond 3m accepted
    second = queue.enqueue_if_not_duplicate(12.97165, 77.59460, 42.0, 0.85, 6.0, "LEFT")
    assert second is True
    assert len(queue.pending) == 2


def test_t_ux_3_payload_validation_and_thumbnails():
    """Verify batch generator produces valid Pydantic models and catches oversized thumbnails."""
    from simulate_ride import generate_mock_thumbnail, LocalSimulationQueue
    from app.schemas import SyncBatchRequest, PotholeDetectionItem
    from pydantic import ValidationError

    queue = LocalSimulationQueue()
    valid_thumb = generate_mock_thumbnail(size_bytes=5000)
    queue.enqueue_if_not_duplicate(
        12.9720, 77.5950, 40.0, 0.92, 8.0, "CENTER", image_base64=valid_thumb
    )
    batch_dict = queue.to_sync_batch_payload("test-bike-01")

    # Validate against actual FastAPI Pydantic schema
    batch = SyncBatchRequest.model_validate(batch_dict)
    assert batch.device_id == "test-bike-01"
    assert len(batch.detections) == 1
    assert batch.detections[0].severity_score == 8.0

    # Oversized thumbnail validation
    oversized_thumb = generate_mock_thumbnail(size_bytes=105000)
    with pytest.raises(ValidationError):
        PotholeDetectionItem(
            event_uuid="test-uuid",
            latitude=12.9720,
            longitude=77.5950,
            speed_kmh=40.0,
            confidence=0.9,
            severity_score=8.0,
            lane_position="CENTER",
            captured_at="2026-11-19T14:45:00Z",
            image_base64=oversized_thumb,
        )


def test_t_ux_4_web_console_structure():
    """Verify web console includes critical tactical HUD & telemetry controls."""
    html_path = os.path.join(parent_dir, "web_console", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "id=\"ping-btn\"" in html or "id='ping-btn'" in html
    assert "id=\"hud-canvas\"" in html or "id='hud-canvas'" in html
    assert "CRITICAL" in html
    assert "sync-btn" in html or "sync_btn" in html


def test_t_ux_5_pwa_and_mobile_camera():
    """Verify web console includes native mobile camera capture and PWA manifest for iOS."""
    html_path = os.path.join(parent_dir, "web_console", "index.html")
    manifest_path = os.path.join(parent_dir, "web_console", "manifest.json")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert os.path.isfile(manifest_path)
    assert 'apple-mobile-web-app-capable' in html
    assert 'id="camera-file-input"' in html or "id='camera-file-input'" in html
    assert 'capture="environment"' in html or "capture='environment'" in html
    assert 'id="snap-photo-btn"' in html or "id='snap-photo-btn'" in html


def test_t_ux_6_video_evaluator_files_and_controls():
    """Verify dual video evaluation web app contains synchronized controls, curtain slider, and telemetry HUD."""
    eval_dir = os.path.join(parent_dir, "video_evaluator")
    html_path = os.path.join(eval_dir, "index.html")
    js_path = os.path.join(eval_dir, "app.js")
    css_path = os.path.join(eval_dir, "style.css")
    server_path = os.path.join(eval_dir, "server.py")

    assert os.path.isfile(html_path)
    assert os.path.isfile(js_path)
    assert os.path.isfile(css_path)
    assert os.path.isfile(server_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Dual Video Player & Curtain Slider
    assert 'id="video-raw"' in html
    assert 'id="video-hud"' in html
    assert 'id="curtain-toggle-btn"' in html
    assert 'id="curtain-slider"' in html

    # 2. Synchronized Playback Controls
    assert 'id="play-pause-btn"' in html
    assert 'id="timeline-scrubber"' in html
    assert 'id="step-back-btn"' in html
    assert 'id="step-fwd-btn"' in html
    assert 'id="speed-selector"' in html

    # 3. Video Loading & Live Pipeline Processing
    assert 'id="raw-file-input"' in html
    assert 'id="hud-file-input"' in html
    assert 'id="process-upload-btn"' in html

    # 4. Synchronized Telemetry & Evaluation HUD
    assert 'id="timecode-display"' in html
    assert 'id="metric-hazard-count"' in html
    assert 'id="metric-closest-distance"' in html
    assert 'id="metric-assigned-lane"' in html
    assert 'id="metric-fps-latency"' in html

    # 5. Modular 6-Layer HUD Toggles & Interactive Live Overlay Canvas
    assert 'id="hud-overlay-canvas"' in html
    assert 'id="layers-all-on-btn"' in html
    assert 'id="layers-potholes-only-btn"' in html
    assert 'id="layers-all-off-btn"' in html
    for toggle_id in (
        "toggle-horizon",
        "toggle-boundaries",
        "toggle-carpet",
        "toggle-potholes",
        "toggle-distance-ticks",
        "toggle-telemetry",
    ):
        assert f'id="{toggle_id}"' in html


def test_t_ux_7_video_evaluator_live_rust_pipeline_on_uploaded_video(tmp_path):
    """Verify video evaluator HTTP server parses Range headers and processes real uploaded video frames through Rust FFI."""
    import cv2
    import numpy as np
    from video_evaluator.server import (
        parse_byte_range,
        get_available_presets,
        process_video_through_rust_pipeline,
        VideoEvalRequestHandler,
    )
    from types import SimpleNamespace

    assert parse_byte_range("bytes=0-1023", 5000) == (0, 1023)
    assert parse_byte_range("bytes=100-", 5000) == (100, 4999)
    assert parse_byte_range(None, 5000) is None

    mock_handler = SimpleNamespace(
        headers={"X-Hud-Layers": '{"horizon": false, "potholes": true}', "X-Conf-Threshold": "0.50"}
    )
    parsed_layers, parsed_thresh = VideoEvalRequestHandler._parse_request_layer_options(mock_handler)
    assert parsed_layers == {"horizon": False, "potholes": True}
    assert parsed_thresh >= 0.90

    presets = get_available_presets()
    assert isinstance(presets, list)
    assert len(presets) >= 1

    # Create a real 10-frame road video with a textured pothole crater inside the road corridor
    in_video = str(tmp_path / "raw_upload.mp4")
    out_video = str(tmp_path / "processed_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(in_video, fourcc, 30.0, (640, 360))
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 6, (360, 640, 3)).astype(np.int16)
    for i in range(10):
        frame = np.clip(122 + noise, 80, 165).astype(np.uint8)
        frame[:144, :] = (200, 180, 150)
        cv2.line(frame, (260, 144), (40, 360), (245, 245, 245), 3)
        cv2.line(frame, (380, 144), (600, 360), (245, 245, 245), 3)
        py = 216 + i
        cv2.ellipse(frame, (320, py), (28, 16), 5, 0, 360, (38, 40, 44), -1)
        cv2.ellipse(frame, (320, py), (28, 16), 5, 0, 360, (94, 98, 104), 2)
        cv2.ellipse(frame, (318, py + 2), (20, 11), 3, 0, 360, (22, 24, 28), -1)
        writer.write(frame)
    writer.release()

    result = process_video_through_rust_pipeline(in_video, out_video, max_frames=10, convert_h264=False)
    assert os.path.isfile(result["output_path"])
    assert os.path.getsize(result["output_path"]) > 0
    assert result["frames_processed"] == 10
    assert len(result["telemetry"]) == 10
    assert result["telemetry"][0]["hazard_count"] >= 1
    assert result["telemetry"][0]["conf_threshold"] >= 0.90
    assert all(p["confidence"] >= 0.90 for p in result["telemetry"][0]["potholes"])
    assert 1 <= len(result["telemetry"][0]["distance_ticks"]) <= 4
    assert "vp" in result["telemetry"][0]
    assert "carpet_ys" in result["telemetry"][0]
    assert "closest_distance_m" in result["telemetry"][0]
    assert "assigned_lane" in result["telemetry"][0]
    assert "postprocess_ms" in result["telemetry"][0]


def test_t_ux_8_phase6_curved_road_boundaries_and_offroad_clutter_rejection():
    """Verify Phase 6 road-level horizon (50-58%), 2nd-degree curved boundaries, off-road clutter filter, and trajectory path collision alert."""
    import cv2
    import numpy as np
    from video_evaluator.server import (
        calibrate_road_horizon_norm,
        fit_curved_road_boundaries,
        classify_hazard_in_corridor_ffi,
    )

    # Synthetic 360x640 road scene: sky/trees above y=0.54 (194px), gray asphalt road polygon below
    frame = np.full((360, 640, 3), (40, 95, 50), dtype=np.uint8)  # green vegetation/walls off-road
    frame[:194, :] = (190, 175, 155)  # sky/canopy above road horizon
    road_pts = np.array([[250, 194], [390, 194], [580, 310], [60, 310]], dtype=np.int32)
    cv2.fillPoly(frame, [road_pts], (115, 115, 118))  # gray asphalt surface

    horizon_norm = calibrate_road_horizon_norm(frame)
    assert 0.50 <= horizon_norm <= 0.58

    horizon_y = int(360 * horizon_norm)
    left_poly, right_poly, steering_curv = fit_curved_road_boundaries(frame, horizon_y)
    assert len(left_poly) == 3
    assert len(right_poly) == 3

    # 1. Off-road parked car / fence at x=[0.02, 0.10], y=0.75 -> rejected (0)
    code_offroad = classify_hazard_in_corridor_ffi(
        0.02, 0.10, 0.75, horizon_norm, 0.85, left_poly, right_poly, 0.50, steering_curv
    )
    assert code_offroad == 0

    # 2. On-road hazard outside bike's tire-track corridor at x=[0.24, 0.30], y=0.75 -> faint dot (1)
    code_side = classify_hazard_in_corridor_ffi(
        0.24, 0.30, 0.75, horizon_norm, 0.85, left_poly, right_poly, 0.50, steering_curv
    )
    assert code_side == 1

    # 3. On-road pothole directly in bike's trajectory corridor at x=[0.47, 0.53], y=0.75 -> collision path (2)
    code_in_path = classify_hazard_in_corridor_ffi(
        0.47, 0.53, 0.75, horizon_norm, 0.85, left_poly, right_poly, 0.50, steering_curv
    )
    assert code_in_path == 2


def test_t_ux_9_phase8_green_carpet_tilting_horizon_and_obstacle_weaving():
    """Verify Phase 8/9 vanishing point, 0.90 gate, obstacle-weaving green carpet, and independent 6-layer rendering."""
    import cv2
    import numpy as np
    from video_evaluator.server import (
        DEFAULT_HUD_LAYERS,
        compute_phase8_horizon_and_boundaries,
        compute_phase8_green_carpet_polygon,
        render_phase8_ar_hud,
    )

    # Synthetic 360x640 road scene: sky above y=0.52 (187px), gray asphalt converging at (0.50, 0.52)
    frame = np.full((360, 640, 3), (40, 95, 50), dtype=np.uint8)
    frame[:187, :] = (190, 175, 155)
    road_pts = np.array([[280, 187], [360, 187], [580, 324], [60, 324]], dtype=np.int32)
    cv2.fillPoly(frame, [road_pts], (115, 115, 118))

    x_vp, y_vp, theta_bank, x_left_bot, x_right_bot, kappa_road = compute_phase8_horizon_and_boundaries(
        frame, prev_vp=(0.50, 0.54)
    )
    assert 0.15 <= x_vp <= 0.85
    assert 0.45 <= y_vp <= 0.58
    assert -0.40 <= theta_bank <= 0.40
    assert 0.02 <= x_left_bot < x_right_bot <= 0.98

    # Clear path: carpet extends all the way to near y_vp + 0.02 and boundaries never cross
    clear_res = compute_phase8_green_carpet_polygon(
        frame, x_vp, y_vp, x_left_bot, x_right_bot, kappa_road, detections=[]
    )
    assert clear_res["y_target_norm"] <= y_vp + 0.04
    assert clear_res["is_critical_collision"] is False
    for lx, rx in zip(clear_res["left_bound_xs"], clear_res["right_bound_xs"]):
        assert lx <= rx

    # Sub-0.90 pothole must be strictly ignored (no deflection, truncation, or collision alert)
    sub_90_pothole = [
        {
            "x_min": 0.38,
            "y_min": 0.62,
            "x_max": 0.62,
            "y_max": 0.76,
            "distance_m": 5.5,
            "confidence": 0.85,
            "class_id": 0,
            "in_path": True,
        }
    ]
    sub90_res = compute_phase8_green_carpet_polygon(
        frame, x_vp, y_vp, x_left_bot, x_right_bot, kappa_road, detections=sub_90_pothole
    )
    assert sub90_res["is_critical_collision"] is False
    assert sub90_res["y_target_norm"] <= y_vp + 0.04

    # Obstacle on left-center (x=[0.36, 0.52], y_max=0.68): carpet must weave right & stop before obstacle
    obs_detections = [
        {
            "x_min": 0.36,
            "y_min": 0.56,
            "x_max": 0.52,
            "y_max": 0.68,
            "distance_m": 9.5,
            "confidence": 0.88,
            "class_id": 1,
            "in_path": True,
        }
    ]
    weave_res = compute_phase8_green_carpet_polygon(
        frame, x_vp, y_vp, x_left_bot, x_right_bot, kappa_road, detections=obs_detections
    )
    assert weave_res["target_gap_x"] > 0.50
    far_center_x = 0.5 * (weave_res["carpet_left_xs"][-1] + weave_res["carpet_right_xs"][-1])
    assert far_center_x > 0.50
    for cl, cr, xl, xr in zip(
        weave_res["carpet_left_xs"],
        weave_res["carpet_right_xs"],
        weave_res["carpet_bound_left_xs"],
        weave_res["carpet_bound_right_xs"],
    ):
        assert xl <= cl <= cr <= xr

    # Imminent unavoidable obstacle directly ahead (>= 0.90, < 8.0m) -> critical flag True
    imminent_obs = [
        {
            "x_min": 0.38,
            "y_min": 0.62,
            "x_max": 0.62,
            "y_max": 0.76,
            "distance_m": 5.5,
            "confidence": 0.92,
            "class_id": 0,
            "in_path": True,
        }
    ]
    crit_res = compute_phase8_green_carpet_polygon(
        frame, x_vp, y_vp, x_left_bot, x_right_bot, kappa_road, detections=imminent_obs
    )
    assert crit_res["is_critical_collision"] is True
    assert crit_res["y_target_norm"] >= max(0.76, y_vp + 0.03) - 1e-3

    # Independent 6-layer toggling: all OFF == raw frame byte-identical; each single layer ON draws independently
    all_off = {k: False for k in DEFAULT_HUD_LAYERS}
    raw_out = render_phase8_ar_hud(frame, x_vp, y_vp, theta_bank, crit_res, imminent_obs, header_text="HDR", layers=all_off)
    assert np.array_equal(raw_out, frame)

    for layer_key in DEFAULT_HUD_LAYERS:
        single_layer = {k: (k == layer_key) for k in DEFAULT_HUD_LAYERS}
        layered_out = render_phase8_ar_hud(
            frame, x_vp, y_vp, theta_bank, crit_res, imminent_obs, header_text="HDR", layers=single_layer
        )
        assert not np.array_equal(layered_out, frame), f"Layer {layer_key} produced no visual change when enabled"


def test_t_ux_10_pyresearch_pothole_detector_achieves_over_80_percent_output():
    """Verify pure PyResearch YOLOv12s best.pt detector achieves >=80% positive recall at >=90% confidence with 0 FPs on smooth road."""
    import numpy as np
    import torch
    from video_evaluator.pyresearch_pothole_detector import (
        load_pyresearch_yolo_model,
        detect_pyresearch_and_road_potholes,
        evaluate_pothole_benchmark_accuracy,
    )

    model = load_pyresearch_yolo_model()
    assert getattr(model, "names", None) == {0: "Pothole"}
    dummy = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    assert tuple(out.shape) == (1, 5, 8400)

    # Verify use_yolo=False honors neural-only policy with zero heuristic fallback
    assert detect_pyresearch_and_road_potholes(np.zeros((360, 640, 3), dtype=np.uint8), use_yolo=False) == ([], 0)

    metrics = evaluate_pothole_benchmark_accuracy(use_yolo=True)
    assert abs((metrics["random_window_end_s"] - metrics["random_window_start_s"]) - 60.0) < 0.25
    assert len(metrics["sampled_timestamps_s"]) >= 10
    assert metrics["total_frames"] >= 10
    assert metrics["detection_rate_pct"] >= 80.0
    assert metrics["mean_confidence_pct"] >= 90.0
    for item in metrics["per_frame"]:
        if item["detections"] > 0:
            assert item["top_confidence"] >= 0.90
    assert metrics.get("smooth_road_fp_count", 0) == 0
    assert metrics["windshield_fp_count"] == 0
    assert metrics["offroad_fp_count"] == 0





