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
