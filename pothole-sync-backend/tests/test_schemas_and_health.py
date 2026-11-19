import base64
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import (
    MAX_THUMBNAIL_BYTES,
    HealthResponse,
    PotholeDetectionItem,
    SyncBatchRequest,
)

client = TestClient(app)


def test_t_3_1_1_sync_batch_request_valid_schema():
    """T-3.1.1: SyncBatchRequest parses valid payloads with single and multiple detections."""
    tiny_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    valid_b64 = base64.b64encode(tiny_png_bytes).decode("ascii")

    payload = {
        "device_id": "bike-sensor-alpha",
        "detections": [
            {
                "event_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "latitude": 9.931200,
                "longitude": 76.267300,
                "speed_kmh": 42.5,
                "confidence": 0.92,
                "severity_score": 7.8,
                "lane_position": "CENTER",
                "captured_at": "2026-11-19T14:30:00Z",
                "image_base64": valid_b64,
            },
            {
                "event_uuid": "550e8400-e29b-41d4-a716-446655440001",
                "latitude": 9.931250,
                "longitude": 76.267320,
                "speed_kmh": 38.0,
                "confidence": 0.85,
                "severity_score": 5.4,
                "lane_position": "RIGHT",
                "captured_at": "2026-11-19T14:30:05Z",
                "image_base64": None,
            },
        ],
    }

    batch = SyncBatchRequest.model_validate(payload)
    assert batch.device_id == "bike-sensor-alpha"
    assert len(batch.detections) == 2
    assert batch.detections[0].event_uuid == "550e8400-e29b-41d4-a716-446655440000"
    assert batch.detections[0].confidence == 0.92
    assert batch.detections[0].lane_position == "CENTER"
    assert batch.detections[1].image_base64 is None


def test_t_3_1_2_pothole_detection_item_rejects_large_image():
    """T-3.1.2: PotholeDetectionItem enforces <= 100 KB decoded image_base64 thumbnails."""
    # 100 KB = 102,400 bytes. Exactly 100 KB should be allowed.
    borderline_bytes = b"A" * MAX_THUMBNAIL_BYTES
    borderline_b64 = base64.b64encode(borderline_bytes).decode("ascii")

    item_ok = PotholeDetectionItem(
        event_uuid="test-uuid-ok",
        latitude=10.0,
        longitude=76.0,
        speed_kmh=40.0,
        confidence=0.8,
        severity_score=5.0,
        lane_position="CENTER",
        captured_at="2026-11-19T14:30:00Z",
        image_base64=borderline_b64,
    )
    assert item_ok.image_base64 == borderline_b64

    # 102,401 bytes (> 100 KB) must be rejected with ValidationError
    oversized_bytes = b"A" * (MAX_THUMBNAIL_BYTES + 1)
    oversized_b64 = base64.b64encode(oversized_bytes).decode("ascii")

    with pytest.raises(ValidationError) as exc_info:
        PotholeDetectionItem(
            event_uuid="test-uuid-fail",
            latitude=10.0,
            longitude=76.0,
            speed_kmh=40.0,
            confidence=0.8,
            severity_score=5.0,
            lane_position="CENTER",
            captured_at="2026-11-19T14:30:00Z",
            image_base64=oversized_b64,
        )
    assert "exceeds maximum allowed thumbnail limit of 102400 bytes" in str(exc_info.value)

    # Invalid base64 must also be rejected
    with pytest.raises(ValidationError) as exc_info_invalid:
        PotholeDetectionItem(
            event_uuid="test-uuid-invalid",
            latitude=10.0,
            longitude=76.0,
            speed_kmh=40.0,
            confidence=0.8,
            severity_score=5.0,
            lane_position="CENTER",
            captured_at="2026-11-19T14:30:00Z",
            image_base64="!!!NOT_VALID_BASE64@@@",
        )
    assert "Invalid base64 payload" in str(exc_info_invalid.value)


def test_t_3_1_3_health_endpoint_returns_200():
    """T-3.1.3: GET /health returns HTTP 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "healthy"}
