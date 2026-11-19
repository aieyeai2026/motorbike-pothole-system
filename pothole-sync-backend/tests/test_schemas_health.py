import base64
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from schemas import SyncBatchRequest, PotholeDetectionItem, SyncResponse
from main import app

client = TestClient(app)

# T-3.1.1: SyncBatchRequest validates valid payloads with single and multiple detections
def test_t_3_1_1_valid_sync_batch_request():
    single_detection = {
        "event_uuid": "11111111-1111-4111-8111-111111111111",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "speed_kmh": 42.5,
        "confidence": 0.91,
        "severity_score": 6.8,
        "lane_position": "CENTER",
        "captured_at": "2026-11-19T14:45:00.000Z",
        "image_base64": None,
    }

    payload_single = {
        "device_id": "device-alpha-01",
        "detections": [single_detection],
    }

    req_single = SyncBatchRequest.model_validate(payload_single)
    assert req_single.device_id == "device-alpha-01"
    assert len(req_single.detections) == 1
    assert req_single.detections[0].event_uuid == "11111111-1111-4111-8111-111111111111"
    assert req_single.detections[0].latitude == 12.9716
    assert req_single.detections[0].longitude == 77.5946
    assert req_single.detections[0].speed_kmh == 42.5
    assert req_single.detections[0].confidence == 0.91
    assert req_single.detections[0].severity_score == 6.8
    assert req_single.detections[0].lane_position == "CENTER"

    # Multi-item batch
    multi_detection = {
        "event_uuid": "22222222-2222-4222-8222-222222222222",
        "latitude": 12.9717,
        "longitude": 77.5947,
        "speed_kmh": 35.0,
        "confidence": 0.82,
        "severity_score": 4.5,
        "lane_position": "LEFT",
        "captured_at": "2026-11-19T14:45:01.000Z",
        "image_base64": base64.b64encode(b"sample-tiny-image").decode("utf-8"),
    }
    payload_multi = {
        "device_id": "device-beta-02",
        "detections": [single_detection, multi_detection],
    }
    req_multi = SyncBatchRequest.model_validate(payload_multi)
    assert len(req_multi.detections) == 2
    assert req_multi.detections[1].lane_position == "LEFT"

    # SyncResponse schema validation
    resp = SyncResponse(
        status="success",
        synced_count=2,
        synced_uuids=[req_multi.detections[0].event_uuid, req_multi.detections[1].event_uuid],
    )
    assert resp.synced_count == 2
    assert len(resp.synced_uuids) == 2


# T-3.1.2: PotholeDetectionItem rejects thumbnail payloads where decoded byte length > 100 KB
def test_t_3_1_2_reject_oversized_thumbnail():
    # Exactly 100 KB = 102,400 bytes (should pass)
    valid_bytes = b"X" * 102400
    valid_b64 = base64.b64encode(valid_bytes).decode("utf-8")

    item_valid = PotholeDetectionItem(
        event_uuid="33333333-3333-4333-8333-333333333333",
        latitude=12.9720,
        longitude=77.5950,
        speed_kmh=50.0,
        confidence=0.88,
        severity_score=7.0,
        lane_position="RIGHT",
        captured_at="2026-11-19T14:45:02.000Z",
        image_base64=valid_b64,
    )
    assert item_valid.image_base64 is not None

    # 100 KB + 1 byte = 102,401 bytes (must be rejected)
    oversized_bytes = b"Y" * 102401
    oversized_b64 = base64.b64encode(oversized_bytes).decode("utf-8")

    with pytest.raises(ValidationError) as exc_info:
        PotholeDetectionItem(
            event_uuid="44444444-4444-4444-8444-444444444444",
            latitude=12.9721,
            longitude=77.5951,
            speed_kmh=50.0,
            confidence=0.88,
            severity_score=7.0,
            lane_position="RIGHT",
            captured_at="2026-11-19T14:45:03.000Z",
            image_base64=oversized_b64,
        )
    assert "100 KB" in str(exc_info.value) or "102400" in str(exc_info.value)


# T-3.1.3: GET /health returns status 200 OK with {"status": "healthy"}
def test_t_3_1_3_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
