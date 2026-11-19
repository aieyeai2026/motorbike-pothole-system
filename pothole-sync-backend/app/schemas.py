import base64
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

MAX_THUMBNAIL_BYTES = 100 * 1024  # 100 KB limit = 102,400 bytes


class PotholeDetectionItem(BaseModel):
    event_uuid: str = Field(..., description="Unique UUID for this pothole detection event")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 longitude")
    speed_kmh: float = Field(..., ge=0.0, description="Speed of the vehicle in km/h")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO detection confidence score [0.0, 1.0]")
    severity_score: float = Field(..., ge=0.0, le=10.0, description="Computed severity score [0.0, 10.0]")
    lane_position: str = Field(..., description="Detected lane: LEFT, CENTER, or RIGHT")
    captured_at: str = Field(..., description="ISO 8601 timestamp string when captured")
    image_base64: Optional[str] = Field(None, description="Optional base64-encoded thumbnail image (<= 100 KB decoded)")

    @field_validator("image_base64")
    @classmethod
    def validate_image_base64_size(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        # Handle optional data URI prefix e.g. "data:image/jpeg;base64,..."
        clean_v = v.split(",", 1)[-1] if "," in v else v
        
        try:
            decoded = base64.b64decode(clean_v, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 payload: {exc}") from exc
        
        if len(decoded) > MAX_THUMBNAIL_BYTES:
            raise ValueError(
                f"Decoded image size {len(decoded)} bytes exceeds maximum allowed thumbnail limit of {MAX_THUMBNAIL_BYTES} bytes (100 KB)"
            )
        
        return v


class SyncBatchRequest(BaseModel):
    device_id: str = Field(..., min_length=1, description="Identifier of the publishing bike edge device")
    detections: List[PotholeDetectionItem] = Field(default_factory=list, description="Batch of pothole detections")


class SyncResponse(BaseModel):
    status: str = Field("success", description="Status indicator")
    synced_uuids: List[str] = Field(default_factory=list, description="UUIDs successfully processed and persisted")
    total_processed: int = Field(0, ge=0, description="Count of detections processed in this batch")
    synced_count: int = Field(0, ge=0, description="Number of hazards successfully synchronized")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service health status")
