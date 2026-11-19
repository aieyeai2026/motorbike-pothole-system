import math
import uuid
from typing import NamedTuple, Optional
from sqlalchemy.orm import Session

from app.models import PotholeEvent, ClusteredHazard
from app.schemas import PotholeDetectionItem

EARTH_RADIUS_METERS = 6371000.0


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two coordinates via Haversine formula."""
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


class IngestResult(NamedTuple):
    event_uuid: str
    cluster_id: Optional[int]
    is_new_event: bool


class SpatialClusteringEngine:
    """3.0-meter Haversine spatial clustering and event idempotency engine."""

    def __init__(self, db: Session, cluster_radius_meters: float = 3.0):
        self.db = db
        self.cluster_radius_meters = cluster_radius_meters

    def ingest_detection(self, item: PotholeDetectionItem, device_id: str) -> IngestResult:
        # Step 1: Idempotency check on event_uuid
        existing_event = (
            self.db.query(PotholeEvent).filter_by(event_uuid=item.event_uuid).first()
        )
        if existing_event is not None:
            return IngestResult(
                event_uuid=item.event_uuid,
                cluster_id=existing_event.cluster_id,
                is_new_event=False,
            )

        # Step 2: Query candidate clusters within cluster_radius_meters
        clusters = self.db.query(ClusteredHazard).all()
        best_cluster: Optional[ClusteredHazard] = None
        min_dist = float("inf")

        for c in clusters:
            dist = haversine_distance_meters(
                item.latitude, item.longitude, c.centroid_lat, c.centroid_lon
            )
            if dist <= self.cluster_radius_meters and dist < min_dist:
                min_dist = dist
                best_cluster = c

        # Step 3: Update existing cluster or instantiate new cluster
        if best_cluster is not None:
            n = best_cluster.detection_count
            best_cluster.centroid_lat = (best_cluster.centroid_lat * n + item.latitude) / (n + 1)
            best_cluster.centroid_lon = (best_cluster.centroid_lon * n + item.longitude) / (n + 1)
            best_cluster.detection_count = n + 1
            best_cluster.max_severity = max(best_cluster.max_severity, item.severity_score)
            best_cluster.avg_confidence = (best_cluster.avg_confidence * n + item.confidence) / (n + 1)
            best_cluster.last_detected_at = item.captured_at
            target_cluster_id = best_cluster.id
        else:
            new_cluster = ClusteredHazard(
                cluster_uuid=str(uuid.uuid4()),
                centroid_lat=item.latitude,
                centroid_lon=item.longitude,
                detection_count=1,
                max_severity=item.severity_score,
                avg_confidence=item.confidence,
                predominant_lane=item.lane_position,
                first_detected_at=item.captured_at,
                last_detected_at=item.captured_at,
            )
            self.db.add(new_cluster)
            self.db.flush()
            target_cluster_id = new_cluster.id

        # Step 4: Persist detection event linked to cluster
        event = PotholeEvent(
            event_uuid=item.event_uuid,
            device_id=device_id,
            cluster_id=target_cluster_id,
            latitude=item.latitude,
            longitude=item.longitude,
            speed_kmh=item.speed_kmh,
            confidence=item.confidence,
            severity_score=item.severity_score,
            lane_position=item.lane_position,
            captured_at=item.captured_at,
            image_base64=item.image_base64,
        )
        self.db.add(event)
        self.db.commit()

        return IngestResult(
            event_uuid=item.event_uuid,
            cluster_id=target_cluster_id,
            is_new_event=True,
        )
