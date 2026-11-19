from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class ClusteredHazard(Base):
    __tablename__ = "clustered_hazards"

    id = Column(Integer, primary_key=True, index=True)
    cluster_uuid = Column(String(64), unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    centroid_lat = Column(Float, nullable=False)
    centroid_lon = Column(Float, nullable=False)
    detection_count = Column(Integer, default=1, nullable=False)
    max_severity = Column(Float, nullable=False)
    avg_confidence = Column(Float, nullable=False)
    predominant_lane = Column(String(16), nullable=False)
    first_detected_at = Column(String(64), nullable=False)
    last_detected_at = Column(String(64), nullable=False)

    events = relationship("PotholeEvent", back_populates="cluster")


class PotholeEvent(Base):
    __tablename__ = "pothole_events"

    id = Column(Integer, primary_key=True, index=True)
    event_uuid = Column(String(64), unique=True, index=True, nullable=False)
    device_id = Column(String(64), nullable=False, index=True)
    cluster_id = Column(Integer, ForeignKey("clustered_hazards.id"), nullable=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed_kmh = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    severity_score = Column(Float, nullable=False)
    lane_position = Column(String(16), nullable=False)
    captured_at = Column(String(64), nullable=False)
    image_base64 = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    cluster = relationship("ClusteredHazard", back_populates="events")
