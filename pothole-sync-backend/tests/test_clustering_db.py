import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import PotholeEvent, ClusteredHazard
from app.clustering import SpatialClusteringEngine, haversine_distance_meters
from app.schemas import PotholeDetectionItem


@pytest.fixture
def db_session():
    """Create in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# T-3.2.1: Spatial clustering creates new clustered_hazards entry and updates running centroid within 3.0m
def test_t_3_2_1_cluster_within_3_meters(db_session):
    engine = SpatialClusteringEngine(db_session, cluster_radius_meters=3.0)

    # Initial detection at (12.971600, 77.594600)
    det1 = PotholeDetectionItem(
        event_uuid="event-001",
        latitude=12.971600,
        longitude=77.594600,
        speed_kmh=45.0,
        confidence=0.88,
        severity_score=6.0,
        lane_position="CENTER",
        captured_at="2026-11-19T14:40:00Z",
    )
    res1 = engine.ingest_detection(det1, device_id="bike-01")
    assert res1.is_new_event is True
    assert res1.cluster_id is not None

    # Verify cluster created
    cluster = db_session.query(ClusteredHazard).filter_by(id=res1.cluster_id).one()
    assert cluster.detection_count == 1
    assert pytest.approx(cluster.centroid_lat, 1e-6) == 12.971600
    assert pytest.approx(cluster.centroid_lon, 1e-6) == 77.594600
    assert cluster.max_severity == 6.0

    # Second detection 1.5 meters away (0.000013 deg lat ~ 1.44m)
    det2 = PotholeDetectionItem(
        event_uuid="event-002",
        latitude=12.971613,
        longitude=77.594600,
        speed_kmh=42.0,
        confidence=0.92,
        severity_score=7.5,
        lane_position="CENTER",
        captured_at="2026-11-19T14:40:05Z",
    )
    res2 = engine.ingest_detection(det2, device_id="bike-02")
    assert res2.is_new_event is True
    assert res2.cluster_id == cluster.id

    # Refresh cluster and verify centroid updated & count incremented
    db_session.refresh(cluster)
    assert cluster.detection_count == 2
    assert cluster.max_severity == 7.5
    # Running average centroid
    expected_lat = (12.971600 + 12.971613) / 2.0
    assert pytest.approx(cluster.centroid_lat, 1e-6) == expected_lat


# T-3.2.2: Spatial clustering treats detections beyond 3.0 meters as distinct independent hazard clusters
def test_t_3_2_2_distinct_clusters_beyond_3_meters(db_session):
    engine = SpatialClusteringEngine(db_session, cluster_radius_meters=3.0)

    # First point
    det1 = PotholeDetectionItem(
        event_uuid="event-alpha",
        latitude=12.971600,
        longitude=77.594600,
        speed_kmh=40.0,
        confidence=0.90,
        severity_score=5.0,
        lane_position="LEFT",
        captured_at="2026-11-19T14:41:00Z",
    )
    res1 = engine.ingest_detection(det1, device_id="bike-01")

    # Second point 6.0 meters away (0.000054 deg lat ~ 6.0m)
    det2 = PotholeDetectionItem(
        event_uuid="event-beta",
        latitude=12.971654,
        longitude=77.594600,
        speed_kmh=38.0,
        confidence=0.85,
        severity_score=6.2,
        lane_position="RIGHT",
        captured_at="2026-11-19T14:41:02Z",
    )
    res2 = engine.ingest_detection(det2, device_id="bike-01")

    assert res1.cluster_id != res2.cluster_id

    total_clusters = db_session.query(ClusteredHazard).count()
    assert total_clusters == 2


# T-3.2.3: Ingestion enforces event_uuid idempotency
def test_t_3_2_3_event_uuid_idempotency(db_session):
    engine = SpatialClusteringEngine(db_session, cluster_radius_meters=3.0)

    det = PotholeDetectionItem(
        event_uuid="event-unique-999",
        latitude=12.971600,
        longitude=77.594600,
        speed_kmh=50.0,
        confidence=0.95,
        severity_score=8.5,
        lane_position="CENTER",
        captured_at="2026-11-19T14:42:00Z",
    )

    # Ingest first time
    res1 = engine.ingest_detection(det, device_id="bike-01")
    assert res1.is_new_event is True

    cluster = db_session.query(ClusteredHazard).one()
    assert cluster.detection_count == 1
    total_events_1 = db_session.query(PotholeEvent).count()
    assert total_events_1 == 1

    # Ingest identical event_uuid second time
    res2 = engine.ingest_detection(det, device_id="bike-01")
    assert res2.is_new_event is False
    assert res2.cluster_id == cluster.id

    # Verify no duplicate event or cluster increment
    db_session.refresh(cluster)
    assert cluster.detection_count == 1
    total_events_2 = db_session.query(PotholeEvent).count()
    assert total_events_2 == 1
