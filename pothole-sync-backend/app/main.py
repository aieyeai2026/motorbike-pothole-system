import os
from typing import Any, Dict, List
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.clustering import SpatialClusteringEngine, haversine_distance_meters
from app.database import get_db, init_db
from app.models import ClusteredHazard
from app.schemas import HealthResponse, SyncBatchRequest, SyncResponse

init_db()

app = FastAPI(
    title="Motorbike Pothole Telemetry & Spatial Clustering Backend",
    version="1.0.0",
    description="Telemetry ingestion and 3-meter spatial clustering backend for edge vision pothole detections.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/console"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response



@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status of the service."""
    return HealthResponse(status="healthy")


@app.post("/api/v1/telemetry/sync", response_model=SyncResponse)
async def sync_telemetry(
    batch: SyncBatchRequest, db: Session = Depends(get_db)
) -> SyncResponse:
    """Ingest a batch of detections, executing 3.0m spatial clustering and event idempotency."""
    engine = SpatialClusteringEngine(db, cluster_radius_meters=3.0)
    synced_uuids: List[str] = []

    for item in batch.detections:
        result = engine.ingest_detection(item, device_id=batch.device_id)
        synced_uuids.append(result.event_uuid)

    return SyncResponse(
        status="success",
        synced_uuids=synced_uuids,
        total_processed=len(synced_uuids),
        synced_count=len(synced_uuids),
    )


@app.get("/api/v1/hazards/nearby")
async def get_nearby_hazards(
    lat: float, lon: float, radius_meters: float = 100.0, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Query clustered hazards within radius_meters of lat/lon coordinate."""
    all_clusters = db.query(ClusteredHazard).all()
    nearby_list = []

    for c in all_clusters:
        dist = haversine_distance_meters(lat, lon, c.centroid_lat, c.centroid_lon)
        if dist <= radius_meters:
            nearby_list.append({
                "cluster_uuid": c.cluster_uuid,
                "centroid_lat": c.centroid_lat,
                "centroid_lon": c.centroid_lon,
                "distance_meters": round(dist, 1),
                "detection_count": c.detection_count,
                "max_severity": c.max_severity,
                "avg_confidence": round(c.avg_confidence, 2),
                "predominant_lane": c.predominant_lane,
                "last_detected_at": c.last_detected_at,
            })

    nearby_list.sort(key=lambda x: x["distance_meters"])

    return {
        "status": "success",
        "query": {"latitude": lat, "longitude": lon, "radius_meters": radius_meters},
        "hazards": nearby_list,
        "total_count": len(nearby_list),
    }


# Mount web console static assets
console_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "user_test_experience", "web_console")
)
if os.path.isdir(console_dir):
    app.mount("/console", StaticFiles(directory=console_dir, html=True), name="console")


@app.get("/")
async def root() -> RedirectResponse:
    """Redirect root directly to interactive web console."""
    return RedirectResponse(url="/console/")
