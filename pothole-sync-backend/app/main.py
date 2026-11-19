import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.schemas import HealthResponse, SyncBatchRequest, SyncResponse

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


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status of the service."""
    return HealthResponse(status="healthy")


@app.post("/api/v1/telemetry/sync", response_model=SyncResponse)
async def sync_telemetry(batch: SyncBatchRequest) -> SyncResponse:
    """Ingest a batch of pothole detections from an edge client device."""
    synced_uuids = [d.event_uuid for d in batch.detections]
    return SyncResponse(
        status="success",
        synced_uuids=synced_uuids,
        total_processed=len(synced_uuids),
        synced_count=len(synced_uuids),
    )


@app.get("/api/v1/hazards/nearby")
async def get_nearby_hazards(lat: float, lon: float, radius_meters: float = 100.0) -> dict:
    """Query hazards within radius_meters of lat/lon coordinate."""
    return {
        "status": "success",
        "query": {"latitude": lat, "longitude": lon, "radius_meters": radius_meters},
        "hazards": [],
        "total_count": 0,
    }


# Mount the user test experience web console if directory exists
console_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "user_test_experience", "web_console")
)
if os.path.isdir(console_dir):
    app.mount("/console", StaticFiles(directory=console_dir, html=True), name="console")


@app.get("/")
async def root() -> dict:
    return {
        "service": "Motorbike Pothole Telemetry & Clustering Backend",
        "health": "/health",
        "docs": "/docs",
        "console": "/console",
    }
