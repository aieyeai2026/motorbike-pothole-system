from fastapi import FastAPI
from app.schemas import HealthResponse

app = FastAPI(
    title="Motorbike Pothole Telemetry & Spatial Clustering Backend",
    version="0.1.0",
    description="Telemetry ingestion and 3-meter spatial clustering backend for edge vision pothole detections.",
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status of the service."""
    return HealthResponse(status="healthy")
