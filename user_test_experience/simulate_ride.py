#!/usr/bin/env python3
"""
Motorbike Ride Simulator & Client Device Mock.
Simulates real-world motorbike telemetry, camera detections, 3.0m deduplication,
and batch synchronization with the FastAPI backend.
"""

import argparse
import base64
import math
import sys
import time
import uuid
from typing import Any, Dict, List, Optional
import httpx

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


def generate_mock_thumbnail(size_bytes: int = 4096) -> str:
    """Generate mock binary payload encoded as base64 string."""
    raw = b"POTH" * (size_bytes // 4)
    return base64.b64encode(raw).decode("ascii")


class LocalSimulationQueue:
    """Emulates mobile client SQLite offline queue and 3m deduplication filter."""

    def __init__(self, dedup_radius_meters: float = 3.0):
        self.dedup_radius_meters = dedup_radius_meters
        self.pending: List[Dict[str, Any]] = []

    def enqueue_if_not_duplicate(
        self,
        latitude: float,
        longitude: float,
        speed_kmh: float,
        confidence: float,
        severity_score: float,
        lane_position: str,
        image_base64: Optional[str] = None,
        custom_uuid: Optional[str] = None,
    ) -> bool:
        """Reject if hazard within dedup_radius_meters exists, else enqueue."""
        for item in self.pending:
            dist = haversine_distance_meters(
                latitude, longitude, item["latitude"], item["longitude"]
            )
            if dist < self.dedup_radius_meters:
                return False

        record = {
            "event_uuid": custom_uuid or str(uuid.uuid4()),
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "speed_kmh": round(speed_kmh, 1),
            "confidence": round(confidence, 2),
            "severity_score": round(severity_score, 1),
            "lane_position": lane_position,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_base64": image_base64,
        }
        self.pending.append(record)
        return True

    def to_sync_batch_payload(self, device_id: str, max_items: int = 50) -> Dict[str, Any]:
        """Slice batch according to <= 50 batch constraint."""
        batch = self.pending[:max_items]
        return {
            "device_id": device_id,
            "detections": batch,
        }

    def flush_batch(self, backend_url: str, device_id: str) -> Dict[str, Any]:
        """Send pending batch to backend HTTP endpoint and purge acknowledged UUIDs."""
        if not self.pending:
            return {"status": "empty", "synced_count": 0, "synced_uuids": []}

        payload = self.to_sync_batch_payload(device_id)
        url = f"{backend_url.rstrip('/')}/api/v1/telemetry/sync"

        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            synced_uuids = set(data.get("synced_uuids", []))
            self.pending = [p for p in self.pending if p["event_uuid"] not in synced_uuids]
            return data


def run_simulation(
    backend_url: str,
    device_id: str = "sim-bike-alpha",
    num_hazards: int = 5,
    include_thumbnails: bool = True,
    delay_sec: float = 0.2,
) -> None:
    """Execute simulated ride, inserting detections and flushing to backend."""
    print(f"🏍️ Starting Motorbike Ride Simulation...")
    print(f"📡 Target Backend: {backend_url}")
    print(f"🆔 Client Device ID: {device_id}")

    # Step 1: Health check
    try:
        with httpx.Client(timeout=3.0) as client:
            h_resp = client.get(f"{backend_url.rstrip('/')}/health")
            if h_resp.status_code == 200:
                print(f"✅ Backend connection established: {h_resp.json()}")
            else:
                print(f"⚠️ Warning: /health returned HTTP {h_resp.status_code}")
    except Exception as e:
        print(f"❌ Could not reach backend at {backend_url}: {e}")
        print("💡 Hint: Run `python user_test_experience/start_test_server.py` in another terminal.")
        return

    # Base coordinates (Bangalore Indiranagar 100ft road)
    base_lat = 12.971891
    base_lon = 77.641151

    queue = LocalSimulationQueue(dedup_radius_meters=3.0)
    lanes = ["LEFT", "CENTER", "RIGHT"]

    print(f"\n🛣️ Riding along trajectory and detecting road hazards...")
    for i in range(num_hazards):
        # Move ~12 meters forward each step (~0.0001 deg lat)
        lat = base_lat + (i * 0.00012)
        lon = base_lon + (i * 0.00004)
        speed = 42.0 + (i * 2.5)
        conf = 0.85 + (i * 0.02)
        severity = 5.0 + (i * 0.9)
        lane = lanes[i % len(lanes)]
        thumb = generate_mock_thumbnail(4096) if include_thumbnails else None

        inserted = queue.enqueue_if_not_duplicate(
            latitude=lat,
            longitude=lon,
            speed_kmh=speed,
            confidence=min(conf, 0.98),
            severity_score=min(severity, 9.8),
            lane_position=lane,
            image_base64=thumb,
        )

        status_str = "ENQUEUED" if inserted else "DUPLICATE BLOCKED"
        print(f"  [{i+1}/{num_hazards}] Coord: ({lat:.6f}, {lon:.6f}) | Lane: {lane:6s} | Sev: {severity:.1f} | {status_str}")
        time.sleep(delay_sec)

    print(f"\n📦 Pending in local offline queue: {len(queue.pending)} hazards")
    print(f"🚀 Flushing batch to {backend_url}/api/v1/telemetry/sync ...")

    try:
        result = queue.flush_batch(backend_url, device_id)
        print(f"✅ Sync Successful! Backend Response:")
        print(f"   Status: {result.get('status')}")
        print(f"   Synced Count: {result.get('synced_count', len(result.get('synced_uuids', [])))}")
        print(f"   Acknowledged UUIDs: {result.get('synced_uuids')}")
        print(f"📊 Remaining in offline queue: {len(queue.pending)}")
    except Exception as e:
        print(f"❌ Failed to sync batch: {e}")


def main():
    parser = argparse.ArgumentParser(description="Motorbike Pothole Client Simulator")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="FastAPI backend URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument("--device-id", default="sim-bike-01", help="Device ID")
    parser.add_argument("--num-hazards", type=int, default=5, help="Number of hazards to simulate")
    parser.add_argument("--no-thumbnails", action="store_true", help="Skip thumbnail generation")
    parser.add_argument("--ping-only", action="store_true", help="Only ping /health and exit")

    args = parser.parse_args()

    if args.ping_only:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{args.server.rstrip('/')}/health")
                print(f"HTTP {res.status_code}: {res.json()}")
        except Exception as e:
            print(f"Connection failed: {e}")
            sys.exit(1)
        return

    run_simulation(
        backend_url=args.server,
        device_id=args.device_id,
        num_hazards=args.num_hazards,
        include_thumbnails=not args.no_thumbnails,
    )


if __name__ == "__main__":
    main()
