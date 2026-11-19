#!/usr/bin/env python3
"""
Motorbike Ride Simulator & Client Device Mock.
Simulates real-world motorbike telemetry, road border detection, lateral centering,
camera detections, 3.0m spatial deduplication, and batch synchronization with the FastAPI backend.
"""

import argparse
import base64
import math
import sys
import time
import uuid
from typing import Any, Dict, List, NamedTuple, Optional
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


class LaneCentering(NamedTuple):
    offset_meters: float
    dist_left_m: float
    dist_right_m: float
    status: str  # "CENTERED", "DRIFTING_LEFT", "DRIFTING_RIGHT"


def compute_lane_centering(
    x_ego: float,
    x_left_border: float = 0.0,
    x_right_border: float = 1.0,
    road_width_m: float = 3.6,
) -> LaneCentering:
    """
    Compute motorbike lateral deviation and distances to road boundaries.
    x_ego: Bike lateral position normalized between 0.0 (far left) and 1.0 (far right).
    """
    span = max(0.01, x_right_border - x_left_border)
    norm_pos = max(0.0, min(1.0, (x_ego - x_left_border) / span))
    dist_left = round(norm_pos * road_width_m, 2)
    dist_right = round((1.0 - norm_pos) * road_width_m, 2)
    offset = round((norm_pos - 0.50) * road_width_m, 2)

    if abs(offset) <= 0.25:
        status = "CENTERED"
    elif offset < -0.25:
        status = "DRIFTING_LEFT"
    else:
        status = "DRIFTING_RIGHT"

    return LaneCentering(
        offset_meters=offset,
        dist_left_m=dist_left,
        dist_right_m=dist_right,
        status=status,
    )


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
        x_ego: float = 0.50,
        road_width_m: float = 3.6,
    ) -> bool:
        """Reject if hazard within dedup_radius_meters exists, else enqueue."""
        for item in self.pending:
            dist = haversine_distance_meters(
                latitude, longitude, item["latitude"], item["longitude"]
            )
            if dist < self.dedup_radius_meters:
                return False

        centering = compute_lane_centering(x_ego, road_width_m=road_width_m)

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
            "lane_offset_meters": centering.offset_meters,
            "dist_left_m": centering.dist_left_m,
            "dist_right_m": centering.dist_right_m,
            "centering_status": centering.status,
        }
        self.pending.append(record)
        return True

    def to_sync_batch_payload(self, device_id: str, max_items: int = 50) -> Dict[str, Any]:
        """Slice batch according to <= 50 batch constraint."""
        batch = self.pending[:max_items]
        # Map fields to SyncBatchRequest schema
        clean_batch = []
        for b in batch:
            clean_batch.append({
                "event_uuid": b["event_uuid"],
                "latitude": b["latitude"],
                "longitude": b["longitude"],
                "speed_kmh": b["speed_kmh"],
                "confidence": b["confidence"],
                "severity_score": b["severity_score"],
                "lane_position": b["lane_position"],
                "captured_at": b["captured_at"],
                "image_base64": b.get("image_base64"),
            })
        return {
            "device_id": device_id,
            "detections": clean_batch,
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
    fixed_ego_pos: Optional[float] = None,
) -> None:
    """Execute simulated ride, computing lane centering, boundary clearance, and syncing."""
    print("=" * 76)
    print("       🏍️  MOTORBIKE TELEMETRY & LANE CENTERING SIMULATOR  🏍️")
    print("=" * 76)
    print(f"📡 Target Backend:     {backend_url}")
    print(f"🆔 Client Device ID:   {device_id}")
    print(f"🛣️ Standard Lane Width: 3.6 meters")
    print("-" * 76)

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

    # Base coordinates (Kakkanad SmartCity / Infopark stretch)
    base_lat = 10.009240
    base_lon = 76.363820

    queue = LocalSimulationQueue(dedup_radius_meters=3.0)
    lanes = ["LEFT", "CENTER", "RIGHT"]

    print("\n🛣️ Simulating ride trajectory with ego-centering & road border detection:")
    print("   [Idx] Coordinates       Lane    Sev   Ego Offset & Centering         Clearance (L / R)  Queue Status")
    print("   " + "-" * 90)

    for i in range(num_hazards):
        # Forward progress ~13m per step
        lat = base_lat + (i * 0.00012)
        lon = base_lon + (i * 0.00004)
        speed = 42.0 + (i * 2.5)
        conf = 0.85 + (i * 0.02)
        severity = 5.0 + (i * 0.9)
        lane = lanes[i % len(lanes)]
        thumb = generate_mock_thumbnail(4096) if include_thumbnails else None

        # Simulate natural motorcycle lateral position or use fixed input
        if fixed_ego_pos is not None:
            x_ego = fixed_ego_pos
        else:
            # Sinuous ride trajectory swaying gently between lanes
            x_ego = 0.50 + math.sin(i * 1.25) * 0.22

        inserted = queue.enqueue_if_not_duplicate(
            latitude=lat,
            longitude=lon,
            speed_kmh=speed,
            confidence=min(conf, 0.98),
            severity_score=min(severity, 9.8),
            lane_position=lane,
            image_base64=thumb,
            x_ego=x_ego,
        )

        centering = compute_lane_centering(x_ego)
        offset_sign = "+" if centering.offset_meters > 0 else ""
        offset_str = f"{offset_sign}{centering.offset_meters:.2f}m ({centering.status:14s})"
        clearance_str = f"L: {centering.dist_left_m:.1f}m | R: {centering.dist_right_m:.1f}m"
        status_str = "ENQUEUED" if inserted else "BLOCKED (<3m DUP)"

        print(
            f"   [{i+1:2d}] ({lat:.6f}, {lon:.6f})  {lane:6s}  {severity:.1f}  "
            f"{offset_str:28s}  {clearance_str:18s}  {status_str}"
        )
        time.sleep(delay_sec)

    print("-" * 76)
    print(f"📦 Total pending hazards in local queue: {len(queue.pending)}")
    print(f"🚀 Flushing batch to {backend_url}/api/v1/telemetry/sync ...")

    try:
        result = queue.flush_batch(backend_url, device_id)
        synced_count = result.get('synced_count', len(result.get('synced_uuids', [])))
        print(f"✅ Sync Successful! Status: {result.get('status')} | Synced Count: {synced_count}")
        print(f"   Acknowledged UUIDs: {result.get('synced_uuids')}")
        print(f"📊 Remaining in offline queue: {len(queue.pending)}")
    except Exception as e:
        print(f"❌ Failed to sync batch: {e}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Motorbike Pothole Client & Lane Centering Simulator")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="FastAPI backend URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument("--device-id", default="sim-bike-01", help="Device ID")
    parser.add_argument("--num-hazards", type=int, default=5, help="Number of hazards to simulate")
    parser.add_argument("--no-thumbnails", action="store_true", help="Skip thumbnail generation")
    parser.add_argument("--ping-only", action="store_true", help="Only ping /health and exit")
    parser.add_argument(
        "--ego-pos",
        type=float,
        default=None,
        help="Fixed bike lateral position [0.0 - 1.0] (0.50 = center, <0.43 = left, >0.57 = right)",
    )

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
        fixed_ego_pos=args.ego_pos,
    )


if __name__ == "__main__":
    main()
