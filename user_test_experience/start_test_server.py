#!/usr/bin/env python3
"""
Launcher for FastAPI Telemetry Server with Mobile Device Network Discovery.
Binds to 0.0.0.0:8000 so both local machines and external mobile devices
(via Wi-Fi or USB adb reverse) can connect and test telemetry in real time.
"""

import os
import socket
import sys
import uvicorn


def get_local_lan_ip() -> str:
    """Detect machine local LAN IP on Wi-Fi/Ethernet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Does not send packets, only routes locally to discover default interface IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(host: str, port: int, lan_ip: str) -> None:
    print("=" * 72)
    print("       🏍️  MOTORBIKE POTHOLE SYSTEM — USER TEST EXPERIENCE  🏍️")
    print("=" * 72)
    print(f"🖥️  Local Host URL:        http://127.0.0.1:{port}")
    print(f"📱  Mobile Device (Wi-Fi):  http://{lan_ip}:{port}")
    print(f"🌐  Interactive Console:   http://127.0.0.1:{port}/console")
    print(f"📚  OpenAPI Swagger Docs:  http://127.0.0.1:{port}/docs")
    print("-" * 72)
    print("📲 PHYSICAL MOBILE DEVICE CONNECTION INSTRUCTIONS:")
    print("  Option 1 (Same Wi-Fi):")
    print(f"    • Connect phone to same Wi-Fi network.")
    print(f"    • Open mobile browser or Flutter app.")
    print(f"    • Point backend URL to: http://{lan_ip}:{port}")
    print()
    print("  Option 2 (USB Cable + ADB Reverse Tethering):")
    print(f"    • Plug Android phone via USB and run in terminal:")
    print(f"        adb reverse tcp:{port} tcp:{port}")
    print(f"    • In mobile app, point backend URL to: http://localhost:{port}")
    print()
    print("  Option 3 (Android Emulator):")
    print(f"    • In Android Emulator, point backend URL to: http://10.0.2.2:{port}")
    print("=" * 72)
    print("Starting Uvicorn server on 0.0.0.0:8000 (Press Ctrl+C to stop)...")
    print()


def main():
    host = "0.0.0.0"
    port = 8000
    lan_ip = get_local_lan_ip()

    # Ensure backend app is in pythonpath
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(repo_root, "pothole-sync-backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    print_banner(host, port, lan_ip)

    from app.main import app
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
