#!/usr/bin/env python3
"""
Dual HTTP & HTTPS Launcher for FastAPI Telemetry Server with Mobile Device Network Discovery.
Runs HTTP on port 8000 and HTTPS on port 8443 (enabling native iOS Safari camera WebRTC access).
"""

import os
import socket
import sys
import threading
import uvicorn


def get_local_lan_ip() -> str:
    """Detect machine local LAN IP on Wi-Fi/Ethernet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(host: str, http_port: int, https_port: int, lan_ip: str) -> None:
    print("=" * 76)
    print("       🏍️  MOTORBIKE POTHOLE SYSTEM — DUAL HTTP/HTTPS TEST SERVER  🏍️")
    print("=" * 76)
    print(f"🖥️  Local Host (HTTP):        http://127.0.0.1:{http_port}")
    print(f"📱  Mobile Browser (HTTP):    http://{lan_ip}:{http_port}/console/")
    print(f"🔒  iPhone Live Camera (HTTPS): https://{lan_ip}:{https_port}/console/")
    print(f"📚  OpenAPI Swagger Docs:     http://127.0.0.1:{http_port}/docs")
    print("-" * 76)
    print("📲 IPHONE CAMERA TESTING INSTRUCTIONS:")
    print("  Mode A (Instant Road Photo Capture):")
    print(f"    • Open Safari on iPhone: http://{lan_ip}:{http_port}/console/")
    print(f"    • Tap the yellow '📸 Snap Road Photo' button to take a live photo of the road!")
    print()
    print("  Mode B (Continuous 30 FPS Live Video Streaming):")
    print(f"    • Open Safari on iPhone: https://{lan_ip}:{https_port}/console/")
    print(f"    • Tap 'Advanced' -> 'Proceed to {lan_ip}' (accept self-signed SSL).")
    print(f"    • Tap '🎥 Live Video Stream' for real-time camera HUD tracking!")
    print("=" * 76)


def run_https_server(app, host: str, port: int, cert_file: str, key_file: str):
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level="warning",
        )
    except Exception as e:
        print(f"⚠️ Could not start HTTPS server on port {port}: {e}")


def main():
    host = "0.0.0.0"
    http_port = 8000
    https_port = 8443
    lan_ip = get_local_lan_ip()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(repo_root, "pothole-sync-backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    cert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    print_banner(host, http_port, https_port, lan_ip)

    from app.main import app

    # Start HTTPS in background thread if certificates exist
    if os.path.isfile(cert_file) and os.path.isfile(key_file):
        https_thread = threading.Thread(
            target=run_https_server,
            args=(app, host, https_port, cert_file, key_file),
            daemon=True,
        )
        https_thread.start()

    # Run HTTP on main thread
    uvicorn.run(app, host=host, port=http_port, log_level="info")


if __name__ == "__main__":
    main()
