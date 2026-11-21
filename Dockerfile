# ==============================================================================
# Google Cloud / Container Deployment for Pothole Evaluator (Port 8085)
# Deploy command (Google Cloud Run):
#   gcloud run deploy pothole-evaluator \
#     --source . \
#     --port 8085 \
#     --memory 4Gi \
#     --cpu 2 \
#     --timeout 900 \
#     --no-cpu-throttling \
#     --allow-unauthenticated \
#     --region us-central1
# ==============================================================================

# Stage 1: Build Rust pothole_core C-ABI shared library
FROM rust:1.82-slim AS rust-builder
WORKDIR /build
COPY pothole_core ./pothole_core
WORKDIR /build/pothole_core
RUN cargo build --release

# Stage 2: Runtime image with Python, OpenCV, and PyTorch (CPU)
FROM python:3.11-slim

# System packages: ffmpeg (for web-compatible video output) & glib/libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements (CPU torch for fast download and small container size)
COPY user_test_experience/video_evaluator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy "opencv-python-headless>=4.8.0"

# Copy compiled Rust shared library
COPY --from=rust-builder /build/pothole_core/target/release/libpothole_core.so /app/pothole_core/target/release/libpothole_core.so

# Copy evaluator application code and fixtures structure
COPY user_test_experience /app/user_test_experience
RUN mkdir -p /app/benchmarks/fixtures/uploaded_evals

ENV PORT=8085
ENV HOST=0.0.0.0
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
EXPOSE 8085

CMD ["python3", "user_test_experience/video_evaluator/server.py"]
