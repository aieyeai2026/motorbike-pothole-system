# Google Cloud Deployment Guide: Pothole Video Evaluator (Port 8085)

This guide explains how to deploy the Motorbike Pothole Evaluator to Google Cloud (Project: `talentmatch-ai-505909` or any GCP project) to obtain a live link for testing video uploads.

---

## 1. What to Upload

If uploading directly from your machine, use the pre-packaged zip archive:
* **`deploy_package.zip`** (17 MB)

It includes:
- `user_test_experience/video_evaluator/` (HTML, CSS, JS, PyTorch YOLOv8 neural detector, and model weights)
- `pothole_core/` (Rust C-ABI spatial geometry & corridor classification engine)
- `Dockerfile` & `requirements.txt`
- `run_evaluator.sh`

*(Large benchmark videos and `.git` caches are excluded to keep the package lightweight).*

---

## 2. Fast Deployment via Google Cloud Shell (Recommended)

1. Open **Google Cloud Console** (`https://console.cloud.google.com/`).
2. Click the **Activate Cloud Shell** icon (`>_`) in the top navigation bar.
3. In the Cloud Shell terminal:
   - Click the three dots menu (`⋮`) $\rightarrow$ **Upload** $\rightarrow$ select `deploy_package.zip`.
   - Run the following commands:

```bash
unzip -q deploy_package.zip -d pothole-evaluator
cd pothole-evaluator

# Build and start container
docker build -t pothole-evaluator .
docker run -d -p 8085:8085 --name evaluator pothole-evaluator
```

4. **Access the Live Link**:
   - In the Cloud Shell toolbar, click the **Web Preview** icon.
   - Click **Change Port**, type **`8085`**, and click **Change and Preview**.
   - Your secure live HTTPS testing URL will open immediately.

---

## 3. Permanent Public Deployment via Google Cloud Run

To deploy a permanent public link:

```bash
cd pothole-evaluator

gcloud run deploy pothole-evaluator \
  --source . \
  --port 8085 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --no-cpu-throttling \
  --allow-unauthenticated \
  --region us-central1
```

Once deployment completes, Cloud Run will print the public Service URL (e.g. `https://pothole-evaluator-xxxx.a.run.app`).

---

## 4. Verification

1. Open the live URL.
2. Click **Upload Video** and select an `.mp4` test ride.
3. Verify that the neural model detects potholes with bounding boxes, the green AR trajectory carpet curves dynamically, and telemetry is reported in real time.
