(() => {
  // Raw YOLOv8 confidence gate; must match DEFAULT_CONF_THRESHOLD in pyresearch_pothole_detector.py
  const YOLO_CONF = 0.45;
  const videoRaw = document.getElementById("video-raw");
  const videoHud = document.getElementById("video-hud");
  const playPauseBtn = document.getElementById("play-pause-btn");
  const stepBackBtn = document.getElementById("step-back-btn");
  const stepFwdBtn = document.getElementById("step-fwd-btn");
  const scrubber = document.getElementById("timeline-scrubber");
  const speedSelector = document.getElementById("speed-selector");

  const rawFileInput = document.getElementById("raw-file-input");
  const hudFileInput = document.getElementById("hud-file-input");
  const processUploadBtn = document.getElementById("process-upload-btn");
  const presetRustFfiBtn = document.getElementById("preset-rust-ffi-btn");
  const pipelineStatus = document.getElementById("pipeline-status");
  const dropZone = document.getElementById("drop-zone");

  const playerContainer = document.getElementById("player-container");
  const curtainToggleBtn = document.getElementById("curtain-toggle-btn");
  const curtainControlWrap = document.getElementById("curtain-control-wrap");
  const curtainSlider = document.getElementById("curtain-slider");
  const curtainPctLabel = document.getElementById("curtain-pct-label");

  const timecodeDisplay = document.getElementById("timecode-display");
  const metricHazardCount = document.getElementById("metric-hazard-count");
  const metricClosestDistance = document.getElementById("metric-closest-distance");
  const metricAssignedLane = document.getElementById("metric-assigned-lane");
  const metricFpsLatency = document.getElementById("metric-fps-latency");
  const telemetryBar = document.querySelector(".telemetry-bar");
  const rawMeta = document.getElementById("raw-meta");
  const hudMeta = document.getElementById("hud-meta");
  const hudCanvas = document.getElementById("hud-overlay-canvas");
  const hudCtx = hudCanvas ? hudCanvas.getContext("2d") : null;

  const layerCheckboxes = document.querySelectorAll(".layer-checkbox");
  const layersAllOnBtn = document.getElementById("layers-all-on-btn");
  const layersPotholesOnlyBtn = document.getElementById("layers-potholes-only-btn");
  const layersBoundariesOnlyBtn = document.getElementById("layers-boundaries-only-btn");
  const layersAllOffBtn = document.getElementById("layers-all-off-btn");

  const layerState = {
    horizon: true,
    boundaries: true,
    carpet: true,
    potholes: true,
    distance_ticks: true,
    telemetry: true,
  };

  let videoFps = 30.0;
  let telemetryByFrame = [];
  let selectedRawFile = null;
  let isScrubbing = false;
  let rafId = null;

  function syncLayerCheckboxes() {
    layerCheckboxes.forEach((cb) => {
      const key = cb.getAttribute("data-layer");
      if (key && key in layerState) {
        cb.checked = !!layerState[key];
      }
    });
    if (telemetryBar) {
      telemetryBar.classList.toggle("telemetry-dimmed", !layerState.telemetry);
    }
    renderLiveHudCanvas(videoRaw.currentTime || 0);
  }

  function renderLiveHudCanvas(sec) {
    if (!hudCanvas || !hudCtx || !telemetryByFrame || telemetryByFrame.length === 0) {
      if (hudCanvas) hudCanvas.classList.add("hidden");
      return;
    }
    if (videoRaw.readyState < 2) {
      return;
    }
    const w = videoRaw.videoWidth || 848;
    const h = videoRaw.videoHeight || 478;
    if (hudCanvas.width !== w || hudCanvas.height !== h) {
      hudCanvas.width = w;
      hudCanvas.height = h;
    }
    hudCanvas.classList.remove("hidden");

    // Draw synchronized raw frame cleanly so toggled-off layers vanish completely
    hudCtx.clearRect(0, 0, w, h);
    hudCtx.drawImage(videoRaw, 0, 0, w, h);

    const frameIndex = Math.max(0, Math.floor(sec * videoFps));
    const idx = Math.min(frameIndex, telemetryByFrame.length - 1);
    const t = telemetryByFrame[idx];
    if (!t) return;

    const isCrit = !!t.closest_is_critical;
    const carpetStroke = isCrit ? "#ff1744" : "#00e676";
    const carpetFill = isCrit ? "rgba(255, 23, 68, 0.38)" : "rgba(0, 230, 118, 0.38)";

    // Layer 1: Solid Green/Crimson Trajectory Carpet
    if (layerState.carpet && Array.isArray(t.carpet_ys) && t.carpet_ys.length > 1) {
      hudCtx.save();
      hudCtx.beginPath();
      for (let i = 0; i < t.carpet_ys.length; i++) {
        const px = t.carpet_left_xs[i] * w;
        const py = t.carpet_ys[i] * h;
        if (i === 0) hudCtx.moveTo(px, py);
        else hudCtx.lineTo(px, py);
      }
      for (let i = t.carpet_ys.length - 1; i >= 0; i--) {
        hudCtx.lineTo(t.carpet_right_xs[i] * w, t.carpet_ys[i] * h);
      }
      hudCtx.closePath();
      hudCtx.fillStyle = carpetFill;
      hudCtx.fill();

      hudCtx.lineWidth = 2;
      hudCtx.strokeStyle = carpetStroke;
      hudCtx.beginPath();
      for (let i = 0; i < t.carpet_ys.length; i++) {
        const px = t.carpet_left_xs[i] * w;
        const py = t.carpet_ys[i] * h;
        if (i === 0) hudCtx.moveTo(px, py);
        else hudCtx.lineTo(px, py);
      }
      hudCtx.stroke();

      hudCtx.beginPath();
      for (let i = 0; i < t.carpet_ys.length; i++) {
        const px = t.carpet_right_xs[i] * w;
        const py = t.carpet_ys[i] * h;
        if (i === 0) hudCtx.moveTo(px, py);
        else hudCtx.lineTo(px, py);
      }
      hudCtx.stroke();
      hudCtx.restore();
    }

    // Layer 2: Perspective Distance Ticks (5m - 20m)
    if (layerState.distance_ticks && Array.isArray(t.distance_ticks)) {
      hudCtx.save();
      hudCtx.strokeStyle = "#ffd700";
      hudCtx.lineWidth = 2;
      hudCtx.font = "bold 12px monospace";
      t.distance_ticks.forEach((tick) => {
        const py = tick.y_norm * h;
        const lx = tick.x_left_norm * w;
        const rx = tick.x_right_norm * w;
        hudCtx.beginPath();
        hudCtx.moveTo(lx, py);
        hudCtx.lineTo(rx, py);
        hudCtx.moveTo(lx, py - 4);
        hudCtx.lineTo(lx, py + 4);
        hudCtx.moveTo(rx, py - 4);
        hudCtx.lineTo(rx, py + 4);
        hudCtx.stroke();
        const label = `${tick.distance_m}m`;
        const lxText = Math.min(w - 34, rx + 6);
        hudCtx.lineWidth = 3;
        hudCtx.strokeStyle = "#000000";
        hudCtx.strokeText(label, lxText, py + 4);
        hudCtx.fillStyle = "#ffd700";
        hudCtx.fillText(label, lxText, py + 4);
        hudCtx.lineWidth = 2;
        hudCtx.strokeStyle = "#ffd700";
      });
      hudCtx.restore();
    }

    // Layer 3: Curved Cyan Road Boundaries (measured road edges; falls back to legacy arrays on old telemetry)
    if (layerState.boundaries) {
      const edgeSets = "road_edges" in t
        ? Object.values(t.road_edges || {}).map((e) => e.xs.map((x, i) => [x, e.ys[i]]))
        : Array.isArray(t.bound_ys) && t.bound_ys.length > 1
          ? [t.left_bound_xs, t.right_bound_xs].map((xs) => xs.map((x, i) => [x, t.bound_ys[i]]))
          : [];
      hudCtx.save();
      hudCtx.strokeStyle = "#00e5ff";
      hudCtx.lineWidth = 2;
      for (const pts of edgeSets) {
        if (pts.length < 2) continue;
        hudCtx.beginPath();
        pts.forEach(([x, y], i) => {
          if (i === 0) hudCtx.moveTo(x * w, y * h);
          else hudCtx.lineTo(x * w, y * h);
        });
        hudCtx.stroke();
      }
      hudCtx.restore();
    }

    // Layer 4: Tilting Green Horizon Line centered at VP
    if (layerState.horizon && Array.isArray(t.vp)) {
      const vpX = t.vp[0] * w;
      const vpY = t.vp[1] * h;
      const bank = t.theta_bank || 0;
      const span = w * 0.65;
      const dx = span * Math.cos(bank);
      const dy = span * Math.sin(bank);
      hudCtx.save();
      hudCtx.strokeStyle = "#00e676";
      hudCtx.lineWidth = 2;
      hudCtx.beginPath();
      hudCtx.moveTo(vpX - dx, vpY - dy);
      hudCtx.lineTo(vpX + dx, vpY + dy);
      hudCtx.stroke();
      hudCtx.restore();
    }

    // Layer 5: YOLOv8 pothole boxes (raw conf >= YOLO_CONF) - Pass 1 boxes, Pass 2 label pills
    if (layerState.potholes && Array.isArray(t.potholes)) {
      const verified = t.potholes
        .filter((d) => (d.confidence || 0) >= YOLO_CONF)
        .sort((a, b) => (a.in_path === b.in_path ? b.confidence - a.confidence : a.in_path ? -1 : 1));
      const pills = [];
      hudCtx.save();
      verified.forEach((det) => {
        const bx1 = det.x_min * w;
        const by1 = det.y_min * h;
        const bw = (det.x_max - det.x_min) * w;
        const bh = (det.y_max - det.y_min) * h;
        const cx = bx1 + bw * 0.5;
        const cy = by1 + bh * 0.5;
        const col = !det.in_path ? "#0055ff" : det.distance_m < 8.0 ? "#ff1744" : "#00e5ff";

        if (!det.in_path && pills.length >= 3) {
          hudCtx.fillStyle = "#0055ff";
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, 4, 0, Math.PI * 2);
          hudCtx.fill();
          return;
        }
        if (det.in_path && det.distance_m < 8.0) {
          hudCtx.strokeStyle = "#ff1744";
          hudCtx.lineWidth = 2;
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, 10, 0, Math.PI * 2);
          hudCtx.stroke();
        }
        hudCtx.strokeStyle = col;
        hudCtx.lineWidth = 2;
        hudCtx.strokeRect(bx1, by1, bw, bh);
        if (pills.length < 3) {
          const pct = Math.round((det.confidence || 0) * 100);
          pills.push({ bx1, by1, col, label: `POTHOLE ${Number(det.distance_m).toFixed(1)}m (${pct}%)` });
        }
      });

      const drawnYs = [];
      hudCtx.font = "bold 11px sans-serif";
      pills.forEach(({ bx1, by1, col, label }) => {
        const tw = hudCtx.measureText(label).width;
        let ly = Math.max(28, by1 - 18);
        while (drawnYs.some((py) => Math.abs(ly - py) < 20)) ly += 20;
        if (ly + 16 < h) {
          hudCtx.fillStyle = col;
          hudCtx.fillRect(bx1, ly, tw + 8, 16);
          hudCtx.fillStyle = col === "#00e5ff" ? "#000000" : "#ffffff";
          hudCtx.fillText(label, bx1 + 4, ly + 12);
          drawnYs.push(ly);
        }
      });
      hudCtx.restore();
    }

    // Layer 6: Telemetry Header & Collision Alert Banner
    if (layerState.telemetry) {
      hudCtx.save();
      if (t.header_text) {
        hudCtx.fillStyle = "rgba(12, 16, 24, 0.92)";
        hudCtx.fillRect(0, 0, w, 26);
        hudCtx.fillStyle = "#00e676";
        hudCtx.font = "12px monospace";
        hudCtx.fillText(t.header_text, 10, 17);
      }
      if (isCrit && t.blocking_distance_m !== null && t.blocking_distance_m !== undefined) {
        hudCtx.fillStyle = "#eb122d";
        hudCtx.fillRect(w * 0.2, 30, w * 0.6, 28);
        hudCtx.fillStyle = "#ffffff";
        hudCtx.font = "bold 14px sans-serif";
        hudCtx.fillText(`BRAKE / COLLISION HAZARD: ${Number(t.blocking_distance_m).toFixed(1)}m`, w * 0.2 + 18, 49);
      }
      hudCtx.restore();
    }
  }

  function startRenderLoop() {
    if (rafId !== null) return;
    const tick = () => {
      if (!videoRaw.paused && !videoRaw.ended) {
        renderLiveHudCanvas(videoRaw.currentTime || 0);
        rafId = requestAnimationFrame(tick);
      } else {
        rafId = null;
      }
    };
    rafId = requestAnimationFrame(tick);
  }

  layerCheckboxes.forEach((cb) => {
    cb.addEventListener("change", () => {
      const key = cb.getAttribute("data-layer");
      if (key && key in layerState) {
        layerState[key] = cb.checked;
        syncLayerCheckboxes();
      }
    });
  });

  if (layersAllOnBtn) {
    layersAllOnBtn.addEventListener("click", () => {
      Object.keys(layerState).forEach((k) => { layerState[k] = true; });
      syncLayerCheckboxes();
    });
  }
  if (layersPotholesOnlyBtn) {
    layersPotholesOnlyBtn.addEventListener("click", () => {
      Object.keys(layerState).forEach((k) => { layerState[k] = (k === "potholes"); });
      syncLayerCheckboxes();
    });
  }
  if (layersBoundariesOnlyBtn) {
    layersBoundariesOnlyBtn.addEventListener("click", () => {
      Object.keys(layerState).forEach((k) => { layerState[k] = (k === "boundaries"); });
      syncLayerCheckboxes();
    });
  }
  if (layersAllOffBtn) {
    layersAllOffBtn.addEventListener("click", () => {
      Object.keys(layerState).forEach((k) => { layerState[k] = false; });
      syncLayerCheckboxes();
    });
  }

  function formatTimecode(sec, fps) {
    const totalFrames = Math.max(0, Math.floor(sec * fps));
    const hrs = Math.floor(sec / 3600);
    const mins = Math.floor((sec % 3600) / 60);
    const secs = Math.floor(sec % 60);
    const frameInSec = Math.floor((sec % 1) * fps);
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(hrs)}:${pad(mins)}:${pad(secs)}:${pad(frameInSec)} | Frame ${totalFrames}`;
  }

  function updateTelemetryForTime(sec) {
    const frameIndex = Math.max(0, Math.floor(sec * videoFps));
    timecodeDisplay.textContent = formatTimecode(sec, videoFps);

    if (!telemetryByFrame || telemetryByFrame.length === 0) {
      return;
    }
    const idx = Math.min(frameIndex, telemetryByFrame.length - 1);
    const t = telemetryByFrame[idx];
    if (!t) return;

    metricHazardCount.textContent = `${t.hazard_count} (${t.pothole_count || 0} pothole)`;
    if (t.closest_distance_m !== null && t.closest_distance_m !== undefined) {
      if (t.closest_is_critical) {
        metricClosestDistance.textContent = `${t.closest_distance_m.toFixed(1)}m - CRITICAL (<8m)`;
        metricClosestDistance.classList.add("critical");
      } else {
        metricClosestDistance.textContent = `${t.closest_distance_m.toFixed(1)}m - SAFE`;
        metricClosestDistance.classList.remove("critical");
      }
    } else {
      metricClosestDistance.textContent = "Clear Road";
      metricClosestDistance.classList.remove("critical");
    }

    metricAssignedLane.textContent = `${t.assigned_lane} • ${t.centering_status} (${t.offset_m >= 0 ? "+" : ""}${t.offset_m}m)`;
    metricFpsLatency.textContent = `${videoFps.toFixed(1)} FPS / ${t.postprocess_ms.toFixed(2)} ms`;
    renderLiveHudCanvas(sec);
  }

  function syncHudToRaw(forceSeek = false) {
    const t = videoRaw.currentTime || 0;
    const dur = videoRaw.duration || 0;
    if (dur > 0 && !isScrubbing) {
      scrubber.value = ((t / dur) * 100).toFixed(3);
    }
    if (videoHud.readyState >= 1) {
      const drift = Math.abs((videoHud.currentTime || 0) - t);
      if (forceSeek || drift > 0.035) {
        videoHud.currentTime = Math.min(t, videoHud.duration || t);
      }
    }
    updateTelemetryForTime(t);
  }

  function togglePlayPause() {
    if (videoRaw.paused) {
      videoRaw.play();
      if (videoHud.src) videoHud.play();
      playPauseBtn.textContent = "⏸ Pause";
      startRenderLoop();
    } else {
      videoRaw.pause();
      videoHud.pause();
      playPauseBtn.textContent = "▶ Play";
      renderLiveHudCanvas(videoRaw.currentTime || 0);
    }
  }

  function stepFrames(deltaFrames) {
    videoRaw.pause();
    videoHud.pause();
    playPauseBtn.textContent = "▶ Play";
    const dt = deltaFrames / videoFps;
    const nextTime = Math.max(0, Math.min(videoRaw.duration || 0, (videoRaw.currentTime || 0) + dt));
    videoRaw.currentTime = nextTime;
    if (videoHud.readyState >= 1) {
      videoHud.currentTime = Math.min(nextTime, videoHud.duration || nextTime);
    }
    syncHudToRaw(true);
  }

  playPauseBtn.addEventListener("click", togglePlayPause);
  stepBackBtn.addEventListener("click", () => stepFrames(-1));
  stepFwdBtn.addEventListener("click", () => stepFrames(1));

  scrubber.addEventListener("input", () => {
    isScrubbing = true;
    const dur = videoRaw.duration || 0;
    if (dur > 0) {
      const targetTime = (parseFloat(scrubber.value) / 100.0) * dur;
      videoRaw.currentTime = targetTime;
      if (videoHud.readyState >= 1) {
        videoHud.currentTime = Math.min(targetTime, videoHud.duration || targetTime);
      }
      updateTelemetryForTime(targetTime);
    }
  });

  scrubber.addEventListener("change", () => {
    isScrubbing = false;
    syncHudToRaw(true);
  });

  speedSelector.addEventListener("change", () => {
    const rate = parseFloat(speedSelector.value) || 1.0;
    videoRaw.playbackRate = rate;
    videoHud.playbackRate = rate;
  });

  videoRaw.addEventListener("timeupdate", () => syncHudToRaw(false));
  videoRaw.addEventListener("loadeddata", () => renderLiveHudCanvas(videoRaw.currentTime || 0));
  videoRaw.addEventListener("seeked", () => renderLiveHudCanvas(videoRaw.currentTime || 0));
  videoRaw.addEventListener("ended", () => {
    videoHud.pause();
    playPauseBtn.textContent = "▶ Play";
    renderLiveHudCanvas(videoRaw.currentTime || 0);
  });

  // Curtain slider mode
  curtainToggleBtn.addEventListener("click", () => {
    const isCurtain = playerContainer.classList.toggle("curtain-mode");
    playerContainer.classList.toggle("side-by-side", !isCurtain);
    curtainControlWrap.classList.toggle("hidden", !isCurtain);
  });

  curtainSlider.addEventListener("input", () => {
    const pct = `${curtainSlider.value}%`;
    document.documentElement.style.setProperty("--curtain-split", pct);
    curtainPctLabel.textContent = pct;
  });

  // Keyboard shortcuts: Space (Play/Pause), Left/Right arrows (Frame step)
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      stepFrames(-1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      stepFrames(1);
    }
  });

  // Raw video upload selection
  rawFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    selectedRawFile = file;
    videoRaw.src = URL.createObjectURL(file);
    rawMeta.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    pipelineStatus.textContent = `Loaded ${file.name} — Click 'Run Real Rust Pipeline' to generate HUD output`;
  });

  hudFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    telemetryByFrame = [];
    if (hudCanvas) hudCanvas.classList.add("hidden");
    videoHud.src = URL.createObjectURL(file);
    hudMeta.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  });

  // Drag and drop raw video
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      selectedRawFile = file;
      videoRaw.src = URL.createObjectURL(file);
      rawMeta.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
      runPipelineOnUploadedFile(file);
    }
  });

  async function runPipelineOnUploadedFile(file) {
    pipelineStatus.textContent = `⏳ Running YOLOv8 @ 0.45 + Rust C-ABI pipeline on ${file.name}...`;
    processUploadBtn.disabled = true;
    try {
      const resp = await fetch("/api/process-video", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Max-Frames": "600",
          "X-Conf-Threshold": String(YOLO_CONF),
          "X-Hud-Layers": JSON.stringify(layerState),
        },
        body: file,
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Pipeline execution failed");

      videoFps = data.fps || 30.0;
      telemetryByFrame = data.telemetry || [];
      videoRaw.src = data.raw_url;
      videoHud.src = data.hud_url;
      rawMeta.textContent = `${file.name} (${data.frames_processed} frames)`;
      hudMeta.textContent = `YOLOv8 @ 0.45 + Rust C-ABI (${data.frames_processed} frames @ ${videoFps.toFixed(1)} FPS)`;
      pipelineStatus.textContent = `✓ Processed ${data.frames_processed} frames through PyResearch + libpothole_core.dylib`;
      videoRaw.currentTime = 0;
      videoHud.currentTime = 0;
      updateTelemetryForTime(0);
      renderLiveHudCanvas(0);
    } catch (err) {
      pipelineStatus.textContent = `✗ Error: ${err.message}`;
    } finally {
      processUploadBtn.disabled = false;
    }
  }

  processUploadBtn.addEventListener("click", () => {
    if (!selectedRawFile) {
      pipelineStatus.textContent = "Please select a raw video file first (or click Process Benchmark)";
      return;
    }
    runPipelineOnUploadedFile(selectedRawFile);
  });

  presetRustFfiBtn.addEventListener("click", async () => {
    pipelineStatus.textContent = "⏳ Running YOLOv8 @ 0.45 + Rust C-ABI pipeline on benchmarks/fixtures/ride_test.mp4...";
    presetRustFfiBtn.disabled = true;
    try {
      const resp = await fetch("/api/process-fixture", {
        method: "POST",
        headers: {
          "X-Conf-Threshold": String(YOLO_CONF),
          "X-Hud-Layers": JSON.stringify(layerState),
        },
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Fixture processing failed");

      videoFps = data.fps || 30.0;
      telemetryByFrame = data.telemetry || [];
      videoRaw.src = data.raw_url;
      videoHud.src = data.hud_url;
      rawMeta.textContent = "benchmarks/fixtures/ride_test.mp4";
      hudMeta.textContent = `Live YOLOv8 @ 0.45 + Rust C-ABI (${data.frames_processed} frames)`;
      pipelineStatus.textContent = `✓ Processed ${data.frames_processed} frames from ride_test.mp4 via libpothole_core.dylib`;
      videoRaw.currentTime = 0;
      videoHud.currentTime = 0;
      updateTelemetryForTime(0);
      renderLiveHudCanvas(0);
    } catch (err) {
      pipelineStatus.textContent = `✗ Error: ${err.message}`;
    } finally {
      presetRustFfiBtn.disabled = false;
    }
  });
})();
