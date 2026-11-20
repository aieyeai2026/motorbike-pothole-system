(() => {
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
  const rawMeta = document.getElementById("raw-meta");
  const hudMeta = document.getElementById("hud-meta");

  let videoFps = 30.0;
  let telemetryByFrame = [];
  let selectedRawFile = null;
  let isScrubbing = false;

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
    } else {
      videoRaw.pause();
      videoHud.pause();
      playPauseBtn.textContent = "▶ Play";
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
  videoRaw.addEventListener("ended", () => {
    videoHud.pause();
    playPauseBtn.textContent = "▶ Play";
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
    pipelineStatus.textContent = `⏳ Running real OpenCV + Rust C-ABI pipeline on ${file.name}...`;
    processUploadBtn.disabled = true;
    try {
      const resp = await fetch("/api/process-video", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Max-Frames": "600",
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
      hudMeta.textContent = `Rust C-ABI Output (${data.frames_processed} frames @ ${videoFps.toFixed(1)} FPS)`;
      pipelineStatus.textContent = `✓ Processed ${data.frames_processed} frames through libpothole_core.dylib`;
      videoRaw.currentTime = 0;
      videoHud.currentTime = 0;
      updateTelemetryForTime(0);
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
    pipelineStatus.textContent = "⏳ Running live Rust C-ABI pipeline on benchmarks/fixtures/ride_test.mp4...";
    presetRustFfiBtn.disabled = true;
    try {
      const resp = await fetch("/api/process-fixture", { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Fixture processing failed");

      videoFps = data.fps || 30.0;
      telemetryByFrame = data.telemetry || [];
      videoRaw.src = data.raw_url;
      videoHud.src = data.hud_url;
      rawMeta.textContent = "benchmarks/fixtures/ride_test.mp4";
      hudMeta.textContent = `Live Rust C-ABI Output (${data.frames_processed} frames)`;
      pipelineStatus.textContent = `✓ Processed ${data.frames_processed} frames from ride_test.mp4 via libpothole_core.dylib`;
      videoRaw.currentTime = 0;
      videoHud.currentTime = 0;
      updateTelemetryForTime(0);
    } catch (err) {
      pipelineStatus.textContent = `✗ Error: ${err.message}`;
    } finally {
      presetRustFfiBtn.disabled = false;
    }
  });
})();
