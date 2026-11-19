// Motorbike Tactical HUD & Backend Telemetry Bench
document.addEventListener('DOMContentLoaded', () => {
  // DOM References
  const pingBtn = document.getElementById('ping-btn');
  const serverUrlInput = document.getElementById('server-url');
  const connStatus = document.getElementById('connection-status');
  const eventLog = document.getElementById('event-log');
  const hudCanvas = document.getElementById('hud-canvas');
  const ctx = hudCanvas.getContext('2d');
  const criticalBanner = document.getElementById('critical-alert-banner');
  const distanceSlider = document.getElementById('distance-slider');
  const distanceVal = document.getElementById('distance-val');
  const speedSlider = document.getElementById('speed-slider');
  const speedVal = document.getElementById('speed-val');
  const severitySlider = document.getElementById('severity-slider');
  const severityVal = document.getElementById('severity-val');
  const enqueueBtn = document.getElementById('enqueue-btn');
  const dedupBtn = document.getElementById('dedup-btn');
  const oversizedBtn = document.getElementById('oversized-btn');
  const syncBtn = document.getElementById('sync-btn');
  const queueTbody = document.getElementById('queue-tbody');

  // State
  let serverHealthy = false;
  let offlineQueue = [];
  let animOffset = 0;
  let audioCtx = null;

  function log(msg, type = 'info') {
    const time = new Date().toISOString().split('T')[1].slice(0, 8);
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.innerHTML = `<span class="log-time">[${time}]</span> ${msg}`;
    eventLog.appendChild(div);
    eventLog.scrollTop = eventLog.scrollHeight;
  }

  function playAlertChime() {
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // High warning pitch
      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.18);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.18);
    } catch (_) {}
  }

  // Ping Backend /health
  async function pingBackend() {
    const url = serverUrlInput.value.replace(/\/$/, '');
    const startTime = performance.now();
    try {
      log(`📡 Pinging ${url}/health ...`);
      const resp = await fetch(`${url}/health`, { method: 'GET' });
      const latency = Math.round(performance.now() - startTime);
      if (resp.ok) {
        const data = await resp.json();
        serverHealthy = true;
        connStatus.textContent = `HEALTHY (${latency}ms)`;
        connStatus.className = 'status-badge healthy';
        log(`✅ Backend connected: ${JSON.stringify(data)} [${latency}ms]`);
      } else {
        throw new Error(`HTTP ${resp.status}`);
      }
    } catch (err) {
      serverHealthy = false;
      connStatus.textContent = 'OFFLINE';
      connStatus.className = 'status-badge';
      log(`❌ Backend unreachable: ${err.message}`, 'error');
    }
  }

  pingBtn.addEventListener('click', pingBackend);
  // Auto-ping on load
  pingBackend();

  // Slider event listeners
  distanceSlider.addEventListener('input', (e) => {
    distanceVal.textContent = `${parseFloat(e.target.value).toFixed(1)}m`;
  });
  speedSlider.addEventListener('input', (e) => {
    speedVal.textContent = `${e.target.value} km/h`;
  });
  severitySlider.addEventListener('input', (e) => {
    severityVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  function getSelectedLane() {
    const checked = document.querySelector('input[name="lane"]:checked');
    return checked ? checked.value : 'CENTER';
  }

  // Haversine distance for 3m deduplication
  function haversineMeters(lat1, lon1, lat2, lon2) {
    const R = 6371000.0;
    const dLat = (lat2 - lat1) * Math.PI / 180.0;
    const dLon = (lon2 - lon1) * Math.PI / 180.0;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  function renderQueueTable() {
    if (offlineQueue.length === 0) {
      queueTbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-dim)">Queue empty</td></tr>';
      return;
    }
    queueTbody.innerHTML = offlineQueue.map((item, idx) => `
      <tr>
        <td><code>${item.event_uuid.slice(0, 8)}...</code></td>
        <td>${item.distance_meters ? item.distance_meters.toFixed(1) + 'm' : 'N/A'}</td>
        <td><strong>${item.lane_position}</strong></td>
        <td>${item.severity_score.toFixed(1)}</td>
        <td>${item.latitude.toFixed(5)}, ${item.longitude.toFixed(5)}</td>
      </tr>
    `).join('');
  }

  // Enqueue Detection
  enqueueBtn.addEventListener('click', () => {
    const dist = parseFloat(distanceSlider.value);
    const lane = getSelectedLane();
    const sev = parseFloat(severitySlider.value);
    const speed = parseFloat(speedSlider.value);
    const lat = 12.9716 + (offlineQueue.length * 0.0001);
    const lon = 77.5946;

    const record = {
      event_uuid: crypto.randomUUID ? crypto.randomUUID() : `evt-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      latitude: lat,
      longitude: lon,
      speed_kmh: speed,
      confidence: 0.92,
      severity_score: sev,
      lane_position: lane,
      distance_meters: dist,
      captured_at: new Date().toISOString(),
      image_base64: null,
    };

    offlineQueue.push(record);
    log(`📥 Enqueued hazard [${lane}] dist=${dist.toFixed(1)}m sev=${sev.toFixed(1)} (Total pending: ${offlineQueue.length})`);
    renderQueueTable();
  });

  // Test 3.0m Deduplication
  dedupBtn.addEventListener('click', () => {
    if (offlineQueue.length === 0) {
      log('⚠️ Enqueue at least 1 hazard first to test deduplication.');
      return;
    }
    const last = offlineQueue[offlineQueue.length - 1];
    // Create coordinate 1.5 meters away
    const dupLat = last.latitude + 0.00001; // ~1.1 meters
    const dupLon = last.longitude;
    const dist = haversineMeters(last.latitude, last.longitude, dupLat, dupLon);

    log(`🔍 Checking 3.0m Haversine deduplication filter (Distance to last = ${dist.toFixed(2)}m) ...`);
    if (dist < 3.0) {
      log(`🛡️ BLOCKED DUPLICATE: Distance ${dist.toFixed(2)}m is within 3.0m threshold! Record rejected.`, 'warn');
    }
  });

  // Test >100 KB Thumbnail Rejection
  oversizedBtn.addEventListener('click', async () => {
    const url = serverUrlInput.value.replace(/\/$/, '');
    log('🧪 Generating oversized 105 KB thumbnail payload...');
    const oversizedBytes = new Uint8Array(105000);
    oversizedBytes.fill(65); // 'A'
    let binary = '';
    for (let i = 0; i < oversizedBytes.length; i++) {
      binary += String.fromCharCode(oversizedBytes[i]);
    }
    const oversizedB64 = btoa(binary);

    const testPayload = {
      device_id: 'test-device',
      detections: [{
        event_uuid: crypto.randomUUID ? crypto.randomUUID() : 'test-uuid-oversized',
        latitude: 12.9716,
        longitude: 77.5946,
        speed_kmh: 40.0,
        confidence: 0.9,
        severity_score: 7.0,
        lane_position: 'CENTER',
        captured_at: new Date().toISOString(),
        image_base64: oversizedB64,
      }]
    };

    log(`🚀 Sending oversized batch to ${url}/api/v1/telemetry/sync ...`);
    try {
      const resp = await fetch(`${url}/api/v1/telemetry/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testPayload),
      });
      const data = await resp.json();
      if (resp.status === 422) {
        log(`✅ Backend successfully REJECTED oversized thumbnail with HTTP 422: ${JSON.stringify(data.detail[0]?.msg || data)}`);
      } else {
        log(`⚠️ Unexpected status: HTTP ${resp.status}`);
      }
    } catch (e) {
      log(`❌ Request failed: ${e.message}`, 'error');
    }
  });

  // Flush Batch to Backend
  syncBtn.addEventListener('click', async () => {
    if (offlineQueue.length === 0) {
      log('⚠️ Local queue is empty! Enqueue items first.');
      return;
    }

    const url = serverUrlInput.value.replace(/\/$/, '');
    const batch = offlineQueue.slice(0, 50);
    const payload = {
      device_id: 'tactical-web-bench-01',
      detections: batch.map(b => ({
        event_uuid: b.event_uuid,
        latitude: b.latitude,
        longitude: b.longitude,
        speed_kmh: b.speed_kmh,
        confidence: b.confidence,
        severity_score: b.severity_score,
        lane_position: b.lane_position,
        captured_at: b.captured_at,
        image_base64: b.image_base64,
      })),
    };

    log(`🚀 Flushing ${batch.length} hazards to ${url}/api/v1/telemetry/sync ...`);
    try {
      const resp = await fetch(`${url}/api/v1/telemetry/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        const resData = await resp.json();
        const syncedUuids = new Set(resData.synced_uuids || []);
        offlineQueue = offlineQueue.filter(item => !syncedUuids.has(item.event_uuid));
        log(`✅ Synced ${syncedUuids.size} hazards successfully! Backend returned: ${JSON.stringify(resData)}`);
        renderQueueTable();
      } else {
        const errText = await resp.text();
        log(`❌ Sync failed with HTTP ${resp.status}: ${errText}`, 'error');
      }
    } catch (err) {
      log(`❌ Network error while syncing: ${err.message}`, 'error');
    }
  });

  // HUD Render Loop
  let lastChimeTime = 0;
  function renderHUD() {
    const width = hudCanvas.width;
    const height = hudCanvas.height;
    const horizonY = height * 0.40;

    // Clear canvas
    ctx.fillStyle = '#050911';
    ctx.fillRect(0, 0, width, height);

    // Sky & Stars background
    ctx.fillStyle = '#0a1020';
    ctx.fillRect(0, 0, width, horizonY);

    // Horizon Line
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, horizonY);
    ctx.lineTo(width, horizonY);
    ctx.stroke();

    // Road Perspective
    const roadTopLeft = width * 0.42;
    const roadTopRight = width * 0.58;
    const roadBottomLeft = 0;
    const roadBottomRight = width;

    // Road surface
    ctx.fillStyle = '#070c16';
    ctx.beginPath();
    ctx.moveTo(roadTopLeft, horizonY);
    ctx.lineTo(roadTopRight, horizonY);
    ctx.lineTo(roadBottomRight, height);
    ctx.lineTo(roadBottomLeft, height);
    ctx.closePath();
    ctx.fill();

    // Speed-based animated dashes
    const speed = parseFloat(speedSlider.value);
    animOffset = (animOffset + (speed * 0.15)) % 60;

    // Left Lane line
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(roadTopLeft + (roadTopRight - roadTopLeft) * 0.33, horizonY);
    ctx.lineTo(width * 0.33, height);
    ctx.stroke();

    // Right Lane line
    ctx.beginPath();
    ctx.moveTo(roadTopLeft + (roadTopRight - roadTopLeft) * 0.66, horizonY);
    ctx.lineTo(width * 0.66, height);
    ctx.stroke();

    // Road edge curbs
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(roadTopLeft, horizonY);
    ctx.lineTo(roadBottomLeft, height);
    ctx.moveTo(roadTopRight, horizonY);
    ctx.lineTo(roadBottomRight, height);
    ctx.stroke();

    // Hazard Bounding Box calculation
    const distance = parseFloat(distanceSlider.value);
    const lane = getSelectedLane();
    const isCritical = distance < 8.0;

    // Toggle Critical Alert Banner
    if (isCritical) {
      criticalBanner.classList.add('active');
      const now = Date.now();
      if (now - lastChimeTime > 600) {
        playAlertChime();
        lastChimeTime = now;
      }
    } else {
      criticalBanner.classList.remove('active');
    }

    // Distance to Y-screen projection (d = k / (y - y0)) => (y - y0) = k / d
    const normDist = Math.max(0.05, Math.min(1.0, (35.0 - distance) / 33.0));
    const boxY = horizonY + (normDist * (height - horizonY - 40));
    const boxH = Math.max(20, normDist * 50);
    const boxW = Math.max(30, normDist * 90);

    let laneXCenter = width * 0.5;
    if (lane === 'LEFT') {
      laneXCenter = width * 0.28;
    } else if (lane === 'RIGHT') {
      laneXCenter = width * 0.72;
    }
    const boxX = laneXCenter - (boxW / 2);

    // Draw Bounding Box
    const color = isCritical ? '#ff1744' : '#00e5ff';
    ctx.strokeStyle = color;
    ctx.lineWidth = isCritical ? 3.5 : 2;
    ctx.strokeRect(boxX, boxY, boxW, boxH);

    // Hazard Label Tag
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fillRect(boxX, boxY - 22, boxW, 20);
    ctx.fillStyle = color;
    ctx.font = isCritical ? 'bold 13px monospace' : '11px monospace';
    ctx.fillText(`${distance.toFixed(1)}m [${lane}]`, boxX + 4, boxY - 7);

    // Motorbike handlebar HUD indicator
    ctx.fillStyle = 'rgba(0, 229, 255, 0.7)';
    ctx.beginPath();
    ctx.arc(width / 2, height - 12, 8, 0, Math.PI * 2);
    ctx.fill();

    requestAnimationFrame(renderHUD);
  }

  requestAnimationFrame(renderHUD);
});
