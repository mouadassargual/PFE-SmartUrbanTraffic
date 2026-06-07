"""
Dashboard Flask + Stream MJPEG pour Smart Traffic Agadir.
"""

from __future__ import annotations

import threading
import time
import mimetypes
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file

from pipeline.config import DASHBOARD_HOST, DASHBOARD_PORT, DEFAULT_SCENE, SCENE_PROFILES


class Dashboard:
    """Dashboard web thread-safe avec stream MJPEG et API JSON."""

    def __init__(self, host=DASHBOARD_HOST, port=DASHBOARD_PORT):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.frame_bytes = None
        self.analysis_frame_bytes = self._placeholder_frame()
        self.analysis_frame_version = 0
        self.analysis_frame_id = 0
        self.analysis_state = None
        self.analysis_snapshot_pending = True
        self.source_video_path = None
        self.seek_request = None
        self.scene_request = None
        self.scenes = self._public_scenes()
        self.current_scene_id = (
            DEFAULT_SCENE if DEFAULT_SCENE in self.scenes else next(iter(self.scenes), None)
        )
        self.state = {
            "zones": {},
            "polygons": {},
            "frame_size": {"width": 1280, "height": 720},
            "scores": {"NS": 0.0, "EW": 0.0},
            "decision": {},
            "detections_total": 0,
            "detections_by_class": {},
            "active_tracks": 0,
            "tracks_by_class": {},
            "pedestrians": 0,
            "fps": 0.0,
            "frame_id": 0,
            "emergency": False,
            "anonymized": 0,
            "anonymized_frame": 0,
            "anonymized_total": 0,
            "model_info": {},
            "video": {"fps": 0.0, "total_frames": 0, "current_second": 0.0, "source_ready": False},
            "scenes": deepcopy(self.scenes),
            "current_scene": self.current_scene_id,
            "analysis_frame_version": 0,
            "analysis_frame_id": 0,
            "analysis_status": "idle",
            "analysis": {},
            "timestamp": time.time(),
        }
        self._setup_routes()

    @staticmethod
    def _public_scenes():
        """Expose une version JSON-safe des scenes disponibles."""
        scenes = {}
        for scene_id, scene in SCENE_PROFILES.items():
            zones_json = scene.get("zones_json")
            scenes[scene_id] = {
                "id": scene_id,
                "label": scene.get("label", scene_id),
                "profile": scene.get("profile", scene_id),
                "video": Path(str(scene.get("video", ""))).name,
                "zones_json": Path(str(zones_json)).name if zones_json else None,
                "frame_index": scene.get("frame_index", 0),
            }
        return scenes

    def _setup_routes(self):
        @self.app.get("/")
        def index():
            return self._html()

        @self.app.get("/video_feed")
        def video_feed():
            return Response(
                self.generate_frames(kind="pipeline"),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.get("/source_video")
        def source_video():
            with self.lock:
                path = self.source_video_path
            if not path or not Path(path).exists():
                abort(404)
            mime_type = mimetypes.guess_type(path)[0] or "video/mp4"
            return send_file(path, mimetype=mime_type, conditional=True)

        @self.app.get("/analysis_frame.jpg")
        def analysis_frame():
            with self.lock:
                frame_bytes = self.analysis_frame_bytes
            if frame_bytes is None:
                return Response(status=204)
            return Response(
                frame_bytes,
                mimetype="image/jpeg",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @self.app.get("/Logo.png")
        def dashboard_logo():
            path = Path(__file__).resolve().parents[1] / "Logo.png"
            if not path.exists():
                abort(404)
            return send_file(path, mimetype="image/png", conditional=True)

        @self.app.get("/api/state")
        def api_state():
            with self.lock:
                data = deepcopy(self.state)
            return jsonify(data)

        @self.app.get("/api/scenes")
        def api_scenes():
            with self.lock:
                data = {
                    "current": self.current_scene_id,
                    "scenes": deepcopy(self.scenes),
                }
            return jsonify(data)

        @self.app.post("/api/profile")
        @self.app.post("/api/scene")
        def api_scene():
            payload = request.get_json(silent=True) or {}
            scene_id = payload.get("scene") or payload.get("profile")
            if scene_id not in self.scenes:
                abort(400)
            with self.lock:
                self.scene_request = {
                    "scene": scene_id,
                    "timestamp": time.time(),
                }
                self.current_scene_id = scene_id
                self.source_video_path = None
                self.analysis_snapshot_pending = True
                self.analysis_state = None
                self.analysis_frame_bytes = self._placeholder_frame()
                self.analysis_frame_version += 1
                self.analysis_frame_id = 0
                self.state["current_scene"] = scene_id
                self.state["scenes"] = deepcopy(self.scenes)
                self.state["analysis_status"] = "switching"
                self.state["analysis"] = {}
                self.state["analysis_frame_version"] = self.analysis_frame_version
                self.state["analysis_frame_id"] = 0
                video_state = deepcopy(self.state.get("video", {}))
                video_state["source_ready"] = False
                self.state["video"] = video_state
            return jsonify({"ok": True, "scene": scene_id})

        @self.app.post("/api/analyze")
        @self.app.post("/api/seek")
        def api_seek():
            payload = request.get_json(silent=True) or {}
            with self.lock:
                self.seek_request = {
                    "frame": payload.get("frame"),
                    "second": payload.get("second"),
                    "timestamp": time.time(),
                }
                self.analysis_snapshot_pending = True
                self.state["analysis_status"] = "pending"
            return jsonify({"ok": True, "request": self.seek_request})

        @self.app.get("/stats")
        def stats_alias():
            with self.lock:
                data = deepcopy(self.state)
            return jsonify(data)

        @self.app.get("/health")
        def health():
            return jsonify({"status": "ok", "timestamp": time.time()})

    @staticmethod
    def _placeholder_frame():
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (14, 20, 27)
        cv2.rectangle(frame, (0, 0), (1280, 720), (35, 50, 64), 8)
        cv2.putText(
            frame,
            "IMAGE ANALYSE",
            (430, 300),
            cv2.FONT_HERSHEY_DUPLEX,
            1.6,
            (245, 247, 251),
            2,
        )
        cv2.putText(
            frame,
            "Selectionnez un moment dans la video puis cliquez sur Analyser la situation",
            (275, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (158, 176, 193),
            2,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return encoded.tobytes() if ok else None

    @staticmethod
    def _legacy_html():
        return """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Smart Traffic Agadir</title>
  <style>
    :root {
      --bg:#0e141b; --panel:#17202a; --panel2:#111922; --line:#2a3948;
      --text:#f4f7fb; --muted:#9eb0c1; --green:#25d366; --red:#ff4d4d;
      --amber:#f3b63f; --cyan:#4bc3ff;
    }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Arial,sans-serif; background:var(--bg); color:var(--text); }
    header {
      display:flex; align-items:center; justify-content:space-between;
      padding:12px 18px; background:#15202b; border-bottom:1px solid var(--line);
    }
    header strong { font-size:18px; letter-spacing:.3px; }
    header span { color:var(--muted); font-size:13px; }
    main { display:grid; grid-template-columns:minmax(560px,1.25fr) minmax(480px,.95fr); gap:14px; padding:14px; }
    .video-panel, .side { min-width:0; }
    .video-stack { display:grid; gap:12px; }
    .video-panel { background:var(--panel); border:1px solid var(--line); padding:10px; }
    img, video { width:100%; display:block; background:#000; border:1px solid var(--line); }
    .caption { color:var(--muted); font-size:13px; margin-top:8px; display:flex; justify-content:space-between; gap:10px; }
    .side { display:grid; gap:12px; }
    .panel { background:var(--panel); border:1px solid var(--line); padding:12px; }
    h2, h3 { margin:0 0 10px 0; font-weight:700; }
    h2 { font-size:16px; } h3 { font-size:14px; color:#dbe7f2; }
    .cards { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
    .card { background:var(--panel2); border:1px solid var(--line); padding:10px; min-height:70px; }
    .card .label { color:var(--muted); font-size:12px; }
    .card .value { font-size:24px; margin-top:6px; font-weight:700; }
    .decision { display:grid; gap:12px; align-items:center; }
    .phase { font-size:34px; font-weight:800; margin:4px 0; color:var(--green); }
    .phase.emergency { color:var(--red); animation: blink .8s infinite; }
    .lights { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
    .lightBox { background:var(--panel2); border:1px solid var(--line); padding:10px; text-align:center; }
    .signal {
      width:54px; margin:0 auto 8px; padding:7px; border-radius:12px;
      background:linear-gradient(#252b31,#07090b); border:1px solid #3b4650;
      box-shadow:inset 0 0 10px rgba(255,255,255,.08), 0 8px 18px rgba(0,0,0,.35);
    }
    .lamp { width:34px; height:34px; border-radius:50%; margin:5px auto; background:#242a30; border:2px solid #111; }
    .lamp.red.on { background:#ff3434; box-shadow:0 0 16px rgba(255,52,52,.8); }
    .lamp.yellow.on { background:#ffc83d; box-shadow:0 0 16px rgba(255,200,61,.7); }
    .lamp.green.on { background:#20d76b; box-shadow:0 0 16px rgba(32,215,107,.8); }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    td, th { padding:7px; border-bottom:1px solid var(--line); text-align:left; }
    th { color:var(--muted); font-weight:600; }
    .split { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .class-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; }
    .pill { display:flex; justify-content:space-between; background:var(--panel2); border:1px solid var(--line); padding:7px 8px; font-size:13px; }
    .flow { display:grid; grid-template-columns:repeat(5,1fr); gap:6px; }
    .step { background:var(--panel2); border:1px solid var(--line); padding:8px; font-size:12px; text-align:center; color:#d9e5ef; }
    .seek { display:grid; grid-template-columns:1fr 1fr auto; gap:8px; align-items:end; margin-top:10px; }
    .seek label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
    .seek input {
      width:100%; background:#0e141b; color:var(--text); border:1px solid var(--line);
      padding:8px; font-size:14px;
    }
    .seek button {
      background:#2364aa; color:white; border:0; padding:9px 14px; cursor:pointer;
      font-weight:700;
    }
    .seek button:hover { background:#2d76c6; }
    .review-controls {
      margin-top:10px; background:linear-gradient(180deg,#121b25,#0d141c);
      border:1px solid var(--line); padding:12px; display:grid; gap:10px;
    }
    .review-meta { display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:13px; }
    .review-actions { display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center; }
    .review-actions button {
      background:#1d2b38; color:var(--text); border:1px solid var(--line);
      padding:8px 11px; cursor:pointer; font-weight:700;
    }
    .review-actions button.primary { background:#2364aa; border-color:#2d76c6; color:white; }
    .review-actions button:hover { background:#2d76c6; color:white; }
    .review-status { color:var(--muted); font-size:13px; }
    #mdpCanvas { width:100%; background:#0d1217; border:1px solid var(--line); display:block; }
    #alert { color:var(--red); font-weight:700; min-height:18px; }
    @media (max-width: 1100px) { main { grid-template-columns:1fr; } .cards { grid-template-columns:repeat(2,1fr); } .lights { grid-template-columns:repeat(2,1fr); } }
    @keyframes blink { 50% { opacity:.35; } }
  </style>
</head>
<body>
  <header>
    <strong>SMART TRAFFIC AGADIR</strong>
    <span>Dashboard gouvernemental temps reel - Raspberry Pi 5</span>
  </header>
  <main>
    <section class="video-stack">
      <div class="video-panel">
        <h2>Lecteur video source</h2>
        <video id="sourceVideo" controls preload="metadata"></video>
        <div class="review-controls">
          <div class="review-meta">
            <span id="sourceStatus">Chargement de la video source...</span>
            <span id="sourceClock">00:00 / 00:00</span>
          </div>
          <div class="review-actions">
            <span class="review-status" id="reviewStatus">Place la barre du lecteur sur le moment voulu.</span>
            <button class="primary" onclick="analyzeCurrentMoment()">Analyser la situation</button>
          </div>
        </div>
        <div class="seek">
          <label>Seconde
            <input id="seekSecond" type="number" min="0" step="0.1" placeholder="ex: 42.5">
          </label>
          <label>Frame
            <input id="seekFrame" type="number" min="0" step="1" placeholder="optionnel">
          </label>
          <button onclick="seekVideo()">OK</button>
        </div>
        <div class="caption">
          <span id="model">Modele: --</span>
          <span id="frameText">Frame: --</span>
        </div>
      </div>
      <div class="video-panel">
        <h2>Image analyse: polygones officiels N/E/S/W</h2>
        <img id="analysisImage" src="/analysis_frame.jpg" alt="analysis frame">
        <div class="caption">
          <span>Image fixe utilisee pour compter les objets par zone.</span>
          <span id="zonesSource">Zones: --</span>
        </div>
      </div>
    </section>
    <aside class="side">
      <section class="panel">
        <h2>KPIs operationnels</h2>
        <div class="cards">
          <div class="card"><div class="label">FPS video source</div><div id="sourceFps" class="value">0</div></div>
          <div class="card"><div class="label">Débit analyse</div><div id="fps" class="value">0</div></div>
          <div class="card"><div class="label">Detections</div><div id="detections" class="value">0</div></div>
          <div class="card"><div class="label">Tracks actifs</div><div id="tracks" class="value">0</div></div>
          <div class="card"><div class="label">Anonymises</div><div id="anon" class="value">0</div></div>
          <div class="card"><div class="label">Pietons</div><div id="pedestrians" class="value">0</div></div>
          <div class="card"><div class="label">Etat urgence</div><div id="emergency" class="value">OK</div></div>
          <div class="card"><div class="label">Frames sautees</div><div id="droppedFrames" class="value">0</div></div>
        </div>
      </section>

      <section class="panel">
        <h2>Decision MDP / feux</h2>
        <div class="decision">
          <div>
            <div>Phase courante</div>
            <div id="phase" class="phase">--</div>
            <div id="reason">--</div>
            <div id="alert"></div>
          </div>
          <div class="lights">
            <div class="lightBox">
              <div class="signal">
                <div id="nRed" class="lamp red"></div>
                <div id="nYellow" class="lamp yellow"></div>
                <div id="nGreen" class="lamp green"></div>
              </div>
              <strong>N</strong>
            </div>
            <div class="lightBox">
              <div class="signal">
                <div id="eRed" class="lamp red"></div>
                <div id="eYellow" class="lamp yellow"></div>
                <div id="eGreen" class="lamp green"></div>
              </div>
              <strong>E</strong>
            </div>
            <div class="lightBox">
              <div class="signal">
                <div id="sRed" class="lamp red"></div>
                <div id="sYellow" class="lamp yellow"></div>
                <div id="sGreen" class="lamp green"></div>
              </div>
              <strong>S</strong>
            </div>
            <div class="lightBox">
              <div class="signal">
                <div id="wRed" class="lamp red"></div>
                <div id="wYellow" class="lamp yellow"></div>
                <div id="wGreen" class="lamp green"></div>
              </div>
              <strong>W</strong>
            </div>
          </div>
        </div>
        <canvas id="mdpCanvas" width="420" height="236"></canvas>
      </section>

      <section class="panel split">
        <div>
          <h3>Zones de comptage</h3>
          <table>
            <thead><tr><th>Zone</th><th>Objets</th><th>Score</th></tr></thead>
            <tbody id="zones"></tbody>
          </table>
        </div>
        <div>
          <h3>Classes detectees</h3>
          <div id="classes" class="class-grid"></div>
        </div>
      </section>

      <section class="panel">
        <h3>Lecture frame par frame</h3>
        <div class="flow">
          <div class="step">Frame</div>
          <div class="step">YOLO</div>
          <div class="step">Tracking</div>
          <div class="step">Zones N/E/S/W</div>
          <div class="step">MDP feux</div>
        </div>
      </section>
    </aside>
  </main>
  <script>
    let lastAnalysisVersion = -1;
    let sourceVideoAttached = false;

    function refreshAnalysisImage() {
      const img = document.getElementById('analysisImage');
      img.src = `/analysis_frame.jpg?t=${Date.now()}`;
    }

    function formatTime(seconds) {
      const safe = Math.max(0, Number(seconds || 0));
      const minutes = Math.floor(safe / 60);
      const secs = Math.floor(safe % 60);
      return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    function updateTimelineLabel(current, total) {
      document.getElementById('sourceClock').textContent = `${formatTime(current)} / ${formatTime(total)}`;
    }

    async function postSeek(payload) {
      await fetch('/api/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      document.getElementById('reviewStatus').textContent = 'Analyse demandee, generation de l image...';
    }

    async function seekVideo() {
      const frameValue = document.getElementById('seekFrame').value;
      const secondValue = document.getElementById('seekSecond').value;
      const payload = {};
      if (frameValue !== '') payload.frame = Number(frameValue);
      else if (secondValue !== '') payload.second = Number(secondValue);
      else return;
      await postSeek(payload);
    }

    async function analyzeCurrentMoment() {
      const video = document.getElementById('sourceVideo');
      video.pause();
      const seconds = Number(video.currentTime || 0);
      document.getElementById('seekSecond').value = seconds.toFixed(1);
      document.getElementById('seekFrame').value = '';
      await postSeek({second: seconds});
    }

    function attachSourceVideo() {
      if (sourceVideoAttached) return;
      const video = document.getElementById('sourceVideo');
      video.src = `/source_video?t=${Date.now()}`;
      video.addEventListener('loadedmetadata', () => {
        updateTimelineLabel(video.currentTime, video.duration || 0);
        document.getElementById('sourceStatus').textContent = 'Video source prete';
      });
      video.addEventListener('timeupdate', () => {
        updateTimelineLabel(video.currentTime, video.duration || 0);
      });
      sourceVideoAttached = true;
    }

    function drawMdpMap(s) {
      const canvas = document.getElementById('mdpCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const frame = s.frame_size || {width:1280, height:720};
      const sx = canvas.width / frame.width;
      const sy = canvas.height / frame.height;
      const zoneColors = {
        N: ['rgba(255,180,40,.24)', '#ffb428'],
        E: ['rgba(60,220,255,.24)', '#3cdcfa'],
        S: ['rgba(80,255,120,.24)', '#50ff78'],
        W: ['rgba(255,80,220,.24)', '#ff50dc'],
      };
      const polygons = s.polygons || {};
      ['N','S','E','W'].forEach(z => {
        const poly = polygons[z] || [];
        if (!poly.length) return;
        ctx.beginPath();
        poly.forEach((p, i) => {
          const x = p[0] * sx;
          const y = p[1] * sy;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = zoneColors[z][0];
        ctx.strokeStyle = zoneColors[z][1];
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
        const v = (s.zones || {})[z] || {count:0, score:0};
        const cx = poly.reduce((a,p)=>a+p[0],0) / poly.length * sx;
        const cy = poly.reduce((a,p)=>a+p[1],0) / poly.length * sy;
        ctx.fillStyle = '#fff';
        ctx.font = '13px Arial';
        ctx.fillText(`${z} c:${v.count || 0} s:${Number(v.score || 0).toFixed(1)}`, cx - 34, cy);
      });
    }
    function setZoneLight(zone, go) {
      const key = zone.toLowerCase();
      document.getElementById(`${key}Red`).className = 'lamp red' + (go ? '' : ' on');
      document.getElementById(`${key}Yellow`).className = 'lamp yellow';
      document.getElementById(`${key}Green`).className = 'lamp green' + (go ? ' on' : '');
    }
    async function tick() {
      const r = await fetch('/api/state');
      const s = await r.json();
      const a = (s.analysis && Object.keys(s.analysis).length) ? s.analysis : s;
      const d = a.decision || {};
      const phase = document.getElementById('phase');
      phase.textContent = d.phase || '--';
      phase.className = 'phase' + ((a.emergency || d.phase === 'EMERGENCY') ? ' emergency' : '');
      document.getElementById('reason').textContent = d.reason || '';
      document.getElementById('fps').textContent = Number(s.fps || 0).toFixed(1);
      document.getElementById('detections').textContent = a.detections_total || 0;
      document.getElementById('tracks').textContent = a.active_tracks || 0;
      document.getElementById('anon').textContent = a.anonymized || 0;
      document.getElementById('pedestrians').textContent = a.pedestrians || 0;
      document.getElementById('emergency').textContent = (a.emergency || d.phase === 'EMERGENCY') ? 'ALERTE' : 'OK';
      document.getElementById('alert').textContent = (a.emergency || d.phase === 'EMERGENCY') ? 'ALERTE EMERGENCY' : '';
      const video = s.video || {};
      const currentSec = Number(video.current_second || 0);
      const total = Number(video.total_frames || 0);
      const sourceFps = Number(video.fps || 0);
      const totalSec = sourceFps > 0 && total > 0 ? total / sourceFps : 0;
      if (video.source_ready) {
        attachSourceVideo();
      } else {
        document.getElementById('sourceStatus').textContent = 'Video source indisponible';
        updateTimelineLabel(currentSec, totalSec);
      }
      const dropped = Number(video.dropped_frames || 0);
      const mode = video.playback_mode || 'processed';
      const status = s.analysis_status || 'idle';
      if (status === 'pending') {
        document.getElementById('reviewStatus').textContent = 'Analyse en file, attente du moteur IA...';
      }
      document.getElementById('sourceFps').textContent = sourceFps ? sourceFps.toFixed(1) : '0';
      document.getElementById('droppedFrames').textContent = dropped;
      document.getElementById('frameText').textContent = `Frame: ${s.frame_id || 0}/${total || '--'} | t=${currentSec.toFixed(1)}s | ${mode}`;
      const model = s.model_info || {};
      document.getElementById('model').textContent = `Modele: ${model.name || '--'} | imgsz=${model.imgsz || '--'} | profile=${model.profile || '--'}`;
      document.getElementById('zonesSource').textContent = `Zones: ${model.zones_json || model.profile || '--'} | image F${a.frame_id || 0}`;
      const green = new Set(d.green_dirs || []);
      ['N','E','S','W'].forEach(z => setZoneLight(z, green.has(z) || d.phase === 'EMERGENCY'));
      if (Number(s.analysis_frame_version || 0) !== lastAnalysisVersion) {
        lastAnalysisVersion = Number(s.analysis_frame_version || 0);
        refreshAnalysisImage();
        document.getElementById('reviewStatus').textContent = `Analyse disponible: image F${a.frame_id || 0}, detections, anonymisation, zones et feux.`;
      }
      const zones = a.zones || {};
      document.getElementById('zones').innerHTML = ['N','S','E','W'].map(z => {
        const v = zones[z] || {count:0, score:0};
        return `<tr><td>${z}</td><td>${v.count}</td><td>${Number(v.score).toFixed(1)}</td></tr>`;
      }).join('');
      const classCounts = a.tracks_by_class || a.detections_by_class || {};
      const order = ['car','motorcycle','truck','bus','person','emergency_vehicle'];
      document.getElementById('classes').innerHTML = order.map(c => {
        return `<div class="pill"><span>${c}</span><strong>${classCounts[c] || 0}</strong></div>`;
      }).join('');
      drawMdpMap(a);
    }
    setInterval(tick, 500); tick();
  </script>
</body>
</html>
"""

    @staticmethod
    def _html():
        template = Path(__file__).resolve().parents[1] / "Urbanflow Final Report Evaluation.html"
        if template.exists():
            return template.read_text(encoding="utf-8")

        return """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Traffic Agadir</title>
  <style>
    :root {
      --bg:#f3f5f7; --surface:#ffffff; --surface2:#f6f8fa; --ink:#1d252d;
      --muted:#637080; --line:#d5dde5; --brand:#184e63; --brand2:#0f2f3d;
      --green:#12824a; --red:#bb3434; --amber:#b98000; --blue:#315f8f;
      --teal:#1d7f85; --violet:#7857a8; --shadow:0 1px 2px rgba(16, 24, 32, .08);
    }
    * { box-sizing:border-box; }
    body {
      margin:0; background:var(--bg); color:var(--ink);
      font-family:Inter, Arial, sans-serif; letter-spacing:0;
    }
    .topbar {
      min-height:62px; display:flex; align-items:center; justify-content:space-between;
      padding:12px 18px; background:#13232d;
      border-bottom:1px solid #263a46; box-shadow:0 2px 0 rgba(0,0,0,.08);
      position:sticky; top:0; z-index:10;
    }
    .brand { display:flex; flex-direction:column; gap:3px; }
    .brand strong { font-size:21px; color:#fff; }
    .brand span { font-size:12px; color:#d8e5ef; }
    .statusbar { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
    .chip {
      display:inline-flex; align-items:center; gap:7px; border:1px solid var(--line);
      background:var(--surface2); color:var(--ink); border-radius:6px; padding:7px 9px;
      font-size:12px; white-space:nowrap;
    }
    .topbar .chip {
      background:#1a313e; border-color:#31505e;
      color:#eef6fb;
    }
    .topbar .chip strong { color:#fff; }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--muted); display:inline-block; }
    .dot.ok { background:var(--green); } .dot.warn { background:var(--amber); } .dot.danger { background:var(--red); }
    main { padding:16px; display:grid; gap:14px; }
    .workspace {
      display:grid; grid-template-columns:minmax(520px, 1.05fr) minmax(520px, 1fr);
      gap:14px; align-items:start;
    }
    .panel {
      background:var(--surface); border:1px solid var(--line); border-radius:6px;
      box-shadow:var(--shadow); min-width:0; overflow:hidden;
    }
    .panel-head {
      padding:11px 13px; border-bottom:1px solid var(--line);
      background:#fbfcfd;
      display:flex; justify-content:space-between; align-items:center; gap:10px;
    }
    .panel-head h2 {
      margin:0; font-size:14px; color:var(--brand2);
      display:flex; align-items:center; gap:8px;
    }
    .panel-head h2::before {
      content:""; width:9px; height:9px; border-radius:2px;
      background:var(--teal); display:inline-block;
    }
    .panel-head span { color:var(--muted); font-size:12px; }
    .panel-body { padding:12px; }
    video, img {
      width:100%; display:block; background:#05070a; border:1px solid var(--line);
      border-radius:6px;
    }
    .analysis-image-shell {
      background:#101820; border:1px solid #1d303d; border-radius:6px;
      padding:8px;
    }
    .image-toolbar {
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      margin-bottom:8px; color:#cbd8e5; font-size:12px;
    }
    .image-toolbar span {
      border:1px solid rgba(255,255,255,.14); border-radius:999px;
      padding:4px 8px; color:#d8e5ef; background:rgba(255,255,255,.06);
    }
    .image-toolbar strong { color:#fff; font-size:12px; }
    .analysis-image { min-height:260px; object-fit:contain; border:0; border-radius:5px; }
    .analysis-proof {
      display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:10px;
    }
    .proof-step {
      border:1px solid var(--line); border-radius:6px; background:#fff;
      border-left:4px solid var(--blue); padding:9px; min-width:0;
    }
    .proof-step:nth-child(2) { border-left-color:#475569; }
    .proof-step:nth-child(3) { border-left-color:var(--green); }
    .proof-step span {
      display:block; color:var(--muted); font-size:11px; text-transform:uppercase;
      font-weight:700;
    }
    .proof-step strong {
      display:block; margin-top:5px; color:var(--brand2); font-size:18px;
      line-height:1.1;
    }
    .proof-step small { display:block; margin-top:3px; color:var(--muted); font-size:11px; }
    .analysis-zones {
      display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:10px 0;
    }
    .analysis-zone-card {
      border:1px solid var(--line); border-radius:6px; background:#fff;
      padding:9px; min-width:0;
    }
    .analysis-zone-card.active {
      border-color:rgba(21,153,71,.45); background:rgba(21,153,71,.06);
    }
    .analysis-zone-card.blocked {
      border-color:rgba(207,47,47,.28); background:rgba(207,47,47,.04);
    }
    .analysis-zone-head {
      display:flex; justify-content:space-between; align-items:center; gap:6px;
    }
    .analysis-zone-head strong {
      color:var(--brand2); font-size:13px; overflow:hidden; text-overflow:ellipsis;
      white-space:nowrap;
    }
    .signal-chip {
      border:1px solid var(--line); border-radius:999px; background:var(--surface2);
      color:var(--muted); padding:3px 7px; font-size:11px; font-weight:700;
      white-space:nowrap;
    }
    .signal-chip.green {
      background:rgba(21,153,71,.10); border-color:rgba(21,153,71,.40); color:var(--green);
    }
    .signal-chip.red {
      background:rgba(207,47,47,.08); border-color:rgba(207,47,47,.32); color:var(--red);
    }
    .analysis-zone-score {
      display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:8px;
    }
    .analysis-zone-score span { display:block; color:var(--muted); font-size:11px; }
    .analysis-zone-score strong { display:block; margin-top:3px; color:var(--ink); font-size:17px; }
    .analysis-zone-classes {
      display:grid; grid-template-columns:repeat(2,1fr); gap:5px; margin-top:8px;
    }
    .analysis-zone-classes .zone-class { min-width:0; width:100%; }
    .review-strip {
      margin-top:10px; display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center;
      padding:10px; background:var(--surface2); border:1px solid var(--line); border-radius:6px;
    }
    .review-meta { display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font-size:13px; }
    .btn {
      border:1px solid var(--brand); background:var(--brand); color:#fff; border-radius:5px;
      padding:9px 13px; font-weight:700; cursor:pointer; font-size:13px;
    }
    .btn:hover { background:#103f52; }
    .btn:active { transform:translateY(1px); }
    details.advanced { margin-top:8px; border:1px solid var(--line); border-radius:6px; background:#fff; }
    details.advanced summary { cursor:pointer; padding:9px 10px; color:var(--muted); font-size:13px; }
    .manual-seek { display:grid; grid-template-columns:1fr 1fr auto; gap:8px; padding:0 10px 10px; align-items:end; }
    label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
    input {
      width:100%; border:1px solid var(--line); border-radius:6px; padding:8px;
      background:#fff; color:var(--ink); font-size:13px;
    }
    .kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
    .kpi {
      background:var(--surface); border:1px solid var(--line); border-radius:6px;
      border-top:4px solid var(--blue); padding:10px 11px; box-shadow:var(--shadow); min-height:82px;
    }
    .kpi:nth-child(4), .kpi:nth-child(6) { border-top-color:var(--green); }
    .kpi:nth-child(7) { border-top-color:var(--red); }
    .kpi:nth-child(8) { border-top-color:var(--amber); }
    .kpi span { color:var(--muted); font-size:12px; }
    .kpi strong { display:block; margin-top:8px; font-size:24px; color:var(--brand2); }
    .kpi small { display:block; margin-top:3px; color:var(--muted); font-size:11px; }
    .ops {
      display:grid; grid-template-columns:minmax(440px, .9fr) minmax(440px, 1.1fr);
      gap:14px; align-items:start;
    }
    .phase-card {
      display:grid; grid-template-columns:1fr auto; gap:12px; align-items:center;
      padding:12px; background:var(--surface2); border:1px solid var(--line); border-radius:8px;
    }
    .phase { font-size:36px; font-weight:800; color:var(--green); line-height:1; }
    .phase.emergency { color:var(--red); }
    .reason { color:var(--muted); font-size:13px; margin-top:7px; }
    .score-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
    .score-box { border:1px solid var(--line); border-radius:6px; padding:9px; background:#fff; }
    .score-box span { color:var(--muted); font-size:12px; }
    .score-box strong { display:block; font-size:22px; margin-top:5px; }
    .lights { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:12px; }
    .lightBox { border:1px solid var(--line); background:var(--surface2); border-radius:8px; padding:10px; text-align:center; }
    .signal {
      width:50px; margin:0 auto 8px; padding:7px; border-radius:13px;
      background:linear-gradient(#232b32,#080b0f); border:1px solid #35414b;
      box-shadow:inset 0 0 8px rgba(255,255,255,.10), 0 8px 18px rgba(0,0,0,.20);
    }
    .lamp { width:32px; height:32px; border-radius:50%; margin:5px auto; background:#29313a; border:2px solid #101418; }
    .lamp.red.on { background:#ff3838; box-shadow:0 0 16px rgba(255,56,56,.72); }
    .lamp.yellow.on { background:#ffc83d; box-shadow:0 0 16px rgba(255,200,61,.72); }
    .lamp.green.on { background:#20d76b; box-shadow:0 0 16px rgba(32,215,107,.72); }
    .lightBox strong { font-size:14px; color:var(--brand2); }
    table { width:100%; border-collapse:collapse; font-size:13px; margin-top:10px; }
    th, td { padding:9px 8px; border-bottom:1px solid var(--line); text-align:left; }
    th { color:var(--muted); font-weight:700; background:var(--surface2); }
    tbody tr:hover { background:#f8fafc; }
    .class-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:10px; }
    .pill {
      display:flex; justify-content:space-between; gap:8px; border:1px solid var(--line);
      background:var(--surface2); border-radius:6px; padding:8px 9px; font-size:13px;
    }
    .zone-breakdown { display:flex; flex-wrap:wrap; gap:5px; min-width:190px; }
    .zone-class {
      display:inline-flex; align-items:center; justify-content:space-between; gap:5px;
      min-width:54px; padding:4px 6px; border:1px solid var(--line);
      border-radius:6px; background:var(--surface2); color:var(--muted); font-size:12px;
    }
    .zone-class strong { color:var(--brand2); }
    .compat {
      margin-top:10px; border:1px solid var(--line); border-radius:6px;
      padding:9px 10px; background:var(--surface2); color:var(--muted); font-size:13px;
    }
    .compat.ok { border-color:rgba(21,153,71,.35); background:rgba(21,153,71,.08); color:var(--green); }
    .compat.warn { border-color:rgba(230,169,0,.45); background:rgba(230,169,0,.10); color:#8a6500; }
    #mdpCanvas { width:100%; background:#0b1117; border:1px solid var(--line); border-radius:6px; display:block; }
    .log { display:grid; gap:7px; max-height:174px; overflow:auto; }
    .log-row { border:1px solid var(--line); background:var(--surface2); border-radius:6px; padding:8px 9px; font-size:13px; color:var(--muted); }
    .log-row strong { color:var(--brand2); }
    @media (max-width:1180px) {
      .workspace, .ops { grid-template-columns:1fr; }
      .kpis { grid-template-columns:repeat(2,1fr); }
    }
    @media (max-width:680px) {
      .topbar { align-items:flex-start; flex-direction:column; }
      .statusbar { justify-content:flex-start; }
      .analysis-proof, .analysis-zones, .kpis, .lights, .score-grid, .manual-seek { grid-template-columns:1fr; }
      .review-strip, .image-toolbar { grid-template-columns:1fr; display:grid; }
      main { padding:10px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <strong>Smart Traffic Agadir</strong>
      <span>Supervision intersection</span>
    </div>
    <div class="statusbar">
      <span class="chip"><span id="piDot" class="dot warn"></span><span id="systemStatus">Initialisation</span></span>
      <span class="chip">Mode: <strong id="modeChip">analyse</strong></span>
      <span class="chip">Profil: <strong id="profileChip">--</strong></span>
      <span class="chip">Derniere analyse: <strong id="lastAnalysisChip">--</strong></span>
    </div>
  </header>

  <main>
    <section class="workspace">
      <article class="panel">
        <div class="panel-head">
          <h2>Video source</h2>
          <span id="sourceStatus">Video source</span>
        </div>
        <div class="panel-body">
          <video id="sourceVideo" controls preload="metadata"></video>
          <div class="review-strip">
            <div class="review-meta">
              <span id="sourceClock">00:00 / 00:00</span>
              <span id="reviewStatus">Pret pour analyse</span>
            </div>
            <button class="btn" onclick="analyzeCurrentMoment()">Analyser la situation</button>
          </div>
          <details class="advanced">
            <summary>Recherche precise</summary>
            <div class="manual-seek">
              <label>Seconde
                <input id="seekSecond" type="number" min="0" step="0.1" placeholder="ex: 42.5">
              </label>
              <label>Frame
                <input id="seekFrame" type="number" min="0" step="1" placeholder="optionnel">
              </label>
              <button class="btn" onclick="seekVideo()">Analyser</button>
            </div>
          </details>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <h2>Pipeline traite</h2>
          <span id="zonesSource">Detections + anonymisation</span>
        </div>
        <div class="panel-body">
          <div class="analysis-proof">
            <div class="proof-step"><span>YOLO</span><strong id="proofDetections">0</strong><small>objets detectes</small></div>
            <div class="proof-step"><span>Tracking</span><strong id="proofTracks">0</strong><small>tracks actifs</small></div>
            <div class="proof-step"><span>Anonymisation</span><strong id="proofAnon">0</strong><small id="proofAnonTotal">total protege: 0</small></div>
          </div>
          <div id="analysisZones" class="analysis-zones"></div>
          <div class="analysis-image-shell">
            <div class="image-toolbar">
              <span>Vue IA</span>
              <strong>Detections + anonymisation</strong>
            </div>
            <img id="analysisImage" class="analysis-image" src="/analysis_frame.jpg" alt="analysis frame">
          </div>
        </div>
      </article>
    </section>

    <section class="kpis">
      <div class="kpi"><span>FPS video source</span><strong id="sourceFps">0</strong></div>
      <div class="kpi"><span>Latence analyse</span><strong id="latency">--</strong></div>
      <div class="kpi"><span>Objets detectes</span><strong id="detections">0</strong></div>
      <div class="kpi"><span>Personnes anonymisees / frame</span><strong id="anon">0</strong><small id="anonTotal">total protege: 0</small></div>
      <div class="kpi"><span>Tracks actifs</span><strong id="tracks">0</strong></div>
      <div class="kpi"><span>Pietons</span><strong id="pedestrians">0</strong></div>
      <div class="kpi"><span>Urgence</span><strong id="emergency">OK</strong></div>
      <div class="kpi"><span>Frame analysee</span><strong id="frameKpi">--</strong></div>
    </section>

    <section class="ops">
      <article class="panel">
        <div class="panel-head">
          <h2>Decision MDP et feux</h2>
          <span id="model">Modele: --</span>
        </div>
        <div class="panel-body">
          <div class="phase-card">
            <div>
              <div id="phase" class="phase">--</div>
              <div id="reason" class="reason">En attente d'analyse</div>
            </div>
            <div id="duration" class="chip">-- s</div>
          </div>
          <div class="score-grid">
            <div class="score-box"><span>Score axe N/S</span><strong id="scoreNS">0.0</strong></div>
            <div class="score-box"><span>Score axe E/W</span><strong id="scoreEW">0.0</strong></div>
          </div>
          <div id="compatibility" class="compat warn">Compatibilite feux/statistiques: --</div>
          <div class="lights">
            <div class="lightBox"><div class="signal"><div id="nRed" class="lamp red"></div><div id="nYellow" class="lamp yellow"></div><div id="nGreen" class="lamp green"></div></div><strong>N</strong></div>
            <div class="lightBox"><div class="signal"><div id="eRed" class="lamp red"></div><div id="eYellow" class="lamp yellow"></div><div id="eGreen" class="lamp green"></div></div><strong>E</strong></div>
            <div class="lightBox"><div class="signal"><div id="sRed" class="lamp red"></div><div id="sYellow" class="lamp yellow"></div><div id="sGreen" class="lamp green"></div></div><strong>S</strong></div>
            <div class="lightBox"><div class="signal"><div id="wRed" class="lamp red"></div><div id="wYellow" class="lamp yellow"></div><div id="wGreen" class="lamp green"></div></div><strong>W</strong></div>
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <h2>Zones et repartition</h2>
          <span id="frameText">Frame: --</span>
        </div>
        <div class="panel-body">
          <canvas id="mdpCanvas" width="620" height="348"></canvas>
          <table>
            <thead><tr><th>Zone</th><th>Vehicules</th><th>Pietons</th><th>Details</th><th>Score</th><th>Feu</th></tr></thead>
            <tbody id="zones"></tbody>
          </table>
          <div id="classes" class="class-grid"></div>
        </div>
      </article>
    </section>

    <article class="panel">
      <div class="panel-head">
        <h2>Journal d'analyse</h2>
        <span id="analysisState">idle</span>
      </div>
      <div class="panel-body">
        <div id="analysisLog" class="log">
          <div class="log-row">Aucune analyse pour le moment.</div>
        </div>
      </div>
    </article>
  </main>

  <script>
    let lastAnalysisVersion = -1;
    let sourceVideoAttached = false;
    const analysisLog = [];
    const transportOrder = ['car','motorcycle','truck','bus','emergency_vehicle'];
    const transportLabels = {
      car: 'Voit.',
      motorcycle: 'Moto',
      truck: 'Cam.',
      bus: 'Bus',
      emergency_vehicle: 'Urg.',
    };
    const zoneLabels = {
      N: 'Zone A / Nord',
      E: 'Zone B / Est',
      S: 'Zone C / Sud',
      W: 'Zone D / Ouest',
    };

    function formatTime(seconds) {
      const safe = Math.max(0, Number(seconds || 0));
      const minutes = Math.floor(safe / 60);
      const secs = Math.floor(safe % 60);
      return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    function updateSourceClock(current, total) {
      document.getElementById('sourceClock').textContent = `${formatTime(current)} / ${formatTime(total)}`;
    }

    function refreshAnalysisImage() {
      document.getElementById('analysisImage').src = `/analysis_frame.jpg?t=${Date.now()}`;
    }

    async function postAnalyze(payload) {
      await fetch('/api/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      document.getElementById('reviewStatus').textContent = 'Analyse en cours...';
    }

    async function analyzeCurrentMoment() {
      const video = document.getElementById('sourceVideo');
      video.pause();
      const seconds = Number(video.currentTime || 0);
      document.getElementById('seekSecond').value = seconds.toFixed(1);
      document.getElementById('seekFrame').value = '';
      await postAnalyze({second: seconds});
    }

    async function seekVideo() {
      const frameValue = document.getElementById('seekFrame').value;
      const secondValue = document.getElementById('seekSecond').value;
      const payload = {};
      if (frameValue !== '') payload.frame = Number(frameValue);
      else if (secondValue !== '') payload.second = Number(secondValue);
      else return;
      await postAnalyze(payload);
    }

    function attachSourceVideo() {
      if (sourceVideoAttached) return;
      const video = document.getElementById('sourceVideo');
      video.src = `/source_video?t=${Date.now()}`;
      video.addEventListener('loadedmetadata', () => {
        updateSourceClock(video.currentTime, video.duration || 0);
        document.getElementById('sourceStatus').textContent = 'Source prete';
      });
      video.addEventListener('timeupdate', () => updateSourceClock(video.currentTime, video.duration || 0));
      sourceVideoAttached = true;
    }

    function setZoneLight(zone, go) {
      const key = zone.toLowerCase();
      document.getElementById(`${key}Red`).className = 'lamp red' + (go ? '' : ' on');
      document.getElementById(`${key}Yellow`).className = 'lamp yellow';
      document.getElementById(`${key}Green`).className = 'lamp green' + (go ? ' on' : '');
    }

    function drawMdpMap(s) {
      const canvas = document.getElementById('mdpCanvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const frame = s.frame_size || {width:1280, height:720};
      const sx = canvas.width / frame.width;
      const sy = canvas.height / frame.height;
      const zoneColors = {
        N: ['rgba(255,180,40,.22)', '#ffb428'],
        E: ['rgba(60,220,255,.22)', '#3cdcfa'],
        S: ['rgba(80,255,120,.22)', '#50c878'],
        W: ['rgba(255,80,220,.22)', '#d946ef'],
      };
      const polygons = s.polygons || {};
      ['N','S','E','W'].forEach(z => {
        const poly = polygons[z] || [];
        if (!poly.length) return;
        ctx.beginPath();
        poly.forEach((p, i) => {
          const x = p[0] * sx;
          const y = p[1] * sy;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = zoneColors[z][0];
        ctx.strokeStyle = zoneColors[z][1];
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
        const v = (s.zones || {})[z] || {count:0, score:0};
        const cx = poly.reduce((a,p)=>a+p[0],0) / poly.length * sx;
        const cy = poly.reduce((a,p)=>a+p[1],0) / poly.length * sy;
        ctx.fillStyle = '#f8fafc';
        ctx.font = '13px Arial';
        ctx.fillText(`${z}  ${v.count || 0} / ${Number(v.score || 0).toFixed(1)}`, cx - 34, cy);
        const byClass = v.by_class || {};
        const summary = transportOrder
          .map(className => byClass[className] ? `${transportLabels[className]}${byClass[className]}` : '')
          .filter(Boolean)
          .join(' ');
        const personSummary = v.pedestrians ? ` Piet.${v.pedestrians}` : '';
        const detailText = `${summary}${personSummary}`.trim();
        if (detailText) {
          ctx.fillText(detailText, cx - 42, cy + 16);
        }
      });
    }

    function updateLog(entry) {
      analysisLog.unshift(entry);
      while (analysisLog.length > 6) analysisLog.pop();
      document.getElementById('analysisLog').innerHTML = analysisLog.map(row => (
        `<div class="log-row"><strong>${row.time}</strong> - ${row.text}</div>`
      )).join('');
    }

    function formatZoneBreakdown(zoneState) {
      const byClass = zoneState.by_class || {};
      return transportOrder.map(className => (
        `<span class="zone-class"><span>${transportLabels[className]}</span><strong>${byClass[className] || 0}</strong></span>`
      )).join('');
    }

    function renderAnalysisZones(zones, green, decision) {
      const target = document.getElementById('analysisZones');
      if (!target) return;
      target.innerHTML = ['N','E','S','W'].map(z => {
        const v = zones[z] || {count:0, score:0, pedestrians:0, by_class:{}};
        const go = green.has(z) || decision.phase === 'EMERGENCY';
        const signal = go ? 'Vert' : 'Rouge';
        const axis = (z === 'N' || z === 'S') ? 'NS' : 'EW';
        return `
          <div class="analysis-zone-card ${go ? 'active' : 'blocked'}">
            <div class="analysis-zone-head">
              <strong>${zoneLabels[z] || z}</strong>
              <span class="signal-chip ${go ? 'green' : 'red'}">${signal}</span>
            </div>
            <div class="analysis-zone-score">
              <div><span>Vehicules</span><strong>${v.count || 0}</strong></div>
              <div><span>Pietons</span><strong>${v.pedestrians || 0}</strong></div>
              <div><span>Score</span><strong>${Number(v.score || 0).toFixed(1)}</strong></div>
              <div><span>Axe</span><strong>${axis}</strong></div>
            </div>
            <div class="analysis-zone-classes">${formatZoneBreakdown(v)}</div>
          </div>
        `;
      }).join('');
    }

    function updateCompatibility(decision, analysis) {
      const scoreNS = Number(decision.score_NS || 0);
      const scoreEW = Number(decision.score_EW || 0);
      const expected = scoreNS >= scoreEW ? 'NS' : 'EW';
      const phase = decision.phase || '--';
      const emergency = analysis.emergency || phase === 'EMERGENCY';
      const compatible = emergency || phase === expected || phase === '--';
      const el = document.getElementById('compatibility');
      el.className = 'compat ' + (compatible ? 'ok' : 'warn');
      if (emergency) {
        el.textContent = 'Compatibilite feux/statistiques: urgence prioritaire, tous les axes au vert';
      } else if (phase === '--') {
        el.textContent = 'Compatibilite feux/statistiques: en attente';
      } else if (compatible) {
        el.textContent = `Compatibilite feux/statistiques: OK, phase ${phase} sur axe dominant`;
      } else {
        el.textContent = `Compatibilite feux/statistiques: a verifier, phase ${phase} mais axe ${expected} dominant`;
      }
    }

    function privacyFrameCount(analysis) {
      return Number(analysis.anonymized_frame ?? analysis.anonymized ?? 0);
    }

    function privacyTotalCount(analysis) {
      const frameCount = privacyFrameCount(analysis);
      return Number(analysis.anonymized_total ?? frameCount);
    }

    async function tick() {
      const r = await fetch('/api/state');
      const s = await r.json();
      const a = (s.analysis && Object.keys(s.analysis).length) ? s.analysis : s;
      const d = a.decision || {};
      const video = s.video || {};
      const model = s.model_info || {};
      const sourceFps = Number(video.fps || 0);
      const totalFrames = Number(video.total_frames || 0);
      const currentSec = Number(video.current_second || 0);
      const totalSec = sourceFps > 0 && totalFrames > 0 ? totalFrames / sourceFps : 0;

      document.getElementById('systemStatus').textContent = video.source_ready ? 'Operationnel' : 'Attente source';
      document.getElementById('piDot').className = 'dot ' + (video.source_ready ? 'ok' : 'warn');
      document.getElementById('modeChip').textContent = video.playback_mode || 'review';
      document.getElementById('profileChip').textContent = model.profile || '--';
      document.getElementById('analysisState').textContent = s.analysis_status || 'idle';
      document.getElementById('sourceFps').textContent = sourceFps ? sourceFps.toFixed(1) : '0';
      document.getElementById('model').textContent = `Modele: ${model.name || '--'} | imgsz=${model.imgsz || '--'}`;
      document.getElementById('frameText').textContent = `Frame: ${a.frame_id || 0}/${totalFrames || '--'} | t=${currentSec.toFixed(1)}s`;

      if (video.source_ready) {
        attachSourceVideo();
      } else {
        updateSourceClock(currentSec, totalSec);
      }

      const latencyMs = Number(s.fps || 0) > 0 ? 1000 / Number(s.fps) : 0;
      document.getElementById('latency').textContent = latencyMs ? `${latencyMs.toFixed(0)} ms` : '--';
      document.getElementById('detections').textContent = a.detections_total || 0;
      document.getElementById('tracks').textContent = a.active_tracks || 0;
      const anonymizedFrame = privacyFrameCount(a);
      const anonymizedTotal = privacyTotalCount(a);
      document.getElementById('anon').textContent = anonymizedFrame;
      document.getElementById('anonTotal').textContent = `total protege: ${anonymizedTotal}`;
      document.getElementById('pedestrians').textContent = a.pedestrians || 0;
      document.getElementById('emergency').textContent = (a.emergency || d.phase === 'EMERGENCY') ? 'ALERTE' : 'OK';
      document.getElementById('frameKpi').textContent = a.frame_id || '--';

      const phase = document.getElementById('phase');
      phase.textContent = d.phase || '--';
      phase.className = 'phase' + ((a.emergency || d.phase === 'EMERGENCY') ? ' emergency' : '');
      document.getElementById('reason').textContent = d.reason || 'En attente d analyse';
      document.getElementById('duration').textContent = `${d.duration || '--'} s`;
      document.getElementById('scoreNS').textContent = Number(d.score_NS || 0).toFixed(1);
      document.getElementById('scoreEW').textContent = Number(d.score_EW || 0).toFixed(1);
      updateCompatibility(d, a);
      document.getElementById('proofDetections').textContent = a.detections_total || 0;
      document.getElementById('proofTracks').textContent = a.active_tracks || 0;
      document.getElementById('proofAnon').textContent = anonymizedFrame;
      document.getElementById('proofAnonTotal').textContent = `total protege: ${anonymizedTotal}`;

      const zones = a.zones || {};
      const green = new Set(d.green_dirs || []);
      ['N','E','S','W'].forEach(z => setZoneLight(z, green.has(z) || d.phase === 'EMERGENCY'));
      renderAnalysisZones(zones, green, d);

      if (s.analysis_status === 'pending') {
        document.getElementById('reviewStatus').textContent = 'Analyse en cours...';
      }
      if (Number(s.analysis_frame_version || 0) !== lastAnalysisVersion) {
        lastAnalysisVersion = Number(s.analysis_frame_version || 0);
        refreshAnalysisImage();
        const stamp = new Date().toLocaleTimeString();
        document.getElementById('reviewStatus').textContent = `Analyse disponible - frame ${a.frame_id || 0}`;
        document.getElementById('lastAnalysisChip').textContent = stamp;
        updateLog({
          time: stamp,
          text: `Frame ${a.frame_id || 0}, phase ${d.phase || '--'}, ${a.detections_total || 0} objets, ${anonymizedFrame} anonymises`
        });
      }

      document.getElementById('zones').innerHTML = ['N','E','S','W'].map(z => {
        const v = zones[z] || {count:0, score:0};
        const feu = green.has(z) || d.phase === 'EMERGENCY' ? 'Vert' : 'Rouge';
        return `<tr><td>${z}</td><td>${v.count || 0}</td><td>${v.pedestrians || 0}</td><td><div class="zone-breakdown">${formatZoneBreakdown(v)}</div></td><td>${Number(v.score || 0).toFixed(1)}</td><td>${feu}</td></tr>`;
      }).join('');

      const classCounts = a.tracks_by_class || a.detections_by_class || {};
      const order = ['car','motorcycle','truck','bus','person','emergency_vehicle'];
      document.getElementById('classes').innerHTML = order.map(c => (
        `<div class="pill"><span>${c}</span><strong>${classCounts[c] || 0}</strong></div>`
      )).join('');

      document.getElementById('zonesSource').textContent = `Detections + anonymisation | image F${a.frame_id || 0}`;
      drawMdpMap(a);
    }
    setInterval(tick, 500); tick();
  </script>
</body>
</html>
"""

    def update(self, frame, state, decision, fps, analysis_frame=None, video_meta=None):
        """Met a jour frame + etat."""
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        analysis_bytes = None
        if analysis_frame is not None:
            ok_analysis, encoded_analysis = cv2.imencode(
                ".jpg",
                analysis_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if ok_analysis:
                analysis_bytes = encoded_analysis.tobytes()

        zones = {key: state.get(key, {}) for key in ("N", "S", "E", "W")}
        polygons = state.get("polygons", {})
        frame_size = state.get("frame_size", {"width": frame.shape[1], "height": frame.shape[0]})
        video_data = video_meta or self.state.get("video", {})
        video_data = deepcopy(video_data)
        video_data["source_ready"] = bool(self.source_video_path)

        anonymized_frame = int(state.get("anonymized", 0))
        anonymized_total = int(state.get("anonymized_total", anonymized_frame))

        next_state = {
            "zones": zones,
            "polygons": polygons,
            "frame_size": frame_size,
            "scores": {
                "NS": decision.get("score_NS", 0.0),
                "EW": decision.get("score_EW", 0.0),
            },
            "decision": decision,
            "detections_total": int(state.get("detections_total", 0)),
            "detections_by_class": state.get("detections_by_class", {}),
            "active_tracks": int(state.get("active_tracks", 0)),
            "tracks_by_class": state.get("tracks_by_class", {}),
            "pedestrians": int(state.get("pedestrians", 0)),
            "fps": round(float(fps), 2),
            "frame_id": state.get("frame_id", 0),
            "emergency": bool(state.get("emergency", False)),
            "anonymized": anonymized_frame,
            "anonymized_frame": anonymized_frame,
            "anonymized_total": anonymized_total,
            "model_info": state.get("model_info", {}),
            "video": video_data,
            "scenes": deepcopy(self.scenes),
            "current_scene": self.current_scene_id,
            "timestamp": time.time(),
        }
        with self.lock:
            self.frame_bytes = encoded.tobytes()
            analysis_saved = False
            if analysis_bytes is not None and (
                self.analysis_snapshot_pending or self.analysis_frame_bytes is None
            ):
                self.analysis_frame_bytes = analysis_bytes
                self.analysis_frame_version += 1
                self.analysis_frame_id = next_state["frame_id"]
                self.analysis_state = deepcopy(next_state)
                self.analysis_snapshot_pending = False
                analysis_saved = True
            analysis_state = self.analysis_state or next_state
            next_state["analysis_frame_version"] = self.analysis_frame_version
            next_state["analysis_frame_id"] = self.analysis_frame_id
            next_state["analysis_status"] = "ready" if analysis_saved else self.state.get("analysis_status", "idle")
            next_state["analysis"] = deepcopy(analysis_state)
            next_state["analysis"]["analysis_frame_version"] = self.analysis_frame_version
            next_state["analysis"]["analysis_frame_id"] = self.analysis_frame_id
            self.state = next_state

    def update_video_metadata(self, video_meta, model_info=None):
        """Publie les metadonnees de la video source avant toute analyse IA."""
        with self.lock:
            video_state = deepcopy(self.state.get("video", {}))
            video_state.update(video_meta or {})
            video_state["source_ready"] = bool(self.source_video_path)
            self.state["video"] = video_state
            self.state["scenes"] = deepcopy(self.scenes)
            self.state["current_scene"] = self.current_scene_id
            if model_info is not None:
                self.state["model_info"] = deepcopy(model_info)
            self.state["timestamp"] = time.time()

    def set_source_video(self, source):
        """Expose le fichier video original au lecteur HTML5 du dashboard."""
        if isinstance(source, int):
            return
        path = Path(str(source)).expanduser().resolve()
        if not path.exists():
            return
        with self.lock:
            self.source_video_path = str(path)
            video_state = deepcopy(self.state.get("video", {}))
            video_state["source_ready"] = True
            self.state["video"] = video_state
            self.state["scenes"] = deepcopy(self.scenes)
            self.state["current_scene"] = self.current_scene_id

    def pop_seek_request(self):
        """Retourne puis efface la demande de seek envoyee par le dashboard."""
        with self.lock:
            request_data = self.seek_request
            self.seek_request = None
        return request_data

    def pop_scene_request(self):
        """Retourne puis efface la demande de changement de scene."""
        with self.lock:
            request_data = self.scene_request
            self.scene_request = None
        return request_data

    def set_current_scene(self, scene_id):
        """Publie la scene active dans l'etat JSON du dashboard."""
        if scene_id not in self.scenes:
            return
        with self.lock:
            self.current_scene_id = scene_id
            self.state["current_scene"] = scene_id
            self.state["scenes"] = deepcopy(self.scenes)

    def needs_analysis_snapshot(self):
        """Indique si le pipeline doit recalculer l'image d'analyse fixe."""
        with self.lock:
            return self.analysis_snapshot_pending or self.analysis_frame_bytes is None

    def generate_frames(self, kind="pipeline"):
        """Generator MJPEG."""
        while True:
            with self.lock:
                frame_bytes = self.analysis_frame_bytes if kind == "analysis" else self.frame_bytes
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)

    def start(self):
        """Demarre Flask dans un thread separe."""
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(
            target=self.app.run,
            kwargs={
                "host": self.host,
                "port": self.port,
                "debug": False,
                "threaded": True,
                "use_reloader": False,
            },
            daemon=True,
            name="DashboardThread",
        )
        self.thread.start()
        print(f"✅ Dashboard demarre -> http://{self.host}:{self.port}")

    def stop(self):
        """Arret logique. Le thread Flask daemon se fermera avec le processus."""
        self.running = False


_default_dashboard = None


def start_dashboard_thread(host=DASHBOARD_HOST, port=DASHBOARD_PORT):
    """Compatibilite avec l'ancien main.py."""
    global _default_dashboard
    _default_dashboard = Dashboard(host=host, port=port)
    _default_dashboard.start()
    return _default_dashboard


def update_state(frame, fps, state_mdp, decision, anon_count, frame_idx):
    """Compatibilite avec l'ancien main.py."""
    if _default_dashboard is None:
        return
    state = deepcopy(state_mdp)
    state["frame_id"] = frame_idx
    state["anonymized"] = anon_count
    _default_dashboard.update(frame, state, decision, fps)
