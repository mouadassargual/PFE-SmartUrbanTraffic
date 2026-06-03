/* ═══════════════════════════════════════════════════════
   Smart Traffic Agadir — Dashboard JS
   Mise à jour en temps réel via l'API Flask (/stats)
   ═══════════════════════════════════════════════════════ */

// ─── Configuration Charts (Chart.js) ─────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Inter';

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: { legend: { display: false } },
  scales: {
    x: { display: false },
    y: { border: { display: false }, grid: { color: 'rgba(255,255,255,0.05)' } }
  }
};

let chartFps, chartDensity, chartPhase;

function initCharts() {
  // Chart FPS
  const ctxFps = document.getElementById('chartFps').getContext('2d');
  chartFps = new Chart(ctxFps, {
    type: 'line',
    data: {
      labels: Array(60).fill(''),
      datasets: [{
        data: Array(60).fill(0),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        borderWidth: 2,
        fill: true,
        pointRadius: 0,
        tension: 0.3
      }]
    },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 35 } } }
  });

  // Chart Densité
  const ctxDensity = document.getElementById('chartDensity').getContext('2d');
  chartDensity = new Chart(ctxDensity, {
    type: 'line',
    data: {
      labels: Array(60).fill(''),
      datasets: [
        { label: 'N', data: [], borderColor: '#4ade80', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'S', data: [], borderColor: '#22c55e', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'E', data: [], borderColor: '#f87171', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        { label: 'W', data: [], borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, tension: 0.3 }
      ]
    },
    options: { ...commonOptions, plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 10, usePointStyle: true, pointStyle: 'circle' } } } }
  });

  // Chart Phase (MDP)
  const ctxPhase = document.getElementById('chartPhase').getContext('2d');
  chartPhase = new Chart(ctxPhase, {
    type: 'bar',
    data: {
      labels: Array(60).fill(''),
      datasets: [{
        data: Array(60).fill(0),
        backgroundColor: [],
        borderRadius: 2
      }]
    },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { display: false } } }
  });
}

// ─── Mise à jour de l'UI ─────────────────────────────────

function updateTime() {
  const now = new Date();
  document.getElementById('currentTime').textContent = now.toLocaleTimeString('fr-FR');
}
setInterval(updateTime, 1000);
updateTime();

function updateDashboard() {
  fetch('/stats')
    .then(res => res.json())
    .then(data => {
      // 1. Header & KPIs
      document.getElementById('sessionTime').textContent = `Session : ${data.session_start}`;
      document.getElementById('kpiFps').textContent = data.fps.toFixed(1);
      document.getElementById('kpiVehicles').textContent = Object.values(data.queues).reduce((a, b) => a + b, 0);
      document.getElementById('kpiPrivacy').textContent = data.total_persons_anon;
      document.getElementById('kpiDecisions').textContent = data.total_decisions;
      document.getElementById('kpiEmergency').textContent = data.emergency_events;
      document.getElementById('kpiFrame').textContent = data.frame_idx;
      document.getElementById('kpiSession').textContent = `Démarrage : ${data.session_start}`;
      
      document.getElementById('kpiFpsBar').innerHTML = `<div style="width: ${Math.min(100, data.fps/30*100)}%; height: 2px; background: var(--accent); margin-top: 4px;"></div>`;
      document.getElementById('kpiTracks').textContent = `Files d'attente (total)`;
      document.getElementById('kpiDecSub').textContent = `Urgence: ${data.emergency_events} | Piéton: ${data.pedestrian_events}`;
      document.getElementById('footerFps').textContent = `FPS : ${data.fps.toFixed(1)}`;

      // Alert Urgence globale
      if (data.emergency > 0) {
        document.getElementById('kpi-emergency').classList.add('emg-active');
      } else {
        document.getElementById('kpi-emergency').classList.remove('emg-active');
      }

      // 2. Traffic Lights (Phase)
      const phase = data.phase;
      document.getElementById('phaseLabel').textContent = phase;
      document.getElementById('phaseDuration').textContent = `${data.duration}s`;

      const lNS = document.getElementById('lightNS');
      const lEW = document.getElementById('lightEW');
      
      // Reset
      lNS.className = 'light-circle light-off';
      lEW.className = 'light-circle light-off';

      if (phase === 'emergency') {
        lNS.className = 'light-circle light-orange';
        lEW.className = 'light-circle light-orange';
        document.getElementById('phaseLabel').style.background = 'rgba(245, 158, 11, 0.2)';
        document.getElementById('phaseLabel').style.color = '#f59e0b';
      } else if (phase === 'pedestrian') {
        lNS.className = 'light-circle light-red';
        lEW.className = 'light-circle light-red';
        document.getElementById('phaseLabel').style.background = 'rgba(239, 68, 68, 0.2)';
        document.getElementById('phaseLabel').style.color = '#ef4444';
      } else {
        // NS ou EW
        if (data.green_dirs.includes('N')) lNS.className = 'light-circle light-green';
        else lNS.className = 'light-circle light-red';
        
        if (data.green_dirs.includes('E')) lEW.className = 'light-circle light-green';
        else lEW.className = 'light-circle light-red';
        
        document.getElementById('phaseLabel').style.background = 'rgba(34, 197, 94, 0.2)';
        document.getElementById('phaseLabel').style.color = '#4ade80';
      }

      // 3. Queues (Files d'attente)
      const maxQueue = Math.max(10, ...Object.values(data.queues));
      for (const dir of ['N', 'S', 'E', 'W']) {
        const q = data.queues[dir] || 0;
        document.getElementById(`count${dir}`).textContent = q;
        document.getElementById(`bar${dir}`).style.width = `${(q / maxQueue) * 100}%`;
      }
      
      document.getElementById('countPed').textContent = data.pedestrians;
      document.getElementById('countEmg').textContent = data.emergency;
      if(data.emergency > 0) {
        document.getElementById('emgItem').classList.add('emg-active');
      } else {
        document.getElementById('emgItem').classList.remove('emg-active');
      }

      // 4. Update Charts
      chartFps.data.datasets[0].data = data.fps_history;
      chartFps.update();

      chartDensity.data.datasets[0].data = data.density_history['N'];
      chartDensity.data.datasets[1].data = data.density_history['S'];
      chartDensity.data.datasets[2].data = data.density_history['E'];
      chartDensity.data.datasets[3].data = data.density_history['W'];
      chartDensity.update();

      // Chart Phase colors mapping
      const phaseColors = data.phase_history.map(p => {
        if (p === 'NS') return '#4ade80';
        if (p === 'EW') return '#60a5fa';
        if (p === 'emergency') return '#f59e0b';
        if (p === 'pedestrian') return '#ef4444';
        return '#475569';
      });
      // values just for bar height
      const phaseVals = data.phase_history.map(p => p === 'emergency' ? 3 : p === 'pedestrian' ? 2 : 1);
      chartPhase.data.datasets[0].data = phaseVals;
      chartPhase.data.datasets[0].backgroundColor = phaseColors;
      chartPhase.update();

    })
    .catch(err => console.error("Erreur API :", err));
}

// ─── Lancement ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  setInterval(updateDashboard, 1000); // Poll API every second
});
