/* Browser port of uniformity_model.m (MATLAB) */

// ---------- math helpers ----------
function clampInt(v, min, max) {
  const n = Math.trunc(Number(v));
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function clampNum(v, min, max) {
  const n = Number(v);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

// Seeded RNG (mulberry32)
function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal via Box–Muller
function randn(rng) {
  // avoid log(0)
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function std(arr) {
  const m = mean(arr);
  let s2 = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    s2 += d * d;
  }
  return Math.sqrt(s2 / (arr.length - 1 || 1));
}

function sumFirstN(arr, n) {
  let s = 0;
  const k = Math.min(n, arr.length);
  for (let i = 0; i < k; i++) s += arr[i];
  return s;
}

// ---------- model (matches uniformity_model.m) ----------
// transpiration polynomial: 0.4069 t^4 - 21.061 t^3 + 383.29 t^2 - 2866 t + 7527.4
function trans(t) {
  return (
    0.4069 * t ** 4 -
    21.061 * t ** 3 +
    383.29 * t ** 2 -
    2866 * t +
    7527.4
  );
}

// Antiderivative of trans(t)
function transIntegral(t) {
  return (
    (0.4069 / 5) * t ** 5 -
    (21.061 / 4) * t ** 4 +
    (383.29 / 3) * t ** 3 -
    (2866 / 2) * t ** 2 +
    7527.4 * t
  );
}

function stepConsumption(tEnd, dt, unifactor) {
  // consumption = ∫_{tEnd-dt}^{tEnd} (unifactor * trans(t)) dt
  const a = tEnd - dt;
  const b = tEnd;
  return unifactor * (transIntegral(b) - transIntegral(a));
}

// ---------- UI wiring ----------
const els = {
  phase1: document.getElementById("phase1"),
  limit: document.getElementById("limit"),
  startingMoisture: document.getElementById("startingMoisture"),
  nPlants: document.getElementById("nPlants"),
  nROPlants: document.getElementById("nROPlants"),
  sigma: document.getElementById("sigma"),
  nDays: document.getElementById("nDays"),
  seed: document.getElementById("seed"),
  speed: document.getElementById("speed"),
  speedLabel: document.getElementById("speedLabel"),
  renderEvery: document.getElementById("renderEvery"),
  btnReset: document.getElementById("btnReset"),
  btnStep: document.getElementById("btnStep"),
  btnRun: document.getElementById("btnRun"),
  plot: document.getElementById("plot"),
  plotScale: document.getElementById("plotScale"),
  stats: document.getElementById("stats"),
  log: document.getElementById("log"),
  helpBtn: document.getElementById("helpBtn"),
  helpPopover: document.getElementById("helpPopover"),
  helpClose: document.getElementById("helpClose"),
};

function setHelpOpen(open) {
  if (!els.helpBtn || !els.helpPopover) return;
  els.helpBtn.setAttribute("aria-expanded", open ? "true" : "false");
  els.helpPopover.classList.toggle("hidden", !open);
}

function setRunButton(running) {
  els.btnRun.textContent = running ? "Pause" : "Run";
  els.btnRun.classList.toggle("danger", running);
}

function statCard(k, v) {
  const div = document.createElement("div");
  div.className = "stat";
  div.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
  return div;
}

function fmt(n, digits = 2) {
  if (!Number.isFinite(n)) return "—";
  return Number(n).toFixed(digits);
}

// ---------- Simulation state ----------
const SIM = {
  // parameters
  phase1: 150,
  limit: 6000,
  startingMoisture: 6200,
  nPlants: 100,
  nROPlants: 5,
  sigma: 0.25,
  nDays: 1,
  seed: 1,
  secondsPerStep: 0.5,
  renderEvery: 2,

  // dynamics
  dt: 0.25,
  tStart: 8.0,
  tEnd: 18.5,
  running: false,
  stepIdx: 0,
  irrigationCount: 0,
  dayIdx: 1,
  t: 8.0,
  field: [],
  unifactors: [],
  scaleIndices: [],
  lastFrameTs: 0,
  accMs: 0,
  renderCounter: 0,
  logLines: [],
};

function sampleUniqueIndices(rng, nTotal, k) {
  // Fisher–Yates partial shuffle
  const idx = Array.from({ length: nTotal }, (_, i) => i);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (nTotal - i));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx.slice(0, k);
}

function computeStepsPerDay() {
  return Math.round((SIM.tEnd - SIM.tStart) / SIM.dt) + 1;
}

function computeTotalSteps() {
  return SIM.nDays * computeStepsPerDay();
}

function getStepTime(stepInDay) {
  return SIM.tStart + stepInDay * SIM.dt;
}

function resetSim() {
  SIM.phase1 = clampNum(els.phase1.value, 0, 1e9);
  SIM.limit = clampNum(els.limit.value, 0, 1e9);
  SIM.startingMoisture = clampNum(els.startingMoisture.value, 0, 1e9);
  SIM.nPlants = clampInt(els.nPlants.value, 1, 5000);
  SIM.nROPlants = clampInt(els.nROPlants.value, 1, 5000);
  SIM.sigma = clampNum(els.sigma.value, 0, 10);
  SIM.nDays = clampInt(els.nDays.value, 1, 365);
  SIM.seed = clampInt(els.seed.value, 0, 2 ** 31 - 1);
  SIM.secondsPerStep = clampNum(els.speed.value, 0.01, 10);
  SIM.renderEvery = clampInt(els.renderEvery.value, 1, 200);

  if (SIM.nROPlants > SIM.nPlants) {
    SIM.nROPlants = SIM.nPlants;
    els.nROPlants.value = String(SIM.nROPlants);
  }

  const rng = mulberry32(SIM.seed);
  SIM.unifactors = new Array(SIM.nPlants);
  for (let i = 0; i < SIM.nPlants; i++) {
    SIM.unifactors[i] = 1 + SIM.sigma * randn(rng);
  }

  // Random selection of "Scale" plants from the full crop (seeded/reproducible)
  SIM.scaleIndices = sampleUniqueIndices(rng, SIM.nPlants, SIM.nROPlants);

  SIM.field = new Array(SIM.nPlants).fill(SIM.startingMoisture);
  SIM.running = false;
  SIM.stepIdx = 0;
  SIM.irrigationCount = 0;
  SIM.dayIdx = 1;
  SIM.t = SIM.tStart;
  SIM.lastFrameTs = 0;
  SIM.renderCounter = 0;
  SIM.logLines = [];

  els.speedLabel.textContent = fmt(SIM.secondsPerStep, 2);
  setRunButton(false);
  render(true);
}

function appendLog(line) {
  SIM.logLines.push(line);
  if (SIM.logLines.length > 200) SIM.logLines.shift();
  els.log.textContent = SIM.logLines.join("\n");
}

function simulateOneStep() {
  const stepsPerDay = computeStepsPerDay();
  const totalSteps = computeTotalSteps();
  if (SIM.stepIdx >= totalSteps) {
    SIM.running = false;
    setRunButton(false);
    appendLog("DONE");
    return false;
  }

  const stepInDay = SIM.stepIdx % stepsPerDay;
  SIM.dayIdx = Math.floor(SIM.stepIdx / stepsPerDay) + 1;
  SIM.t = getStepTime(stepInDay);

  // MATLAB loop: for t = 8:.25:18.5
  // consumption = int(crop, t-.25,t);
  const tEnd = SIM.t;
  for (let i = 0; i < SIM.nPlants; i++) {
    SIM.field[i] -= stepConsumption(tEnd, SIM.dt, SIM.unifactors[i]);
  }

  let scaleSum = 0;
  for (let i = 0; i < SIM.scaleIndices.length; i++) {
    scaleSum += SIM.field[SIM.scaleIndices[i]];
  }
  const scaleAvg = scaleSum / (SIM.scaleIndices.length || 1);
  if (scaleAvg <= SIM.limit) {
    for (let i = 0; i < SIM.nPlants; i++) SIM.field[i] += SIM.phase1;
    SIM.irrigationCount += 1;
    appendLog(`Day ${SIM.dayIdx} t=${fmt(SIM.t, 2)}: IRRIGATION (+${SIM.phase1} ml to all)`);
  }

  SIM.stepIdx += 1;
  return true;
}

function render(force = false) {
  SIM.renderCounter += 1;
  const shouldRender = force || SIM.renderCounter % SIM.renderEvery === 0;
  if (!shouldRender) return;

  const y = SIM.field.slice();
  const n = y.length;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    if (y[i] < minV) minV = y[i];
    if (y[i] > maxV) maxV = y[i];
  }

  // Stats
  const mu = mean(y);
  const sd = std(y);
  const stepsPerDay = computeStepsPerDay();
  const totalSteps = computeTotalSteps();
  const stepInDay = SIM.stepIdx % stepsPerDay;
  const pctBelowLimit =
    (SIM.scaleIndices.filter((idx) => y[idx] <= SIM.limit).length / (SIM.scaleIndices.length || 1)) * 100;

  els.stats.innerHTML = "";
  els.stats.appendChild(statCard("Day", `${SIM.dayIdx} / ${SIM.nDays}`));
  els.stats.appendChild(statCard("Time (t)", fmt(SIM.t, 2)));
  els.stats.appendChild(statCard("Step", `${SIM.stepIdx} / ${totalSteps}`));
  els.stats.appendChild(statCard("Irrigations", String(SIM.irrigationCount)));
  els.stats.appendChild(statCard("Mean (ml)", fmt(mu, 1)));
  els.stats.appendChild(statCard("Std (ml)", fmt(sd, 1)));
  els.stats.appendChild(statCard("Min (ml)", fmt(minV, 1)));
  els.stats.appendChild(statCard("Scale ≤ limit (%)", `${fmt(pctBelowLimit, 0)}%`));

  // Plot
  const x = Array.from({ length: n }, (_, i) => i + 1);
  const yMax = Math.max(7000, maxV, SIM.startingMoisture + SIM.phase1 * 3, SIM.limit * 1.1);
  const layoutMain = {
    margin: { l: 50, r: 20, t: 10, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    shapes: [
      // Lower volume limit: thin red horizontal line across the plot
      {
        type: "line",
        layer: "above",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: SIM.limit,
        y1: SIM.limit,
        line: { color: "rgba(255, 92, 122, 0.98)", width: 2 },
      },
    ],
    xaxis: {
      title: "Individually Potted Plants (indexed)",
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "rgba(233,238,252,0.85)" },
      titlefont: { color: "rgba(233,238,252,0.9)" },
    },
    yaxis: {
      title: "Amt. Water in Pot (ml)",
      range: [0, yMax],
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "rgba(233,238,252,0.85)" },
      titlefont: { color: "rgba(233,238,252,0.9)" },
    },
    showlegend: false,
  };

  const traceMain = {
    type: "bar",
    x,
    y,
    width: 0.25,
    marker: { color: "rgba(59, 130, 246, 0.85)" },
  };

  Plotly.react(els.plot, [traceMain], layoutMain, {
    displayModeBar: false,
    responsive: true,
  });

  // Scale plot (random subset of crop)
  const k = SIM.scaleIndices.length;
  const xS = Array.from({ length: k }, (_, i) => i + 1);
  const yS = SIM.scaleIndices.map((idx) => SIM.field[idx]);
  let maxScale = -Infinity;
  for (let i = 0; i < yS.length; i++) maxScale = Math.max(maxScale, yS[i]);
  const yMaxScale = Math.max(yMax, maxScale, SIM.limit * 1.1);

  const layoutScale = {
    margin: { l: 50, r: 10, t: 10, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    shapes: [
      {
        type: "line",
        layer: "above",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: SIM.limit,
        y1: SIM.limit,
        line: { color: "rgba(255, 92, 122, 0.98)", width: 2 },
      },
    ],
    xaxis: {
      title: "Scale pots",
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "rgba(233,238,252,0.85)" },
      titlefont: { color: "rgba(233,238,252,0.9)" },
    },
    yaxis: {
      title: "",
      range: [0, yMaxScale],
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "rgba(233,238,252,0.85)" },
    },
    showlegend: false,
  };

  const traceScale = {
    type: "bar",
    x: xS,
    y: yS,
    width: 0.25,
    marker: { color: "rgba(37, 99, 235, 0.9)" },
  };

  Plotly.react(els.plotScale, [traceScale], layoutScale, {
    displayModeBar: false,
    responsive: true,
  });
}

function tick(ts) {
  if (!SIM.running) return;
  if (!SIM.lastFrameTs) SIM.lastFrameTs = ts;
  const dtMs = ts - SIM.lastFrameTs;
  SIM.lastFrameTs = ts;

  // Stable timing: advance 1 step every N milliseconds (default: 500ms)
  const stepMs = SIM.secondsPerStep * 1000;
  SIM.accMs += dtMs;
  while (SIM.accMs >= stepMs) {
    SIM.accMs -= stepMs;
    const ok = simulateOneStep();
    if (!ok) break;
  }
  render(false);

  if (SIM.running) requestAnimationFrame(tick);
}

// ---------- events ----------
// Help popover
if (els.helpBtn && els.helpPopover) {
  setHelpOpen(false);

  els.helpBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = els.helpBtn.getAttribute("aria-expanded") === "true";
    setHelpOpen(!isOpen);
  });

  if (els.helpClose) {
    els.helpClose.addEventListener("click", (e) => {
      e.stopPropagation();
      setHelpOpen(false);
    });
  }

  // Prevent clicks inside the popover from closing it
  els.helpPopover.addEventListener("click", (e) => e.stopPropagation());

  // Close on outside click
  document.addEventListener("click", () => setHelpOpen(false));

  // Close on Esc
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setHelpOpen(false);
  });
}

els.speed.addEventListener("input", () => {
  els.speedLabel.textContent = fmt(clampNum(els.speed.value, 0.01, 10), 2);
});

els.btnReset.addEventListener("click", () => resetSim());

els.btnStep.addEventListener("click", () => {
  if (SIM.running) return;
  simulateOneStep();
  render(true);
});

els.btnRun.addEventListener("click", () => {
  SIM.running = !SIM.running;
  setRunButton(SIM.running);
  if (SIM.running) {
    appendLog("RUN");
    SIM.lastFrameTs = 0;
    SIM.accMs = 0;
    requestAnimationFrame(tick);
  } else {
    appendLog("PAUSE");
  }
});

// reset on any parameter change (keeps the behavior simple and deterministic)
[
  els.phase1,
  els.limit,
  els.startingMoisture,
  els.nPlants,
  els.nROPlants,
  els.sigma,
  els.nDays,
  els.seed,
  els.renderEvery,
].forEach((el) => el.addEventListener("change", () => resetSim()));

// ---------- boot ----------
resetSim();


