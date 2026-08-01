/* Levencode training dashboard — vanilla JS, polls the FastAPI endpoints. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const STAGE_ORDER = ["sft", "edit", "jepa", "grpo"];
const STAGE_SLOT = { sft: 1, edit: 2, jepa: 3, grpo: 4 }; // color follows the stage, fixed

// Loss-component series: fixed slot per metric (color follows the entity).
const COMPONENTS = [
  ["fill_loss", 1, "fill (SFT)"],
  ["fill_view_loss", 2, "fill (edit view)"],
  ["del_loss", 3, "delete"],
  ["ins_loss", 4, "insert"],
  ["jepa_loss", 5, "jepa"],
  ["ce", 7, "raw CE"],
];

const BENCH_METRICS = [
  { label: "chat masked CE", task: "chat", key: "chat_masked_ce", dir: "down", fmt: f3 },
  { label: "ARC-Easy acc", task: "arc_easy", key: "arc_easy_acc", dir: "up", fmt: pct },
  { label: "GSM8K EM", task: "gsm8k", key: "gsm8k_em", dir: "up", fmt: pct },
  { label: "MBPP pass@1", task: "mbpp", key: "mbpp_pass1", dir: "up", fmt: pct },
  { label: "repair exact", task: "repair", key: "repair_exact", dir: "up", fmt: pct },
  { label: "repair oracle exact", task: "repair", key: "repair_oracle_exact", dir: "up", fmt: pct },
  { label: "repair syntax-valid", task: "repair", key: "repair_syntax_valid", dir: "up", fmt: pct },
  { label: "repair lev-reduction", task: "repair", key: "repair_lev_reduction", dir: "up", fmt: f3 },
  { label: "repair len-ratio", task: "repair", key: "repair_len_ratio", dir: "down", fmt: f3 },
  { label: "infill exact", task: "infill", key: "infill_exact", dir: "up", fmt: pct },
  { label: "gen speed", task: "speed", key: "gen_tok_per_sec", dir: "up", fmt: f1 },
];

const state = {
  experiments: [],
  exp: null,
  focus: null,
  logY: false,
  smooth: true,
  hidden: { loss: new Set(), comp: new Set() },
  asTable: { loss: false, comp: false, speed: false },
  metrics: {}, // stage -> rows
  bench: {},   // stage -> {name: results}
  samples: {},
  samplesStep: null,
  polling: true,
};

// ---------- formatting ----------
function f1(v) { return v == null ? "—" : Number(v).toFixed(1); }
function f3(v) { return v == null ? "—" : Number(v).toFixed(3); }
function pct(v) { return v == null ? "—" : (100 * v).toFixed(1) + "%"; }
function fmtK(v) {
  if (v == null || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return a >= 10 ? v.toFixed(0) : v.toFixed(2);
}
function fmtDur(s) {
  if (s == null || !isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}
function fmtVal(v) {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(2);
  return a >= 100 ? v.toFixed(1) : v.toFixed(3);
}

// ---------- data ----------
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function stagesOf(expName) {
  const exp = state.experiments.find((e) => e.experiment === expName);
  if (!exp) return [];
  return [...exp.stages].sort(
    (a, b) => (STAGE_ORDER.indexOf(a.stage) + 99) - (STAGE_ORDER.indexOf(b.stage) + 99)
  );
}

async function poll() {
  if (!state.polling) return;
  try {
    const data = await getJSON("/api/experiments");
    state.experiments = data.experiments;
    $("#poll-dot").classList.remove("stale");
    if (!state.experiments.length) { renderEmpty(true); return; }
    renderEmpty(false);
    if (!state.exp || !state.experiments.some((e) => e.experiment === state.exp)) {
      state.exp = state.experiments[0].experiment;
    }
    const stages = stagesOf(state.exp);
    const running = stages.find((s) => ["running", "benchmarking"].includes(s.state.status));
    if (!state.focus || !stages.some((s) => s.stage === state.focus)) {
      state.focus = (running || stages[stages.length - 1]).stage;
    } else if (running && stages.find((s) => s.stage === state.focus)?.state.status === "completed") {
      state.focus = running.stage;
    }
    await Promise.all(
      stages.map(async (s) => {
        state.metrics[s.stage] = (await getJSON(`/api/run/${state.exp}/${s.stage}/metrics`)).rows;
        state.bench[s.stage] = await getJSON(`/api/run/${state.exp}/${s.stage}/bench`);
      })
    );
    state.samples[state.focus] = await getJSON(`/api/run/${state.exp}/${state.focus}/samples`);
    render();
  } catch (e) {
    $("#poll-dot").classList.add("stale");
    console.error(e);
  }
}

// ---------- rendering ----------
function renderEmpty(empty) {
  $("#empty-state").hidden = !empty;
  for (const sel of [".pills", ".tiles", ".charts", "#bench-card", "#samples-card"])
    document.querySelectorAll(sel).forEach((el) => (el.style.display = empty ? "none" : ""));
}

function render() {
  renderExpSelect();
  renderPills();
  renderTiles();
  renderLossChart();
  renderCompChart();
  renderSpeedChart();
  renderBench();
  renderSamples();
}

function renderExpSelect() {
  const sel = $("#exp-select");
  const opts = state.experiments.map((e) => e.experiment);
  if (sel.dataset.opts !== opts.join(",")) {
    sel.innerHTML = opts.map((o) => `<option>${o}</option>`).join("");
    sel.dataset.opts = opts.join(",");
  }
  sel.value = state.exp;
}

function renderPills() {
  const stages = stagesOf(state.exp);
  $("#stage-pills").innerHTML = stages
    .map((s) => {
      const st = s.state;
      const pctTxt =
        st.status === "completed" ? "done" :
        st.status === "failed" ? "failed" :
        st.status === "benchmarking" ? "bench…" :
        `${(st.pct || 0).toFixed(0)}%`;
      return `<button class="pill" data-stage="${s.stage}" aria-current="${s.stage === state.focus}">
        <span class="dot ${st.status}"></span>${s.stage}<span class="muted">${pctTxt}</span></button>`;
    })
    .join("");
  document.querySelectorAll(".pill").forEach((el) =>
    el.addEventListener("click", () => { state.focus = el.dataset.stage; render(); })
  );
}

function renderTiles() {
  const s = stagesOf(state.exp).find((x) => x.stage === state.focus);
  if (!s) return;
  const st = s.state;
  const tiles = [
    ["stage", `${state.focus}`, st.status],
    ["progress", `${fmtK(st.step || 0)}<span class="unit">/ ${fmtK(st.total_steps || 0)}</span>`, `${(st.pct || 0).toFixed(1)}%`],
    ["eta", fmtDur(st.eta_s), ""],
    ["loss", st.last && st.last.loss != null ? fmtVal(st.last.loss) : "—", ""],
    ["masked CE", st.last && st.last.ce != null ? fmtVal(st.last.ce) : "—", "window avg"],
    ["tokens/sec", fmtK(st.tok_per_sec), ""],
    ["lr", st.lr != null ? st.lr.toExponential(1) : "—", ""],
    ["tokens seen", st.last ? fmtK(st.last.tokens_seen) : "—", ""],
  ];
  $("#stat-tiles").innerHTML = tiles
    .map(
      ([label, value, sub]) =>
        `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div><div class="muted">${sub}</div></div>`
    )
    .join("");
}

// ---------- charts ----------
function seriesColor(slot) { return `var(--series-${slot})`; }

function downsample(points, n = 700) {
  if (points.length <= n) return points;
  const stride = points.length / n;
  const out = [];
  for (let i = 0; i < n - 1; i++) out.push(points[Math.floor(i * stride)]);
  out.push(points[points.length - 1]);
  return out;
}

// EMA over logged rows. Rows are already window means (see trainer.py); the
// EMA on top makes the trend readable through plateau-level noise.
function emaPoints(points, alpha = 0.8) {
  const out = [];
  let s = null;
  for (const [x, y] of points) {
    if (y == null || !isFinite(y)) continue;
    s = s == null ? y : alpha * s + (1 - alpha) * y;
    out.push([x, s]);
  }
  return out;
}

function slopePer1k(points) {
  const pts = points.filter((p) => p[1] != null && isFinite(p[1]));
  if (pts.length < 8) return null;
  const tail = pts.slice(Math.floor(pts.length * 0.67)); // last third
  const n = tail.length;
  const mx = tail.reduce((a, p) => a + p[0], 0) / n;
  const my = tail.reduce((a, p) => a + p[1], 0) / n;
  let num = 0, den = 0;
  for (const p of tail) { num += (p[0] - mx) * (p[1] - my); den += (p[0] - mx) ** 2; }
  return den > 0 ? (num / den) * 1000 : null;
}

function trendLabel(points) {
  const s = slopePer1k(points);
  if (s == null) return "";
  if (s < -0.003) return `learning, ${s.toFixed(3)} CE per 1k steps`;
  if (s > 0.003) return `regressing, +${s.toFixed(3)} CE per 1k steps`;
  return "flat";
}

// Bold EMA line over a faint raw ghost, per series.
function withSmoothing(seriesList) {
  if (!state.smooth) return seriesList;
  return seriesList.flatMap((s) => [
    { ...s, ghost: true },
    { ...s, points: emaPoints(s.points) },
  ]);
}

function lineChart(container, seriesList, { logY = false } = {}) {
  const visible = seriesList.filter((s) => s.points.length >= 2);
  container.innerHTML = "";
  if (!visible.length) {
    container.innerHTML = `<p class="muted" style="padding:40px 0;text-align:center">waiting for data…</p>`;
    return;
  }
  const W = container.clientWidth || 600, H = 240;
  const M = { l: 52, r: 12, t: 10, b: 24 };
  const xs = visible.flatMap((s) => s.points.map((p) => p[0]));
  let ys = visible.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v != null && isFinite(v));
  if (logY) ys = ys.filter((v) => v > 0);
  if (!ys.length) { container.innerHTML = ""; return; }
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const ty = (v) => (logY ? Math.log10(v) : v);
  const [ly0, ly1] = [ty(y0), ty(y1)];
  const X = (v) => M.l + ((v - x0) / Math.max(x1 - x0, 1e-9)) * (W - M.l - M.r);
  const Y = (v) => M.t + (1 - (ty(v) - ly0) / Math.max(ly1 - ly0, 1e-9)) * (H - M.t - M.b);

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const nTicks = 4;
  for (let i = 0; i <= nTicks; i++) {
    const vy = ly0 + ((ly1 - ly0) * i) / nTicks;
    const y = M.t + (1 - i / nTicks) * (H - M.t - M.b);
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", M.l); line.setAttribute("x2", W - M.r);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("class", i === 0 ? "baseline" : "gridline");
    svg.appendChild(line);
    const t = document.createElementNS(svgNS, "text");
    t.setAttribute("x", M.l - 6); t.setAttribute("y", y + 4);
    t.setAttribute("text-anchor", "end");
    t.setAttribute("class", "axis-label");
    t.textContent = fmtK(logY ? Math.pow(10, vy) : vy);
    svg.appendChild(t);
  }
  for (let i = 0; i <= 4; i++) {
    const vx = x0 + ((x1 - x0) * i) / 4;
    const t = document.createElementNS(svgNS, "text");
    t.setAttribute("x", X(vx)); t.setAttribute("y", H - 6);
    t.setAttribute("text-anchor", i === 0 ? "start" : i === 4 ? "end" : "middle");
    t.setAttribute("class", "axis-label");
    t.textContent = fmtK(vx);
    svg.appendChild(t);
  }

  for (const s of visible) {
    const pts = downsample(s.points.filter((p) => p[1] != null && isFinite(p[1]) && (!logY || p[1] > 0)));
    if (pts.length < 2) continue;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke-width", s.ghost ? "1" : "2");
    if (s.ghost) path.style.opacity = "0.28";
    path.setAttribute("stroke-linejoin", "round");
    path.style.stroke = s.color;
    svg.appendChild(path);
  }

  const cross = document.createElementNS(svgNS, "line");
  cross.setAttribute("class", "crosshair");
  cross.setAttribute("y1", M.t); cross.setAttribute("y2", H - M.b);
  cross.setAttribute("visibility", "hidden");
  svg.appendChild(cross);

  const tooltip = $("#tooltip");
  svg.addEventListener("mousemove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const vx = x0 + ((px - M.l) / Math.max(W - M.l - M.r, 1)) * (x1 - x0);
    let best = null;
    for (const s of visible) {
      for (const p of s.points) {
        if (best === null || Math.abs(p[0] - vx) < Math.abs(best - vx)) best = p[0];
      }
    }
    if (best === null) return;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best));
    cross.setAttribute("visibility", "visible");
    const rows = visible
      .filter((s) => !s.ghost)
      .map((s) => {
        const p = s.points.find((q) => q[0] === best);
        return p && p[1] != null
          ? `<div class="t-row"><span class="t-name"><span class="chip" style="background:${s.color}"></span>${s.name}</span><span class="t-val">${fmtVal(p[1])}</span></div>`
          : "";
      })
      .join("");
    tooltip.innerHTML = `<div class="t-step">step ${fmtK(best)}</div>${rows}`;
    tooltip.hidden = false;
    const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    let tx = ev.clientX + 14, tyy = ev.clientY + 10;
    if (tx + tw > window.innerWidth - 8) tx = ev.clientX - tw - 14;
    if (tyy + th > window.innerHeight - 8) tyy = ev.clientY - th - 10;
    tooltip.style.left = tx + "px"; tooltip.style.top = tyy + "px";
  });
  svg.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    $("#tooltip").hidden = true;
  });

  container.appendChild(svg);
}

function dataTable(container, seriesList, lastN = 30) {
  const steps = [...new Set(seriesList.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b).slice(-lastN);
  const head = `<tr><th>step</th>${seriesList.map((s) => `<th>${s.name}</th>`).join("")}</tr>`;
  const rows = steps
    .map((st) => {
      const cells = seriesList
        .map((s) => {
          const p = s.points.find((q) => q[0] === st);
          return `<td>${p && p[1] != null ? fmtVal(p[1]) : "—"}</td>`;
        })
        .join("");
      return `<tr><td>${st}</td>${cells}</tr>`;
    })
    .join("");
  container.innerHTML = `<div class="table-wrap"><table>${head}${rows}</table></div>`;
}

function legend(el, seriesList, hiddenSet, onToggle) {
  if (seriesList.length < 2) { el.innerHTML = ""; return; }
  el.innerHTML = seriesList
    .map(
      (s) =>
        `<button data-name="${s.name}" aria-pressed="${!hiddenSet.has(s.name)}">
           <span class="chip" style="background:${s.color}"></span>${s.name}</button>`
    )
    .join("");
  el.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => { onToggle(b.dataset.name); })
  );
}

// Headline series: masked-prediction CE — the low-noise "is it learning?"
// signal, and comparable across sft/edit/jepa (same objective, same data mix).
// The 1/t-weighted total `loss` is a high-variance ELBO estimator: it bounces
// ~±0.1 at convergence with zero real change, which reads as "not learning".
// Stages that never log ce (grpo) fall back to their `loss`.
function learnSeries() {
  return stagesOf(state.exp)
    .filter((s) => (state.metrics[s.stage] || []).length)
    .map((s) => ({
      name: s.stage,
      color: seriesColor(STAGE_SLOT[s.stage] || 8),
      points: (state.metrics[s.stage] || []).map((r) => [r.step, r.ce != null ? r.ce : r.loss]),
    }));
}

function renderLossChart() {
  const all = learnSeries();
  const focused = all.find((s) => s.name === state.focus);
  const trend = focused ? trendLabel(emaPoints(focused.points)) : "";
  $("#loss-sub").textContent = `masked CE by stage${trend ? ` — ${state.focus}: ${trend}` : ""}`;
  legend($("#loss-legend"), all, state.hidden.loss, (n) => {
    state.hidden.loss.has(n) ? state.hidden.loss.delete(n) : state.hidden.loss.add(n);
    renderLossChart();
  });
  const shown = all.filter((s) => !state.hidden.loss.has(s.name));
  const el = $("#loss-chart");
  state.asTable.loss ? dataTable(el, shown) : lineChart(el, withSmoothing(shown), { logY: state.logY });
}

function renderCompChart() {
  const rows = state.metrics[state.focus] || [];
  $("#comp-sub").textContent = state.focus || "";
  const series = COMPONENTS
    .map(([key, slot, label]) => ({
      name: label,
      color: seriesColor(slot),
      points: rows.filter((r) => r[key] != null).map((r) => [r.step, r[key]]),
    }))
    .filter((s) => s.points.length >= 2);
  legend($("#comp-legend"), series, state.hidden.comp, (n) => {
    state.hidden.comp.has(n) ? state.hidden.comp.delete(n) : state.hidden.comp.add(n);
    renderCompChart();
  });
  const shown = series.filter((s) => !state.hidden.comp.has(s.name));
  const el = $("#comp-chart");
  state.asTable.comp ? dataTable(el, shown) : lineChart(el, withSmoothing(shown), {});
}

function renderSpeedChart() {
  const rows = state.metrics[state.focus] || [];
  const series = [{
    name: "tok/s",
    color: seriesColor(STAGE_SLOT[state.focus] || 1),
    points: rows.filter((r) => r.tok_per_sec != null).map((r) => [r.step, r.tok_per_sec]),
  }];
  const el = $("#speed-chart");
  state.asTable.speed ? dataTable(el, series) : lineChart(el, series, {});
}

// ---------- bench table ----------
function renderBench() {
  const cols = [];
  for (const s of stagesOf(state.exp)) {
    const per = state.bench[s.stage] || {};
    for (const [name, results] of Object.entries(per)) {
      cols.push({ label: name === s.stage ? s.stage : `${s.stage}/${name}`, results });
    }
  }
  const el = $("#bench-table");
  if (!cols.length) {
    el.innerHTML = `<p class="muted">No benchmark results yet — they appear after each stage finishes.</p>`;
    return;
  }
  const head = `<tr><th>metric</th>${cols.map((c) => `<th>${c.label}</th>`).join("")}</tr>`;
  const body = BENCH_METRICS.map((m) => {
    const vals = cols.map((c) => {
      const task = c.results[m.task];
      if (!task || task.error != null || task[m.key] == null) return null;
      return task[m.key];
    });
    const usable = vals.filter((v) => v != null);
    const best = usable.length
      ? (m.dir === "up" ? Math.max(...usable) : Math.min(...usable))
      : null;
    const cells = vals
      .map((v, i) => {
        if (v == null) {
          const err = cols[i].results[m.task]?.error;
          return `<td class="missing" title="${err ? String(err).replace(/"/g, "'") : "not run"}">—</td>`;
        }
        const isBest = best != null && v === best && usable.length > 1;
        return `<td class="${isBest ? "best" : ""}">${m.fmt(v)}</td>`;
      })
      .join("");
    const arrow = m.dir === "up" ? "↑" : "↓";
    return `<tr><td class="metric">${m.label}<span class="dir" title="${m.dir === "up" ? "higher" : "lower"} is better">${arrow}</span></td>${cells}</tr>`;
  }).join("");
  el.innerHTML = `<table>${head}${body}</table>`;
}

// ---------- samples ----------
function renderSamples() {
  const files = state.samples[state.focus] || {};
  const groups = Object.values(files).flat();
  $("#samples-sub").textContent = state.focus || "";
  const sel = $("#samples-step");
  if (!groups.length) {
    sel.innerHTML = "";
    $("#samples").innerHTML = `<p class="muted">No samples yet — the trainer writes a gallery every <code>sample_every</code> steps.</p>`;
    return;
  }
  const steps = groups.map((g) => g.step).sort((a, b) => b - a);
  const want = state.samplesStep != null && steps.includes(state.samplesStep) ? state.samplesStep : steps[0];
  sel.innerHTML = steps.map((s) => `<option value="${s}" ${s === want ? "selected" : ""}>step ${s}</option>`).join("");
  sel.onchange = () => { state.samplesStep = Number(sel.value); renderSamples(); };
  const group = groups.find((g) => g.step === want);
  $("#samples").innerHTML =
    `<div class="group">` +
    group.samples
      .map(
        (s) => `<div class="sample-card">
          <div class="prompt">${escapeHtml(s.prompt)}</div>
          <pre>${escapeHtml(s.output || "(empty)")}</pre>
          ${s.tok_per_sec ? `<div class="meta">${s.tok_per_sec} tok/s</div>` : ""}
        </div>`
      )
      .join("") +
    `</div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- wiring ----------
$("#exp-select").addEventListener("change", (e) => {
  state.exp = e.target.value;
  state.focus = null;
  poll();
});
$("#logy-btn").addEventListener("click", () => {
  state.logY = !state.logY;
  $("#logy-btn").setAttribute("aria-pressed", state.logY);
  renderLossChart();
});
$("#smooth-btn").addEventListener("click", () => {
  state.smooth = !state.smooth;
  $("#smooth-btn").setAttribute("aria-pressed", state.smooth);
  renderLossChart();
  renderCompChart();
});
document.querySelectorAll(".table-toggle").forEach((b) =>
  b.addEventListener("click", () => {
    const k = b.dataset.chart;
    state.asTable[k] = !state.asTable[k];
    b.setAttribute("aria-pressed", state.asTable[k]);
    render();
  })
);
document.addEventListener("visibilitychange", () => {
  state.polling = !document.hidden;
  if (state.polling) poll();
});

poll();
setInterval(poll, 2500);
