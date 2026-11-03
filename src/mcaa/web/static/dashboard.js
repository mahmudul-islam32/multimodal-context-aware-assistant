/* Analytics dashboard: polls /api/analytics and renders small SVG charts. */
const STATE = { IDLE: "#22c55e", HANDS_BUSY: "#f59e0b", MOVING: "#a855f7", NOISY: "#ef4444", PLAIN: "#64748b", AUTO: "#334155" };
const SERIES = "#3b82f6";
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const ms = (v) => (v == null ? "–" : `${v.toFixed(1)} ms`);
const label = (k) => k.replace(/_/g, " ").toLowerCase();

function kpis(k) {
  const tiles = [
    ["Answers", k.answers, `${k.followups} follow-ups · ${k.briefings} briefings`],
    ["Refusal rate", pct(k.refusal_rate), `${k.blocked} blocked by guardrails`],
    ["Faithfulness", pct(k.faithfulness), "cited sentences supported"],
    ["Latency p50", ms(k.latency_ms_p50), `p95 ${ms(k.latency_ms_p95)}`],
    ["Helpful", pct(k.helpful_rate), `${k.feedback} ratings`],
  ];
  $("#kpis").innerHTML = tiles.map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join("");
}

// horizontal bars: one row per category, value label at the bar end
function hbars(el, data, colorOf = () => SERIES) {
  const entries = Object.entries(data).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { el.innerHTML = '<p class="empty">No data yet.</p>'; return; }
  const W = 380, row = 26, lw = 158, max = Math.max(...entries.map((e) => e[1]));
  const H = entries.length * row;
  const bars = entries.map(([k, v], i) => {
    const w = Math.max(2, ((W - lw - 40) * v) / max), y = i * row;
    return `<g><title>${esc(label(k))}: ${v}</title>
      <text x="${lw - 8}" y="${y + 16}" text-anchor="end">${esc(label(k)).slice(0, 28)}</text>
      <rect x="${lw}" y="${y + 5}" width="${w}" height="16" rx="4" fill="${colorOf(k)}"></rect>
      <text class="val" x="${lw + w + 6}" y="${y + 17}">${v}</text></g>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="bar chart">${bars}</svg>`;
}

function timeline(el, items, now) {
  if (!items.length) { el.innerHTML = '<p class="empty">No context changes yet: stream sensor data or use the simulator.</p>'; return; }
  const W = 1000, H = 54, t0 = items[0].ts, t1 = Math.max(now, items[items.length - 1].ts + 1);
  const x = (t) => ((t - t0) / (t1 - t0)) * W;
  const segs = items.map((it, i) => {
    const end = i + 1 < items.length ? items[i + 1].ts : t1;
    const w = Math.max(1, x(end) - x(it.ts) - 2);
    return `<g><title>${esc(it.state)}${it.override ? " (manual override)" : ""} · ${fmtTime(it.ts)}, ${Math.round(end - it.ts)} s</title>
      <rect x="${x(it.ts)}" y="4" width="${w}" height="30" rx="4" fill="${STATE[it.state] || "#334155"}"${it.override ? ' stroke="#e8eefc" stroke-dasharray="4 3"' : ""}></rect></g>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="context timeline">${segs}
    <text x="0" y="50">${fmtTime(t0)}</text><text x="${W}" y="50" text-anchor="end">${fmtTime(t1)}</text></svg>`;
  $("#tl-legend").innerHTML = ["IDLE", "HANDS_BUSY", "MOVING", "NOISY"].map((s) => `<span><i style="background:${STATE[s]}"></i>${label(s)}</span>`).join("") + '<span><i style="background:none;border:1px dashed #e8eefc"></i>manual override</span>';
}

function columns(el, pts) {
  if (!pts.length) { el.innerHTML = '<p class="empty">No data yet.</p>'; return; }
  const W = 360, H = 140, pad = 22, n = pts.length, max = Math.max(...pts.map((p) => p.n));
  const bw = Math.min(28, (W - pad) / n - 3);
  const cols = pts.map((p, i) => {
    const h = ((H - 30) * p.n) / max, x = pad + i * ((W - pad) / n);
    return `<g><title>${fmtTime(p.t)}: ${p.n} answers</title><rect x="${x}" y="${H - 18 - h}" width="${bw}" height="${h}" rx="3" fill="${SERIES}"></rect></g>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="answers per minute">
    <g class="grid"><line x1="${pad}" x2="${W}" y1="${H - 18}" y2="${H - 18}"></line></g>${cols}
    <text x="${pad}" y="${H - 4}">${fmtTime(pts[0].t)}</text><text x="${W}" y="${H - 4}" text-anchor="end">${fmtTime(pts[n - 1].t)}</text>
    <text class="val" x="${pad - 4}" y="14" text-anchor="end">${max}</text></svg>`;
}

function recent(rows) {
  $("#recent tbody").innerHTML = rows.length ? rows.map((r) => `<tr>
    <td>${fmtTime(r.ts)}</td><td>${esc(r.query)}${r.followup ? ' <span class="pill">follow-up</span>' : ""}</td>
    <td><span class="pill" style="background:${STATE[r.state] || "#334155"};color:#0b1020">${esc(r.state)}</span></td>
    <td>${esc(label(r.modality || ""))}</td>
    <td>${r.refused ? `<span class="pill rejected">refused</span> <small class="muted">${esc(r.reason)}</small>` : '<span class="pill approved">answered</span>'}</td>
    <td>${ms(r.latency_ms)}</td><td>${r.helpful == null ? "–" : r.helpful ? "👍" : "👎"}</td></tr>`).join("")
    : '<tr><td colspan="7" class="muted">No answers yet. Ask something in the assistant.</td></tr>';
}

async function refresh() {
  try {
    const d = await adminFetch("/api/analytics");
    $("#sub").textContent = `${d.live.scenario} · condition ${d.live.condition} · current context ${d.live.context} · HAR p50 ${d.live.har_ms_p50.toFixed(1)} ms`;
    kpis(d.kpis);
    timeline($("#timeline"), d.context_timeline, Date.now() / 1000);
    hbars($("#by-state"), d.by_state, (k) => STATE[k] || SERIES);
    hbars($("#by-modality"), d.by_modality);
    hbars($("#guard"), d.guardrails);
    hbars($("#reasons"), d.refusal_reasons);
    hbars($("#sources"), d.top_sources);
    columns($("#apm"), d.answers_per_minute);
    recent(d.recent);
    $("#live").textContent = `live · ${fmtTime(Date.now() / 1000)}`;
  } catch (e) { $("#live").textContent = e.message; }
}
initAuth(refresh);
refresh();
setInterval(refresh, 3000);
