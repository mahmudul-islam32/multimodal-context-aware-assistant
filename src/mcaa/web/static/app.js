/* Tablet PWA client: voice + touch, context indicator, sources, study mode. */
const $ = (s) => document.querySelector(s);
const api = (path, body, method) =>
  fetch(path, { method: method || (body ? "POST" : "GET"), headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined })
    .then((r) => (r.ok ? r.json() : r.json().then((e) => Promise.reject(e))));

let S = null;            // last server snapshot
let lastFull = "";       // full answer for "show details"
const ACTS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING", "WORKING"];

// ---------------------------------------------------------------- render
function render(s) {
  S = s;
  const plain = s.condition === "B";
  document.body.classList.toggle("plain", plain);
  $("#scenario-name").textContent = s.scenario.name;
  $("#task-title").textContent = s.task.title + (s.participant ? ` · ${s.participant}` : "");
  const w = $("#sim-warning");
  w.hidden = !s.scenario.simulator_only; w.textContent = s.scenario.warning || "";
  const b = $("#condition-badge");
  b.textContent = plain ? "B · plain chatbot" : "A · context-aware"; b.className = "badge " + s.condition;
  // steps
  $("#steps").innerHTML = s.task.steps.map((st, i) =>
    `<li data-i="${i}" class="${i < s.task.step ? "done" : ""} ${i === s.task.step ? "current" : ""}" ${i === s.task.step ? 'aria-current="step"' : ""}>
       <span>${esc(st.text)}</span>${st.manual ? '<span class="hand" title="hands needed for this step"><span aria-hidden="true">✋</span><span class="sr-only">manual step</span></span>' : ""}</li>`).join("");
  // context
  const c = s.context;
  const chip = $("#ctx-chip"); chip.className = "ctx-chip " + c.state;
  $("#ctx-state").textContent = c.state.replace("_", " ");
  $("#ctx-conf").textContent = c.overridden ? "manual" : `${Math.round(c.confidence * 100)}%`;
  $("#ctx-reason").textContent = (c.reason || "") + (c.candidate ? ` · pending → ${c.candidate}` : "");
  document.querySelectorAll("[data-ov]").forEach((btn) => btn.setAttribute("aria-pressed", String(c.overridden && btn.dataset.ov === c.state)));
  const act = s.activity || {};
  $("#activity").innerHTML = Object.keys(act).length
    ? Object.entries(act).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
        `<div class="bar"><span>${k.toLowerCase().replace(/_/g, " ")}</span><span class="track"><span class="fill" style="width:${(v * 100).toFixed(0)}%"></span></span><span>${(v * 100).toFixed(0)}%</span></div>`).join("")
    : '<p class="muted">No sensor data yet: open the phone sensor page or use the simulator.</p>';
  $("#docs-status").textContent = `Indexed: ${s.documents.indexed.join(", ")} · quarantined: ${s.documents.quarantined.length} · rejected: ${s.documents.rejected.length}`;
  $("#cam-consent").checked = s.camera.enabled;
}

const esc = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
const withSups = (t) => esc(t).replace(/\s*\[(\d+)\]/g, '<sup data-n="$1">[$1]</sup>');

function showAnswer(res, query) {
  const { answer: a, plan: p } = res;
  const card = $("#answer-card");
  card.classList.toggle("refused", a.refused);
  card.classList.remove("cue"); void card.offsetWidth;
  if (p.visual_cue) card.classList.add("cue");
  lastFull = a.answer;
  const tags = [`<span class="tag">${p.state}</span>`, `<span class="tag">${p.modality.replace("_", " ")}</span>`,
    a.refused ? `<span class="tag">refused: ${esc(a.refusal_reason)}</span>` : `<span class="tag">grounded ${(a.faithfulness * 100).toFixed(0)}%</span>`,
    `<span class="tag">${a.latency_ms} ms</span>`];
  $("#answer-meta").innerHTML = tags.join("");
  $("#answer-text").innerHTML = withSups(p.screen_text);
  $("#show-details").hidden = !(p.screen_detail === "summary" && p.screen_text !== a.answer);
  $("#voice-line").hidden = !p.speak;
  $("#voice-text").textContent = p.voice_text;
  if (p.speak) speak(p.voice_text);
  // sources
  $("#sources").innerHTML = a.citations.length
    ? a.citations.map((c) => `<div class="src" id="src-${c.n}"><b>[${c.n}] ${esc(c.title)}</b><span class="ver">${esc(c.section)} · v${esc(c.version)}</span><p>${esc(c.text)}</p></div>`).join("")
    : `<p class="muted">${a.refused ? "No approved source supports an answer." : ""}</p>`;
  const h = document.createElement("div");
  h.innerHTML = `<div class="q">Q: ${esc(query)}</div><div class="a">${withSups(a.short_answer)}</div>`;
  $("#history").prepend(h);
}

// ---------------------------------------------------------------- voice out (F-14)
function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05; u.lang = "en-US";
  speechSynthesis.speak(u);
}
function bargeIn() { if ("speechSynthesis" in window) speechSynthesis.cancel(); }

// ---------------------------------------------------------------- ask
async function ask(text, source = "text", extra = {}) {
  bargeIn();
  const res = await api("/api/ask", { text, source, ...extra });
  if (res.barge_in) return;
  showAnswer(res, text);
}

// ---------------------------------------------------------------- voice in (F-03)
// Preferred: local Whisper on the backend (audio never leaves the lab server).
// Fallback: browser Web Speech API.
let rec = null, chunks = [], ctxAudio = null, proc = null, stream = null;
async function pttStart() {
  bargeIn();
  $("#ptt").classList.add("rec");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    ctxAudio = new AudioContext();
    const src = ctxAudio.createMediaStreamSource(stream);
    proc = ctxAudio.createScriptProcessor(4096, 1, 1);
    chunks = [];
    proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(ctxAudio.destination);
  } catch (e) { $("#ptt").classList.remove("rec"); alert("Microphone not available: " + e.message); }
}
async function pttStop() {
  $("#ptt").classList.remove("rec");
  if (!proc) return;
  proc.disconnect(); stream.getTracks().forEach((t) => t.stop());
  const sr = ctxAudio.sampleRate; ctxAudio.close(); proc = null;
  const pcm = flatten(chunks);
  const fd = new FormData(); fd.append("file", new Blob([wav(pcm, sr)], { type: "audio/wav" }), "ptt.wav");
  const r = await fetch("/api/speech/transcribe", { method: "POST", body: fd });
  if (r.ok) {
    const t = await r.json();
    if (t.text) ask(t.text, "voice", { asr_confidence: t.confidence, noise_dbfs: t.noise_dbfs });
  } else webSpeechOnce();
}
function webSpeechOnce() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return alert("No speech recognition available. Install faster-whisper on the server.");
  const r = new SR(); r.lang = "en-US";
  r.onresult = (e) => ask(e.results[0][0].transcript, "voice", { asr_confidence: e.results[0][0].confidence });
  r.start();
}
const flatten = (cs) => { const n = cs.reduce((a, c) => a + c.length, 0), o = new Float32Array(n); let i = 0; cs.forEach((c) => { o.set(c, i); i += c.length; }); return o; };
function wav(pcm, sr) {
  const b = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(b);
  const ws = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  ws(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); ws(8, "WAVE"); ws(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); ws(36, "data"); v.setUint32(40, pcm.length * 2, true);
  pcm.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
  return b;
}

// ---------------------------------------------------------------- camera (F-16)
let camTimer = null;
async function camera(on) {
  await api("/api/camera/consent", { consent: on });
  const v = $("#cam");
  if (on) {
    v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } }); v.hidden = false;
    const c = document.createElement("canvas"); c.width = 160; c.height = 120;
    camTimer = setInterval(() => { c.getContext("2d").drawImage(v, 0, 0, 160, 120); api("/api/camera/frame", { jpeg_base64: c.toDataURL("image/jpeg", 0.6) }).catch(() => {}); }, 1000);
  } else {
    clearInterval(camTimer); if (v.srcObject) v.srcObject.getTracks().forEach((t) => t.stop()); v.hidden = true;
  }
}

// ---------------------------------------------------------------- wiring
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/ui`);
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === "state") render(m); };
  ws.onclose = () => setTimeout(connect, 1500);
  window._ws = ws;
  document.addEventListener("pointerdown", () => ws.readyState === 1 && ws.send(JSON.stringify({ type: "touch" })), { passive: true });
}

$("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (q) { ask(q); $("#q").value = ""; } });
const ptt = $("#ptt");
ptt.addEventListener("pointerdown", pttStart); ptt.addEventListener("pointerup", pttStop); ptt.addEventListener("pointerleave", () => proc && pttStop());
$("#stop-voice").addEventListener("click", () => { bargeIn(); api("/api/touch", {}); });
$("#answer-card").addEventListener("click", bargeIn);
$("#show-details").addEventListener("click", () => { $("#answer-text").innerHTML = withSups(lastFull); $("#show-details").hidden = true; });
$("#steps").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) api("/api/task/step", { index: +li.dataset.i }).then(render); });
$("#next-step").addEventListener("click", () => api("/api/task/step", { index: S.task.step + 1 }).then(render));
$("#prev-step").addEventListener("click", () => api("/api/task/step", { index: S.task.step - 1 }).then(render));
document.querySelectorAll("[data-ov]").forEach((b) => b.addEventListener("click", () => api("/api/context/override", { state: b.dataset.ov || null }).then(() => api("/api/state").then(render))));
$("#researcher-toggle").addEventListener("click", (e) => { const d = $("#researcher"); d.hidden = !d.hidden; e.target.setAttribute("aria-expanded", String(!d.hidden)); });
$("#start-task").addEventListener("click", async () => {
  const pid = $("#pid").value.trim() || null;
  if (pid) { const sch = await api(`/api/study/schedule/${pid}`); $("#schedule").textContent = "Counterbalanced schedule: " + sch.blocks.map((b) => `block ${b.block}: condition ${b.condition}, task set ${b.task_set}`).join(" · "); }
  render(await api("/api/task/start", { task_set: $("#tset").value, condition: $("#cond").value, participant: pid }));
  $("#history").innerHTML = "";
});
$("#complete-task").addEventListener("click", () => api("/api/task/complete", {}).then(render));
$("#log-error").addEventListener("click", () => { const k = prompt("Error type (wrong / skipped / unsafe step)", "skipped"); if (k) api("/api/task/error", { kind: k }); });
$("#export").addEventListener("click", () => api("/api/study/export").then((r) => alert("Exported:\n" + r.files.join("\n"))));
$("#cam-consent").addEventListener("change", (e) => camera(e.target.checked).catch((err) => { alert(err.message || err.detail); e.target.checked = false; }));
$("#sim-buttons").innerHTML = ACTS.map((a) => `<button data-sim="${a}">${a.toLowerCase().replace(/_/g, " ")}</button>`).join("") + '<button data-sim="" class="secondary">stop</button>';
$("#sim-buttons").addEventListener("click", (e) => { const b = e.target.closest("[data-sim]"); if (b) api("/api/sim/activity", { activity: b.dataset.sim || null }); });
$("#scenario-select").addEventListener("change", (e) => api(`/api/scenarios/${e.target.value}`, {}).then(render));
api("/api/scenarios").then((r) => { $("#scenario-select").innerHTML = r.available.map((n) => `<option ${n === r.active ? "selected" : ""}>${n}</option>`).join(""); });
api("/api/state").then(render);
connect();
// fallback: poll state if the WebSocket is not open (e.g. behind a proxy without WS support)
setInterval(() => { if (!window._ws || window._ws.readyState !== 1) api("/api/state").then(render).catch(() => {}); }, 2000);
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
