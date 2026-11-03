/* Shared helpers for the admin and dashboard pages. */
const $ = (s, el = document) => el.querySelector(s);
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const store = {
  get(k) { try { return sessionStorage.getItem(k) || ""; } catch { return ""; } },
  set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};
async function adminFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const tok = store.get("mcaa-admin-token");
  if (tok) headers["X-Admin-Token"] = tok;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(path, { ...opts, headers });
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (r.status === 401) { showAuth(true); throw new Error("Admin token required"); }
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
function showAuth(needed) {
  const bar = $("#auth");
  if (!bar) return;
  bar.hidden = !needed;
}
function initAuth(onSave) {
  $("#token").value = store.get("mcaa-admin-token");
  $("#auth-form").addEventListener("submit", (e) => { e.preventDefault(); store.set("mcaa-admin-token", $("#token").value.trim()); showAuth(false); onSave(); });
  fetch("/api/health").then((r) => r.json()).then((h) => { $("#auth-mode").textContent = `auth: ${h.auth}`; });
}
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
