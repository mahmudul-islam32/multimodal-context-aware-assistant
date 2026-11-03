// Minimal offline shell cache for the PWA (API calls always go to the network).
const CACHE = "mcaa-v1";
const SHELL = ["/", "/style.css", "/app.js", "/manifest.webmanifest", "/icon.svg"];
self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL))));
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (u.pathname.startsWith("/api") || u.pathname.startsWith("/ws")) return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
