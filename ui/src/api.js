// Thin client for the PathSnap FastAPI backend running on Laptop 1.
const KEY = "pathsnap_api_url";

export function getApiUrl() {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved) return saved;
  } catch (_) { /* storage unavailable */ }
  return import.meta.env.VITE_API_URL || "http://localhost:8000";
}

export function setApiUrl(url) {
  try { localStorage.setItem(KEY, url); } catch (_) { /* ignore */ }
}

async function call(base, path, opts = {}) {
  const res = await fetch(base.replace(/\/$/, "") + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const health = (base) => call(base, "/api/health");
export const listTrips = (base) => call(base, "/api/trips");
export const predict = (base, tripId) =>
  call(base, "/api/predict", { method: "POST", body: JSON.stringify({ trip_id: tripId }) });
