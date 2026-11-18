import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, LayersControl, LayerGroup, Pane, useMap } from "react-leaflet";
import { getApiUrl, setApiUrl, health, listTrips, predict } from "./api.js";

const pct = (v) => (v == null || Number.isNaN(v) ? "–" : `${(100 * v).toFixed(1)}%`);
const num = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (!bounds) return;
    // the metric cards appear with the result and shrink the map: resize first, then fit
    const id = setTimeout(() => {
      map.invalidateSize();
      map.fitBounds(bounds, { padding: [24, 24] });
    }, 80);
    return () => clearTimeout(id);
  }, [bounds, map]);
  return null;
}

function Metric({ label, value, sub, accent }) {
  return (
    <div className={`metric ${accent ? "accent" : ""}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

export default function App() {
  const [apiUrl, setUrl] = useState(getApiUrl());
  const [status, setStatus] = useState({ state: "idle" });
  const [trips, setTrips] = useState([]);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  async function connect(url = apiUrl) {
    setError(null);
    setStatus({ state: "connecting" });
    try {
      const h = await health(url);
      const t = await listTrips(url);
      setTrips(t.trips);
      setStatus({ state: "ok", info: h });
      setApiUrl(url);
    } catch (e) {
      setStatus({ state: "error" });
      setError(`Cannot reach API at ${url} (${e.message})`);
      setTrips([]);
    }
  }

  useEffect(() => { connect(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    if (!selected) return;
    setRunning(true); setError(null); setResult(null);
    const t0 = performance.now();
    try {
      const r = await predict(apiUrl, selected);
      const rtt = (performance.now() - t0) / 1000;
      setResult({ ...r, rtt });
      setHistory((h) => [{ trip: selected, acc: r.evaluation?.hmm_accuracy, f1: r.evaluation?.route_f1, rtt,
                            at: new Date().toLocaleTimeString() }, ...h].slice(0, 12));
    } catch (e) {
      setError(`Prediction failed: ${e.message}`);
    } finally {
      setRunning(false);
    }
  }

  const ev = result?.evaluation || {};
  const gps = useMemo(() => result?.gps || [], [result]);

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">◎</span>
          <div>
            <h1>PathSnap</h1>
            <p>Map-matching &amp; route reconstruction · evaluator console</p>
          </div>
        </div>
        <div className="conn">
          <input value={apiUrl} onChange={(e) => setUrl(e.target.value)} placeholder="http://<laptop-1-ip>:8000"
                 onKeyDown={(e) => e.key === "Enter" && connect()} />
          <button onClick={() => connect()}>Connect</button>
          <span className={`dot ${status.state}`} title={status.state} />
          <span className="conn-label">
            {status.state === "ok" ? "API online" : status.state === "connecting" ? "Connecting…" : status.state === "error" ? "Offline" : ""}
          </span>
        </div>
      </header>

      <main>
        <aside>
          <h2>Evaluation trips</h2>
          <p className="hint">Select a trip, then run the full pipeline on the API server.</p>
          <div className="trips">
            {trips.map((t) => (
              <button key={t.trip_id} className={`trip ${selected === t.trip_id ? "sel" : ""}`}
                      onClick={() => setSelected(t.trip_id)}>
                <span className="tid">Trip {t.trip_id}</span>
                <span className="tn">{t.n_fixes.toLocaleString()} GPS fixes</span>
              </button>
            ))}
            {trips.length === 0 && <div className="empty">No trips loaded.</div>}
          </div>
          <button className="run" disabled={!selected || running || status.state !== "ok"} onClick={run}>
            {running ? "Running pipeline…" : selected ? `Run PathSnap on ${selected}` : "Select a trip"}
          </button>
          {error && <div className="error">{error}</div>}

          {history.length > 0 && (
            <div className="history">
              <h3>Run log</h3>
              <table>
                <thead><tr><th>Time</th><th>Trip</th><th>Acc.</th><th>Route F1</th><th>s</th></tr></thead>
                <tbody>
                  {history.map((h, i) => (
                    <tr key={i}><td>{h.at}</td><td>{h.trip}</td><td>{pct(h.acc)}</td><td>{num(h.f1, 3)}</td><td>{num(h.rtt, 1)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </aside>

        <section className="content">
          {result && (
            <div className="metrics">
              <Metric accent label="Per-fix accuracy (HMM)" value={pct(ev.hmm_accuracy)}
                      sub={`classifier only: ${pct(ev.classifier_accuracy)}`} />
              <Metric label="Route F1 (length-weighted)" value={num(ev.route_f1, 3)}
                      sub={`P ${num(ev.route_precision, 3)} · R ${num(ev.route_recall, 3)}`} />
              <Metric label="Route continuity" value={pct(result.stats.hmm_fix_continuity)}
                      sub={`${result.stats.route_breaks} break(s) · clf ${pct(result.stats.classifier_fix_continuity)}`} />
              <Metric label="Predicted route" value={`${num(result.stats.route_length_km, 2)} km`}
                      sub={`${result.stats.route_arcs} road segments`} />
              <Metric label="Server time" value={`${num(result.timings.total_prediction_s, 2)} s`}
                      sub={`round trip ${num(result.rtt, 2)} s`} />
            </div>
          )}
          <div className="mapwrap">
            <MapContainer center={[48.8, 9]} zoom={4} preferCanvas className="map">
              {/* route drawn in its own pane above the GPS dots */}
              <Pane name="routePane" style={{ zIndex: 450 }} />
              <LayersControl position="topright">
                <LayersControl.BaseLayer checked name="OpenStreetMap">
                  <TileLayer attribution="&copy; OpenStreetMap contributors"
                             url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                </LayersControl.BaseLayer>
                {result && (
                  <LayersControl.Overlay checked name="Predicted route (HMM)">
                    <LayerGroup>
                      {result.predicted_route.map((pl, i) => (
                        <Polyline key={i} positions={pl} pane="routePane" pathOptions={{ color: "#d62728", weight: 4, opacity: 0.9 }} />
                      ))}
                    </LayerGroup>
                  </LayersControl.Overlay>
                )}
                {result && (
                  <LayersControl.Overlay checked name="Raw GPS fixes">
                    <LayerGroup>
                      {gps.map((p, i) => (
                        <CircleMarker key={i} center={p} radius={2}
                                      pathOptions={{ color: "#1f4fd8", weight: 0, fillOpacity: 0.55 }} />
                      ))}
                    </LayerGroup>
                  </LayersControl.Overlay>
                )}
              </LayersControl>
              {result && <FitBounds bounds={result.bounds} />}
            </MapContainer>
            <div className="legend">
              <div><span className="sw gps" />Raw GPS fixes</div>
              <div><span className="sw pred" />Predicted route (HMM + Viterbi)</div>
            </div>
            {running && <div className="overlay">Matching {selected}…</div>}
          </div>
          {result && (
            <div className="details">
              <div><b>{result.n_fixes.toLocaleString()}</b> fixes · <b>{result.n_segments}</b> track segment(s) ·
                <b> {ev.n_labelled_fixes?.toLocaleString?.() ?? "–"}</b> labelled fixes scored server-side ·
                <b> {result.stats.fixes_without_candidates}</b> fixes without candidates</div>
              <div className="timing">
                load {num(result.timings.load_s)}s · candidates+features {num(result.timings.candidates_features_s)}s ·
                classifier {num(result.timings.classifier_s)}s · Viterbi {num(result.timings.viterbi_s)}s ·
                stitching {num(result.timings.stitch_s)}s
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
