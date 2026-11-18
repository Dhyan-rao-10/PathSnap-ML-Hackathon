"""Folium map generation (GPS fixes, road network, reference route, predicted route)."""
import numpy as np
import folium
from branca.element import Template, MacroElement

COLORS = {"gps": "#1f4fd8", "network": "#9a9a9a", "reference": "#1a9e3a",
          "predicted": "#d62728", "classifier": "#ff9f1c"}


def _arc_lines(net, arcs):
    lines = []
    for a in arcs:
        f, t = net.a_from[a], net.a_to[a]
        lines.append([[net.nodes_lonlat[f, 1], net.nodes_lonlat[f, 0]],
                      [net.nodes_lonlat[t, 1], net.nodes_lonlat[t, 0]]])
    return lines


def _route_polyline(net, arcs):
    """Chain consecutive arcs into polylines (break where they are not connected)."""
    polys, cur = [], []
    for i, a in enumerate(arcs):
        f, t = net.a_from[a], net.a_to[a]
        pf = [net.nodes_lonlat[f, 1], net.nodes_lonlat[f, 0]]
        pt = [net.nodes_lonlat[t, 1], net.nodes_lonlat[t, 0]]
        if cur and net.a_to[arcs[i - 1]] == f:
            cur.append(pt)
        else:
            if cur:
                polys.append(cur)
            cur = [pf, pt]
    if cur:
        polys.append(cur)
    return polys


def network_arcs_near_track(net, df, margin_m=150.0, max_arcs=40000):
    x0, x1 = df["x"].min() - margin_m, df["x"].max() + margin_m
    y0, y1 = df["y"].min() - margin_m, df["y"].max() + margin_m
    inside = ((np.minimum(net.ax, net.bx) <= x1) & (np.maximum(net.ax, net.bx) >= x0)
              & (np.minimum(net.ay, net.by) <= y1) & (np.maximum(net.ay, net.by) >= y0))
    arcs = np.where(inside)[0]
    # draw two-way streets once
    arcs = arcs[(net.twin[arcs] < 0) | (arcs < net.twin[arcs])]
    return arcs[:max_arcs]


def _legend(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in items)
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
         padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font: 13px sans-serif;">
      <b>Legend</b>{rows}
    </div>
    {{% endmacro %}}"""
    m = MacroElement()
    m._template = Template(html)
    return m


def trip_map(trip, reference_arcs=None, predicted_arcs=None, classifier_arcs=None,
             title=None, show_network=True, gps_every=1):
    df, net = trip.df, trip.net
    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap",control_scale=True)
    legend = []
    if show_network:
        arcs = network_arcs_near_track(net, df)
        fg = folium.FeatureGroup(name=f"Road network ({len(arcs)} segments)")
        folium.PolyLine(_arc_lines(net, arcs), color=COLORS["network"], weight=1.5,
                        opacity=0.6).add_to(fg)
        fg.add_to(m); legend.append(("Road network", COLORS["network"]))
    if reference_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route")
        for pl in _route_polyline(net, list(reference_arcs)):
            folium.PolyLine(pl, color=COLORS["reference"], weight=6, opacity=0.75).add_to(fg)
        fg.add_to(m); legend.append(("Reference route", COLORS["reference"]))
    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only route", show=False)
        for pl in _route_polyline(net, list(classifier_arcs)):
            folium.PolyLine(pl, color=COLORS["classifier"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m); legend.append(("Classifier-only (top-1)", COLORS["classifier"]))
    if predicted_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM)")
        for pl in _route_polyline(net, list(predicted_arcs)):
            folium.PolyLine(pl, color=COLORS["predicted"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m); legend.append(("Predicted route (HMM)", COLORS["predicted"]))
    fg = folium.FeatureGroup(name="GPS observations")
    sub = df.iloc[::gps_every]
    for lat, lon, t in zip(sub["lat"], sub["lon"], sub["t"]):
        folium.CircleMarker([lat, lon], radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"t={t:.0f}s").add_to(fg)
    fg.add_to(m); legend.append(("GPS fixes", COLORS["gps"]))
    folium.Marker([df["lat"].iloc[0], df["lon"].iloc[0]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([df["lat"].iloc[-1], df["lon"].iloc[-1]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[df["lat"].min(), df["lon"].min()], [df["lat"].max(), df["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(_legend(legend))
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<h4 style="position:fixed;top:8px;left:60px;z-index:9999;background:white;'
            f'padding:4px 10px;border-radius:4px;font-family:sans-serif">{title}</h4>'))
    return m
