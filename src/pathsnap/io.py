"""Raw file readers. Each trip folder <id>/ holds <id>.track/.nodes/.arcs[/.route]."""
from pathlib import Path
import numpy as np


def _load(path, dtype=float, ncols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def list_trips(part_dir):
    part_dir = Path(part_dir)
    if not part_dir.exists():
        return []
    return sorted(p.name for p in part_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def load_track(trip_dir):
    """GPS fixes: columns lon, lat, t (seconds)."""
    trip_dir = Path(trip_dir)
    return _load(trip_dir / f"{trip_dir.name}.track", ncols=3)


def load_network(trip_dir):
    """Road network: nodes (lon, lat) and directed arcs (from_node, to_node)."""
    trip_dir = Path(trip_dir)
    nodes = _load(trip_dir / f"{trip_dir.name}.nodes", ncols=2)
    arcs = _load(trip_dir / f"{trip_dir.name}.arcs", dtype=np.int64, ncols=2)
    return nodes, arcs


def load_route(trip_dir):
    """Reference route: chronological arc ids. ONLY used for labels/evaluation,
    never by the prediction pipeline."""
    trip_dir = Path(trip_dir)
    p = trip_dir / f"{trip_dir.name}.route"
    if not p.exists():
        return None
    return np.loadtxt(p, dtype=np.int64, ndmin=1)
