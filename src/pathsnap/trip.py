"""Trip container used by both training and inference."""
from pathlib import Path
import numpy as np

from . import io
from .geo import LocalProjection
from .network import RoadNetwork
from .track import clean_track, build_track_frame


class Trip:
    """Prediction-side view of a trip: GPS track + road network only.
    The reference route is deliberately NOT loaded here."""

    def __init__(self, trip_dir):
        trip_dir = Path(trip_dir)
        self.trip_id = trip_dir.name
        self.dir = trip_dir
        raw = io.load_track(trip_dir)
        nodes, arcs = io.load_network(trip_dir)
        self.proj = LocalProjection.for_points(raw[:, :2])
        tr, self.clean_report = clean_track(raw)
        self.df = build_track_frame(tr, self.proj)
        self.net = RoadNetwork(nodes, arcs, self.proj)
        self.net.build_index()
