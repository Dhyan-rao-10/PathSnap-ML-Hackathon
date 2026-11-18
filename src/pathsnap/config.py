"""Project paths and default parameters (overridable via artifacts/config.json)."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "raw"))
TRAIN_DIR = DATA_RAW / "train"
STUDENT_TEST_DIR = DATA_RAW / "student_test"
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "eval"))
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))

SEED = 42

# ---- track cleaning / segmentation ----
GAP_BREAK_S = 120.0       # time gap that breaks the HMM chain into separate segments
MAX_SPEED_MS = 70.0       # implied speed above this between fixes = GPS jump -> break
HEADING_HALF_WINDOW = 2   # heading computed from fix i-k to i+k

# ---- candidate generation ----
CAND_RADIUS_M = 50.0      # search radius for candidate arcs (tuned in build_dataset)
CAND_K = 8                # max candidates per GPS fix
CAND_RADIUS_MAX_M = 300.0 # fallback radius when nothing lies within CAND_RADIUS_M
INDEX_STEP_M = 10.0       # arcs are densified every INDEX_STEP_M for the KD-tree

# ---- labels ----
LABEL_MAX_DIST_M = 60.0   # fixes further than this from their aligned route arc are "unlabelled"

# ---- features ----
CTX_HALF_WINDOW = 5       # temporal context window (fixes on each side)

# ---- HMM ----
HMM_BETA = 10.0           # transition scale (m) for |route_dist - gps_dist|
HMM_EMISSION_WEIGHT = 1.0 # weight of log classifier probability
HMM_MAX_STATES = 8        # candidates kept per fix (top by classifier prob)
HMM_UNREACHABLE_LOGP = -40.0
HMM_SAME_ARC_BACK_TOL_M = 10.0
