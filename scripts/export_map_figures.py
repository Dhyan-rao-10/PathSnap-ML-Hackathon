"""Export the committed Folium HTML maps to PNG for the report (headless Microsoft Edge).

Full-extent views are plain screenshots of results/**/map_*.html. The zoomed view
re-opens the same map, switches on the classifier-only layer and zooms to the
stretch of student-test trip 00000076 where the HMM corrected the most classifier
errors (from results/hmm_analysis/per_fix_errors.csv). Nothing is drawn by hand.

Run after scripts/eda.py, scripts/tune_eval_hmm.py and scripts/hmm_error_analysis.py.
Outputs -> report/figures/
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from pathsnap import config as C
from pathsnap.trip import Trip

EDGE_CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
EDGE = next((p for p in EDGE_CANDIDATES if Path(p).exists()), None) or shutil.which("msedge")
FIG = C.ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MAPS = ["eda/map_train_00000003", "eda/map_student_test_00000009",
        "hmm/map_student_test_00000009", "hmm/map_student_test_00000076"]


def shoot(html: Path, png: Path, size=(1400, 900)):
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=15000",
                    f"--screenshot={png}", html.resolve().as_uri()],
                   check=True, capture_output=True, timeout=120)
    print("wrote", png)


for m in MAPS:
    shoot(C.RESULTS / f"{m}.html", FIG / f"{m.replace('/', '_')}.png")

# zoomed view: densest run of fixes the HMM corrected on trip 76
tid = "00000076"
err = pd.read_csv(C.RESULTS / "hmm_analysis" / "per_fix_errors.csv", dtype={"trip": str})
t = err[err.trip == tid].sort_values("fix").reset_index(drop=True)
fixed = (~t.clf_correct) & t.hmm_correct
centre_fix = int(t.fix[int(fixed.rolling(30, center=True).sum().idxmax())])
df = Trip(C.STUDENT_TEST_DIR / tid).df
lat, lon = float(df.lat.iat[centre_fix]), float(df.lon.iat[centre_fix])

src = (C.RESULTS / f"hmm/map_student_test_{tid}.html").read_text(encoding="utf-8")
# same map, only the initial bounds changed: a ~250 m box around the corrected stretch
box = f"[[{lat - 0.0012}, {lon - 0.002}], [{lat + 0.0012}, {lon + 0.002}]]"
src = re.sub(r"(\.fitBounds\(\s*)\[\[[^\]]+\], \[[^\]]+\]\]", lambda m: m.group(1) + box, src, count=1)
inject = """<script>
window.addEventListener('load', function () {
  document.querySelectorAll('.leaflet-control-layers-overlays label').forEach(function (l) {
    var cb = l.querySelector('input');
    if (l.textContent.indexOf('Classifier-only') >= 0 && !cb.checked) cb.click();
  });
});
</script>"""
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d) / "zoom.html"
    tmp.write_text(src.replace("</html>", inject + "</html>"), encoding="utf-8")
    shoot(tmp, FIG / f"hmm_map_student_test_{tid}_zoom.png")
print(f"zoom centre: trip {tid} fix {centre_fix} ({lat:.5f}, {lon:.5f})")
