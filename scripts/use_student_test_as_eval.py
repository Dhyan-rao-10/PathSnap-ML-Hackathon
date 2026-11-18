"""Rehearsal helper: copy the 10 student-test trips into data/eval/ so the API/UI
can be exercised end-to-end before the real evaluator trips are available."""
import shutil

import _common  # noqa: F401
from pathsnap import config as C, io

C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
for tid in io.list_trips(C.STUDENT_TEST_DIR):
    dst = C.EVAL_DIR / tid
    if not dst.exists():
        shutil.copytree(C.STUDENT_TEST_DIR / tid, dst)
print("data/eval now holds:", io.list_trips(C.EVAL_DIR))
