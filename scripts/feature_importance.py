"""Feature importance of the tuned LightGBM classifier (gain = total loss reduction
contributed by splits on each feature; split = how often the feature is used).

Usage: python scripts/feature_importance.py
Outputs -> results/models/feature_importance_tuned_lgbm.{csv,png}
"""
import _common  # noqa: F401
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathsnap import config as C
from pathsnap.features import FEATURES

OUT = C.RESULTS / "models"


def main():
    model = joblib.load(C.ARTIFACTS / "classifier_lgbm_tuned.joblib")
    booster = model.booster_
    imp = pd.DataFrame({"feature": FEATURES,
                        "gain": booster.feature_importance("gain"),
                        "splits": booster.feature_importance("split")})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain_pct", ascending=False)
    imp.to_csv(OUT / "feature_importance_tuned_lgbm.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp["feature"][::-1], imp["gain_pct"][::-1])
    ax.set(title="Tuned LightGBM feature importance (gain %)", xlabel="% of total gain")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance_tuned_lgbm.png", dpi=120)
    print(imp[["feature", "gain_pct", "splits"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
