"""Step 2b: post-ad funnel among compliers (visit -> conversion), unadjusted and covariate-adjusted.

Complier means come from IV with the same covariates as Step 2:
  complier mean of Y with the ad    = IV of Y*D on D
  complier mean of Y without the ad = IV of Y*(1-D) on (1-D)
Funnel ratios (conversions per visit) are also IV estimates, so they get robust SEs directly:
  with the ad:          IV of conv*D on visit*D
  without the ad:       IV of conv*(1-D) on visit*(1-D)
  incremental (marginal): IV of conv on visit   (= conversion ITT / visit ITT)
Every conversion in the data has a visit, so conversions per visit = conversion rate among visitors.
"""
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

t0 = time.time()
df = pd.read_parquet("data/criteo.parquet")
F = [f"f{i}" for i in range(12)]
Z = df.treatment.to_numpy(np.float64)
D = df.exposure.to_numpy(np.float64)
X = df[F].to_numpy(np.float64)
V = df.visit.to_numpy(np.float64)
C = df.conversion.to_numpy(np.float64)
n, p = len(df), Z.mean()


def residualize(v, W):
    beta, *_ = np.linalg.lstsq(W, v, rcond=None)
    return v - W @ beta


def iv_robust(y, d, z):
    beta = (z @ y) / (z @ d)
    e = y - beta * d
    se = np.sqrt(np.sum(z**2 * e**2)) / abs(z @ d)
    return beta, se


def crossfit_prediction(y, k=2, seed=0):
    pred = np.zeros(n)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        idx = np.random.default_rng(seed).choice(tr, size=min(2_000_000, len(tr)), replace=False)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1).fit(X[idx], y[idx])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


ones = np.ones((n, 1))
Zc = (Z - p)[:, None]
Xc = X - X.mean(0)
mv = crossfit_prediction(V)
mcv = crossfit_prediction(C)
M = np.column_stack([mv - mv.mean(), mcv - mcv.mean()])
designs = {
    "unadjusted": ones,
    "linear (12 features)": np.hstack([ones, Xc, Xc * Zc]),
    # both predicted outcomes, so visit and conversion are adjusted with the same covariates
    "ML-predicted outcomes": np.hstack([ones, M, M * Zc]),
}
print(f"predictions done ({time.time() - t0:.0f}s)")

quantities = {
    # name: (outcome, "treatment" for IV)
    "visit, with ad (Y1)": (V * D, D),
    "visit, without ad (Y0)": (V * (1 - D), 1 - D),
    "visit, CACE": (V, D),
    "conversion, with ad (Y1)": (C * D, D),
    "conversion, without ad (Y0)": (C * (1 - D), 1 - D),
    "conversion, CACE": (C, D),
    "conv per visit, with ad": (C * D, V * D),
    "conv per visit, without ad": (C * (1 - D), V * (1 - D)),
    "conv per incremental visit": (C, V),
}

rows = []
for label, W in designs.items():
    z = residualize(Z, W)
    for q, (y, d) in quantities.items():
        b, se = iv_robust(residualize(y, W), residualize(d, W), z)
        rows.append(dict(adjustment=label, quantity=q, est_pct=100 * b,
                         lo=100 * (b - 1.96 * se), hi=100 * (b + 1.96 * se), se_pct=100 * se))
    print(f"{label} done ({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv("reports/step2b_funnel.csv", index=False)
print(f"total {time.time() - t0:.0f}s")
