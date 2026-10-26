"""Step 2: ITT and CACE with confidence intervals, unadjusted and covariate-adjusted."""
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
n, p = len(df), Z.mean()
pi = D[Z == 1].mean() - D[Z == 0].mean()
print(f"n={n:,}  treated share={p:.4f}  first stage pi={pi:.5f}")


def diff_in_means(y):
    """Unadjusted ITT with Welch standard error."""
    y1, y0 = y[Z == 1], y[Z == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return est, se


def residualize(v, W):
    """Residual of v after OLS on W (Frisch-Waugh-Lovell)."""
    beta, *_ = np.linalg.lstsq(W, v, rcond=None)
    return v - W @ beta


def iv_robust(y, d, z):
    """Just-identified IV of y on d with instrument z (inputs already residualized); HC0 SE."""
    beta = (z @ y) / (z @ d)
    e = y - beta * d
    se = np.sqrt(np.sum(z**2 * e**2)) / abs(z @ d)
    return beta, se


def adjusted_itt(y, W):
    """ITT from OLS of y on [Z, W] via FWL, with HC0 SE (same as IV with d = z)."""
    return iv_robust(residualize(y, W), residualize(Z, W), residualize(Z, W))


def crossfit_prediction(y, k=2, seed=0):
    """Out-of-fold predicted outcome from features only (no treatment), used as one covariate."""
    pred = np.zeros(n)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        idx = np.random.default_rng(seed).choice(tr, size=min(2_000_000, len(tr)), replace=False)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1).fit(X[idx], y[idx])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


ones = np.ones((n, 1))
Xc = X - X.mean(0)
W_lin = np.hstack([ones, Xc, Xc * (Z - p)[:, None]])  # Lin (2013): covariates + interactions

rows = []
for outcome in ["visit", "conversion"]:
    y = df[outcome].to_numpy(np.float64)
    m_hat = crossfit_prediction(y)
    mc = (m_hat - m_hat.mean())[:, None]
    W_ml = np.hstack([ones, mc, mc * (Z - p)[:, None]])
    for label, W in [("unadjusted", None), ("linear (12 features)", W_lin), ("ML-predicted outcome", W_ml)]:
        if W is None:
            itt, itt_se = diff_in_means(y)
            Wc = ones
        else:
            itt, itt_se = adjusted_itt(y, W)
            Wc = W
        cace, cace_se = iv_robust(residualize(y, Wc), residualize(D, Wc), residualize(Z, Wc))
        rows.append(dict(outcome=outcome, adjustment=label,
                         itt_pp=100 * itt, itt_lo=100 * (itt - 1.96 * itt_se), itt_hi=100 * (itt + 1.96 * itt_se),
                         cace_pp=100 * cace, cace_lo=100 * (cace - 1.96 * cace_se), cace_hi=100 * (cace + 1.96 * cace_se),
                         itt_se=itt_se))
    print(f"{outcome} done ({time.time() - t0:.0f}s)")

res = pd.DataFrame(rows)
base = res[res.adjustment == "unadjusted"].set_index("outcome").itt_se
res["ci_width_vs_unadj"] = res.apply(lambda r: r.itt_se / base[r.outcome], axis=1)
res["equiv_sample_gain"] = 1 / res.ci_width_vs_unadj**2
pd.set_option("display.width", 200)
print(res.drop(columns="itt_se").round(4).to_string(index=False))
res.to_csv("reports/step2_itt_cace.csv", index=False)
print(f"total {time.time() - t0:.0f}s")
