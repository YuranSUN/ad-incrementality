"""Step 3: who does the ad work for? Uplift models vs response targeting, evaluated on a held-out half.

Train half: fit models. Test half: rank users by each score and evaluate with doubly robust (AIPW) scores,
which stay valid when treatment probability varies slightly with the features (the Step 2 imbalance).
  T-learner:  tau(x) = mu1(x) - mu0(x)
  X-learner:  impute effects with cross-fitted mu's, regress them, combine with the propensity e(x)
  Response:   mu1(x), "target who is most likely to visit when shown the campaign" (common practice)
Outputs: reports/step3_uplift_curve.csv, reports/step3_deciles.csv, reports/step3_summary.csv
"""
import time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

t0 = time.time()
rng = np.random.default_rng(0)
df = pd.read_parquet("data/criteo.parquet")
F = [f"f{i}" for i in range(12)]
X = df[F].to_numpy(np.float32)
Z = df.treatment.to_numpy(np.int8)
D = df.exposure.to_numpy(np.float64)
Y = {"visit": df.visit.to_numpy(np.float64), "conversion": df.conversion.to_numpy(np.float64)}
n = len(df)
test = rng.random(n) < 0.5
tr, te = np.where(~test)[0], rng.permutation(np.where(test)[0])  # shuffle: the file is stored in test-like blocks, so ties must not follow row order
CAP = 2_000_000


def sub(idx, k=CAP):
    return idx if len(idx) <= k else rng.choice(idx, k, replace=False)


def clf(idx, y):
    i = sub(idx)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1).fit(X[i], y[i])


def reg(idx, y):
    i = sub(idx)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1).fit(X[i], y[i])


def proba(m, idx):
    return m.predict_proba(X[idx])[:, 1]


# propensity e(x) = P(T=1 | x): fit on train, used for X-learner weights and AIPW scores on test
e_model = clf(tr, Z)
e_te = np.clip(proba(e_model, te), 0.01, 0.99)
print(f"propensity on test: mean {e_te.mean():.4f}, sd {e_te.std():.4f}  ({time.time() - t0:.0f}s)")

summary, curves, deciles = [], [], []
for name, y in Y.items():
    t1, t0_ = tr[Z[tr] == 1], tr[Z[tr] == 0]
    mu1, mu0 = clf(t1, y), clf(t0_, y)
    m1, m0 = proba(mu1, te), proba(mu0, te)
    scores = {"response mu1(x)": m1, "T-learner": m1 - m0}

    # X-learner with cross-fitted imputations inside the training half
    imp = np.zeros(len(tr))
    for a, b in KFold(2, shuffle=True, random_state=1).split(tr):
        ia, ib = tr[a], tr[b]
        g1 = clf(ia[Z[ia] == 1], y); g0 = clf(ia[Z[ia] == 0], y)
        zb = Z[ib] == 1
        imp[b] = np.where(zb, y[ib] - proba(g0, ib), proba(g1, ib) - y[ib])
    imp_full = np.zeros(n)
    imp_full[tr] = imp
    tau1 = reg(t1, imp_full)
    tau0 = reg(t0_, imp_full)
    e_x = e_te
    scores["X-learner"] = e_x * tau0.predict(X[te]) + (1 - e_x) * tau1.predict(X[te])

    # AIPW pseudo-outcome on test (ITT scale) and the same for exposure (first stage)
    yt, zt = y[te], Z[te]
    psi = m1 - m0 + zt * (yt - m1) / e_te - (1 - zt) * (yt - m0) / (1 - e_te)
    d1 = proba(clf(t1, D), te)  # exposure is 0 in control, so mu0 for D is 0
    psi_d = d1 + zt * (D[te] - d1) / e_te
    ate = psi.mean()
    N = len(te)
    print(f"{name}: test AIPW ITT {100 * ate:.4f} pp ({time.time() - t0:.0f}s)")

    for s_name, s in scores.items():
        order = np.argsort(-s)
        cum = np.cumsum(psi[order])
        for q in np.arange(0.05, 1.0001, 0.05):
            k = int(round(q * N))
            curves.append(dict(outcome=name, score=s_name, target_share=round(q, 2),
                               incremental_per_1M=1e6 * cum[k - 1] / N,
                               random_per_1M=1e6 * q * ate))
        # gain at 10% and 20% targeting with bootstrap-free SE (influence of the top-k mean)
        for q in (0.1, 0.2):
            k = int(round(q * N)); top = psi[order[:k]]
            share_cap = top.sum() / psi.sum()
            summary.append(dict(outcome=name, score=s_name, target_share=q,
                                uplift_top_pp=100 * top.mean(), se_pp=100 * top.std() / np.sqrt(k),
                                share_of_all_incremental=share_cap, lift_vs_random=top.mean() / ate))
        # deciles: observed uplift, exposure rate, implied CACE, baseline visit rate
        dec = pd.qcut(pd.Series(s).rank(method="first", ascending=False), 10, labels=False)
        g = pd.DataFrame(dict(dec=dec, psi=psi, psi_d=psi_d, base=m0, pred=s)).groupby("dec")
        out = g.agg(pred=("pred", "mean"), itt=("psi", "mean"), itt_sd=("psi", "std"), n=("psi", "size"),
                    exposure=("psi_d", "mean"), baseline=("base", "mean")).reset_index()
        out["itt_se"] = out.itt_sd / np.sqrt(out.n)
        out["cace"] = out.itt / out.exposure
        out.insert(0, "score", s_name); out.insert(0, "outcome", name)
        deciles.append(out.drop(columns="itt_sd"))

    # stability: rank agreement of T-learner scores from two independent training halves
    if name == "visit":
        a, b = np.array_split(rng.permutation(tr), 2)
        ta = proba(clf(a[Z[a] == 1], y), te) - proba(clf(a[Z[a] == 0], y), te)
        tb = proba(clf(b[Z[b] == 1], y), te) - proba(clf(b[Z[b] == 0], y), te)
        rho = pd.Series(ta).rank().corr(pd.Series(tb).rank())
        top_a, top_b = ta >= np.quantile(ta, 0.9), tb >= np.quantile(tb, 0.9)
        overlap = (top_a & top_b).sum() / top_a.sum()
        summary.append(dict(outcome=name, score="stability (T-learner, two halves)", target_share=0.1,
                            uplift_top_pp=np.nan, se_pp=np.nan, share_of_all_incremental=overlap, lift_vs_random=rho))
        rr = pd.Series(m1).rank().corr(pd.Series(m1 - m0).rank())
        summary.append(dict(outcome=name, score="rank corr: response vs T-learner", target_share=np.nan,
                            uplift_top_pp=np.nan, se_pp=np.nan, share_of_all_incremental=np.nan, lift_vs_random=rr))

pd.set_option("display.width", 220)
S = pd.DataFrame(summary); C = pd.DataFrame(curves); Dd = pd.concat(deciles)
S.to_csv("reports/step3_summary.csv", index=False)
C.to_csv("reports/step3_uplift_curve.csv", index=False)
Dd.to_csv("reports/step3_deciles.csv", index=False)
print(S.round(4).to_string(index=False))
print(Dd[Dd.outcome == "visit"].round(5).to_string(index=False))
print(Dd[Dd.outcome == "conversion"].round(6).to_string(index=False))
print(f"total {time.time() - t0:.0f}s")
