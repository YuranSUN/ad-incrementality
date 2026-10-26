import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/criteo.parquet")
F = [f"f{i}" for i in range(12)]
n = len(df)
print("rows", n, "| missing", int(df.isna().sum().sum()))
print("exact duplicate rows", int(df.duplicated().sum()), "| duplicate feature vectors", int(df.duplicated(F).sum()))

print("\n-- group shares --")
print(df.treatment.value_counts(normalize=True).round(4))

print("\n-- outcome rates by group --")
g = df.groupby("treatment")[["visit", "conversion", "exposure"]].mean()
print(g.round(5))
print("exposure in control:", int(df.loc[df.treatment == 0, "exposure"].sum()))
print("conversion without visit:", int(((df.conversion == 1) & (df.visit == 0)).sum()))

print("\n-- covariate balance (standardized mean difference) --")
t, c = df[df.treatment == 1], df[df.treatment == 0]
smd = (t[F].mean() - c[F].mean()) / np.sqrt((t[F].var() + c[F].var()) / 2)
print(smd.round(4).to_string())

print("\n-- can features predict treatment? (AUC 0.5 = good randomization) --")
s = df.sample(1_000_000, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(s[F], s.treatment, test_size=0.3, random_state=0)
m = HistGradientBoostingClassifier(max_iter=100).fit(Xtr, ytr)
print("AUC", round(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), 4))

print("\n-- unique values per feature --")
print(df[F].nunique().to_string())
