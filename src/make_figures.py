"""Figures for the README, built from the CSVs in reports/."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = "reports", "reports/figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, ORANGE, GREEN = "#2563eb", "#9ca3af", "#ea580c", "#16a34a"

# 1. Step 2: adjustment moves the visit effect, not just its interval
s2 = pd.read_csv(f"{R}/step2_itt_cace.csv")
fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
for ax, outcome in zip(axes, ["visit", "conversion"]):
    d = s2[s2.outcome == outcome].reset_index(drop=True)
    y = range(len(d))[::-1]
    ax.errorbar(d.itt_pp, y, xerr=[d.itt_pp - d.itt_lo, d.itt_hi - d.itt_pp], fmt="o", color=BLUE, capsize=3)
    ax.set_yticks(list(y), d.adjustment)
    ax.set_xlabel("ITT (percentage points, 95% CI)")
    ax.set_title(f"{outcome.capitalize()} ITT by adjustment")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_adjustment.png")

# 2. Step 2b: funnel among compliers, with and without the ad
f = pd.read_csv(f"{R}/step2b_funnel.csv").set_index(["adjustment", "quantity"]).est_pct
labels = ["unadjusted", "linear (12 features)", "ML-predicted outcomes"]
short = ["Unadjusted", "Linear", "ML"]
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
w = 0.38
for i, (lab, s) in enumerate(zip(labels, short)):
    axes[0].bar(i - w / 2, f[lab, "visit, without ad (Y0)"], w, color=GREY, label="Without ad" if i == 0 else None)
    axes[0].bar(i + w / 2, f[lab, "visit, with ad (Y1)"], w, color=BLUE, label="With ad" if i == 0 else None)
    axes[1].bar(i - w / 2, f[lab, "conv per visit, without ad"], w, color=GREY)
    axes[1].bar(i + w / 2, f[lab, "conv per visit, with ad"], w, color=BLUE)
    axes[1].plot([i - 0.45, i + 0.45], [f[lab, "conv per incremental visit"]] * 2, color=ORANGE, lw=2,
                 label="Incremental visits" if i == 0 else None)
for ax, t in zip(axes, ["Visit rate of compliers (%)", "Conversions per visit (%)"]):
    ax.set_xticks(range(3), short)
    ax.set_title(t)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_funnel.png")

# 3. Step 3: who to target (uplift curve on the held-out half)
c = pd.read_csv(f"{R}/step3_uplift_curve.csv")
fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
colors = {"response mu1(x)": GREY, "T-learner": ORANGE, "X-learner": BLUE}
names = {"response mu1(x)": "Most likely to respond", "T-learner": "Uplift (T-learner)", "X-learner": "Uplift (X-learner)"}
for ax, outcome in zip(axes, ["visit", "conversion"]):
    d = c[c.outcome == outcome]
    for s, g in d.groupby("score"):
        ax.plot(100 * g.target_share, g.incremental_per_1M, color=colors[s], label=names[s])
    r = d[d.score == "X-learner"]
    ax.plot(100 * r.target_share, r.random_per_1M, color="black", ls="--", lw=1, label="Random")
    ax.set_xlabel("Share of users targeted (%)")
    ax.set_ylabel(f"Incremental {outcome}s per 1M users")
    ax.set_title(f"{outcome.capitalize()}: who to target")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_uplift_curve.png")

# 4. Step 3: uplift deciles are mostly about who can be reached
dd = pd.read_csv(f"{R}/step3_deciles.csv")
d = dd[(dd.outcome == "visit") & (dd.score == "X-learner")]
fig, ax1 = plt.subplots(figsize=(6, 2.8))
ax1.bar(d.dec + 1, 100 * d.itt, color=BLUE, label="Observed visit uplift (pp)")
ax1.set_xlabel("Decile of predicted uplift (1 = highest)")
ax1.set_ylabel("Visit uplift (pp)")
ax2 = ax1.twinx()
ax2.plot(d.dec + 1, 100 * d.exposure, color=ORANGE, marker="o", label="Share exposed (%)")
ax2.set_ylabel("Share exposed (%)")
ax2.spines["top"].set_visible(False)
ax1.set_xticks(range(1, 11))
ax1.set_title("Top uplift decile = users the campaign can actually reach")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_deciles.png")

# 5. Step 1b: row order reveals test-like blocks
b = pd.read_csv(f"{R}/step1b_row_blocks.csv")
fig, ax1 = plt.subplots(figsize=(6.5, 3.0))
ax1.bar(b.chunk + 1, 100 * b.treatment_share, color=GREY, label="Treatment share (%)")
ax1.set_ylabel("Treatment share (%)")
ax1.set_xlabel("Position in file (20 equal chunks)")
ax2 = ax1.twinx()
ax2.plot(b.chunk + 1, 100 * b.visit_rate, color=BLUE, marker="o", label="Visit rate (%)")
ax2.set_ylabel("Visit rate (%)")
ax2.spines["top"].set_visible(False)
ax1.set_xticks(range(1, 21))
ax1.set_title("The released file is stored in test-like blocks")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_row_blocks.png")
print("figures saved to", OUT)
