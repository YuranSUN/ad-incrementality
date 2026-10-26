"""Step 1b: the released file is stored in blocks with very different treatment shares and baselines.

Splits the original row order into 20 equal chunks and reports treatment share, visit, conversion and
exposure rates. Large jumps between chunks (some 100% treatment, some 0%) are consistent with several
tests concatenated in the file, the pooling behind the Step 2 imbalance. Test IDs are not released,
so this is supporting evidence, not proof.
"""
import numpy as np, pandas as pd

df = pd.read_parquet("data/criteo.parquet", columns=["treatment", "exposure", "visit", "conversion"])
df["chunk"] = np.arange(len(df)) * 20 // len(df)
out = df.groupby("chunk").agg(treatment_share=("treatment", "mean"), visit_rate=("visit", "mean"),
                              conversion_rate=("conversion", "mean"), exposure_rate=("exposure", "mean"),
                              n=("treatment", "size"))
print((out.drop(columns="n") * 100).round(3).to_string())
out.to_csv("reports/step1b_row_blocks.csv")
