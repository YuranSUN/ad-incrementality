"""Convert the raw Criteo CSV to a compact parquet file (float32 features, int8 flags)."""
import pandas as pd

dtypes = {f"f{i}": "float32" for i in range(12)}
dtypes.update({c: "int8" for c in ["treatment", "conversion", "visit", "exposure"]})
df = pd.read_csv("data/raw/criteo-uplift-v2.1.csv.gz", dtype=dtypes)
df.to_parquet("data/criteo.parquet", index=False)
print(df.shape, round(df.memory_usage().sum() / 1e9, 2), "GB in memory")
