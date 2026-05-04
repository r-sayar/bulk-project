"""Annotate the per-stimulation CV>1 CSVs with gene symbols."""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
mapping = pd.read_csv(HERE / "ensembl_to_symbol.tsv", sep="\t")
sym = mapping.dropna(subset=["symbol"]).drop_duplicates("ensembl_id").set_index("ensembl_id")["symbol"]

for stim in ["null", "lps", "poly_ic", "seb"]:
    p = HERE / f"gse279480_{stim}_cv_gt1.csv"
    df = pd.read_csv(p)
    df.insert(1, "symbol", df["ensembl_id"].map(sym))
    df.to_csv(p, index=False)
    print(f"  {p.name}: {len(df)} rows, {df['symbol'].notna().sum()} with symbol")

shared = HERE / "gse279480_cv_gt1_shared_all_stims.csv"
df = pd.read_csv(shared)
df.insert(1, "symbol", df["ensembl_id"].map(sym))
df.to_csv(shared, index=False)
print(f"  {shared.name}: {len(df)} rows, {df['symbol'].notna().sum()} with symbol")

print("\nTop 30 highest-CV genes (Null condition):")
null = pd.read_csv(HERE / "gse279480_null_cv_gt1.csv").head(30)
print(null[["symbol", "ensembl_id", "cv", "mean_log2cpm", "mean_cpm", "pct_zero_samples"]].to_string(index=False))

print("\n\nTop 30 highest-CV genes shared across ALL 4 stimulations (sorted by Null CV):")
shared_df = pd.read_csv(HERE / "gse279480_cv_gt1_shared_all_stims.csv")
null_full = pd.read_csv(HERE / "gse279480_null_cv_gt1.csv")
joined = shared_df.merge(null_full[["ensembl_id", "cv", "mean_log2cpm", "pct_zero_samples"]],
                         on="ensembl_id").sort_values("cv", ascending=False).head(30)
print(joined[["symbol", "ensembl_id", "cv", "mean_log2cpm", "pct_zero_samples"]].to_string(index=False))
