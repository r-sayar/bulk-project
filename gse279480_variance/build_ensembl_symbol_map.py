"""Build a local Ensembl -> gene symbol map using mygene.info.

Reads the gene IDs from GSE279480_P441_genecounts.csv.gz and queries mygene
in batches. Caches the result to ensembl_to_symbol.tsv so downstream scripts
can join without re-querying.
"""

from pathlib import Path
import pandas as pd
import mygene

HERE = Path(__file__).parent
COUNTS_CSV = HERE.parent / "data/GSE279480/GSE279480_P441_genecounts.csv.gz"
OUT = HERE / "ensembl_to_symbol.tsv"

ids = pd.read_csv(COUNTS_CSV, index_col=0, usecols=[0]).index.tolist()
print(f"Querying mygene for {len(ids):,} Ensembl IDs...")

mg = mygene.MyGeneInfo()
res = mg.querymany(
    ids, scopes="ensembl.gene", fields="symbol,name,type_of_gene",
    species="human", as_dataframe=True, returnall=False,
)
res = res.reset_index().rename(columns={"query": "ensembl_id"})
res = res[["ensembl_id", "symbol", "name", "type_of_gene"]]
res.to_csv(OUT, sep="\t", index=False)
print(f"  {len(res):,} rows  ->  {OUT}")
print(f"  with symbol: {res['symbol'].notna().sum():,}")
print(res.head())
