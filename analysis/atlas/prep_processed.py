"""Atlas inputs from the processed DROPS and SoyNAM releases (data/processed/<name>/*.parquet).

Reading parquet needs pyarrow, which the analysis environment lacks, so this one step runs in any
Python with pandas + pyarrow and writes plain files the rest of the atlas reads:
  analysis/atlas/cache/<name>_pheno.csv.gz   genotype_id, environment_id, trait_id, value
                                             (genotype x environment means over plots)
  analysis/atlas/cache/<name>_pc.npz         genotype ids and the first 80 marker principal components
Marker dosages are used as released (DROPS: Beagle-imputed; SoyNAM: founder-inferred fill, see
analysis/crosscrop/results/soynam_DATA_SOURCE.md); remaining gaps take the marker mean.
"""
import numpy as np, pandas as pd
NPC = 80
for name in ("drops", "soynam"):
    ph = pd.read_parquet(f"data/processed/{name}/phenotype.parquet")
    ph["phenotype_value"] = pd.to_numeric(ph.phenotype_value, errors="coerce")
    ge = (ph.dropna(subset=["phenotype_value"])
            .groupby(["genotype_id", "environment_id", "trait_id"], as_index=False).phenotype_value.mean()
            .rename(columns={"phenotype_value": "value"}))
    ge.to_csv(f"analysis/atlas/cache/{name}_pheno.csv.gz", index=False)
    g = pd.read_parquet(f"data/processed/{name}/genotype.parquet")
    ids = pd.Index(sorted(g.genotype_id.unique())); mk = pd.Index(sorted(g.marker_id.unique()))
    X = np.full((len(ids), len(mk)), np.nan, dtype=np.float32)
    X[ids.get_indexer(g.genotype_id), mk.get_indexer(g.marker_id)] = g.allele_dosage.to_numpy(np.float32)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = X[:, np.nanstd(X, axis=0) > 0]
    Z = (X - X.mean(0)) / X.std(0)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    PC = (U[:, :NPC] * S[:NPC]).astype(np.float64)
    np.savez_compressed(f"analysis/atlas/cache/{name}_pc.npz", ids=np.array(ids, dtype=object), PC=PC,
                        explained=(S ** 2 / np.sum(S ** 2))[:NPC].sum())
    print(name, "pheno", ge.shape, ge.trait_id.unique().tolist(), "| markers", X.shape,
          f"PC1-{NPC} explain {(S**2/np.sum(S**2))[:NPC].sum()*100:.1f}%")
