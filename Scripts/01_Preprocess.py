import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Resolve inputs/outputs from Snakemake or standalone defaults
if "snakemake" in globals():
    circ_path = snakemake.input.circ
    mrna_path = snakemake.input.mrna
    out_circ = snakemake.output.circ_norm
    out_mrna = snakemake.output.mrna_norm
    out_scores = snakemake.output.scores
    out_sigs = snakemake.output.sigs
else:
    circ_path = "Data/GSE236932_circ_countmatrix.tsv.gz"
    mrna_path = "Data/GSE236932_mRNA_countmatrix.csv.gz"
    out_circ = "Processed/GSE236932_circRNA_log2cpm.csv"
    out_mrna = "Processed/GSE236932_mRNA_log2.csv"
    out_scores = "Processed/GSE236932_target_scores.csv"
    out_sigs = "Processed/signatures.json"

Path(out_circ).parent.mkdir(parents=True, exist_ok=True)

# 1. Load raw matrices
circ_raw = pd.read_csv(circ_path, sep="\t", index_col=0)
mrna_raw = pd.read_csv(mrna_path, index_col=0)

# Harmonize sample columns across circRNA and mRNA matrices
common_samples = [c for c in circ_raw.columns if c in mrna_raw.columns]
circ_raw = circ_raw[common_samples]
mrna_raw = mrna_raw[common_samples]

# 2. Library-size log2(CPM + 1) normalization for circRNAs; log2(x + 1) for mRNAs
lib_sizes = circ_raw.sum(axis=0)
circ_cpm = circ_raw.div(lib_sizes, axis=1) * 1e6
circ_log2cpm = np.log2(circ_cpm + 1.0)
mrna_log2 = np.log2(mrna_raw + 1.0)

# 3. Assign tissue groups from sample prefixes (BT = Tumor, BP = Para_Carcinoma, BN = Normal)
def parse_group(sample_id):
    sid = str(sample_id).upper()
    if sid.startswith("BT"):
        return "Tumor"
    if sid.startswith("BP"):
        return "Para_Carcinoma"
    return "Normal"

groups = [parse_group(s) for s in common_samples]

# 4. Curated pathway signatures & single-sample mean Z-score calculation
signatures = {
    "Hypoxia_Buffa": [
        "VEGFA", "SLC2A1", "PGAM1", "ENO1", "LDHA", "TPI1", "ALDOA",
        "PGK1", "GPI", "P4HA1", "MRPS17", "CDKN3", "ADM", "NDRG1", "TUBB6"
    ],
    "EMT_Invasion": [
        "VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "MMP2", "MMP9"
    ],
    "Immune_Exhaustion_TME": [
        "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "FOXP3"
    ]
}

scores_df = pd.DataFrame({"Group": groups}, index=common_samples)
for sig_name, genes in signatures.items():
    valid_genes = [g for g in genes if g in mrna_log2.index]
    sub = mrna_log2.loc[valid_genes]
    z_sub = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    scores_df[sig_name] = z_sub.mean(axis=0)

# Print Tumor vs Normal separation check
tum = scores_df[scores_df["Group"] == "Tumor"]["Hypoxia_Buffa"]
norm = scores_df[scores_df["Group"] == "Normal"]["Hypoxia_Buffa"]
t_stat, p_val = ttest_ind(tum, norm, equal_var=False)
print(f"[01_preprocess] Aligned {len(common_samples)} samples | Hypoxia Tumor vs Normal t={t_stat:.2f}, p={p_val:.2e}")

circ_log2cpm.to_csv(out_circ)
mrna_log2.to_csv(out_mrna)
scores_df.to_csv(out_scores)
with open(out_sigs, "w") as f:
    json.dump(signatures, f, indent=2)