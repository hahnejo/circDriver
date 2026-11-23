import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore, ttest_ind

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

# 1. Load compressed matrices exactly as in Notebook 01
df_circ = pd.read_csv(circ_path, sep="\t")
df_mrna = pd.read_csv(mrna_path)

mrna_sample_cols = [c for c in df_mrna.columns if c not in ["genename", "geneID"]]
circ_sample_cols = df_circ.columns.tolist()
assert set(circ_sample_cols) == set(mrna_sample_cols), "Sample IDs do not match!"

# Parse tissue group ('BT' = Tumor, 'BP' = Para-carcinoma, 'BN' = Normal)
def get_tissue_group(sample_id):
    prefix = str(sample_id).split("_")[0]
    if prefix.endswith("BT"):
        return "Tumor"
    elif prefix.endswith("BP"):
        return "Para_Carcinoma"
    elif prefix.endswith("BN"):
        return "Normal"
    return "Unknown"

sample_meta = pd.DataFrame({
    "Sample_ID": circ_sample_cols,
    "Group": [get_tissue_group(s) for s in circ_sample_cols]
}).set_index("Sample_ID")

# Exact sample ordering from Notebook 01 (critical for identical 5-fold CV splits)
sample_meta = sample_meta.sort_values(by=["Group", "Sample_ID"], ascending=[False, True])
ordered_samples = sample_meta.index.tolist()

print("[01_preprocess] Sample Breakdown:", sample_meta["Group"].value_counts().to_dict())

# 2. Normalize circRNA raw counts -> log2(CPM + 1)
circ_counts = df_circ[ordered_samples].astype(float)
circ_lib_sizes = circ_counts.sum(axis=0)
X_circ_logcpm = np.log2(circ_counts.divide(circ_lib_sizes, axis=1) * 1e6 + 1)
X_circ_logcpm.index.name = "circRNA_id"

# 3. Aggregate duplicate gene symbols in mRNA & log2(x + 1) transform (already FPKM/TPM)
df_mrna_clean = df_mrna.dropna(subset=["genename"]).groupby("genename")[ordered_samples].sum()
mrna_log2 = np.log2(df_mrna_clean.astype(float) + 1)

# 4. Compute continuous target scores (Y) across all 63 samples
signatures = {
    "Hypoxia_Buffa": [
        "VEGFA", "SLC2A1", "PGAM1", "ENO1", "LDHA", "TPI1", "ALDOA",
        "PGK1", "GPI", "P4HA1", "MRPS17", "CDKN3", "ADM", "NDRG1", "TUBB6"
    ],
    "EMT_Invasion": [
        "VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "TWIST1", "ZEB1",
        "ZEB2", "MMP2", "MMP9", "COL1A1", "COL3A1", "TGFB1", "SPARC"
    ],
    "Immune_Exhaustion_TME": [
        "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
        "IDO1", "TGFB1", "IL10", "FOXP3", "CCL2", "CXCL8"
    ]
}

scores_df = sample_meta.copy()
tumor_ids = scores_df[scores_df["Group"] == "Tumor"].index
normal_ids = scores_df[scores_df["Group"] == "Normal"].index

for sig_name, gene_list in signatures.items():
    present_genes = [g for g in gene_list if g in mrna_log2.index]
    sig_matrix = mrna_log2.loc[present_genes]
    sig_matrix = sig_matrix.loc[sig_matrix.std(axis=1) > 0]

    sig_z = pd.DataFrame(
        zscore(sig_matrix.values, axis=1),
        index=sig_matrix.index,
        columns=sig_matrix.columns
    )
    scores_df[sig_name] = sig_z.mean(axis=0)

    t_stat, p_val = ttest_ind(
        scores_df.loc[tumor_ids, sig_name],
        scores_df.loc[normal_ids, sig_name]
    )
    print(f"[01_preprocess] {sig_name} ({len(present_genes)}/{len(gene_list)}): Tumor vs Normal t={t_stat:.2f}, p={p_val:.4e}")

X_circ_logcpm.to_csv(out_circ)
mrna_log2.to_csv(out_mrna)
scores_df.to_csv(out_scores)
with open(out_sigs, "w") as f:
    json.dump(signatures, f, indent=2)