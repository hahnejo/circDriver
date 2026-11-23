import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

if "snakemake" in globals():
    circ_path = snakemake.input.circ_norm
    mrna_path = snakemake.input.mrna_norm
    scores_path = snakemake.input.scores
    out_lifr = snakemake.output.lifr_partners
    out_l3mbtl4 = snakemake.output.l3mbtl4_partners
    lead_circ = snakemake.params.lead_circ
    sec_circ = snakemake.params.sec_circ
    fdr_alpha = snakemake.params.fdr
else:
    circ_path = "Processed/GSE236932_circRNA_log2cpm.csv"
    mrna_path = "Processed/GSE236932_mRNA_log2.csv"
    scores_path = "Processed/GSE236932_target_scores.csv"
    out_lifr = "Processed/GSE236932_circLIFR_ceRNA_partners.csv"
    out_l3mbtl4 = "Processed/GSE236932_circL3MBTL4_ceRNA_partners.csv"
    lead_circ = "chr5:38523419|38530666"
    sec_circ = "chr18:6213149|6312056"
    fdr_alpha = 0.01

X_circ_logcpm = pd.read_csv(circ_path, index_col=0)
mrna_log2 = pd.read_csv(mrna_path, index_col=0)
scores_df = pd.read_csv(scores_path, index_col=0)

sample_cols = scores_df.index.tolist()
X_circ_logcpm = X_circ_logcpm[sample_cols]
mrna_log2 = mrna_log2[sample_cols]

expressed_mrnas = mrna_log2.loc[(mrna_log2 > 1.0).sum(axis=1) >= 20]
hyp_vec = scores_df["Hypoxia_Buffa"].values
hyp_rhos = [spearmanr(expressed_mrnas.iloc[i].values, hyp_vec)[0] for i in range(len(expressed_mrnas))]

def screen_cerna_partners(circ_id):
    c_vec = X_circ_logcpm.loc[circ_id].values
    c_rank = pd.Series(c_vec).rank().values
    c_rank_centered = c_rank - c_rank.mean()

    m_ranks = expressed_mrnas.rank(axis=1).values
    m_ranks_centered = m_ranks - m_ranks.mean(axis=1, keepdims=True)
    rhos = np.dot(m_ranks_centered, c_rank_centered) / np.sqrt(
        np.sum(m_ranks_centered**2, axis=1) * np.sum(c_rank_centered**2)
    )

    pvals = [spearmanr(c_vec, expressed_mrnas.iloc[i].values)[1] for i in range(len(expressed_mrnas))]
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")

    res = pd.DataFrame({
        "rho_with_circ": rhos,
        "FDR_qval": qvals,
        "rho_with_Hypoxia": hyp_rhos
    }, index=expressed_mrnas.index)
    res.index.name = "mRNA_Gene"

    cascade_hits = res[
        (res["rho_with_circ"] > 0.5) & (res["FDR_qval"] < fdr_alpha) & (res["rho_with_Hypoxia"] < -0.3)
    ].sort_values(by="rho_with_circ", ascending=False)
    return cascade_hits

lifr_partners = screen_cerna_partners(lead_circ)
l3mbtl4_partners = screen_cerna_partners(sec_circ)

lifr_partners.to_csv(out_lifr)
l3mbtl4_partners.to_csv(out_l3mbtl4)
print(f"[04_cerna_partner_screen] Saved {len(lifr_partners)} circLIFR partners and {len(l3mbtl4_partners)} circL3MBTL4 partners.")