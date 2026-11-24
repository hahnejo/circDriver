import json
import time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless-safe backend for CLI/Snakemake execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

if "snakemake" in globals():
    circ_path = snakemake.input.circ_norm
    mrna_path = snakemake.input.mrna_norm
    scores_path = snakemake.input.scores
    lifr_partners_path = snakemake.input.lifr_partners
    out_summary = snakemake.output.summary
    out_fig = snakemake.output.fig_cerna
    lead_circ = snakemake.params.lead_circ
    partner_gene = snakemake.params.partner
    mediator_gene = snakemake.params.mediator
    n_perm = snakemake.params.n_perm
    seed = snakemake.params.seed
else:
    circ_path = "Processed/GSE236932_circRNA_log2cpm.csv"
    mrna_path = "Processed/GSE236932_mRNA_log2.csv"
    scores_path = "Processed/GSE236932_target_scores.csv"
    lifr_partners_path = "Processed/GSE236932_circLIFR_ceRNA_partners.csv"
    out_summary = "Processed/GSE236932_cerna_triangulation_summary.csv"
    out_fig = "Processed/GSE236932_ceRNA_Triangulation.png"
    lead_circ = "chr5:38523419|38530666"
    partner_gene = "HIF3A"
    mediator_gene = "MIR205HG"
    n_perm = 10000
    seed = 42

X_circ_logcpm = pd.read_csv(circ_path, index_col=0)
mrna_log2 = pd.read_csv(mrna_path, index_col=0)
scores_df = pd.read_csv(scores_path, index_col=0)
lifr_partners = pd.read_csv(lifr_partners_path, index_col=0)

sample_cols = scores_df.index.tolist()
X_circ_logcpm = X_circ_logcpm[sample_cols]
mrna_log2 = mrna_log2[sample_cols]
mrna_log2 = mrna_log2[~mrna_log2.index.duplicated(keep="first")]

# 1. Local sequence cache so Ensembl API timeouts never interrupt re-runs
SEQ_CACHE_FILE = Path(out_summary).parent / "ensembl_seq_cache.json"
seq_cache = json.load(open(SEQ_CACHE_FILE)) if SEQ_CACHE_FILE.exists() else {}

session = requests.Session()
session.headers.update({"User-Agent": "circDriver-Triangulation/1.0"})

def safe_get(url, content_type="text/plain", retries=5, timeout=45):
    for attempt in range(retries):
        try:
            r = session.get(url, headers={"Content-Type": content_type}, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 3.0)))
            else:
                time.sleep(2.0 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
    return None

def fetch_ensembl_seq(chrom, start, end, strand=-1):
    cache_key = f"region_{chrom}:{start}..{end}:{strand}"
    if cache_key in seq_cache and seq_cache[cache_key]:
        return seq_cache[cache_key]
    url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:{strand}"
    r = safe_get(url, content_type="text/plain")
    seq = r.text.strip().upper() if r else ""
    if seq:
        seq_cache[cache_key] = seq
    return seq

def fetch_true_3utr(gene_symbol):
    cache_key = f"utr3_{gene_symbol}"
    if cache_key in seq_cache and seq_cache[cache_key]:
        return seq_cache[cache_key]
    r1 = safe_get(
        f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?expand=1",
        content_type="application/json"
    )
    if not r1:
        return ""
    transcripts = [t for t in r1.json().get("Transcript", []) if t.get("biotype") == "protein_coding"]
    canonical = [t for t in transcripts if t.get("is_canonical") == 1]
    target_t = canonical[0] if canonical else max(transcripts, key=lambda x: x.get("length", 0))
    enst = target_t["id"]

    r_cdna = safe_get(f"https://rest.ensembl.org/sequence/id/{enst}?type=cdna", content_type="text/plain")
    r_cds = safe_get(f"https://rest.ensembl.org/sequence/id/{enst}?type=cds", content_type="text/plain")
    if not r_cdna or not r_cds:
        return ""
    cdna_seq = r_cdna.text.strip().upper()
    cds_seq = r_cds.text.strip().upper()
    idx = cdna_seq.find(cds_seq)
    utr3 = cdna_seq[idx + len(cds_seq):] if idx != -1 else ""
    if utr3:
        seq_cache[cache_key] = utr3
    return utr3

circ_lifr_seq = fetch_ensembl_seq("5", 38523419, 38530666, strand=-1)
utr_seqs = {}
for g in ["HIF3A", "GAS1", "SCARA5", "LIFR"]:
    utr_seqs[g] = fetch_true_3utr(g)
    time.sleep(0.2)

with open(SEQ_CACHE_FILE, "w") as f:
    json.dump(seq_cache, f)

# 2. Exact 8 data-driven miRNA seeds
data_driven_seeds = {
    "hsa-miR-205-5p (MIR205HG)":      "ATGAAGG",
    "hsa-miR-200b/c/429 (MIR200CHG)": "CAGTATT",
    "hsa-miR-200a/141 (MIR200A)":     "CAGTGTT",
    "hsa-miR-210-3p (MIR210HG)":      "ACGCACA",
    "hsa-miR-34a-5p (MIR34AHG)":      "CACTGCC",
    "hsa-miR-17-5p/20a (MIR17HG)":    "GCACTTT",
    "hsa-miR-145-5p (MIR145)":        "AACTGGA",
    "hsa-miR-1-3p (MIR1-1HG)":        "ACATTCC"
}

# 3. Triplet Sensitivity Correlation (SensCorr) + 10,000-gene Permutation Test
c_vec = X_circ_logcpm.loc[lead_circ].values
m_vec = mrna_log2.loc[mediator_gene].values
t_vec = mrna_log2.loc[partner_gene].values
hyp_vec = scores_df["Hypoxia_Buffa"].values

r_ct, _ = spearmanr(c_vec, t_vec)
r_cm, _ = spearmanr(c_vec, m_vec)
r_tm, _ = spearmanr(t_vec, m_vec)
r_ct_given_m = (r_ct - r_cm * r_tm) / (np.sqrt(1 - r_cm**2) * np.sqrt(1 - r_tm**2))
sens_corr = r_ct - r_ct_given_m

bg_pool = mrna_log2.loc[(mrna_log2 > 1.0).sum(axis=1) >= 20]

np.random.seed(seed)
rand_genes = np.random.choice(bg_pool.index, size=n_perm, replace=True)

null_sens = []
for rg in rand_genes:
    z_vec = bg_pool.loc[rg].values
    r_cz, _ = spearmanr(c_vec, z_vec)
    r_tz, _ = spearmanr(t_vec, z_vec)
    denom = np.sqrt(1 - r_cz**2) * np.sqrt(1 - r_tz**2)
    if denom > 1e-6:
        null_sens.append(r_ct - (r_ct - r_cz * r_tz) / denom)

null_sens = np.array(null_sens)
emp_pval = float(np.mean(null_sens >= sens_corr))

summary_rows = []
for mir, seed_7mer in data_driven_seeds.items():
    summary_rows.append({
        "Candidate_miRNA": mir,
        "circLIFR_7mer_MREs": circ_lifr_seq.count(seed_7mer),
        "circLIFR_8mer_MREs": circ_lifr_seq.count(seed_7mer + "A"),
        "HIF3A_3UTR_7mer_MREs": utr_seqs["HIF3A"].count(seed_7mer),
        "GAS1_3UTR_7mer_MREs": utr_seqs["GAS1"].count(seed_7mer),
        "SCARA5_3UTR_7mer_MREs": utr_seqs["SCARA5"].count(seed_7mer),
        "Triplet_Direct_rho": round(r_ct, 4),
        "Triplet_Partial_rho_given_MIR205HG": round(r_ct_given_m, 4),
        "SensCorr_Delta_rho": round(sens_corr, 4),
        "Null_Mean": round(float(null_sens.mean()), 4),
        "Null_SD": round(float(null_sens.std()), 4),
        "Permutation_p_value": round(emp_pval, 4)
    })

pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

# 4. Figure 3: 2-Panel ceRNA Triangulation Showcase
def vectorized_spearman(mat_df, target_vec):
    t_rank = pd.Series(target_vec).rank().values
    t_centered = t_rank - t_rank.mean()
    m_ranks = mat_df.rank(axis=1).values
    m_centered = m_ranks - m_ranks.mean(axis=1, keepdims=True)
    num = np.dot(m_centered, t_centered)
    den = np.sqrt(np.sum(m_centered**2, axis=1) * np.sum(t_centered**2))
    return np.where(den > 0, num / den, 0.0)

quad_df = pd.DataFrame({
    "rho_circ": vectorized_spearman(bg_pool, c_vec),
    "rho_hyp": vectorized_spearman(bg_pool, hyp_vec)
}, index=bg_pool.index)

is_hit = quad_df.index.isin(lifr_partners.index)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

# Panel A: Genome-Wide ceRNA Partner Quadrant Plot
ax0 = axes[0]
ax0.add_patch(plt.Rectangle((0.5, -0.95), 0.5, 0.65, facecolor="#2980b9", alpha=0.08, zorder=0))
ax0.axhline(-0.3, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.8)
ax0.axvline(0.5, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.8)
ax0.axhline(0, color="#bdc3c7", linestyle=":", linewidth=0.8)
ax0.axvline(0, color="#bdc3c7", linestyle=":", linewidth=0.8)

ax0.scatter(
    quad_df.loc[~is_hit, "rho_circ"], quad_df.loc[~is_hit, "rho_hyp"],
    c="#95a5a6", s=8, alpha=0.18, edgecolors="none", rasterized=True,
    label=f"Background mRNAs (n={int((~is_hit).sum()):,})"
)
ax0.scatter(
    quad_df.loc[is_hit, "rho_circ"], quad_df.loc[is_hit, "rho_hyp"],
    c="#2980b9", s=45, alpha=0.9, edgecolor="k", linewidth=0.5, zorder=3,
    label=f"circLIFR Cascade Hits (FDR < 0.01, n={int(is_hit.sum())})"
)

# Highlight HIF3A distinctly as the lead brake-release effector
if partner_gene in quad_df.index:
    ax0.scatter(
        quad_df.loc[partner_gene, "rho_circ"], quad_df.loc[partner_gene, "rho_hyp"],
        c="#e74c3c", s=110, marker="*", edgecolor="k", linewidth=0.6, zorder=5,
        label=f"Lead Partner ({partner_gene})"
    )

# Clean non-overlapping gene callouts
callout_offsets = {
    "HIF3A":    (0.67, -0.58),
    "LIFR-AS1": (0.30, -0.82),
    "GAS1":     (0.20, -0.71),
    "SCARA5":   (0.67, -0.43),
}
for gene, (tx, ty) in callout_offsets.items():
    if gene in quad_df.index:
        gx, gy = quad_df.loc[gene, "rho_circ"], quad_df.loc[gene, "rho_hyp"]
        ax0.annotate(
            gene, xy=(gx, gy), xytext=(tx, ty),
            fontsize=9.5, fontweight="bold" if gene == partner_gene else "semibold",
            color="#c0392b" if gene == partner_gene else "#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#34495e", lw=0.9, shrinkA=0, shrinkB=4),
            zorder=6
        )

ax0.set_xlim(-0.75, 0.88)
ax0.set_ylim(-0.88, 0.85)
ax0.set_title(
    f"A. Genome-Wide ceRNA Partner Screen ({len(bg_pool):,} mRNAs)\n"
    f"circLIFR Co-Expression vs. Hypoxia Suppression",
    fontweight="bold", fontsize=11
)
ax0.set_xlabel("Spearman ρ with circLIFR (hsa_circ_0072309)", fontweight="bold")
ax0.set_ylabel("Spearman ρ with Hypoxia_Buffa Score", fontweight="bold")
ax0.legend(loc="upper left", frameon=True, fontsize=8.5)
ax0.text(
    0.85, -0.84, "Brake-Release\nQuadrant", ha="right", va="bottom",
    fontsize=8.5, style="italic", color="#2980b9", fontweight="bold"
)

# Panel B: Empirical Permutation Null Distribution (10,000 background transcripts)
ax1 = axes[1]
sns.histplot(
    null_sens, bins=55, stat="density", kde=True,
    color="#5dade2", edgecolor="white", alpha=0.55,
    line_kws={"color": "#1f618d", "lw": 2.0}, ax=ax1
)

# Shade empirical p-value tail region
tail_vals = null_sens[null_sens >= sens_corr]
if len(tail_vals) > 0:
    ax1.axvspan(sens_corr, max(null_sens.max(), sens_corr + 0.04), color="#e74c3c", alpha=0.12)

ax1.axvline(
    null_sens.mean(), color="#34495e", linestyle=":", linewidth=1.6,
    label=f"Null Mean ({null_sens.mean():+.4f} ± {null_sens.std():.4f})"
)
ax1.axvline(
    sens_corr, color="#c0392b", linestyle="--", linewidth=2.2,
    label=f"Observed {mediator_gene} (Δρ = {sens_corr:+.4f})"
)

ax1.text(
    0.97, 0.72,
    f"Triplet: circLIFR ⊣ {mediator_gene} ⊣ {partner_gene}\n"
    f"Direct ρ(circ, {partner_gene}) = {r_ct:+.3f}\n"
    f"Partial ρ(| {mediator_gene}) = {r_ct_given_m:+.3f}\n"
    f"SensCorr Δρ = {sens_corr:+.4f}\n"
    f"Empirical p = {emp_pval:.4f}",
    transform=ax1.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#c0392b", alpha=0.92)
)

ax1.set_title(
    f"B. Empirical Permutation Test for {mediator_gene} Mediation\n"
    f"({n_perm:,} Random Background Transcripts)",
    fontweight="bold", fontsize=11
)
ax1.set_xlabel("Conditional Sensitivity Correlation (SensCorr Δρ)", fontweight="bold")
ax1.set_ylabel("Density", fontweight="bold")
ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

plt.tight_layout()
plt.savefig(out_fig, dpi=300)
plt.close(fig)

print(
    f"[05_cerna_triangulation] Saved figure & table | "
    f"SensCorr=+{sens_corr:.4f} | Null={null_sens.mean():+.4f} +/- {null_sens.std():.4f} | p={emp_pval:.4e}"
)
