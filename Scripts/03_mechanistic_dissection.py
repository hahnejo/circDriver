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
from sklearn.metrics import r2_score

if "snakemake" in globals():
    circ_path = snakemake.input.circ_norm
    mrna_path = snakemake.input.mrna_norm
    scores_path = snakemake.input.scores
    sigs_path = snakemake.input.sigs
    preds_path = snakemake.input.preds
    drivers_path = snakemake.input.drivers
    out_cache = snakemake.params.cache
    out_fig_cv = snakemake.output.fig_cv
    out_fig_heat = snakemake.output.fig_heat
    out_indep = snakemake.output.indep
    min_freq = snakemake.params.min_freq
    lead_circ = snakemake.params.lead_circ
else:
    circ_path = "Processed/GSE236932_circRNA_log2cpm.csv"
    mrna_path = "Processed/GSE236932_mRNA_log2.csv"
    scores_path = "Processed/GSE236932_target_scores.csv"
    sigs_path = "Processed/signatures.json"
    preds_path = "Processed/GSE236932_predictions_Hypoxia_Buffa.csv"
    drivers_path = "Processed/GSE236932_top_drivers_Hypoxia_Buffa.csv"
    out_cache = "Processed/coord_to_gene_cache.json"
    out_fig_cv = "Processed/GSE236932_Hypoxia_ElasticNet_Results.png"
    out_fig_heat = "Processed/GSE236932_circRNA_vs_Hypoxia_Genes_Heatmap.png"
    out_indep = "Processed/GSE236932_host_independence_check.csv"
    min_freq = 0.6
    lead_circ = "chr5:38523419|38530666"

X_circ_logcpm = pd.read_csv(circ_path, index_col=0)
mrna_log2 = pd.read_csv(mrna_path, index_col=0)
scores_df = pd.read_csv(scores_path, index_col=0)
eval_df = pd.read_csv(preds_path, index_col=0)
summary_circ = pd.read_csv(drivers_path, index_col=0)

with open(sigs_path) as f:
    signatures = json.load(f)

sample_cols = scores_df.index.tolist()
X_circ_logcpm = X_circ_logcpm[sample_cols]
mrna_log2 = mrna_log2[sample_cols]

# 1. Persistent cache + Multi-provider GRCh38 gene mapper (Ensembl -> MyGene.info -> UCSC)
cache_file = Path(out_cache)
gene_cache = json.load(open(cache_file)) if cache_file.exists() else {}

session = requests.Session()
session.headers.update({"Accept": "application/json", "User-Agent": "circDriver-Mapper/2.0"})

def query_ensembl(chrom, start, end, retries=3, timeout=30):
    url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{start}-{end}?feature=gene"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                genes = [g.get("external_name") for g in r.json() if g.get("external_name")]
                return ",".join(dict.fromkeys(genes)) if genes else "Intergenic"
            elif r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2.0)))
            else:
                time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None

def query_mygene(chrom, start, end, timeout=20):
    url = f"https://mygene.info/v3/query?q=chr{chrom}:{start}-{end}&species=human&fields=symbol,type_of_gene&size=50"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            coding = [h["symbol"] for h in hits if h.get("symbol") and h.get("type_of_gene") == "protein-coding"]
            all_syms = [h["symbol"] for h in hits if h.get("symbol")]
            genes = coding if coding else all_syms
            return ",".join(dict.fromkeys(genes)) if genes else "Intergenic"
    except requests.RequestException:
        pass
    return None

def query_ucsc(chrom, start, end, timeout=20):
    url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=ncbiRefSeqCurated;chrom=chr{chrom};start={start};end={end}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 200:
            items = r.json().get("ncbiRefSeqCurated", [])
            genes = [item.get("name2") for item in items if item.get("name2")]
            return ",".join(dict.fromkeys(genes)) if genes else "Intergenic"
    except requests.RequestException:
        pass
    return None

def map_coords_to_gene(coord_str):
    cached = gene_cache.get(coord_str)
    if cached and cached != "Unknown":
        return cached

    chrom, pos = coord_str.split(":")
    chrom_clean = chrom.replace("chr", "")
    start, end = pos.split("|")

    # Try Ensembl first, then MyGene.info, then UCSC Genome Browser
    for provider_name, fn in [("Ensembl", query_ensembl), ("MyGene.info", query_mygene), ("UCSC", query_ucsc)]:
        res = fn(chrom_clean, start, end)
        if res is not None:
            gene_cache[coord_str] = res
            time.sleep(0.15)
            return res
        print(f"[03_mechanistic_dissection] {provider_name} timed out for {coord_str}, trying fallback...")

    raise RuntimeError(f"All annotation APIs failed for {coord_str}. Check internet connection and re-run.")

top_neg_ids = summary_circ[summary_circ["Selection_Freq"] >= min_freq].index.tolist()
top_pos_ids = summary_circ[summary_circ["Mean_Coef"] > 0].index.tolist()
selected_circ_ids = top_neg_ids + top_pos_ids

for cid in selected_circ_ids:
    map_coords_to_gene(cid)

with open(cache_file, "w") as f:
    json.dump(gene_cache, f, indent=2)

# 2. Figure 1: Out-of-Fold Prediction & Lead Driver Scatter
r2 = r2_score(eval_df["Actual_Hypoxia"], eval_df["Predicted_Hypoxia"])
r_s, p_s = spearmanr(eval_df["Actual_Hypoxia"], eval_df["Predicted_Hypoxia"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
palette = {"Tumor": "#e74c3c", "Para_Carcinoma": "#f39c12", "Normal": "#2980b9"}

sns.scatterplot(
    data=eval_df, x="Actual_Hypoxia", y="Predicted_Hypoxia",
    hue="Group", palette=palette, s=80, alpha=0.85, edgecolor="k", ax=axes[0]
)
sns.regplot(
    data=eval_df, x="Actual_Hypoxia", y="Predicted_Hypoxia",
    scatter=False, color="black", line_kws={"linestyle": "--", "alpha": 0.7}, ax=axes[0]
)
axes[0].set_title(f"5-Fold CV Out-of-Fold Hypoxia Prediction (n=63)\nSpearman ρ = {r_s:.3f} (p = {p_s:.2e}), R² = {r2:.3f}", fontweight="bold")
axes[0].set_xlabel("Actual Hypoxia Score (Buffa Z-score)")
axes[0].set_ylabel("ElasticNet Predicted Hypoxia Score")

eval_df["circLIFR_log2CPM"] = X_circ_logcpm.loc[lead_circ, eval_df.index]
sns.scatterplot(
    data=eval_df, x="circLIFR_log2CPM", y="Actual_Hypoxia",
    hue="Group", palette=palette, s=80, alpha=0.85, edgecolor="k", ax=axes[1]
)
sns.regplot(
    data=eval_df, x="circLIFR_log2CPM", y="Actual_Hypoxia",
    scatter=False, color="#c0392b", line_kws={"linestyle": "--"}, ax=axes[1]
)
rho_lifr, p_lifr = spearmanr(eval_df["circLIFR_log2CPM"], eval_df["Actual_Hypoxia"])
axes[1].set_title(f"hsa_circ_0072309 (circLIFR) vs. Hypoxia Score\nSpearman ρ = {rho_lifr:.3f} (p = {p_lifr:.2e})", fontweight="bold")
axes[1].set_xlabel("hsa_circ_0072309 (circLIFR) Expression [log2(CPM+1)]")
axes[1].set_ylabel("Actual Hypoxia Score (Buffa Z-score)")

plt.tight_layout()
plt.savefig(out_fig_cv, dpi=300)
plt.close(fig)

# 3. Figure 2: Spearman Heatmap vs. Individual Buffa Hypoxia Effector Genes
circ_labels = []
for cid in selected_circ_ids:
    host = gene_cache.get(cid, "Unknown").split(",")[-1]
    short_coord = cid.split("|")[0]
    direction = "[+]" if summary_circ.loc[cid, "Mean_Coef"] > 0 else "[-]"
    circ_labels.append(f"{direction} circ{host} ({short_coord})")

hypoxia_genes = [g for g in signatures["Hypoxia_Buffa"] if g in mrna_log2.index]
rho_matrix = pd.DataFrame(index=circ_labels, columns=hypoxia_genes, dtype=float)
pval_matrix = pd.DataFrame(index=circ_labels, columns=hypoxia_genes, dtype=float)

for cid, label in zip(selected_circ_ids, circ_labels):
    c_expr = X_circ_logcpm.loc[cid]
    for h_gene in hypoxia_genes:
        rho, pval = spearmanr(c_expr, mrna_log2.loc[h_gene])
        rho_matrix.loc[label, h_gene] = rho
        pval_matrix.loc[label, h_gene] = pval

fig2, ax2 = plt.subplots(figsize=(13, 6.5))
annot_matrix = rho_matrix.round(2).astype(str)
for r in range(rho_matrix.shape[0]):
    for c in range(rho_matrix.shape[1]):
        p = pval_matrix.iloc[r, c]
        val = f"{rho_matrix.iloc[r, c]:.2f}"
        annot_matrix.iloc[r, c] = val + ("**" if p < 0.001 else ("*" if p < 0.05 else ""))

sns.heatmap(
    rho_matrix, cmap="vlag", center=0, vmin=-0.65, vmax=0.65,
    annot=annot_matrix, fmt="", cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
    linewidths=0.5, annot_kws={"size": 8.5}, ax=ax2
)
ax2.set_title("Spearman Correlation: Top ElasticNet circRNA Drivers vs. Individual Buffa Hypoxia Effector Genes (n=63)\n(* p < 0.05, ** p < 0.001)", fontweight="bold", fontsize=11)
ax2.set_xlabel("Hypoxia_Buffa Signature Effector mRNAs", fontweight="bold")
ax2.set_ylabel("Candidate circRNA Drivers", fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(out_fig_heat, dpi=300)
plt.close(fig2)

# 4. Circular vs. Linear Host mRNA Partial Correlation Independence Check
check_records = []
for cid in selected_circ_ids:
    hosts = [h.strip() for h in gene_cache.get(cid, "").split(",") if h.strip() in mrna_log2.index]
    c_expr = X_circ_logcpm.loc[cid]
    rho_circ_hyp, _ = spearmanr(c_expr, scores_df["Hypoxia_Buffa"])
    if hosts:
        host_sym = hosts[-1]
        h_expr = mrna_log2.loc[host_sym]
        rho_circ_host, _ = spearmanr(c_expr, h_expr)
        rho_host_hyp, _ = spearmanr(h_expr, scores_df["Hypoxia_Buffa"])
        p_rho = (rho_circ_hyp - rho_circ_host * rho_host_hyp) / (
            np.sqrt(1 - rho_circ_host**2) * np.sqrt(1 - rho_host_hyp**2)
        )
        check_records.append({
            "circRNA_id": cid,
            "Host_Gene": host_sym,
            "rho(circ, Hypoxia)": round(rho_circ_hyp, 3),
            "rho(circ, Linear_Host)": round(rho_circ_host, 3),
            "rho(Linear_Host, Hypoxia)": round(rho_host_hyp, 3),
            "Partial_rho(circ|Host)": round(p_rho, 3)
        })

pd.DataFrame(check_records).to_csv(out_indep, index=False)
print("[03_mechanistic_dissection] Saved heatmaps and host-gene independence table.")