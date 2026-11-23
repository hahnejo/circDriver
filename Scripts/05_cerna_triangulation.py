import json
import time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if "snakemake" in globals():
    circ_path = snakemake.input.circ_norm
    mrna_path = snakemake.input.mrna_norm
    scores_path = snakemake.input.scores
    out_summary = snakemake.output.summary
    lead_circ = snakemake.params.lead_circ
    partner_gene = snakemake.params.partner
    mediator_gene = snakemake.params.mediator
    n_perm = snakemake.params.n_perm
    seed = snakemake.params.seed
else:
    circ_path = "Processed/GSE236932_circRNA_log2cpm.csv"
    mrna_path = "Processed/GSE236932_mRNA_log2.csv"
    scores_path = "Processed/GSE236932_target_scores.csv"
    out_summary = "Processed/GSE236932_cerna_triangulation_summary.csv"
    lead_circ = "chr5:38523419|38530666"
    partner_gene = "HIF3A"
    mediator_gene = "MIR205HG"
    n_perm = 10000
    seed = 42

X_circ_logcpm = pd.read_csv(circ_path, index_col=0)
mrna_log2 = pd.read_csv(mrna_path, index_col=0)
scores_df = pd.read_csv(scores_path, index_col=0)

sample_cols = scores_df.index.tolist()
X_circ_logcpm = X_circ_logcpm[sample_cols]
mrna_log2 = mrna_log2[sample_cols]

# Local sequence cache so Ensembl API timeouts never interrupt re-runs
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

# Exact 8 data-driven miRNA seeds from Notebook 05
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

# Compute Triplet Sensitivity Correlation (SensCorr) + 10,000-gene Permutation Test
c_vec = X_circ_logcpm.loc[lead_circ].values
m_vec = mrna_log2.loc[mediator_gene].values
t_vec = mrna_log2.loc[partner_gene].values

r_ct, _ = spearmanr(c_vec, t_vec)
r_cm, _ = spearmanr(c_vec, m_vec)
r_tm, _ = spearmanr(t_vec, m_vec)
r_ct_given_m = (r_ct - r_cm * r_tm) / (np.sqrt(1 - r_cm**2) * np.sqrt(1 - r_tm**2))
sens_corr = r_ct - r_ct_given_m

np.random.seed(seed)
bg_pool = mrna_log2.loc[(mrna_log2 > 1.0).sum(axis=1) >= 20]
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
print(f"[05_cerna_triangulation] SensCorr=+{sens_corr:.4f} | Null={null_sens.mean():+.4f} +/- {null_sens.std():.4f} | p={emp_pval:.4e}")