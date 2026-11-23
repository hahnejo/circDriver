configfile: "config/config.yaml"

TARGET = config["target_signature"]
DATA_ID = config["dataset_id"]

rule all:
    input:
        f"Processed/{DATA_ID}_Hypoxia_ElasticNet_Results.png",
        f"Processed/{DATA_ID}_circRNA_vs_Hypoxia_Genes_Heatmap.png",
        f"Processed/{DATA_ID}_host_independence_check.csv",
        f"Processed/{DATA_ID}_circLIFR_ceRNA_partners.csv",
        f"Processed/{DATA_ID}_cerna_triangulation_summary.csv"

rule preprocess:
    input:
        circ=config["raw_data"]["circ_counts"],
        mrna=config["raw_data"]["mrna_counts"]
    output:
        circ_norm=f"Processed/{DATA_ID}_circRNA_log2cpm.csv",
        mrna_norm=f"Processed/{DATA_ID}_mRNA_log2.csv",
        scores=f"Processed/{DATA_ID}_target_scores.csv",
        sigs="Processed/signatures.json"
    conda:
        "envs/circdriver.yaml"
    script:
        "Scripts/01_preprocess.py"

rule train_elasticnet:
    input:
        circ_norm=rules.preprocess.output.circ_norm,
        scores=rules.preprocess.output.scores,
        sigs=rules.preprocess.output.sigs
    output:
        preds=f"Processed/{DATA_ID}_predictions_{TARGET}.csv",
        drivers=f"Processed/{DATA_ID}_top_drivers_{TARGET}.csv"
    params:
        target=TARGET,
        min_cpm=config["filtering"]["min_circ_log2cpm"],
        min_samples=config["filtering"]["min_circ_samples"],
        n_folds=config["model"]["n_folds"],
        k_best=config["model"]["k_best"],
        l1_ratios=config["model"]["l1_ratios"],
        seed=config["model"]["random_state"]
    conda:
        "envs/circdriver.yaml"
    script:
        "Scripts/02_train_elasticnet.py"

rule mechanistic_dissection:
    input:
        circ_norm=rules.preprocess.output.circ_norm,
        mrna_norm=rules.preprocess.output.mrna_norm,
        scores=rules.preprocess.output.scores,
        sigs=rules.preprocess.output.sigs,
        preds=rules.train_elasticnet.output.preds,
        drivers=rules.train_elasticnet.output.drivers
    output:
        cache="Processed/coord_to_gene_cache.json",
        fig_cv=f"Processed/{DATA_ID}_Hypoxia_ElasticNet_Results.png",
        fig_heat=f"Processed/{DATA_ID}_circRNA_vs_Hypoxia_Genes_Heatmap.png",
        indep=f"Processed/{DATA_ID}_host_independence_check.csv"
    params:
        min_freq=config["model"]["min_selection_freq"],
        lead_circ=config["cerna"]["lead_circrna"]
    conda:
        "envs/circdriver.yaml"
    script:
        "Scripts/03_mechanistic_dissection.py"

rule cerna_partner_screen:
    input:
        circ_norm=rules.preprocess.output.circ_norm,
        mrna_norm=rules.preprocess.output.mrna_norm,
        scores=rules.preprocess.output.scores
    output:
        lifr_partners=f"Processed/{DATA_ID}_circLIFR_ceRNA_partners.csv",
        l3mbtl4_partners=f"Processed/{DATA_ID}_circL3MBTL4_ceRNA_partners.csv"
    params:
        lead_circ=config["cerna"]["lead_circrna"],
        sec_circ=config["cerna"]["secondary_circrna"],
        fdr=config["cerna"]["fdr_alpha"]
    conda:
        "envs/circdriver.yaml"
    script:
        "Scripts/04_cerna_partner_screen.py"

rule cerna_triangulation:
    input:
        circ_norm=rules.preprocess.output.circ_norm,
        mrna_norm=rules.preprocess.output.mrna_norm,
        scores=rules.preprocess.output.scores,
        lifr_partners=rules.cerna_partner_screen.output.lifr_partners
    output:
        summary=f"Processed/{DATA_ID}_cerna_triangulation_summary.csv"
    params:
        lead_circ=config["cerna"]["lead_circrna"],
        partner=config["cerna"]["lead_partner"],
        mediator=config["cerna"]["mediator_proxy"],
        n_perm=config["cerna"]["n_permutations"],
        seed=config["model"]["random_state"]
    conda:
        "envs/circdriver.yaml"
    script:
        "Scripts/05_cerna_triangulation.py"