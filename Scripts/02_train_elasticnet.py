import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

if "snakemake" in globals():
    circ_path = snakemake.input.circ_norm
    scores_path = snakemake.input.scores
    out_preds = snakemake.output.preds
    out_drivers = snakemake.output.drivers
    target_sig = snakemake.params.target
    min_cpm = snakemake.params.min_cpm
    min_samples = snakemake.params.min_samples
    n_folds = snakemake.params.n_folds
    k_best = snakemake.params.k_best
    l1_ratios = list(snakemake.params.l1_ratios)
    seed = snakemake.params.seed
else:
    circ_path = "Processed/GSE236932_circRNA_log2cpm.csv"
    scores_path = "Processed/GSE236932_target_scores.csv"
    out_preds = "Processed/GSE236932_predictions_Hypoxia_Buffa.csv"
    out_drivers = "Processed/GSE236932_top_drivers_Hypoxia_Buffa.csv"
    target_sig = "Hypoxia_Buffa"
    min_cpm, min_samples, n_folds, k_best = 1.0, 19, 5, 50
    l1_ratios = [0.3, 0.5, 0.7, 0.9]
    seed = 42

X_circ_logcpm = pd.read_csv(circ_path, index_col=0)
scores_df = pd.read_csv(scores_path, index_col=0)

sample_cols = scores_df.index.tolist()
X_circ_logcpm = X_circ_logcpm[sample_cols]

# Filter for consistently expressed circRNAs
expressed_mask = (X_circ_logcpm >= min_cpm).sum(axis=1) >= min_samples
valid_circs = X_circ_logcpm.loc[expressed_mask]

X = valid_circs.T
y = scores_df[target_sig].values
groups = scores_df["Group"].values

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
y_preds = np.zeros_like(y)
coef_matrix = np.zeros((n_folds, X.shape[1]))

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, groups)):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_regression, k=k_best)),
        ("model", ElasticNetCV(
            l1_ratio=l1_ratios,
            cv=5,
            max_iter=10000,
            random_state=seed
        ))
    ])
    pipe.fit(X.iloc[train_idx], y[train_idx])
    y_preds[test_idx] = pipe.predict(X.iloc[test_idx])

    selected_mask = pipe.named_steps["select"].get_support()
    coef_matrix[fold_idx, selected_mask] = pipe.named_steps["model"].coef_

r2 = r2_score(y, y_preds)
r_p, p_p = pearsonr(y, y_preds)
r_s, p_s = spearmanr(y, y_preds)
mae = mean_absolute_error(y, y_preds)
print(f"[02_train_elasticnet] Out-of-Fold R2={r2:.3f}, Spearman rho={r_s:.3f} (p={p_s:.2e}), MAE={mae:.3f}")

coef_df = pd.DataFrame(coef_matrix, columns=X.columns)
summary_circ = pd.DataFrame({
    "Selection_Freq": (coef_df != 0).mean(axis=0),
    "Mean_Coef": coef_df.mean(axis=0),
    "Abs_Mean_Coef": coef_df.mean(axis=0).abs()
})
summary_circ = summary_circ[summary_circ["Selection_Freq"] > 0].sort_values(
    by=["Selection_Freq", "Abs_Mean_Coef"], ascending=False
)

eval_df = pd.DataFrame({
    "Group": groups,
    "Actual_Hypoxia": y,
    "Predicted_Hypoxia": y_preds
}, index=sample_cols)

eval_df.to_csv(out_preds)
summary_circ.to_csv(out_drivers)