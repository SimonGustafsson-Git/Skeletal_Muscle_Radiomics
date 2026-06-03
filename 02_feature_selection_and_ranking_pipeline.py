# Feature selection and ranking pipeline
# Based on csv files of features extracted subjects using PyRadiomics
# One csv file for individual subject groups, image contrasts, muscle groups and sex
# Subsequent pipeline steps:
# 1) Correlation clustering
# 2) VIF filtering
# 3) Boruta feature selection
# 4) Permutation importance ranking


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------
# Inputs 
# ---------------------------------------------------------------------
FEATURE_DIR_PATH = Path("")

GROUP_A = "" # The subjects with label = 'A'
GROUP_B = ""    # The subjects with label = 'B'
GROUP_A_LABEL = 1 
GROUP_B_LABEL = 0
CONTRAST = "" # The image contrast looked at

# Correlation clustering looks at an averaged correlation matrix for sexes and muscles
CORR_THRESHOLD = 0.1
CORR_MUSCLE_GROUPS = ("A", "B", "C",...)
SEXES = ("F", "M")

# The rest of the script looks at one muscle group at a time
MUSCLE_GROUP = ""  # The muscle looked at
N_SPLITS = 5
LABEL_COL = "Group"

VIF_THRESHOLD = 10

BORUTA_N_ITER = 30
BORUTA_N_ESTIMATORS = 100
BORUTA_MAX_DEPTH = 5
BORUTA_SHADOW_QUANTILE = 0.80
BORUTA_WIN_FRACTION = 0.80

N_REPEATS = 50
TOP_N_PLOT = 10
PLOT_TOP_RANKING = True

META_COLS = {"subject_id", "sex", "muscle_group", "contrast"}
IMAGE_TYPE_ORDER = ["original", "log", "wavelet", "square", "squareroot", "logarithm", "exponential", "gradient", "lbp-2D", "lbp-3D"]


# ---------------------------------------------------------------------
# Feature-name parsing / sorting helpers
# ---------------------------------------------------------------------

def parse_feature_class_from_name(feature_name: str) -> str:
    """Return the PyRadiomics feature class from a feature column name.

    Example: "original_firstorder_Mean" -> "firstorder"
    """
    parts = feature_name.split("_", 2)
    if len(parts) < 2:
        return "unknown"
    return parts[1]


def cluster_representative_sort_key(feature_name: str):
    """Cluster representative priority: Original -> LoG -> Wavelet -> alphabetical."""
    fname = feature_name.lower()
    if fname.startswith("original_"):
        return (0, feature_name)
    if fname.startswith("log_"):
        return (1, feature_name)
    if fname.startswith("wavelet_"):
        return (2, feature_name)
    return (3, feature_name)


def feature_name_sort_key(feature_name: str):
    """Sort features by image-type prefix (Original/LoG/Wavelet/...)."""
    image_type = feature_name.split("_", 1)[0].lower()
    for i, prefix in enumerate(IMAGE_TYPE_ORDER):
        if image_type.startswith(prefix):
            return (i, feature_name)
    return (len(IMAGE_TYPE_ORDER), feature_name)


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (excluding metadata columns)."""
    candidate_cols = [c for c in df.columns if c not in META_COLS]
    return [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]


def build_feature_csv_path(feature_dir: Path, group_name: str, contrast: str, muscle: str, sex: str) -> Path:
    """Build the expected feature CSV path for one stratum."""
    return feature_dir / f"{group_name}_{contrast}_{muscle}_{sex}_PyFeatures.csv"


def load_strata_feature_tables(
    feature_dir: Path,
    group_a: str,
    group_b: str,
    contrast: str,
    muscles: Iterable[str],
    sexes: Iterable[str],
) -> tuple[list[pd.DataFrame], list[str]]:
    """Load all requested (muscle, sex) strata and concatenate group_a + group_b row-wise."""
    strata_dfs: list[pd.DataFrame] = []
    strata_labels: list[str] = []

    for muscle in muscles:
        for sex in sexes:
            file_a = build_feature_csv_path(feature_dir, group_a, contrast, muscle, sex)
            file_b = build_feature_csv_path(feature_dir, group_b, contrast, muscle, sex)

            missing = [str(p) for p in (file_a, file_b) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "Missing required CSV file(s) for "
                    f"{group_a}-{group_b}, contrast={contrast}, muscle={muscle}, sex={sex}:\n"
                    + "\n".join(missing)
                )

            df_a = pd.read_csv(file_a)
            df_b = pd.read_csv(file_b)
            combined = pd.concat([df_a, df_b], ignore_index=True)

            strata_dfs.append(combined)
            strata_labels.append(f"{muscle}_{sex}")
            print(f"Loaded {muscle}_{sex}")

    return strata_dfs, strata_labels


def run_correlation_feature_selection(
    feature_dir: str | Path,
    group_a: str,
    group_b: str,
    contrast: str,
    corr_threshold: float = CORR_THRESHOLD,
    muscles: Iterable[str] = CORR_MUSCLE_GROUPS,
    sexes: Iterable[str] = SEXES,
) -> tuple[list[str], pd.DataFrame]:
    feature_dir = Path(feature_dir).expanduser().resolve()
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    strata_dfs, _strata_labels = load_strata_feature_tables(
        feature_dir=feature_dir,
        group_a=group_a,
        group_b=group_b,
        contrast=contrast,
        muscles=muscles,
        sexes=sexes,
    )
    print(f"Built {len(strata_dfs)} combined strata for correlation.")

    data_all = pd.concat(strata_dfs, ignore_index=True)
    print("Combined all strata shape for constant feature reduction:", data_all.shape)

    feature_cols = get_numeric_feature_columns(data_all)
    data0 = data_all[feature_cols].copy()
    data1 = data0.loc[:, data0.std() > 1e-8]  # exclude globally constant
    feature_cols = data1.columns.tolist()
    print("Non-constant numeric feature count:", len(feature_cols))

    corr_mats: list[pd.DataFrame] = []
    for sdf in strata_dfs:
        xs = sdf.reindex(columns=feature_cols)
        corr = xs.corr().abs()
        corr = corr.reindex(index=feature_cols, columns=feature_cols)
        np.fill_diagonal(corr.values, 1.0)
        corr_mats.append(corr)

    arr = np.stack([m.to_numpy(dtype=float) for m in corr_mats], axis=0)
    avg_arr = np.nanmean(arr, axis=0)
    correlation_matrix = pd.DataFrame(avg_arr, index=feature_cols, columns=feature_cols)
    correlation_matrix = correlation_matrix.fillna(0.0)
    np.fill_diagonal(correlation_matrix.values, 1.0)

    distance_matrix = (1 - correlation_matrix).clip(lower=0)

    feat_class_series = pd.Series(feature_cols, index=feature_cols).apply(parse_feature_class_from_name)
    feature_groups: dict[str, list[str]] = {}
    for col, fc in feat_class_series.items():
        feature_groups.setdefault(fc, []).append(col)

    selected_features: list[str] = []
    for cols in feature_groups.values():
        if len(cols) == 1:
            selected_features.append(cols[0])
            continue

        sub_distance = distance_matrix.loc[cols, cols]
        sub_distance = (sub_distance + sub_distance.T) / 2.0
        np.fill_diagonal(sub_distance.values, 0.0)
        condensed_distance = squareform(sub_distance.values, checks=False)

        z = linkage(condensed_distance, method="centroid")
        clusters = fcluster(z, t=corr_threshold, criterion="distance")
        cluster_series = pd.Series(clusters, index=cols)

        for cid in sorted(cluster_series.unique()):
            members = cluster_series[cluster_series == cid].index.tolist()
            members_sorted = sorted(members, key=cluster_representative_sort_key)
            selected_features.append(members_sorted[0])

    selected_features = sorted(set(selected_features), key=feature_name_sort_key)
    print("Selected representative features:", len(selected_features))
    return selected_features, correlation_matrix


# ---------------------------------------------------------------------
# VIF/Boruta/permutation helpers
# ---------------------------------------------------------------------
def group_features_by_class(feature_columns: Iterable[str]) -> dict[str, list[str]]:
    """Group feature column names by their parsed feature class."""
    groups: dict[str, list[str]] = {}
    for col in feature_columns:
        groups.setdefault(parse_feature_class_from_name(col), []).append(col)
    return groups


def load_combined_one_muscle_two_groups(
    feature_dir: Path,
    group_a: str,
    group_b: str,
    contrast: str,
    muscle_group: str,
    sexes: tuple[str, ...],
) -> pd.DataFrame:
    """Load one muscle group (both sexes) and assign binary labels."""
    parts: list[pd.DataFrame] = []
    for sex in sexes:
        file_a = build_feature_csv_path(feature_dir, group_a, contrast, muscle_group, sex)
        file_b = build_feature_csv_path(feature_dir, group_b, contrast, muscle_group, sex)

        missing = [str(p) for p in (file_a, file_b) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required CSV file(s) for "
                f"{group_a}-{group_b}, contrast={contrast}, muscle={muscle_group}, sex={sex}:\n"
                + "\n".join(missing)
            )

        df_a = pd.read_csv(file_a).copy()
        df_b = pd.read_csv(file_b).copy()

        df_a[LABEL_COL] = GROUP_A_LABEL
        df_b[LABEL_COL] = GROUP_B_LABEL

        parts.append(pd.concat([df_a, df_b], ignore_index=True))
        print(f"Loaded {muscle_group}_{sex}: {group_a} vs {group_b}")

    return pd.concat(parts, ignore_index=True)


def compute_vif(x: pd.DataFrame) -> pd.Series:
    """Compute VIF for each column using linear regression R^2."""
    vif_dict: dict[str, float] = {}
    for col in x.columns:
        y = x[col].values
        x_others = x.drop(columns=[col]).values

        if x_others.shape[1] == 0:
            vif_dict[col] = 1.0
            continue

        model = LinearRegression()
        model.fit(x_others, y)
        r2 = model.score(x_others, y)
        vif_dict[col] = np.inf if r2 >= 0.9999 else 1.0 / (1.0 - r2)

    return pd.Series(vif_dict)


def vif_selection_per_group(df_group: pd.DataFrame) -> list[str]:
    """Iteratively drop the highest-VIF feature until all VIF <= threshold."""
    x = df_group.copy()
    while True:
        if x.shape[1] <= 1:
            break

        vif_values = compute_vif(x)
        if vif_values.max() <= VIF_THRESHOLD:
            break

        feature_to_remove = vif_values.idxmax()
        x = x.drop(columns=[feature_to_remove])

    return x.columns.tolist()


def boruta_simple_rf(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = BORUTA_N_ITER,
    random_state: int = 0,
    n_estimators: int = BORUTA_N_ESTIMATORS,
    max_depth: int = BORUTA_MAX_DEPTH,
) -> tuple[list[str], pd.Series]:
    """Boruta-like selection using shadow features and RF importances."""
    rng = np.random.default_rng(random_state)
    feat_names = np.array(x_train.columns)
    n_feat = x_train.shape[1]

    wins = np.zeros(n_feat, dtype=int)
    x_train_values = x_train.values

    for t in range(n_iter):
        x_shadow = x_train_values.copy()
        for j in range(n_feat):
            rng.shuffle(x_shadow[:, j])

        x_boruta = np.hstack([x_train_values, x_shadow])

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion="gini",
            random_state=random_state + t,
            n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(x_boruta, y_train)

        importances = rf.feature_importances_
        imp_real = importances[:n_feat]
        imp_shadow = importances[n_feat:]
        shadow_thr = np.quantile(imp_shadow, BORUTA_SHADOW_QUANTILE)
        wins += imp_real > shadow_thr

    keep_mask = wins >= int(np.ceil(BORUTA_WIN_FRACTION * n_iter))
    kept_features = feat_names[keep_mask].tolist()
    win_series = pd.Series(wins, index=feat_names).sort_values(ascending=False)
    return kept_features, win_series


def permutation_rank_one_fold(
    x_train_full: pd.DataFrame,
    y_train: np.ndarray,
    x_val_full: pd.DataFrame,
    y_val: np.ndarray,
    feat_list: list[str],
    fold: int,
) -> pd.Series:
    """Compute mean normalized rank per feature from permutation importances."""
    if len(feat_list) == 0:
        return pd.Series(dtype=float)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        criterion="gini",
        random_state=100 + fold,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(x_train_full[feat_list], y_train)

    perm = permutation_importance(
        rf,
        x_val_full[feat_list],
        y_val,
        n_repeats=N_REPEATS,
        random_state=200 + fold,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    imp_arr = np.asarray(perm.importances)
    n_selected = len(feat_list)
    if imp_arr.shape[0] != n_selected and imp_arr.shape[1] == n_selected:
        imp_arr = imp_arr.T

    norm_rank_sum = pd.Series(0.0, index=feat_list, dtype=float)
    for r in range(N_REPEATS):
        imp_r = imp_arr[:, r]
        s_r = pd.Series(imp_r, index=feat_list)
        ranks_r = s_r.rank(ascending=False, method="average")

        if n_selected == 1:
            norm_r = pd.Series({feat_list[0]: 0.0}, dtype=float)
        else:
            norm_r = (ranks_r - 1) / (n_selected - 1)
        norm_rank_sum += norm_r

    return norm_rank_sum / N_REPEATS


def run_vif_boruta_permutation_pipeline(
    data: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LABEL_COL not in data.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in data.")

    y = data[LABEL_COL].astype(int).values
    available_selected = [f for f in selected_features if f in data.columns and f != LABEL_COL]
    if len(available_selected) == 0:
        raise ValueError("No correlation-selected features available in this dataset.")

    x_base = data[available_selected].copy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(skf.split(x_base, y))

    selected_per_fold: list[set[str]] = []
    n_selected_per_fold: list[int] = []
    n_vif_per_fold: list[int] = []
    norm_rank_selected_per_fold: list[pd.Series] = []

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        x_train_full = x_base.iloc[train_idx]
        y_train = y[train_idx]
        x_val_full = x_base.iloc[val_idx]
        y_val = y[val_idx]

        groups = group_features_by_class(x_train_full.columns)

        # Step: VIF reduction per feature class (training fold only)
        fold_vif_features: list[str] = []
        for cols in groups.values():
            fold_vif_features.extend(vif_selection_per_group(x_train_full[cols]))

        fold_vif_features = sorted(set(fold_vif_features), key=feature_name_sort_key)
        n_vif_per_fold.append(len(fold_vif_features))
        x_train = x_train_full[fold_vif_features]

        # Step: Boruta-like selection
        kept, _wins = boruta_simple_rf(
            x_train=x_train,
            y_train=y_train,
            n_iter=BORUTA_N_ITER,
            random_state=1000 + fold,
            n_estimators=BORUTA_N_ESTIMATORS,
            max_depth=BORUTA_MAX_DEPTH,
        )
        kept = sorted(kept, key=feature_name_sort_key)

        selected_per_fold.append(set(kept))
        n_selected_per_fold.append(len(kept))
        print(f"Fold {fold}: VIF kept {len(fold_vif_features)}, Boruta kept {len(kept)}")

        # Step: permutation ranking on Boruta-kept features
        if len(kept) == 0:
            norm_rank_selected_per_fold.append(pd.Series(dtype=float))
            continue

        norm_rank_selected = permutation_rank_one_fold(
            x_train_full=x_train_full,
            y_train=y_train,
            x_val_full=x_val_full,
            y_val=y_val,
            feat_list=kept,
            fold=fold,
        )
        norm_rank_selected_per_fold.append(norm_rank_selected)
        print(f"Fold {fold}: permutation ranking completed for {len(kept)} features.")

    counts = Counter()
    for s in selected_per_fold:
        counts.update(s)

    fixed_feats = sorted(set().union(*selected_per_fold), key=feature_name_sort_key)
    if len(fixed_feats) == 0:
        empty_summary = pd.DataFrame(columns=["mean_norm_rank", "std_norm_rank"])
        empty_rank_df = pd.DataFrame()
        return empty_summary, empty_rank_df

    # Assemble rank_df and summary
    rank_per_fold: list[pd.Series] = []
    for norm_rank_selected in norm_rank_selected_per_fold:
        fold_rank = pd.Series(1.0, index=fixed_feats, dtype=float)
        if not norm_rank_selected.empty:
            fold_rank.loc[norm_rank_selected.index.tolist()] = norm_rank_selected.values
        rank_per_fold.append(fold_rank)

    rank_df = pd.concat(rank_per_fold, axis=1).reindex(fixed_feats)
    summary = pd.DataFrame(
        {"mean_norm_rank": rank_df.mean(axis=1), "std_norm_rank": rank_df.std(axis=1)}
    ).sort_values("mean_norm_rank", ascending=True)

    return summary, rank_df


# Function used in future scripts to get ranked feature summary
def get_ranked_feature_summary(
    feature_dir: Path = FEATURE_DIR_PATH,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
    contrast: str = CONTRAST,
    muscle_group: str = MUSCLE_GROUP,
    sexes: tuple[str, ...] = SEXES,
    corr_muscles: tuple[str, ...] = CORR_MUSCLE_GROUPS,
    corr_sexes: tuple[str, ...] = SEXES,
    corr_threshold: float = CORR_THRESHOLD,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_features, correlation_matrix = run_correlation_feature_selection(
        feature_dir=feature_dir,
        group_a=group_a,
        group_b=group_b,
        contrast=contrast,
        corr_threshold=corr_threshold,
        muscles=corr_muscles,
        sexes=corr_sexes,
    )
    data = load_combined_one_muscle_two_groups(
        feature_dir=feature_dir,
        group_a=group_a,
        group_b=group_b,
        contrast=contrast,
        muscle_group=muscle_group,
        sexes=sexes,
    )
    ranking_summary, rank_df = run_vif_boruta_permutation_pipeline(data, selected_features)
    return selected_features, correlation_matrix, ranking_summary, rank_df


def plot_top_feature_importance(summary: pd.DataFrame, rank_df: pd.DataFrame, top_n: int = TOP_N_PLOT) -> None:
    if summary.empty or rank_df.empty:
        print("No ranked features available to plot.")
        return

    top = summary.head(top_n)
    labels = top.index.tolist()
    means = top["mean_norm_rank"].values
    y_pos = np.arange(len(labels))

    box_data = [rank_df.loc[f].dropna().values for f in labels]

    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.60, right=0.97, top=0.92, bottom=0.12)
    ax.boxplot(
        box_data,
        vert=False,
        labels=labels,
        whis=(0, 100),
        medianprops={"color": "red", "linewidth": 3},
        boxprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"markerfacecolor": "white", "markeredgecolor": "black"},
    )
    ax.scatter(means, y_pos + 1, marker="^", color="blue", zorder=3, s=80)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized rank (0 = best)")
    plt.show()


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def main() -> None:
    selected_features, _corr, ranking_summary, rank_df = get_ranked_feature_summary()
    print("Selected representative feature count:", len(selected_features))
    print("\nTop ranked features:")
    print(ranking_summary.head(TOP_N_PLOT))

    if PLOT_TOP_RANKING:
        plot_top_feature_importance(ranking_summary, rank_df, top_n=TOP_N_PLOT)


if __name__ == "__main__":
    main()
