# Statistical significance test for one selected case
# Uses Mann-Whitney U for the top 10 ranked features from script 02.


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
FEATURE_DIR = Path("")

GROUP_A = "" # The subjects with label = 'A'
GROUP_B = "" # The subjects with label = 'B'
CONTRAST = ""
MUSCLE_GROUP = ""
CORR_MUSCLES = ("A", "B", "C",...) # Correlation clustering looks at an averaged correlation matrix for sexes and muscles
SEXES = ("F", "M")
N_FEATURES_TO_TEST = 10


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_feature_selection_module() -> ModuleType:
    module_path = Path(__file__).with_name("02_feature_selection_and_ranking_pipeline.py")
    spec = importlib.util.spec_from_file_location("feature_selection_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_group_data(group_name: str) -> pd.DataFrame:
    parts = []
    for sex in SEXES:
        csv_path = FEATURE_DIR / f"{group_name}_{CONTRAST}_{MUSCLE_GROUP}_{sex}_PyFeatures.csv"
        parts.append(pd.read_csv(csv_path))
    return pd.concat(parts, ignore_index=True)


def get_top_features() -> list[str]:
    pipeline = load_feature_selection_module()
    _selected, _corr, summary, _rank_df = pipeline.get_ranked_feature_summary(
        feature_dir=FEATURE_DIR,
        group_a=GROUP_A,
        group_b=GROUP_B,
        contrast=CONTRAST,
        muscle_group=MUSCLE_GROUP,
        sexes=SEXES,
        corr_muscles=CORR_MUSCLES,
        corr_sexes=SEXES,
    )
    return summary.head(N_FEATURES_TO_TEST).index.tolist()


def run_mann_whitney_tests(group_a_df: pd.DataFrame, group_b_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        u_stat, p_value = mannwhitneyu(
            group_a_df[feature],
            group_b_df[feature],
            alternative="two-sided",
            method="exact",
        )
        rows.append({"feature": feature, "u_statistic": float(u_stat), "p_value": float(p_value)})
    return pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Case: {GROUP_A} vs {GROUP_B}, contrast={CONTRAST}, muscle={MUSCLE_GROUP}")

    group_a_df = load_group_data(GROUP_A)
    group_b_df = load_group_data(GROUP_B)
    top_features = get_top_features()

    print(f"Top features tested: {len(top_features)}")
    results = run_mann_whitney_tests(group_a_df, group_b_df, top_features)

    print("\nMann-Whitney U results (sorted by p-value):")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
