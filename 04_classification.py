# RadiomiX classification for one selected case
# Uses top 10 features from script 02
# A temporary CSV is built for RadiomiX to use as input

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pandas as pd


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
FEATURE_DIR = Path("")

GROUP_A = "" # The subjects with label = 'A'
GROUP_B = "" # The subjects with label = 'B'
GROUP_A_LABEL = 1
GROUP_B_LABEL = 0
CONTRAST = ""
MUSCLE_GROUP = ""
CORR_MUSCLES = ("A", "B", "C",...) # Correlation clustering looks at an averaged correlation matrix for sexes and muscles
SEXES = ("F", "M")
N_SELECTED_FEATURES = 10
OUTPUT_CSV_NAME = ""

# ---------------------------------------------------------------------
# RadiomiX
# ---------------------------------------------------------------------
RADIOMIX_DIR = Path("")
PYTHON_EXE = "/opt/miniconda3/envs/RadiomiX/bin/python"
RADIOMIX_SCRIPT = "radiomix_edit.py"
TARGET_COL = "Group"
ID_COL = "ID"
PARAMS_PROFILE = "custom_params"
CSV_NAME = OUTPUT_CSV_NAME


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
    return summary.head(N_SELECTED_FEATURES).index.tolist()


def build_radiomix_dataframe() -> pd.DataFrame:
    selected_features = get_top_features()
    print("Selected features:", selected_features)

    group_a_df = load_group_data(GROUP_A)
    group_b_df = load_group_data(GROUP_B)
    group_a_df["Group"] = GROUP_A_LABEL
    group_b_df["Group"] = GROUP_B_LABEL
    data = pd.concat([group_a_df, group_b_df], ignore_index=True)

    radiomix_df = data[selected_features].copy()
    radiomix_df.insert(0, "ID", data["subject_id"])
    radiomix_df["Group"] = data["Group"]
    return radiomix_df


def run_radiomix(radiomix_df: pd.DataFrame) -> None:
    tmp_dir = RADIOMIX_DIR / "tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / CSV_NAME
    radiomix_df.to_csv(tmp_path, index=False)
    print(f"Using temporary RadiomiX CSV: {tmp_path}")

    cmd = [
        PYTHON_EXE,
        str(RADIOMIX_DIR / RADIOMIX_SCRIPT),
        "--path",
        str(tmp_path),
        "--target",
        TARGET_COL,
        "--id",
        ID_COL,
        "--params",
        PARAMS_PROFILE,
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=RADIOMIX_DIR, check=True)
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Case: {GROUP_A} vs {GROUP_B}, contrast={CONTRAST}, muscle={MUSCLE_GROUP}")
    radiomix_df = build_radiomix_dataframe()
    run_radiomix(radiomix_df)


if __name__ == "__main__":
    main()
