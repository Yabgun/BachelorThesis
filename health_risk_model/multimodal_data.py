import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd


def compute_disease_risk(xray_df: pd.DataFrame) -> pd.DataFrame:
    df = xray_df.copy()
    df["disease_risk"] = 1.0 - df["risk_score"].astype(float)
    return df


def encode_stay_category(stay_series: pd.Series) -> pd.Series:
    order = [
        "0-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
        "91-100",
        "More than 100 Days",
    ]
    mapping = {cat: idx for idx, cat in enumerate(order)}
    return stay_series.map(mapping).fillna(-1).astype(int)


def align_by_risk_and_stay(
    xray_df: pd.DataFrame,
    stay_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    x_df = compute_disease_risk(xray_df)
    n_x = len(x_df)
    if n_x == 0:
        return pd.DataFrame()

    stay_df = stay_df.copy()
    stay_df = stay_df.loc[stay_df["Stay"].notna()]
    if len(stay_df) == 0:
        return pd.DataFrame()

    x_df_sorted = x_df.sort_values("disease_risk").reset_index(drop=True)

    stay_df["stay_order"] = encode_stay_category(stay_df["Stay"])
    available = stay_df[stay_df["stay_order"] >= 0]
    if len(available) < n_x:
        n_x = len(available)
        x_df_sorted = x_df_sorted.iloc[:n_x].reset_index(drop=True)

    stay_subset = (
        available.sample(n=n_x, random_state=random_state)
        .sort_values("stay_order")
        .reset_index(drop=True)
    )

    multimodal = pd.concat(
        [
            stay_subset.reset_index(drop=True),
            x_df_sorted[["image_path", "label_index", "risk_score", "disease_risk"]].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    multimodal = multimodal.drop(columns=["stay_order"])
    return multimodal


def align_without_stay(
    xray_df: pd.DataFrame,
    stay_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    x_df = compute_disease_risk(xray_df)
    n_x = len(x_df)
    if n_x == 0:
        return pd.DataFrame()

    stay_df = stay_df.copy()
    if len(stay_df) == 0:
        return pd.DataFrame()

    if len(stay_df) < n_x:
        n_x = len(stay_df)
        x_df = x_df.iloc[:n_x].reset_index(drop=True)

    stay_subset = stay_df.sample(n=n_x, random_state=random_state).reset_index(drop=True)

    multimodal = pd.concat(
        [
            stay_subset,
            x_df[["image_path", "label_index", "risk_score", "disease_risk"]]
            .reset_index(drop=True),
        ],
        axis=1,
    )
    return multimodal


def build_multimodal_datasets(
    xray_train_path: str | None = None,
    xray_test_path: str | None = None,
    stay_train_path: str | None = None,
    stay_test_path: str | None = None,
) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    xray_train_path = xray_train_path or os.path.join(base_dir, "XrayData", "xray_features_train.csv")
    xray_test_path = xray_test_path or os.path.join(base_dir, "XrayData", "xray_features_test.csv")
    stay_train_path = stay_train_path or os.path.join(base_dir, "StayData", "train_data.csv")
    stay_test_path = stay_test_path or os.path.join(base_dir, "StayData", "test_data.csv")

    xray_train = pd.read_csv(xray_train_path)
    xray_test = pd.read_csv(xray_test_path)
    stay_train = pd.read_csv(stay_train_path)
    stay_test = pd.read_csv(stay_test_path)
    if "Stay" not in stay_test.columns and "Stay" in stay_train.columns:
        stay_test = stay_test.copy()
        stay_test["Stay"] = np.nan

    multimodal_train = align_by_risk_and_stay(xray_train, stay_train, random_state=42)
    if stay_test["Stay"].notna().any():
        multimodal_test = align_by_risk_and_stay(xray_test, stay_test, random_state=1337)
    else:
        multimodal_test = align_without_stay(xray_test, stay_test, random_state=1337)

    output_dir = os.path.join(base_dir, "StayData")
    os.makedirs(output_dir, exist_ok=True)

    multimodal_train_path = os.path.join(output_dir, "multimodal_train.csv")
    multimodal_test_path = os.path.join(output_dir, "multimodal_test.csv")

    multimodal_train.to_csv(multimodal_train_path, index=False)
    multimodal_test.to_csv(multimodal_test_path, index=False)

    return {
        "multimodal_train_path": multimodal_train_path,
        "multimodal_test_path": multimodal_test_path,
        "n_xray_train_used": len(multimodal_train),
        "n_xray_test_used": len(multimodal_test),
    }


if __name__ == "__main__":
    build_multimodal_datasets()

