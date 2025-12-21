import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path(__file__).resolve().parent / "healthcare-dataset-stroke-data.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    strategy: str,
    target_recall: float,
    grid_size: int = 401,
) -> dict:
    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(np.unique(y_true)) < 2:
        return best

    target_recall = float(np.clip(target_recall, 0.0, 1.0))

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if strategy == "f1":
            if f1 > best["f1"] or (f1 == best["f1"] and p > best["precision"]):
                best = {"threshold": float(t), "precision": p, "recall": r, "f1": f1}
        elif strategy == "precision_at_recall":
            if r >= target_recall:
                if p > best["precision"] or (p == best["precision"] and f1 > best["f1"]):
                    best = {"threshold": float(t), "precision": p, "recall": r, "f1": f1}
        else:
            raise ValueError(f"Unsupported threshold strategy: {strategy}")

    if strategy == "precision_at_recall" and best["precision"] == 0.0 and best["recall"] == 0.0:
        for t in thresholds:
            y_pred = (proba >= t).astype(int)
            p = float(precision_score(y_true, y_pred, zero_division=0))
            r = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            if r > best["recall"] or (r == best["recall"] and p > best["precision"]):
                best = {"threshold": float(t), "precision": p, "recall": r, "f1": f1}

    return best


def build_stroke_classifier_model(model_key: str, random_state: int) -> object:
    if model_key == "logreg":
        return LogisticRegression(max_iter=20000, solver="saga", n_jobs=-1, class_weight="balanced")
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_key == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=900,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model_key: {model_key}")


def bootstrap_oversample(
    X: pd.DataFrame, y: np.ndarray, target_pos_ratio: float, random_state: int
) -> tuple[pd.DataFrame, np.ndarray]:
    target_pos_ratio = float(np.clip(target_pos_ratio, 0.0, 1.0))
    y_arr = np.asarray(y, dtype=int)
    if len(y_arr) == 0:
        return X, y_arr

    pos_idx = np.flatnonzero(y_arr == 1)
    neg_idx = np.flatnonzero(y_arr == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y_arr

    current_ratio = float(len(pos_idx) / len(y_arr))
    if target_pos_ratio <= current_ratio:
        return X, y_arr

    target_pos_ratio = float(np.clip(target_pos_ratio, 1e-6, 1.0 - 1e-6))
    n_neg = int(len(neg_idx))
    n_pos_target = int(np.ceil((target_pos_ratio / (1.0 - target_pos_ratio)) * n_neg))
    n_pos_extra = max(0, n_pos_target - int(len(pos_idx)))
    if n_pos_extra == 0:
        return X, y_arr

    rng = np.random.default_rng(random_state)
    extra_pos_idx = rng.choice(pos_idx, size=n_pos_extra, replace=True)
    all_idx = np.concatenate([neg_idx, pos_idx, extra_pos_idx])
    rng.shuffle(all_idx)
    return X.iloc[all_idx].reset_index(drop=True), y_arr[all_idx]


def engineer_features(X: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return X
    if mode != "basic":
        raise ValueError(f"Unsupported feature engineering mode: {mode}")

    out = X.copy()

    if "age" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
        out["age_squared"] = age * age

    if "avg_glucose_level" in out.columns:
        glucose = pd.to_numeric(out["avg_glucose_level"], errors="coerce")
        out["glucose_log1p"] = np.log1p(glucose.clip(lower=0))

    if "bmi" in out.columns:
        bmi = pd.to_numeric(out["bmi"], errors="coerce")
        out["bmi_squared"] = bmi * bmi

    if "age" in out.columns and "avg_glucose_level" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
        glucose = pd.to_numeric(out["avg_glucose_level"], errors="coerce")
        out["age_x_glucose"] = age * glucose

    if "age" in out.columns and "hypertension" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
        hyp = pd.to_numeric(out["hypertension"], errors="coerce")
        out["age_x_hypertension"] = age * hyp

    if "age" in out.columns and "heart_disease" in out.columns:
        age = pd.to_numeric(out["age"], errors="coerce")
        hd = pd.to_numeric(out["heart_disease"], errors="coerce")
        out["age_x_heart_disease"] = age * hd

    if "hypertension" in out.columns and "heart_disease" in out.columns:
        hyp = pd.to_numeric(out["hypertension"], errors="coerce")
        hd = pd.to_numeric(out["heart_disease"], errors="coerce")
        out["hypertension_x_heart_disease"] = hyp * hd

    return out


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def load_stroke_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"].replace("N/A", np.nan), errors="coerce")
    if "stroke" in df.columns:
        df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce").astype("Int64")
    return df


def train_stroke_classifier(df: pd.DataFrame, random_state: int = 42) -> dict:
    return train_stroke_classifier_with_model(df, model_key="logreg", random_state=random_state)


def train_stroke_classifier_with_model(
    df: pd.DataFrame,
    model_key: str,
    feature_engineering: str = "none",
    resample: str = "none",
    target_pos_ratio: float = 0.2,
    threshold_strategy: str = "precision_at_recall",
    target_recall: float = 0.8,
    threshold_grid_size: int = 401,
    random_state: int = 42,
) -> dict:
    target = "stroke"
    df = df.dropna(subset=[target]).copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").astype(int)

    X = df.drop(columns=[target, "id"], errors="ignore")
    y = df[target].to_numpy(dtype=int)
    X = engineer_features(X, mode=feature_engineering)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state + 1,
        stratify=y_train,
    )

    if resample == "bootstrap":
        X_subtrain_fit, y_subtrain_fit = bootstrap_oversample(
            X=X_subtrain,
            y=np.asarray(y_subtrain, dtype=int),
            target_pos_ratio=target_pos_ratio,
            random_state=random_state + 10_000,
        )
    elif resample == "none":
        X_subtrain_fit, y_subtrain_fit = X_subtrain, y_subtrain
    else:
        raise ValueError(f"Unsupported resample: {resample}")

    preprocessor_inner, _, _ = build_preprocessor(X_subtrain_fit)
    model_inner = build_stroke_classifier_model(model_key=model_key, random_state=random_state)
    pipeline_inner = Pipeline(steps=[("prep", preprocessor_inner), ("model", model_inner)])
    pipeline_inner.fit(X_subtrain_fit, y_subtrain_fit)
    proba_val = pipeline_inner.predict_proba(X_val)[:, 1]

    threshold_result = _choose_threshold(
        y_true=np.asarray(y_val, dtype=int),
        proba=np.asarray(proba_val, dtype=float),
        strategy=threshold_strategy,
        target_recall=target_recall,
        grid_size=threshold_grid_size,
    )
    threshold = float(threshold_result["threshold"])

    if resample == "bootstrap":
        X_train_fit, y_train_fit = bootstrap_oversample(
            X=X_train,
            y=np.asarray(y_train, dtype=int),
            target_pos_ratio=target_pos_ratio,
            random_state=random_state + 20_000,
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train_fit)
    model = build_stroke_classifier_model(model_key=model_key, random_state=random_state)
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train_fit, y_train_fit)

    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "task": "stroke_classification",
        "target": target,
        "model": model_key,
        "feature_engineering": feature_engineering,
        "resample": resample,
        "target_pos_ratio": float(target_pos_ratio) if resample == "bootstrap" else None,
        "threshold_strategy": threshold_strategy,
        "target_recall": float(target_recall) if threshold_strategy == "precision_at_recall" else None,
        "threshold_grid_size": int(threshold_grid_size),
        "selected_threshold": float(threshold),
        "val_precision": float(threshold_result["precision"]),
        "val_recall": float(threshold_result["recall"]),
        "val_f1": float(threshold_result["f1"]),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "stroke_rate": float(np.mean(y)),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }

    return {"pipeline": pipeline, "metrics": metrics}


def save_artifacts(result: dict, out_dir: Path) -> Path:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    task = result["metrics"]["task"]
    model_path = out_dir / f"{task}_pipeline.joblib"
    metrics_path = out_dir / f"{task}_metrics.json"

    joblib.dump(result["pipeline"], model_path)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    return model_path


def load_artifacts(task: str, artifacts_dir: Path) -> tuple[object, dict]:
    import joblib

    model_path = artifacts_dir / f"{task}_pipeline.joblib"
    metrics_path = artifacts_dir / f"{task}_metrics.json"
    pipeline = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return pipeline, metrics


def predict_stroke(
    pipeline: object,
    metrics: dict,
    df: pd.DataFrame,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    target = str(metrics.get("target", "stroke"))
    feature_engineering = str(metrics.get("feature_engineering", "none"))
    threshold_value = float(metrics.get("selected_threshold", 0.5)) if threshold is None else float(threshold)

    has_target = target in df.columns
    y_true = None
    if has_target:
        y_true = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)

    ids = df["id"].copy() if "id" in df.columns else None
    X = df.drop(columns=[target, "id"], errors="ignore")
    X = engineer_features(X, mode=feature_engineering)

    proba = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
    y_pred = (proba >= threshold_value).astype(int)

    out = pd.DataFrame({"stroke_proba": proba, "stroke_pred": y_pred})
    if ids is not None:
        out.insert(0, "id", ids.to_numpy())
    if has_target and y_true is not None:
        out["stroke_true"] = y_true

    eval_metrics = None
    if has_target and y_true is not None and len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        eval_metrics = {
            "threshold": float(threshold_value),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "pr_auc": float(average_precision_score(y_true, proba)),
        }

    return out, eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold-strategy", choices=["precision_at_recall", "f1"], default="precision_at_recall")
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--threshold-grid-size", type=int, default=401)
    parser.add_argument("--model", choices=["logreg", "random_forest", "extra_trees"], default="logreg")
    parser.add_argument("--feature-engineering", choices=["none", "basic"], default="none")
    parser.add_argument("--resample", choices=["none", "bootstrap"], default="none")
    parser.add_argument("--target-pos-ratio", type=float, default=0.2)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict-input", default=None)
    parser.add_argument("--predict-output", default=None)
    parser.add_argument("--predict-threshold", type=float, default=None)
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    if args.predict:
        artifacts_dir = Path(args.artifacts_dir)
        pipeline, metrics = load_artifacts(task="stroke_classification", artifacts_dir=artifacts_dir)
        predict_input = Path(args.predict_input) if args.predict_input else Path(args.dataset)
        predict_df = load_stroke_dataset(predict_input)
        preds, eval_metrics = predict_stroke(
            pipeline=pipeline, metrics=metrics, df=predict_df, threshold=args.predict_threshold
        )
        if args.predict_output:
            out_path = Path(args.predict_output)
            out_path.write_text(preds.to_csv(index=False), encoding="utf-8")
            print(f"Saved: {out_path}")
        result = {"task": "stroke_prediction", "rows": int(len(preds)), "eval": eval_metrics}
        print(json.dumps(result, indent=2))
        return

    dataset_path = Path(args.dataset)
    df = load_stroke_dataset(dataset_path)
    result = train_stroke_classifier_with_model(
        df,
        model_key=args.model,
        feature_engineering=args.feature_engineering,
        resample=args.resample,
        target_pos_ratio=args.target_pos_ratio,
        threshold_strategy=args.threshold_strategy,
        target_recall=args.target_recall,
        threshold_grid_size=args.threshold_grid_size,
        random_state=args.random_state,
    )
    print(json.dumps(result["metrics"], indent=2))
    if args.save:
        artifacts_dir = Path(args.artifacts_dir)
        model_path = save_artifacts(result, artifacts_dir)
        print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()

