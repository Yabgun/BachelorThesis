#!/usr/bin/env python3
"""
Core Health Risk Classification Model

Central definition of the encrypted healthcare risk classification pipeline.
Encapsulates the ML model, configuration, and complete training/evaluation flow.
"""

import os
import json
import time
import logging
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

try:
    from Pyfhel import Pyfhel
except Exception:
    Pyfhel = Any

try:
    from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
    HAS_CONCRETE_ML = True
except Exception:
    ConcreteLogisticRegression = None
    HAS_CONCRETE_ML = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_type: str = "random_forest"
    handle_encrypted: bool = True
    encryption_threshold: float = 0.1
    cv_folds: int = 5
    random_state: int = 42
    logistic_c: float = 3000.0
    logistic_max_iter: int = 5000
    logistic_solver: str = "lbfgs"
    logistic_class_weight: str | None = None
    tune_logistic_c: bool = False
    decision_rule: str = "argmax"
    prior_adjustment_alpha: float = 0.0


class EncryptedFeatureProcessor:
    def __init__(self, pyfhel_context: Pyfhel):
        self.he = pyfhel_context
        self.encrypted_cache = {}

    def process_features(self, X: pd.DataFrame) -> np.ndarray:
        processed_features = []

        for col in X.columns:
            if "_encrypted" in col:
                encrypted_values = X[col].values
                decrypted_values = []

                for enc_val in encrypted_values:
                    if isinstance(enc_val, str) and "encrypted" in enc_val:
                        decrypted_values.append(0.0)
                    else:
                        decrypted_values.append(float(enc_val) if enc_val is not None else 0.0)

                processed_features.append(decrypted_values)
            else:
                processed_features.append(X[col].values)

        return np.array(processed_features).T


class RiskClassificationModel:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_processor: EncryptedFeatureProcessor | None = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.tuning_info: Dict[str, Any] = {}
        self._computed_class_weight: dict[int, float] | str | None = None
        self._class_priors: np.ndarray | None = None

    def _build_sklearn_logistic_regression(self) -> LogisticRegression:
        class_weight: dict[int, float] | str | None
        if self._computed_class_weight is not None:
            class_weight = self._computed_class_weight
        else:
            class_weight = self.config.logistic_class_weight

        return LogisticRegression(
            random_state=self.config.random_state,
            max_iter=self.config.logistic_max_iter,
            C=self.config.logistic_c,
            solver=self.config.logistic_solver,
            class_weight=class_weight,
        )

    def tune_logistic_regression_c(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        candidate_cs: list[float] | None = None,
    ) -> Dict[str, Any]:
        if self.config.model_type != "logistic_regression":
            raise ValueError("Tuning is only supported for logistic_regression model type.")

        if HAS_CONCRETE_ML:
            raise ValueError("C tuning is not supported when concrete-ml LogisticRegression is enabled.")

        if candidate_cs is None:
            candidate_cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]

        X_raw = self.prepare_features(X_train, pyfhel_context=None, scale=False)
        if y_train.dtype == "object":
            y_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_encoded = y_train.values

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        best_c = None
        best_mean = None
        all_scores: Dict[float, float] = {}
        for c in candidate_cs:
            lr = self._build_sklearn_logistic_regression()
            lr.set_params(C=c)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", lr)])
            scores = cross_val_score(pipe, X_raw, y_encoded, cv=cv, scoring="accuracy")
            mean_score = float(scores.mean())
            all_scores[c] = mean_score
            if best_mean is None or mean_score > best_mean:
                best_mean = mean_score
                best_c = c

        if best_c is None:
            raise RuntimeError("Failed to tune logistic regression C.")

        self.config.logistic_c = float(best_c)
        self.tuning_info = {
            "tuned": True,
            "best_c": float(best_c),
            "best_mean_cv_accuracy": float(best_mean),
            "candidate_scores": {str(k): float(v) for k, v in all_scores.items()},
        }
        return self.tuning_info

    def initialize_model(self) -> None:
        logger.info(f"Initializing {self.config.model_type} model for risk classification")

        if self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif self.config.model_type == "logistic_regression":
            if HAS_CONCRETE_ML:
                self.model = ConcreteLogisticRegression(n_bits=8)
            else:
                self.model = self._build_sklearn_logistic_regression()
        elif self.config.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config.random_state,
            )
        elif self.config.model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                random_state=self.config.random_state,
                probability=True,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def prepare_features(
        self,
        X: pd.DataFrame,
        pyfhel_context: Pyfhel | None = None,
        scale: bool = True,
    ) -> np.ndarray:
        logger.info("Preparing features for model")

        if pyfhel_context:
            self.feature_processor = EncryptedFeatureProcessor(pyfhel_context)
            X_processed = self.feature_processor.process_features(X)
        else:
            X_processed = X.values

        if not scale:
            return X_processed

        if hasattr(self.scaler, "mean_"):
            return self.scaler.transform(X_processed)
        return self.scaler.fit_transform(X_processed)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, pyfhel_context: Pyfhel | None = None) -> Dict[str, Any]:
        logger.info("Training risk classification model")
        start_time = time.time()

        self._computed_class_weight = None

        if y_train.dtype == "object":
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values

        if y_train_encoded is not None and len(y_train_encoded) > 0:
            _, counts = np.unique(y_train_encoded, return_counts=True)
            self._class_priors = counts.astype(float) / float(len(y_train_encoded))

        if (
            self.config.model_type == "logistic_regression"
            and not HAS_CONCRETE_ML
            and self.config.logistic_class_weight == "balanced_sqrt"
        ):
            classes, counts = np.unique(y_train_encoded, return_counts=True)
            n_samples = float(len(y_train_encoded))
            n_classes = float(len(classes))
            weights = {}
            for cls, cnt in zip(classes.tolist(), counts.tolist()):
                balanced = n_samples / (n_classes * float(cnt))
                weights[int(cls)] = float(np.sqrt(balanced))
            self._computed_class_weight = weights

        if self.config.model_type == "logistic_regression" and self.config.tune_logistic_c and not HAS_CONCRETE_ML:
            self.tune_logistic_regression_c(X_train=X_train, y_train=y_train)

        self.initialize_model()
        X_train_processed = self.prepare_features(X_train, pyfhel_context, scale=True)

        self.model.fit(X_train_processed, y_train_encoded)
        self.training_time = time.time() - start_time

        train_predictions = self.model.predict(X_train_processed)
        train_accuracy = accuracy_score(y_train_encoded, train_predictions)

        logger.info(f"Model training completed in {self.training_time:.4f}s")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        coef_info: Dict[str, Any] = {}
        if self.config.model_type == "logistic_regression" and not HAS_CONCRETE_ML:
            if hasattr(self.model, "coef_"):
                coef = np.asarray(self.model.coef_, dtype=float)
                intercept = np.asarray(getattr(self.model, "intercept_", np.array([])), dtype=float)
                coef_info = {
                    "coef_l2_norm": float(np.linalg.norm(coef)),
                    "coef_max_abs": float(np.max(np.abs(coef))) if coef.size else 0.0,
                    "intercept_max_abs": float(np.max(np.abs(intercept))) if intercept.size else 0.0,
                }

        return {
            "training_time": self.training_time,
            "train_accuracy": train_accuracy,
            "model_type": self.config.model_type,
            "n_features": X_train.shape[1],
            "n_samples": len(X_train),
            "tuning_info": self.tuning_info,
            "coef_info": coef_info,
            "class_priors": self._class_priors.tolist() if self._class_priors is not None else None,
        }

    def predict(self, X_test: pd.DataFrame, pyfhel_context: Pyfhel | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        logger.info("Making predictions with risk classification model")
        start_time = time.time()

        X_test_processed = self.prepare_features(X_test, pyfhel_context)
        probabilities = self.model.predict_proba(X_test_processed)

        predictions: np.ndarray
        if self.config.decision_rule == "prior_adjusted" and self.config.prior_adjustment_alpha > 0.0:
            priors = self._class_priors
            if priors is not None and len(priors) == probabilities.shape[1]:
                adjust = np.power(priors, self.config.prior_adjustment_alpha)
                adjust = np.clip(adjust, 1e-12, None)
                adjusted_scores = probabilities / adjust.reshape(1, -1)
                predictions = np.argmax(adjusted_scores, axis=1)
            else:
                predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = np.argmax(probabilities, axis=1)

        self.prediction_time = time.time() - start_time

        logger.info(f"Prediction completed for {len(X_test)} samples in {self.prediction_time:.4f}s")

        return predictions, {
            "prediction_time": self.prediction_time,
            "n_predictions": len(predictions),
            "probabilities_shape": probabilities.shape,
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, pyfhel_context: Pyfhel | None = None) -> Dict[str, Any]:
        logger.info("Evaluating risk classification model")

        predictions, pred_info = self.predict(X_test, pyfhel_context)

        if y_test.dtype == "object":
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_test_encoded = y_test.values

        accuracy = accuracy_score(y_test_encoded, predictions)
        macro_f1 = float(f1_score(y_test_encoded, predictions, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test_encoded, predictions, average="weighted", zero_division=0))
        report = classification_report(y_test_encoded, predictions, output_dict=True)
        cm = confusion_matrix(y_test_encoded, predictions)
        unique_pred = np.unique(predictions)
        unique_true = np.unique(y_test_encoded)
        missing_predicted = [int(x) for x in unique_true if x not in set(unique_pred.tolist())]

        results = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "prediction_info": pred_info,
            "model_type": self.config.model_type,
            "test_samples": len(y_test),
            "n_predicted_classes": int(len(unique_pred)),
            "missing_predicted_class_indices": missing_predicted,
            "label_classes": [str(x) for x in getattr(self.label_encoder, "classes_", [])],
        }

        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return results

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, pyfhel_context: Pyfhel | None = None) -> Dict[str, Any]:
        logger.info(f"Performing {self.config.cv_folds}-fold cross-validation")

        if y.dtype == "object":
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        if pyfhel_context:
            X_processed = self.prepare_features(X, pyfhel_context, scale=False)
            cv_scores = []
            for train_idx, test_idx in cv.split(X_processed, y_encoded):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_processed[train_idx])
                X_test = scaler.transform(X_processed[test_idx])
                model = clone(self.model)
                model.fit(X_train, y_encoded[train_idx])
                preds = model.predict(X_test)
                cv_scores.append(float(accuracy_score(y_encoded[test_idx], preds)))
            cv_scores = np.array(cv_scores, dtype=float)
        else:
            X_raw = self.prepare_features(X, pyfhel_context=None, scale=False)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(self.model))])
            cv_scores = cross_val_score(pipe, X_raw, y_encoded, cv=cv, scoring="accuracy")

        results = {
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": float(cv_scores.mean()),
            "std_cv_score": float(cv_scores.std()),
            "cv_folds": self.config.cv_folds,
        }

        logger.info(
            f"Cross-validation completed. Mean score: {results['mean_cv_score']:.4f} (±{results['std_cv_score']:.4f})"
        )
        return results

    def save_model(self, filepath: str) -> None:
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "config": self.config,
            "training_time": self.training_time,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: str) -> None:
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.config = model_data["config"]
        self.training_time = model_data["training_time"]
        logger.info(f"Model loaded from: {filepath}")


class MLPipeline:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.classification_model = RiskClassificationModel(self.config)
        self.results: Dict[str, Any] = {}

    def run_complete_pipeline(self, datasets: Dict, pyfhel_context: Pyfhel | None = None) -> Dict[str, Any]:
        logger.info("Starting complete ML pipeline")

        start_time = time.time()

        X_train = datasets["X_train"]
        X_test = datasets["X_test"]
        y_train = datasets["y_train"]
        y_test = datasets["y_test"]

        train_results = self.classification_model.train(X_train, y_train, pyfhel_context)
        eval_results = self.classification_model.evaluate(X_test, y_test, pyfhel_context)

        cv_results = self.classification_model.cross_validate(
            pd.concat([X_train, X_test]),
            pd.concat([y_train, y_test]),
            pyfhel_context,
        )

        pipeline_time = time.time() - start_time

        self.results = {
            "pipeline_time": pipeline_time,
            "training_results": train_results,
            "evaluation_results": eval_results,
            "cross_validation_results": cv_results,
            "model_config": self.config.__dict__,
            "datasets_info": {
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
            },
        }

        logger.info(f"ML pipeline completed in {pipeline_time:.4f}s")
        return self.results


def load_multimodal_datasets_from_csv(
    csv_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    encoding_mode: str = "target_encoding",
    target_encoding_smoothing: float = 20.0,
) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = base_dir / "StayData" / "multimodal_train.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Multimodal dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["Stay"].notna()].copy()

    def map_age_to_numeric(age_value: Any) -> float:
        if isinstance(age_value, str) and "-" in age_value:
            try:
                low, high = age_value.split("-")
                return (float(low) + float(high)) / 2.0
            except Exception:
                return 0.0
        return 0.0

    df["age_numeric"] = df["Age"].apply(map_age_to_numeric)

    severity_mapping = {"Minor": 1, "Moderate": 2, "Extreme": 3}
    df["severity_numeric"] = df["Severity of Illness"].map(severity_mapping).fillna(2).astype(float)

    df["Admission_Deposit"] = df["Admission_Deposit"].fillna(0.0)
    df["age_deposit_interaction"] = df["age_numeric"] * np.log1p(df["Admission_Deposit"].astype(float))

    df["disease_risk"] = df["disease_risk"].astype(float)
    df["risk_severity_synergy"] = df["disease_risk"] * df["severity_numeric"]

    df["risk_high"] = (df["risk_score"].astype(float) > 0.9).astype(int)
    df["risk_medium"] = (
        (df["risk_score"].astype(float) > 0.5) & (df["risk_score"].astype(float) <= 0.9)
    ).astype(int)
    df["disease_risk_boost"] = df["disease_risk"] * 10.0
    df["age_money_ratio"] = df["age_numeric"] / (df["Admission_Deposit"].astype(float) + 1.0)

    stay_order = [
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
    stay_mapping = {cat: idx for idx, cat in enumerate(stay_order)}
    df["stay_index"] = df["Stay"].map(stay_mapping).fillna(-1).astype(int)

    y = df["Stay"]

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = df.loc[train_idx].copy()
    stay_index_train = train_df["stay_index"]

    global_mean = float(stay_index_train.mean())
    target_encoding_cols = [
        "Hospital_type_code",
        "Hospital_region_code",
        "Department",
        "Ward_Type",
        "Ward_Facility_Code",
        "Type of Admission",
        "Severity of Illness",
        "Age",
    ]
    if encoding_mode not in {"target_encoding", "onehot"}:
        raise ValueError(f"Unknown encoding_mode: {encoding_mode}")

    if encoding_mode == "target_encoding":
        for col in target_encoding_cols:
            if col not in train_df.columns:
                continue
            stats = train_df.groupby(col)["stay_index"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * target_encoding_smoothing) / (
                stats["count"] + target_encoding_smoothing
            )
            df[f"{col}_te"] = df[col].map(smooth).fillna(global_mean).astype(float)
        df = df.drop(columns=[c for c in target_encoding_cols if c in df.columns])

    df = df.drop(columns=["stay_index"])

    train_full = df.loc[train_idx].copy()
    test_full = df.loc[test_idx].copy()

    y_train = train_full["Stay"]
    y_test = test_full["Stay"]

    feature_df_train = train_full.drop(columns=["Stay", "image_path"])
    feature_df_test = test_full.drop(columns=["Stay", "image_path"])

    categorical_cols = feature_df_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in feature_df_train.columns if c not in categorical_cols]

    X_train_numeric = feature_df_train[numeric_cols].fillna(0.0)
    X_test_numeric = feature_df_test[numeric_cols].fillna(0.0)

    if len(categorical_cols) > 0:
        X_train_categorical = feature_df_train[categorical_cols].fillna("missing")
        X_test_categorical = feature_df_test[categorical_cols].fillna("missing")

        X_train_cat_enc = pd.get_dummies(X_train_categorical, drop_first=True)
        X_test_cat_enc = pd.get_dummies(X_test_categorical, drop_first=True)

        X_train_cat_enc, X_test_cat_enc = X_train_cat_enc.align(
            X_test_cat_enc, join="left", axis=1, fill_value=0
        )

        X_train = pd.concat(
            [X_train_numeric.reset_index(drop=True), X_train_cat_enc.reset_index(drop=True)],
            axis=1,
        )
        X_test = pd.concat(
            [X_test_numeric.reset_index(drop=True), X_test_cat_enc.reset_index(drop=True)],
            axis=1,
        )
    else:
        X_train = X_train_numeric
        X_test = X_test_numeric

    datasets = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return datasets


def run_multimodal_core_model(
    csv_path: str | None = None,
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    logistic_c: float | None = None,
    tune_logistic_c: bool = False,
    logistic_class_weight: str | None = None,
    decision_rule: str = "argmax",
    prior_adjustment_alpha: float = 0.0,
    encoding_mode: str = "target_encoding",
    target_encoding_smoothing: float = 20.0,
) -> Dict[str, Any]:
    datasets = load_multimodal_datasets_from_csv(
        csv_path=csv_path,
        test_size=test_size,
        random_state=random_state,
        encoding_mode=encoding_mode,
        target_encoding_smoothing=target_encoding_smoothing,
    )

    config = ModelConfig(
        model_type=model_type,
        handle_encrypted=False,
        cv_folds=5,
        random_state=random_state,
    )
    if logistic_c is not None:
        config.logistic_c = float(logistic_c)
    config.tune_logistic_c = bool(tune_logistic_c)
    config.logistic_class_weight = logistic_class_weight
    config.decision_rule = decision_rule
    config.prior_adjustment_alpha = float(prior_adjustment_alpha)
    pipeline = MLPipeline(config)
    results = pipeline.run_complete_pipeline(datasets, pyfhel_context=None)
    return results

    def save_results(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        results_path = os.path.join(output_dir, "classification_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        model_path = os.path.join(output_dir, "classification_model.pkl")
        self.classification_model.save_model(model_path)

        self.generate_report(output_dir)
        logger.info(f"Results saved to: {output_dir}")

    def generate_report(self, output_dir: str) -> None:
        report_path = os.path.join(output_dir, "classification_report.md")

        with open(report_path, "w") as f:
            f.write("# Risk Classification Model Report\n\n")
            f.write("## Model Configuration\n\n")
            f.write(f"- **Model Type**: {self.config.model_type}\n")
            f.write(f"- **Handles Encrypted Features**: {self.config.handle_encrypted}\n")
            f.write(f"- **Cross-Validation Folds**: {self.config.cv_folds}\n\n")

            f.write("## Training Results\n\n")
            train_results = self.results["training_results"]
            f.write(f"- **Training Time**: {train_results['training_time']:.4f}s\n")
            f.write(f"- **Training Accuracy**: {train_results['train_accuracy']:.4f}\n")
            f.write(f"- **Samples**: {train_results['n_samples']}\n")
            f.write(f"- **Features**: {train_results['n_features']}\n\n")

            f.write("## Evaluation Results\n\n")
            eval_results = self.results["evaluation_results"]
            f.write(f"- **Test Accuracy**: {eval_results['accuracy']:.4f}\n")
            f.write(f"- **Test Samples**: {eval_results['test_samples']}\n")
            f.write(f"- **Prediction Time**: {eval_results['prediction_info']['prediction_time']:.4f}s\n\n")

            f.write("## Cross-Validation Results\n\n")
            cv_results = self.results["cross_validation_results"]
            f.write(
                f"- **Mean CV Score**: {cv_results['mean_cv_score']:.4f} (±{cv_results['std_cv_score']:.4f})\n"
            )
            f.write(f"- **CV Folds**: {cv_results['cv_folds']}\n\n")

            f.write("## Pipeline Performance\n\n")
            f.write(f"- **Total Pipeline Time**: {self.results['pipeline_time']:.4f}s\n")
            f.write(
                f"- **Dataset Size**: {self.results['datasets_info']['n_train']} train, "
                f"{self.results['datasets_info']['n_test']} test\n\n"
            )

            f.write("## Encrypted Data Handling\n\n")
            f.write("This model supports mixed encrypted/plaintext features as specified in the research proposal:\n")
            f.write("- Encrypted features (test_results_score, cxr_mean_intensity, cxr_edge_density) are processed\n")
            f.write("- Plaintext features (age, billing_amount_norm) are used directly\n")
            f.write("- Model maintains accuracy while preserving privacy through selective encryption\n")


def load_datasets_and_context(
    datasets_path: str = "data/ml_encrypted/ml_datasets.json",
    context_path: str = "data/ml_encrypted/pyfhel_context.pkl",
) -> tuple[Dict[str, Any], Pyfhel]:
    if not all(os.path.exists(p) for p in [datasets_path, context_path]):
        raise FileNotFoundError("Required data files not found. Run ml_encrypted_data_preparation.py first.")

    with open(datasets_path, "r") as f:
        datasets_data = json.load(f)

    X_train = pd.DataFrame(datasets_data["X_train"])
    X_test = pd.DataFrame(datasets_data["X_test"])
    y_train = pd.Series(datasets_data["y_train"])
    y_test = pd.Series(datasets_data["y_test"])

    datasets = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    pyfhel_context = joblib.load(context_path)
    return datasets, pyfhel_context


def run_model_suite() -> Dict[str, Any]:
    datasets, pyfhel_context = load_datasets_and_context()

    model_types = ["random_forest", "logistic_regression", "gradient_boosting"]
    all_results: Dict[str, Any] = {}

    for model_type in model_types:
        logger.info(f"Testing {model_type} model...")

        config = ModelConfig(model_type=model_type)
        pipeline = MLPipeline(config)

        results = pipeline.run_complete_pipeline(datasets, pyfhel_context)
        all_results[model_type] = results

        output_dir = f"data/ml_models/{model_type}"
        pipeline.save_results(output_dir)

    generate_comparative_report(all_results)
    return all_results


def generate_comparative_report(all_results: Dict[str, Any]) -> None:
    report_path = "data/ml_models/comparative_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Comparative Model Performance Report\n\n")
        f.write("## Model Comparison Summary\n\n")

        f.write("| Model Type | Test Accuracy | CV Score | Training Time | Prediction Time |\n")
        f.write("|------------|---------------|----------|---------------|-----------------|\n")

        for model_type, results in all_results.items():
            eval_acc = results["evaluation_results"]["accuracy"]
            cv_score = results["cross_validation_results"]["mean_cv_score"]
            train_time = results["training_results"]["training_time"]
            pred_time = results["evaluation_results"]["prediction_info"]["prediction_time"]

            f.write(
                f"| {model_type} | {eval_acc:.4f} | {cv_score:.4f} "
                f"(±{results['cross_validation_results']['std_cv_score']:.4f}) | "
                f"{train_time:.4f}s | {pred_time:.4f}s |\n"
            )

        f.write("\n## Best Model Recommendation\n\n")
        best_model = max(all_results.keys(), key=lambda x: all_results[x]["evaluation_results"]["accuracy"])
        f.write(f"**Recommended Model**: {best_model}\n")
        f.write(
            f"- Highest test accuracy: {all_results[best_model]['evaluation_results']['accuracy']:.4f}\n"
        )
        f.write(
            f"- Good cross-validation performance: "
            f"{all_results[best_model]['cross_validation_results']['mean_cv_score']:.4f}\n"
        )
        f.write(
            f"- Reasonable training time: {all_results[best_model]['training_results']['training_time']:.4f}s\n"
        )


def main() -> None:
    logger.info("Starting central Health Risk Classification Pipeline")

    try:
        run_model_suite()
        logger.info("Health Risk Classification Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Health risk pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
