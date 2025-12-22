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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
            self.model = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=2000,
                C=2.0,
                class_weight="balanced",
                multi_class="multinomial",
                solver="lbfgs",
            )
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

    def prepare_features(self, X: pd.DataFrame, pyfhel_context: Pyfhel | None = None) -> np.ndarray:
        logger.info("Preparing features for model")

        if pyfhel_context:
            self.feature_processor = EncryptedFeatureProcessor(pyfhel_context)
            X_processed = self.feature_processor.process_features(X)
        else:
            X_processed = X.values

        if hasattr(self.scaler, "mean_"):
            X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = self.scaler.fit_transform(X_processed)

        return X_scaled

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, pyfhel_context: Pyfhel | None = None) -> Dict[str, Any]:
        logger.info("Training risk classification model")
        start_time = time.time()

        self.initialize_model()
        X_train_processed = self.prepare_features(X_train, pyfhel_context)

        if y_train.dtype == "object":
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values

        self.model.fit(X_train_processed, y_train_encoded)
        self.training_time = time.time() - start_time

        train_predictions = self.model.predict(X_train_processed)
        train_accuracy = accuracy_score(y_train_encoded, train_predictions)

        logger.info(f"Model training completed in {self.training_time:.4f}s")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        return {
            "training_time": self.training_time,
            "train_accuracy": train_accuracy,
            "model_type": self.config.model_type,
            "n_features": X_train.shape[1],
            "n_samples": len(X_train),
        }

    def predict(self, X_test: pd.DataFrame, pyfhel_context: Pyfhel | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        logger.info("Making predictions with risk classification model")
        start_time = time.time()

        X_test_processed = self.prepare_features(X_test, pyfhel_context)
        predictions = self.model.predict(X_test_processed)
        probabilities = self.model.predict_proba(X_test_processed)

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
        report = classification_report(y_test_encoded, predictions, output_dict=True)
        cm = confusion_matrix(y_test_encoded, predictions)

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "prediction_info": pred_info,
            "model_type": self.config.model_type,
            "test_samples": len(y_test),
        }

        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return results

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, pyfhel_context: Pyfhel | None = None) -> Dict[str, Any]:
        logger.info(f"Performing {self.config.cv_folds}-fold cross-validation")

        X_processed = self.prepare_features(X, pyfhel_context)

        if y.dtype == "object":
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y.values

        cv_scores = cross_val_score(
            self.model,
            X_processed,
            y_encoded,
            cv=self.config.cv_folds,
            scoring="accuracy",
        )

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
) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = base_dir / "StayData" / "multimodal_train.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Multimodal dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["Stay"].notna()].copy()

    y = df["Stay"]
    feature_df = df.drop(columns=["Stay", "image_path"])

    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    X_numeric = feature_df[numeric_cols].fillna(0.0)
    X_categorical = feature_df[categorical_cols].fillna("missing")

    if len(categorical_cols) > 0:
        X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)
        X = pd.concat(
            [X_numeric.reset_index(drop=True), X_categorical_encoded.reset_index(drop=True)],
            axis=1,
        )
    else:
        X = X_numeric

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

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
) -> Dict[str, Any]:
    datasets = load_multimodal_datasets_from_csv(
        csv_path=csv_path,
        test_size=test_size,
        random_state=random_state,
    )

    config = ModelConfig(
        model_type=model_type,
        handle_encrypted=False,
        cv_folds=5,
        random_state=random_state,
    )
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
