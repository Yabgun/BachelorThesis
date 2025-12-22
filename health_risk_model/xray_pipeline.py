import os
from typing import Dict, Any

import pandas as pd

from .xray_data import create_dataloaders
from .xray_model import XrayRiskModel


def run_xray_training_feature_export(
    data_root: str | None = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_epochs: int = 5,
) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xr_root = data_root or os.path.join(base_dir, "XrayData")

    train_loader, test_loader, class_names = create_dataloaders(
        xr_root,
        batch_size=batch_size,
        img_size=img_size,
    )

    model = XrayRiskModel(num_classes=len(class_names))
    model.fit(train_loader, num_epochs=num_epochs)

    train_acc = model.evaluate(train_loader)
    test_acc = model.evaluate(test_loader)

    train_paths, train_labels, train_probs = model.predict_proba(train_loader)
    test_paths, test_labels, test_probs = model.predict_proba(test_loader)

    train_df = pd.DataFrame(
        {
            "image_path": train_paths,
            "label_index": train_labels,
            "risk_score": train_probs,
        }
    )
    test_df = pd.DataFrame(
        {
            "image_path": test_paths,
            "label_index": test_labels,
            "risk_score": test_probs,
        }
    )

    features_dir = os.path.join(base_dir, "XrayData")
    os.makedirs(features_dir, exist_ok=True)

    train_csv = os.path.join(features_dir, "xray_features_train.csv")
    test_csv = os.path.join(features_dir, "xray_features_test.csv")

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return {
        "class_names": class_names,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_features_path": train_csv,
        "test_features_path": test_csv,
    }


if __name__ == "__main__":
    run_xray_training_feature_export()

