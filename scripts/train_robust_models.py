import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_FILE = MODELS_DIR / 'he_model_weights.json'

def train_models():
    logger.info("ADIM 2: HE Uyumlu Model Eğitimi Başlıyor...")
    
    # 1. Load Data
    try:
        X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
        X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
        # Ensure classification targets are integers
        y_class_train = pd.read_csv(PROCESSED_DIR / 'y_class_train.csv').values.ravel().astype(int)
        y_class_test = pd.read_csv(PROCESSED_DIR / 'y_class_test.csv').values.ravel().astype(int)
        y_reg_train = pd.read_csv(PROCESSED_DIR / 'y_reg_train.csv').values.ravel()
        y_reg_test = pd.read_csv(PROCESSED_DIR / 'y_reg_test.csv').values.ravel()
    except FileNotFoundError:
        logger.error("İşlenmiş veri bulunamadı. Önce scripts/prepare_robust_data.py çalıştırın.")
        return

    # --- PART A: CLASSIFICATION (Risk) ---
    logger.info("--- Sınıflandırma Modeli (Logistic Regression) ---")
    # HE ile uyumlu olması için 'sigmoid' aktivasyonu kullanılır, 
    # ancak HE tarafında genellikle lineer yaklaşım veya polinom kullanılır.
    # Biz burada standart Logistic Regression eğitiyoruz, 
    # HE tarafında "w*x + b" hesaplayıp sonucu sigmoid'e sokacağız (client side) veya approx.
    
    clf = LogisticRegression(random_state=42)
    
    # Cross Validation (Ezber kontrolü)
    cv_scores = cross_val_score(clf, X_train, y_class_train, cv=5)
    logger.info(f"5-Katlı Çapraz Doğrulama Skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    clf.fit(X_train, y_class_train)
    
    # Test Evaluation
    y_pred_class = clf.predict(X_test)
    test_acc = accuracy_score(y_class_test, y_pred_class)
    logger.info(f"Test Seti Doğruluğu: {test_acc:.4f}")
    
    if test_acc < 0.6:
        logger.warning("UYARI: Model performansı düşük. Veri sayısı az olabilir.")
    
    # --- PART B: REGRESSION (Cost) ---
    logger.info("--- Regresyon Modeli (Ridge Regression) ---")
    # Ridge, katsayıları küçülterek overfitting'i engeller (L2 Regularization).
    
    reg = Ridge(alpha=1.0, random_state=42)
    
    # Cross Validation
    cv_mse = -cross_val_score(reg, X_train, y_reg_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(cv_mse)
    logger.info(f"CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")
    
    reg.fit(X_train, y_reg_train)
    
    # Test Evaluation
    y_pred_reg = reg.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    test_r2 = r2_score(y_reg_test, y_pred_reg)
    logger.info(f"Test Seti RMSE: {test_rmse:.2f}")
    logger.info(f"Test Seti R2 Skoru: {test_r2:.4f}")
    
    # --- PART C: SAVE WEIGHTS FOR HE ---
    # HE tarafında sadece Toplama ve Çarpma yapabiliyoruz.
    # Bu yüzden modelin katsayılarını (Weights) ve Sabitini (Intercept/Bias) alıyoruz.
    
    model_weights = {
        "feature_names": list(X_train.columns),
        "classification": {
            "algorithm": "LogisticRegression",
            "weights": clf.coef_[0].tolist(),
            "bias": float(clf.intercept_[0]),
            "classes": clf.classes_.tolist()
        },
        "regression": {
            "algorithm": "Ridge",
            "weights": reg.coef_.tolist(),
            "bias": float(reg.intercept_)
        },
        "meta": {
            "train_samples": len(X_train),
            "test_acc": test_acc,
            "test_r2": test_r2
        }
    }
    
    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(model_weights, f, indent=2)
        
    logger.info("ADIM 2 Tamamlandı: Model katsayıları kaydedildi.")
    logger.info(f"Katsayı Dosyası: {WEIGHTS_FILE}")
    
    # Check for overfitting logic
    if cv_scores.mean() > test_acc + 0.15:
        logger.warning("DİKKAT: Sınıflandırma modelinde Overfitting riski var (CV >> Test)")
    else:
        logger.info("Sınıflandırma modeli dengeli görünüyor (Ezber yok).")

if __name__ == "__main__":
    train_models()

