import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg') # Set backend to Agg for file saving only
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# Paths
PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('models')
WEIGHTS_FILE = MODELS_DIR / 'he_model_weights.json'
SCALER_FILE = PROCESSED_DIR / 'feature_scaler.joblib'

def verify_learning():
    print("--- MODEL ÖĞRENME DOĞRULAMASI ---")
    
    # 1. Load Data and Model
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    y_reg_test = pd.read_csv(PROCESSED_DIR / 'y_reg_test.csv').values.ravel()
    
    with open(WEIGHTS_FILE, 'r') as f:
        weights_data = json.load(f)
        
    reg_weights = np.array(weights_data['regression']['weights'])
    reg_bias = weights_data['regression']['bias']
    
    scaler = joblib.load(SCALER_FILE)
    
    # 2. Make Predictions (Plaintext for verification)
    # y_pred = X * W + b
    y_pred = np.dot(X_test.values, reg_weights) + reg_bias
    
    # 3. Interpret Weights (Reverse Scaling)
    # The weights we have are for SCALED data. To see if they match the formula (Age*100, etc.),
    # we need to consider the variance of the features.
    # However, a simpler check is just looking at the relative importance.
    
    print("\n[1] Ağırlık Analizi (Hangi özellik maliyeti ne kadar artırıyor?)")
    feature_names = X_test.columns.tolist()
    
    # Combine feature names with their learned weights
    importance = list(zip(feature_names, reg_weights))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, weight in importance:
        print(f"  - {name}: {weight:.4f} (Modelin öğrendiği katsayı)")
        
    print("\n  YORUM: 'CXR_Opacity', 'Smoking' ve 'Age' katsayıları en yüksek olmalı.")
    print("  Çünkü formülümüzde en büyük çarpanlar onlardı.")

    # 4. Visualization
    print("\n[2] Grafik Oluşturuluyor: Gerçek vs Tahmin...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_reg_test, y_pred, alpha=0.6, color='blue', label='Hasta Verisi')
    
    # Perfect prediction line
    min_val = min(y_reg_test.min(), y_pred.min())
    max_val = max(y_reg_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Mükemmel Tahmin Çizgisi')
    
    plt.xlabel('Gerçek Maliyet (USD)')
    plt.ylabel('Model Tahmini (USD)')
    plt.title('Model Doğrulaması: Gerçek vs Tahmin Edilen Maliyet')
    plt.legend()
    plt.grid(True)
    
    output_plot = 'verification_plot.png'
    plt.savefig(output_plot)
    print(f"  Grafik kaydedildi: {output_plot}")
    print("  Eğer noktalar kırmızı çizgi üzerindeyse, model MÜKEMMEL öğrenmiş demektir.")

if __name__ == "__main__":
    verify_learning()

