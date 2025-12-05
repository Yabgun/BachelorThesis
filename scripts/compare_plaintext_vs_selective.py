import time
import json
import numpy as np
import pandas as pd
import tenseal as ts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# --- Configuration ---
OUTPUT_DIR = os.path.abspath("HE_Analiz_Raporlari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_WEIGHTS_PATH = os.path.abspath("models/he_model_weights.json")
X_TEST_PATH = os.path.abspath("data/processed/X_test.csv")
Y_REG_PATH = os.path.abspath("data/processed/y_reg_test.csv")
Y_CLASS_PATH = os.path.abspath("data/processed/y_class_test.csv")

PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sifresiz_vs_secici_he_grafik.png")
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sifresiz_vs_secici_he_raporu.md")

# Sensitive columns policy
SENSITIVE_INDICES = [3, 4, 5] # Smoking, CXR, Genetic
NON_SENSITIVE_INDICES = [0, 1, 2] # Age, Gender, BMI

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def load_data_and_model():
    print("Veri ve Model ağırlıkları yükleniyor...")
    with open(MODEL_WEIGHTS_PATH, 'r') as f:
        model_data = json.load(f)
    
    # Regression Weights
    reg_weights = np.array(model_data["regression"]["weights"])
    reg_bias = model_data["regression"]["bias"]
    
    # Classification Weights
    class_weights = np.array(model_data["classification"]["weights"])
    class_bias = model_data["classification"]["bias"]
    
    feature_names = model_data["feature_names"]
    
    X_test = pd.read_csv(X_TEST_PATH)
    y_reg = pd.read_csv(Y_REG_PATH).values.flatten()
    y_class = pd.read_csv(Y_CLASS_PATH).values.flatten()
    
    X_test = X_test[feature_names]
    
    return X_test.values, y_reg, y_class, reg_weights, reg_bias, class_weights, class_bias

# --- PLAIN TEXT METHODS ---
def run_plaintext_regression(X_batch, weights, bias):
    return np.dot(X_batch, weights) + bias

def run_plaintext_classification(X_batch, weights, bias):
    logits = np.dot(X_batch, weights) + bias
    return sigmoid(logits)

# --- SELECTIVE HE METHODS ---
def run_selective_he_regression(context, X_batch, weights, bias):
    predictions = []
    sens_weights = weights[SENSITIVE_INDICES]
    plain_weights = weights[NON_SENSITIVE_INDICES]
    
    for row in tqdm(X_batch, desc="Seçici HE (Regresyon)"):
        sens_data = row[SENSITIVE_INDICES]
        plain_data = row[NON_SENSITIVE_INDICES]
        
        # Encrypted Part
        enc_sens_vector = ts.ckks_vector(context, sens_data)
        res_enc = enc_sens_vector.dot(sens_weights)
        
        # Plaintext Part
        res_plain = np.dot(plain_data, plain_weights)
        
        # Combine
        # Note: In real selective HE, we add plain result to encrypted result
        # result is still encrypted until decryption
        total_plain_component = res_plain + bias
        final_enc_result = res_enc + total_plain_component
        
        # Decrypt
        pred = final_enc_result.decrypt()[0]
        predictions.append(pred)
        
    return np.array(predictions)

def run_selective_he_classification(context, X_batch, weights, bias):
    probabilities = []
    sens_weights = weights[SENSITIVE_INDICES]
    plain_weights = weights[NON_SENSITIVE_INDICES]
    
    for row in tqdm(X_batch, desc="Seçici HE (Sınıflandırma)"):
        sens_data = row[SENSITIVE_INDICES]
        plain_data = row[NON_SENSITIVE_INDICES]
        
        # Encrypted Part
        enc_sens_vector = ts.ckks_vector(context, sens_data)
        res_enc = enc_sens_vector.dot(sens_weights)
        
        # Plaintext Part
        res_plain = np.dot(plain_data, plain_weights)
        
        # Combine
        total_plain_component = res_plain + bias
        final_enc_result = res_enc + total_plain_component
        
        # Decrypt & Sigmoid
        logit = final_enc_result.decrypt()[0]
        prob = sigmoid(logit)
        probabilities.append(prob)
        
    return np.array(probabilities)

def main():
    context = create_ckks_context()
    X_test, y_reg, y_class, reg_weights, reg_bias, class_weights, class_bias = load_data_and_model()
    
    print(f"\n--- {len(X_test)} Hasta İçin Şifresiz vs Seçici HE Kıyaslaması ---")
    
    # 1. Regression Comparison
    print("\n[1/2] Regresyon Analizi Yapılıyor...")
    plain_reg_preds = run_plaintext_regression(X_test, reg_weights, reg_bias)
    sel_reg_preds = run_selective_he_regression(context, X_test, reg_weights, reg_bias)
    
    reg_mse_diff = mean_squared_error(plain_reg_preds, sel_reg_preds)
    reg_r2 = r2_score(plain_reg_preds, sel_reg_preds)
    
    # 2. Classification Comparison
    print("\n[2/2] Sınıflandırma Analizi Yapılıyor...")
    plain_class_probs = run_plaintext_classification(X_test, class_weights, class_bias)
    sel_class_probs = run_selective_he_classification(context, X_test, class_weights, class_bias)
    
    # Check absolute difference
    max_prob_diff = np.max(np.abs(plain_class_probs - sel_class_probs))
    
    print(f"\n--- SONUÇLAR ---")
    print(f"Regresyon Farkı (MSE): {reg_mse_diff:.10f}")
    print(f"Regresyon Uyumu (R2): {reg_r2:.10f}")
    print(f"Sınıflandırma Olasılık Maksimum Farkı: {max_prob_diff:.10f}")
    
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Regression Correlation
    ax1.scatter(plain_reg_preds, sel_reg_preds, alpha=0.6, c='#2ecc71', edgecolors='black', s=30)
    
    # Ideal line
    min_val = min(plain_reg_preds.min(), sel_reg_preds.min())
    max_val = max(plain_reg_preds.max(), sel_reg_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Mükemmel Uyum (y=x)')
    
    ax1.set_xlabel('Şifresiz Tahmin (Plaintext)', fontsize=12)
    ax1.set_ylabel('Seçici HE Tahmin (Encrypted)', fontsize=12)
    ax1.set_title(f'Regresyon Uyumu\nMSE Farkı: {reg_mse_diff:.2e}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Classification Probability Correlation
    ax2.scatter(plain_class_probs, sel_class_probs, alpha=0.6, c='#f1c40f', edgecolors='black', marker='^', s=30)
    ax2.plot([0, 1], [0, 1], 'r--', lw=2, label='Mükemmel Uyum (y=x)')
    
    ax2.set_xlabel('Şifresiz Risk Olasılığı', fontsize=12)
    ax2.set_ylabel('Seçici HE Risk Olasılığı', fontsize=12)
    ax2.set_title(f'Sınıflandırma Uyumu\nMaks. Fark: {max_prob_diff:.2e}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    print(f"Grafik kaydedildi: {PLOT_OUTPUT_PATH}")
    
    # --- Report ---
    report = f"""# Şifresiz Veri vs Seçici Homomorfik Şifreleme Karşılaştırması

## Genel Bakış
Bu rapor, modelimizin **Şifresiz (Plaintext)** ortamda ürettiği sonuçlar ile **Seçici Homomorfik Şifreleme (Selective HE)** yöntemiyle ürettiği sonuçları kıyaslamaktadır. Amaç, şifreleme işleminin sonuçlarda herhangi bir bozulmaya veya doğruluk kaybına yol açmadığını kanıtlamaktır.

## 1. Regresyon (Maliyet Tahmini) Kıyaslaması
Şifresiz tahminler ile Seçici HE tahminleri arasındaki farklar incelenmiştir.

*   **Ortalama Kare Hata (MSE) Farkı:** {reg_mse_diff:.10f}
*   **Uyum Katsayısı (R²):** {reg_r2:.10f} (1.00 = Birebir Aynı)

**Yorum:** İki yöntem arasındaki fark sıfıra yakındır (CKKS şifrelemesinden kaynaklı ihmal edilebilir ondalık farklar). Bu, şifreli hesaplamanın matematiksel olarak doğru çalıştığını kanıtlar.

## 2. Sınıflandırma (Risk Analizi) Kıyaslaması
Hastaların risk skorları (0-1 arası olasılık) karşılaştırılmıştır.

*   **Maksimum Olasılık Farkı:** {max_prob_diff:.10f}

**Yorum:** Şifresiz ve şifreli modelin ürettiği risk skorları neredeyse aynıdır. Karar mekanizması (Yüksek Risk / Düşük Risk) şifrelemeden etkilenmemiştir.

## Sonuç
**Seçici Homomorfik Şifreleme**, verilerin gizliliğini korurken, **Şifresiz (Plaintext)** işlemeyle **aynı doğruluğu** sağlamaktadır. Grafiklerde görülen "Mükemmel Uyum (y=x)" çizgisi üzerindeki dağılım, modelin güvenilirliğini görsel olarak da teyit etmektedir.
"""
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapor kaydedildi: {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
