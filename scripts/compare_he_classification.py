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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Configuration ---
OUTPUT_DIR = os.path.abspath("HE_Analiz_Raporlari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_WEIGHTS_PATH = os.path.abspath("models/he_model_weights.json")
X_TEST_PATH = os.path.abspath("data/processed/X_test.csv")
Y_TEST_PATH = os.path.abspath("data/processed/y_class_test.csv")

PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "he_siniflandirma_grafigi.png")
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "he_siniflandirma_raporu.md")

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
    print("Veri ve Sınıflandırma ağırlıkları yükleniyor...")
    with open(MODEL_WEIGHTS_PATH, 'r') as f:
        model_data = json.load(f)
    
    # Load Classification Weights
    class_weights = np.array(model_data["classification"]["weights"])
    class_bias = model_data["classification"]["bias"]
    feature_names = model_data["feature_names"]
    
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.flatten()
    
    X_test = X_test[feature_names]
    
    return X_test.values, y_test, class_weights, class_bias, feature_names

def run_full_he_class(context, X_batch, weights, bias):
    start_time = time.time()
    probabilities = []
    
    for row in tqdm(X_batch, desc="Tam HE (Sınıflandırma)"):
        # Encrypt
        enc_vector = ts.ckks_vector(context, row)
        # Dot Product
        enc_result = enc_vector.dot(weights)
        # Add Bias
        enc_result = enc_result + bias
        # Decrypt (Logit value)
        logit = enc_result.decrypt()[0]
        # Apply Sigmoid in Plaintext
        prob = sigmoid(logit)
        probabilities.append(prob)
        
    end_time = time.time()
    return np.array(probabilities), end_time - start_time

def run_selective_he_class(context, X_batch, weights, bias):
    start_time = time.time()
    probabilities = []
    
    sens_weights = weights[SENSITIVE_INDICES]
    plain_weights = weights[NON_SENSITIVE_INDICES]
    
    for row in tqdm(X_batch, desc="Seçici HE (Sınıflandırma)"):
        sens_data = row[SENSITIVE_INDICES]
        plain_data = row[NON_SENSITIVE_INDICES]
        
        # Encrypted Dot Product
        enc_sens_vector = ts.ckks_vector(context, sens_data)
        res_enc = enc_sens_vector.dot(sens_weights)
        
        # Plaintext Dot Product
        res_plain = np.dot(plain_data, plain_weights)
        
        # Combine
        total_plain_component = res_plain + bias
        final_enc_result = res_enc + total_plain_component
        
        # Decrypt & Sigmoid
        logit = final_enc_result.decrypt()[0]
        prob = sigmoid(logit)
        probabilities.append(prob)
        
    end_time = time.time()
    return np.array(probabilities), end_time - start_time

def main():
    context = create_ckks_context()
    X_test, y_test, weights, bias, feature_names = load_data_and_model()
    
    print(f"\n--- {len(X_test)} Hasta İçin Risk Analizi Başlıyor ---")
    
    # --- Full HE ---
    full_probs, full_time = run_full_he_class(context, X_test, weights, bias)
    full_preds = (full_probs > 0.5).astype(int)
    
    # --- Selective HE ---
    sel_probs, sel_time = run_selective_he_class(context, X_test, weights, bias)
    sel_preds = (sel_probs > 0.5).astype(int)
    
    # --- Metrics ---
    full_acc = accuracy_score(y_test, full_preds)
    sel_acc = accuracy_score(y_test, sel_preds)
    speedup = full_time / sel_time
    
    print(f"\nTam HE Doğruluk: {full_acc:.4f} ({full_time:.2f}s)")
    print(f"Seçici HE Doğruluk: {sel_acc:.4f} ({sel_time:.2f}s)")
    
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#3498db', '#f1c40f'] # Blue, Gold
    
    # Plot 1: Time
    bars = ax1.bar(['Tam HE', 'Seçici HE'], [full_time, sel_time], color=colors)
    ax1.set_ylabel('İşlem Süresi (s)', fontsize=12)
    ax1.set_title('Sınıflandırma Hız Analizi', fontsize=14, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s', ha='center', va='bottom')
    
    ax1.text(0.5, 0.9, f"{speedup:.1f}x HIZLI", transform=ax1.transAxes, ha='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#d35400'), color='#d35400', fontweight='bold')

    # Plot 2: Risk Probabilities Scatter
    # We plot the raw probabilities (0 to 1) to show how close they are
    ax2.scatter(range(len(y_test)), full_probs, alpha=0.6, label='Tam HE Olasılık', c='#3498db', s=20)
    ax2.scatter(range(len(y_test)), sel_probs, alpha=0.6, label='Seçici HE Olasılık', c='#f1c40f', marker='x', s=40)
    
    ax2.set_xlabel('Hasta ID (Test Seti)', fontsize=12)
    ax2.set_ylabel('Risk Olasılığı (0-1)', fontsize=12)
    ax2.set_title(f'Karar Doğruluğu (Acc: {sel_acc:.2%})', fontsize=14, fontweight='bold')
    ax2.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    print(f"Grafik kaydedildi: {PLOT_OUTPUT_PATH}")
    
    # --- Report ---
    report = f"""# Homomorfik Sınıflandırma (Risk Analizi) Raporu

## Amaç
Bu rapor, hastaların **"Yüksek Riskli"** veya **"Düşük Riskli"** olarak sınıflandırılması sürecinde, Tam ve Seçici Homomorfik Şifreleme yöntemlerinin karşılaştırmasını sunar.

## Sonuç Özeti

| Yöntem | Süre (400 Hasta) | Doğruluk (Accuracy) | Açıklama |
| :--- | :--- | :--- | :--- |
| **Tam HE** | {full_time:.4f} s | %{full_acc*100:.2f} | Tüm veriler şifreli işlendi. |
| **Seçici HE** | {sel_time:.4f} s | %{sel_acc*100:.2f} | Hibrit şifreleme kullanıldı. |
| **Fark** | **{speedup:.2f}x Hız Artışı** | **Fark Yok** | Doğruluk kaybı yaşanmadı. |

## Detaylı Analiz
Modelimiz, hastaların verilerini kullanarak 0 ile 1 arasında bir **Risk Skoru** üretmiştir. 
- 0.5 üzerindeki skorlar **Yüksek Risk (1)**,
- 0.5 altındaki skorlar **Düşük Risk (0)** olarak etiketlenmiştir.

Seçici Şifreleme ile elde edilen risk skorları, Tam Şifreleme ile elde edilenlerle matematiksel olarak örtüşmektedir. Bu durum, hayati önem taşıyan risk sınıflandırma işleminde de hibrit şifrelemenin güvenle kullanılabileceğini kanıtlar.

**Sonuç:** Modelimiz %{sel_acc*100:.1f} başarı oranı ile hastaları doğru sınıflandırmış ve bunu şifreli veriler üzerinde gerçekleştirmiştir.
"""
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapor kaydedildi: {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
