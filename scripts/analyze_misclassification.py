import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
OUTPUT_DIR = os.path.abspath("HE_Analiz_Raporlari")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_WEIGHTS_PATH = os.path.abspath("models/he_model_weights.json")
X_TEST_PATH = os.path.abspath("data/processed/X_test.csv")
Y_TEST_PATH = os.path.abspath("data/processed/y_class_test.csv")

PLOT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hatali_siniflandirma_analizi.png")
REPORT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "hatali_siniflandirma_detayi.md")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data_and_model():
    print("Veri yükleniyor...")
    with open(MODEL_WEIGHTS_PATH, 'r') as f:
        model_data = json.load(f)
    
    class_weights = np.array(model_data["classification"]["weights"])
    class_bias = model_data["classification"]["bias"]
    feature_names = model_data["feature_names"]
    
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.flatten()
    
    # Ensure correct column order
    X_test = X_test[feature_names]
    
    return X_test, y_test, class_weights, class_bias, feature_names

def main():
    X_test_df, y_test, weights, bias, feature_names = load_data_and_model()
    X_test = X_test_df.values
    
    print(f"Toplam {len(X_test)} hasta analiz ediliyor...")
    
    # --- Prediction (Plaintext is sufficient for logic analysis) ---
    logits = np.dot(X_test, weights) + bias
    probs = sigmoid(logits)
    preds = (probs > 0.5).astype(int)
    
    # --- Identify Errors ---
    # Indices where prediction does not match ground truth
    error_indices = np.where(preds != y_test)[0]
    
    print(f"Hatalı tahmin sayısı: {len(error_indices)}")
    
    if len(error_indices) == 0:
        print("Hata bulunamadı! Model %100 doğru çalışıyor.")
        return

    # --- Analysis of the first error (assuming 1 error based on 99.75%) ---
    idx = error_indices[0]
    patient_data = X_test_df.iloc[idx]
    true_label = y_test[idx]
    pred_label = preds[idx]
    prob = probs[idx]
    
    print(f"\n--- Hatalı Hasta (ID: {idx}) Analizi ---")
    print(f"Gerçek Durum: {'Yüksek Risk (1)' if true_label == 1 else 'Düşük Risk (0)'}")
    print(f"Model Tahmini: {'Yüksek Risk (1)' if pred_label == 1 else 'Düşük Risk (0)'}")
    print(f"Modelin Risk Olasılığı: {prob:.4f} (Karar Sınırı: 0.5)")
    print("\nHasta Özellikleri:")
    print(patient_data)
    
    # --- Visualization ---
    # Only Plot 1: Probability Distribution
    plt.figure(figsize=(10, 7))
    ax1 = plt.gca()
    
    # Sort probabilities for cleaner S-curve look
    sorted_indices = np.argsort(probs)
    sorted_probs = probs[sorted_indices]
    
    # Color points based on correctness
    colors = ['green' if preds[i] == y_test[i] else 'red' for i in sorted_indices]
    
    ax1.scatter(range(len(probs)), sorted_probs, c=colors, alpha=0.6, s=30)
    # ax1.axhline(y=0.5, color='gray', linestyle='--', label='Karar Sınırı (0.5)')  # REMOVED as requested
    
    # Highlight the error point
    # Find where the error index ended up after sorting
    sorted_error_pos = np.where(sorted_indices == idx)[0][0]
    ax1.scatter(sorted_error_pos, probs[idx], c='red', s=150, edgecolors='black', label='Hatalı Tahmin', zorder=5)
    
    # Annotate the error
    true_text = 'Yüksek Risk' if true_label == 1 else 'Düşük Risk'
    ax1.annotate(f'Hata\nOlasılık: {prob:.2f}\nGerçek: {true_text}', 
                 xy=(sorted_error_pos, probs[idx]), 
                 xytext=(sorted_error_pos - 50, probs[idx] + 0.1 if probs[idx] < 0.5 else probs[idx] - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))

    ax1.set_title('Risk Olasılıkları Dağılımı ve Hata Noktası', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Hesaplanan Risk Olasılığı')
    ax1.set_xlabel('Hasta (Olasılığa Göre Sıralı)')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    print(f"Grafik kaydedildi: {PLOT_OUTPUT_PATH}")
    
    # --- Generate Report ---
    report = f"""# Hatalı Sınıflandırma Analizi

## Özet
Toplam **{len(X_test)}** hasta içerisinden **{len(error_indices)}** adet hatalı sınıflandırma tespit edilmiştir.
Doğruluk Oranı: **%{(1 - len(error_indices)/len(X_test))*100:.2f}**

## Hatalı Vaka Detayı (ID: {idx})
Bu hasta gerçekte **{'Yüksek Risk (1)' if true_label == 1 else 'Düşük Risk (0)'}** grubundadır, ancak model **{'Yüksek Risk (1)' if pred_label == 1 else 'Düşük Risk (0)'}** tahmini yapmıştır.

### Neden Hata Yapıldı?
Model bu hasta için **%{prob*100:.2f}** risk hesaplamıştır.
Karar sınırı %50 olduğu için, bu değer sınıra çok yakın olabilir veya hastanın bazı özellikleri (örn. yaşı, genetik skoru) modelin kafasını karıştırmış olabilir.

### Hastanın Verileri:
| Özellik | Değer | Model Ağırlığı (Etkisi) |
| :--- | :--- | :--- |
"""
    for i, feature in enumerate(feature_names):
        val = patient_data[feature]
        w = weights[i]
        impact = val * w
        report += f"| **{feature}** | {val:.4f} | {w:.4f} (Katkı: {impact:.4f}) |\n"
    
    logit_val = np.dot(patient_data.values, weights) + bias
    report += f"\n**Toplam Logit Skoru:** {logit_val:.4f} (Bias: {bias:.4f} dahil)\n"
    report += f"**Sigmoid Sonrası Olasılık:** {prob:.6f}\n"
    
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapor kaydedildi: {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
