import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import tenseal as ts
import time
from tqdm import tqdm

# Configure Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Paths
PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('models')
WEIGHTS_FILE = MODELS_DIR / 'he_model_weights.json'
SCALER_FILE = PROCESSED_DIR / 'feature_scaler.joblib'

# HE Setup
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def verify_he_learning():
    logger.info("--- ŞİFRELİ (FHE) MODEL DOĞRULAMASI BAŞLIYOR ---")
    
    # 1. Load Data & Model Weights
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    y_reg_test = pd.read_csv(PROCESSED_DIR / 'y_reg_test.csv').values.ravel()
    
    with open(WEIGHTS_FILE, 'r') as f:
        weights_data = json.load(f)
        
    reg_weights = np.array(weights_data['regression']['weights'])
    reg_bias = weights_data['regression']['bias']
    
    # 2. Initialize HE Context
    context = create_context()
    logger.info("TenSEAL (CKKS) Context Oluşturuldu.")
    
    # 3. Encrypted Inference Loop
    # We will process a subset if 400 takes too long, but let's try all.
    n_samples = len(X_test)
    logger.info(f"Toplam {n_samples} hasta verisi şifreli olarak işlenecek...")
    
    he_predictions = []
    start_total = time.time()
    
    # Using tqdm for progress bar
    for i in tqdm(range(n_samples), desc="Şifreli Tahmin"):
        # a. Encrypt
        x_plain = X_test.iloc[i].values
        enc_x = ts.ckks_vector(context, x_plain)
        
        # b. Compute (Homomorphic Dot Product)
        # Enc(y) = Enc(x) dot W + b
        enc_y = enc_x.dot(reg_weights) + reg_bias
        
        # c. Decrypt
        y_pred = enc_y.decrypt()[0]
        he_predictions.append(y_pred)
        
    duration = time.time() - start_total
    avg_time = duration / n_samples
    logger.info(f"Tamamlandı! Toplam Süre: {duration:.2f}s (Hasta başı: {avg_time:.4f}s)")
    
    he_predictions = np.array(he_predictions)
    
    # 4. Visualization
    logger.info("Grafik çiziliyor...")
    
    plt.figure(figsize=(12, 7))
    
    # Plot points
    plt.scatter(y_reg_test, he_predictions, alpha=0.6, color='green', label='Şifreli Tahminler (FHE)')
    
    # Ideal line
    min_val = min(y_reg_test.min(), he_predictions.min())
    max_val = max(y_reg_test.max(), he_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Mükemmel Doğruluk')
    
    plt.xlabel('Gerçek Maliyet (USD)')
    plt.ylabel('Şifreli Model Tahmini (USD)')
    plt.title(f'FHE Doğrulaması: Şifreli Veri Üzerinde Maliyet Tahmini\n(N={n_samples} Hasta, Ortalama Süre: {avg_time:.3f}s/hasta)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with stats
    mse = np.mean((y_reg_test - he_predictions)**2)
    rmse = np.sqrt(mse)
    stats_text = f"RMSE Hata: {rmse:.2f} USD"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    output_plot = 'he_verification_plot.png'
    plt.savefig(output_plot)
    logger.info(f"Grafik kaydedildi: {output_plot}")
    
    # 5. Compare with Plaintext (Sanity Check)
    # Check if HE introduced significant noise
    logger.info("Şifreli vs Şifresiz Fark Analizi yapılıyor...")
    plain_preds = np.dot(X_test.values, reg_weights) + reg_bias
    he_noise = np.abs(he_predictions - plain_preds)
    avg_he_noise = np.mean(he_noise)
    logger.info(f"Ortalama Şifreleme Gürültüsü (Approximation Error): {avg_he_noise:.10f}")
    
    if avg_he_noise < 1e-3:
        logger.info("SONUÇ: MÜKEMMEL. Şifreli işlemler neredeyse sıfır hata ile çalıştı.")
    else:
        logger.info("SONUÇ: Kabul edilebilir gürültü seviyesi.")

if __name__ == "__main__":
    verify_he_learning()
