import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import tenseal as ts
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path('models')
WEIGHTS_FILE = MODELS_DIR / 'he_model_weights.json'
PROCESSED_DIR = Path('data/processed')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class HEClient:
    def __init__(self):
        # Setup TenSEAL Context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        
    def encrypt_features(self, feature_vector):
        """Encrypts a numpy array of features"""
        return ts.ckks_vector(self.context, feature_vector)

    def decrypt_result(self, enc_result):
        """Decrypts the result"""
        # Result is a list with one element
        return enc_result.decrypt()[0]

class HEServer:
    def __init__(self, weights_path):
        with open(weights_path, 'r') as f:
            self.model_data = json.load(f)
        self.class_weights = np.array(self.model_data['classification']['weights'])
        self.class_bias = self.model_data['classification']['bias']
        self.reg_weights = np.array(self.model_data['regression']['weights'])
        self.reg_bias = self.model_data['regression']['bias']
        
    def secure_inference(self, enc_features, task='classification'):
        """
        Performs Homomorphic Dot Product: W * X + b
        """
        weights = self.class_weights if task == 'classification' else self.reg_weights
        bias = self.class_bias if task == 'classification' else self.reg_bias
        
        # TenSEAL supports dot product directly
        # enc_features is a vector, weights is a vector
        enc_dot = enc_features.dot(weights)
        
        # Add bias
        enc_result = enc_dot + bias
        
        return enc_result

def run_simulation():
    logger.info("ADIM 3: Client-Server Şifreli Analiz Simülasyonu Başlıyor (TenSEAL)...")
    
    # 1. Setup
    client = HEClient()
    server = HEServer(WEIGHTS_FILE)
    
    # 2. Load Test Data (Simulate New Patient)
    try:
        X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
        y_class_test = pd.read_csv(PROCESSED_DIR / 'y_class_test.csv').values.ravel()
        y_reg_test = pd.read_csv(PROCESSED_DIR / 'y_reg_test.csv').values.ravel()
    except FileNotFoundError:
        logger.error("Veri dosyaları bulunamadı. Lütfen önce prepare_robust_data.py ve train_robust_models.py çalıştırın.")
        return

    # Pick a random patient
    idx = np.random.randint(0, len(X_test))
    patient_features = X_test.iloc[idx].values
    true_class = y_class_test[idx]
    true_cost = y_reg_test[idx]
    
    logger.info(f"Hasta Seçildi (Index: {idx})")
    logger.info(f"Gerçek Veriler -> Risk Sınıfı: {true_class}, Maliyet: {true_cost:.2f}")
    logger.info("--------------------------------------------------")
    
    # 3. CLIENT: Encryption
    logger.info("[CLIENT] Veriler şifreleniyor...")
    start_time = time.time()
    enc_features = client.encrypt_features(patient_features)
    enc_time = time.time() - start_time
    logger.info(f"[CLIENT] Şifreleme Tamamlandı ({enc_time:.4f}s)")
    
    # 4. SERVER: Secure Inference (Classification)
    logger.info("[SERVER] Şifreli Risk Analizi yapılıyor...")
    start_time = time.time()
    # Server doesn't need client object, just the encrypted vector
    enc_risk_logit = server.secure_inference(enc_features, task='classification')
    risk_inf_time = time.time() - start_time
    logger.info(f"[SERVER] Analiz Bitti ({risk_inf_time:.4f}s)")
    
    # 5. SERVER: Secure Inference (Regression)
    logger.info("[SERVER] Şifreli Maliyet Hesabı yapılıyor...")
    start_time = time.time()
    enc_cost_pred = server.secure_inference(enc_features, task='regression')
    cost_inf_time = time.time() - start_time
    logger.info(f"[SERVER] Analiz Bitti ({cost_inf_time:.4f}s)")
    
    # 6. CLIENT: Decryption & Interpretation
    logger.info("[CLIENT] Sonuçlar alınıyor ve çözülüyor...")
    
    # Risk
    risk_logit = client.decrypt_result(enc_risk_logit)
    risk_prob = sigmoid(risk_logit)
    risk_pred = 1 if risk_prob > 0.5 else 0
    
    # Cost
    cost_pred = client.decrypt_result(enc_cost_pred)
    
    logger.info("--------------------------------------------------")
    logger.info("SONUÇ RAPORU:")
    logger.info(f"Risk Analizi (Tahmin): {risk_pred} (Olasılık: {risk_prob:.4f}) | Gerçek: {true_class}")
    logger.info(f"Maliyet Tahmini: {cost_pred:.2f} | Gerçek: {true_cost:.2f}")
    logger.info(f"Maliyet Hatası: {abs(cost_pred - true_cost):.2f}")
    
    if risk_pred == true_class:
        logger.info("✅ Risk Sınıflandırması DOĞRU")
    else:
        logger.info("❌ Risk Sınıflandırması YANLIŞ")
        
    logger.info("--------------------------------------------------")
    logger.info("Bu model ezberlememiş, şifreli uzayda matematiksel işlem yaparak sonucu bulmuştur.")

if __name__ == "__main__":
    run_simulation()