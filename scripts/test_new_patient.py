import json
import numpy as np
import pandas as pd
import tenseal as ts
import joblib
import os

# --- Configuration ---
MODEL_WEIGHTS_PATH = os.path.abspath("models/he_model_weights.json")
SCALER_PATH = os.path.abspath("data/processed/feature_scaler.joblib")

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

def load_resources():
    print("Model ve Scaler yükleniyor...")
    
    # Load Weights
    with open(MODEL_WEIGHTS_PATH, 'r') as f:
        model_data = json.load(f)
        
    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    return model_data, scaler

def run_plaintext_prediction(patient_data_norm, model_data):
    """
    Performs prediction using standard unencrypted arithmetic for comparison.
    """
    # 1. Regression (Cost)
    reg_weights = np.array(model_data["regression"]["weights"])
    reg_bias = model_data["regression"]["bias"]
    
    reg_res = np.dot(patient_data_norm, reg_weights) + reg_bias
    
    # 2. Classification (Risk)
    class_weights = np.array(model_data["classification"]["weights"])
    class_bias = model_data["classification"]["bias"]
    
    class_logit = np.dot(patient_data_norm, class_weights) + class_bias
    risk_prob = sigmoid(class_logit)
    
    return reg_res, risk_prob

def run_selective_he_prediction(context, patient_data_norm, model_data):
    # 1. Regression (Cost) Prediction
    reg_weights = np.array(model_data["regression"]["weights"])
    reg_bias = model_data["regression"]["bias"]
    
    # 2. Classification (Risk) Prediction
    class_weights = np.array(model_data["classification"]["weights"])
    class_bias = model_data["classification"]["bias"]
    
    # Split Data
    sens_data = patient_data_norm[SENSITIVE_INDICES]
    plain_data = patient_data_norm[NON_SENSITIVE_INDICES]
    
    # --- REGRESSION CALCULATION ---
    # Encrypted Part
    enc_sens_vector = ts.ckks_vector(context, sens_data)
    reg_res_enc = enc_sens_vector.dot(reg_weights[SENSITIVE_INDICES])
    
    # Plaintext Part
    reg_res_plain = np.dot(plain_data, reg_weights[NON_SENSITIVE_INDICES])
    
    # Combine & Decrypt
    reg_final_enc = reg_res_enc + (reg_res_plain + reg_bias)
    predicted_cost = reg_final_enc.decrypt()[0]
    
    # --- CLASSIFICATION CALCULATION ---
    # Encrypted Part
    class_res_enc = enc_sens_vector.dot(class_weights[SENSITIVE_INDICES])
    
    # Plaintext Part
    class_res_plain = np.dot(plain_data, class_weights[NON_SENSITIVE_INDICES])
    
    # Combine & Decrypt
    class_final_enc = class_res_enc + (class_res_plain + class_bias)
    logit = class_final_enc.decrypt()[0]
    risk_prob = sigmoid(logit)
    
    return predicted_cost, risk_prob

def main():
    context = create_ckks_context()
    model_data, scaler = load_resources()
    feature_names = model_data["feature_names"]
    
    print("\n--- GERÇEKÇİ (ORTA RİSKLİ) HASTA SENARYOSU ---")
    
    # Define a "Medium Risk" profile
    # Senaryo: 45 yaşında, kilolu ama sigara içmiyor, akciğeri temiz, sadece genetik yatkınlığı var.
    new_patient_raw = {
        "Age": 45,              # Orta Yaş
        "Gender": 0,            # Kadın
        "BMI": 27.5,            # Hafif Kilolu
        "Smoking": 0,           # Sigara YOK (Risk Düşürücü)
        "CXR_Opacity": 0,       # Akciğer Temiz (Risk Düşürücü)
        "Genetic_Marker": 1     # Genetik Yatkınlık VAR (Risk Artırıcı)
    }
    
    print("\n[1] Hasta Profili (Ham Veri):")
    for k, v in new_patient_raw.items():
        status = "VAR" if v == 1 and k not in ["Age", "BMI", "Gender"] else ("YOK" if v == 0 and k not in ["Age", "BMI", "Gender"] else v)
        print(f"  - {k}: {status}")
        
    # Convert to DataFrame for Scaler
    df_new = pd.DataFrame([new_patient_raw], columns=feature_names)
    
    # Scale the data (Normalize)
    # Important: We must use the SAME scaler used during training!
    patient_data_norm = scaler.transform(df_new)[0]
    
    print("\n[2] Veri Ön İşleme:")
    print(f"  - Normalizasyon işlemi başarıyla uygulandı.")
    print(f"  - İşlenecek Vektör: {patient_data_norm}")
    
    # Run Prediction
    print("\n[3] Seçici Homomorfik Tahmin Başlıyor...")
    print("  -> Hassas veriler (Smoking, CXR, Genetic) ŞİFRELENDİ.")
    print("  -> Açık veriler (Age, Gender, BMI) işleme alındı.")
    print("  -> Şifreli ortamda hesaplama yapılıyor...")
    
    he_cost, he_risk = run_selective_he_prediction(context, patient_data_norm, model_data)
    
    print("\n[4] Şifresiz (Plaintext) Doğrulama Başlıyor...")
    print("  -> Kontrol amaçlı standart Python hesaplaması yapılıyor...")
    plain_cost, plain_risk = run_plaintext_prediction(patient_data_norm, model_data)

    print("\n--- KARŞILAŞTIRMALI SONUÇLAR ---")
    print(f"{'Metrik':<20} | {'Şifreli (HE)':<20} | {'Şifresiz (Plain)':<20} | {'Fark':<20}")
    print("-" * 90)
    print(f"{'Sağlık Harcaması':<20} | ${he_cost:<19.4f} | ${plain_cost:<19.4f} | {abs(he_cost - plain_cost):.10f}")
    print(f"{'Risk Olasılığı':<20} | %{he_risk*100:<19.6f} | %{plain_risk*100:<19.6f} | {abs(he_risk - plain_risk):.10f}")
    
    print("\n--- YORUM ---")
    if abs(he_risk - plain_risk) < 1e-5:
        print("✅ MÜKEMMEL SONUÇ: Şifreli hesaplama, şifresiz hesaplama ile BİREBİR AYNI sonucu verdi.")
        print("   (Fark sıfıra o kadar yakın ki ihmal edilebilir.)")
    else:
        print("⚠️ UYARI: Sonuçlar arasında beklenenden fazla fark var.")

    print("\n--- KLİNİK KARAR (Ayşe Hanım İçin) ---")
    if he_risk > 0.5:
        print(f"[KARAR]: YÜKSEK RİSKLİ HASTA 🚨 (Risk: %{he_risk*100:.2f})")
    else:
        print(f"[KARAR]: Düşük Riskli Hasta ✅ (Risk: %{he_risk*100:.2f})")
        print("  -> Hastanın genetik yatkınlığı olsa da, sigara içmemesi ve akciğerlerinin temiz olması riski düşürdü.")

if __name__ == "__main__":
    main()

