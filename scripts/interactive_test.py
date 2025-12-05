import json
import numpy as np
import pandas as pd
import tenseal as ts
import joblib
import os
import sys

# --- Configuration ---
# Paths are relative to the project root
MODEL_WEIGHTS_PATH = os.path.abspath("models/he_model_weights.json")
SCALER_PATH = os.path.abspath("data/processed/feature_scaler.joblib")

# Sensitive columns policy
SENSITIVE_INDICES = [3, 4, 5] # Smoking, CXR, Genetic
NON_SENSITIVE_INDICES = [0, 1, 2] # Age, Gender, BMI

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_ckks_context():
    """Creates the homomorphic encryption context."""
    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    except Exception as e:
        print(f"❌ HATA: Şifreleme bağlamı oluşturulamadı: {e}")
        sys.exit(1)

def load_resources():
    """Loads the trained model weights and scaler."""
    if not os.path.exists(MODEL_WEIGHTS_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ HATA: Model dosyaları bulunamadı! Lütfen önce eğitimi tamamlayın.")
        print(f"Aranan yollar:\n- {MODEL_WEIGHTS_PATH}\n- {SCALER_PATH}")
        sys.exit(1)

    print("⏳ Sistem başlatılıyor... (Model ve Anahtarlar Yükleniyor)")
    
    # Load Weights
    with open(MODEL_WEIGHTS_PATH, 'r') as f:
        model_data = json.load(f)
        
    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    return model_data, scaler

def run_selective_he_prediction(context, patient_data_norm, model_data):
    """Performs the prediction using Selective Homomorphic Encryption."""
    
    # 1. Get Weights & Biases
    reg_weights = np.array(model_data["regression"]["weights"])
    reg_bias = model_data["regression"]["bias"]
    class_weights = np.array(model_data["classification"]["weights"])
    class_bias = model_data["classification"]["bias"]
    
    # 2. Split Data (Sensitive vs Plain)
    sens_data = patient_data_norm[SENSITIVE_INDICES]
    plain_data = patient_data_norm[NON_SENSITIVE_INDICES]
    
    # 3. Encrypt Sensitive Data
    enc_sens_vector = ts.ckks_vector(context, sens_data)
    
    # --- REGRESSION (COST) ---
    # Encrypted Dot Product
    reg_res_enc = enc_sens_vector.dot(reg_weights[SENSITIVE_INDICES])
    # Plaintext Dot Product
    reg_res_plain = np.dot(plain_data, reg_weights[NON_SENSITIVE_INDICES])
    # Combine
    reg_final_enc = reg_res_enc + (reg_res_plain + reg_bias)
    # Decrypt Result
    predicted_cost = reg_final_enc.decrypt()[0]
    
    # --- CLASSIFICATION (RISK) ---
    # Encrypted Dot Product
    class_res_enc = enc_sens_vector.dot(class_weights[SENSITIVE_INDICES])
    # Plaintext Dot Product
    class_res_plain = np.dot(plain_data, class_weights[NON_SENSITIVE_INDICES])
    # Combine
    class_final_enc = class_res_enc + (class_res_plain + class_bias)
    # Decrypt Result & Apply Sigmoid
    logit = class_final_enc.decrypt()[0]
    risk_prob = sigmoid(logit)
    
    return predicted_cost, risk_prob

def get_user_input():
    """Interactively prompts the user for patient data."""
    print("\n" + "="*50)
    print("🏥 HASTA VERİ GİRİŞ EKRANI")
    print("="*50)
    print("Lütfen aşağıdaki bilgileri sırasıyla giriniz:\n")
    
    try:
        # Age
        while True:
            try:
                age = float(input("1. Yaş (Örn: 45): "))
                if 0 < age < 120: break
                else: print("   ⚠️ Lütfen geçerli bir yaş giriniz.")
            except ValueError: print("   ⚠️ Lütfen sayısal bir değer giriniz.")

        # Gender
        while True:
            g_input = input("2. Cinsiyet (E: Erkek / K: Kadın): ").strip().upper()
            if g_input in ['E', 'K']:
                gender = 1 if g_input == 'E' else 0
                break
            else: print("   ⚠️ Lütfen 'E' veya 'K' giriniz.")

        # BMI
        while True:
            try:
                bmi = float(input("3. Vücut Kitle İndeksi (BMI) (Örn: 27.5): "))
                if 10 < bmi < 60: break
                else: print("   ⚠️ Lütfen geçerli bir BMI giriniz (10-60 arası).")
            except ValueError: print("   ⚠️ Lütfen sayısal bir değer giriniz.")

        # Smoking
        while True:
            s_input = input("4. Sigara Kullanıyor mu? (E: Evet / H: Hayır): ").strip().upper()
            if s_input in ['E', 'H']:
                smoking = 1 if s_input == 'E' else 0
                break
            else: print("   ⚠️ Lütfen 'E' veya 'H' giriniz.")

        # CXR (Lung Opacity)
        while True:
            c_input = input("5. Akciğer Röntgeninde Opasite (Leke) Var mı? (E: Evet / H: Hayır): ").strip().upper()
            if c_input in ['E', 'H']:
                cxr = 1 if c_input == 'E' else 0
                break
            else: print("   ⚠️ Lütfen 'E' veya 'H' giriniz.")

        # Genetic Marker
        while True:
            gm_input = input("6. Ailede Genetik Hastalık Öyküsü Var mı? (E: Evet / H: Hayır): ").strip().upper()
            if gm_input in ['E', 'H']:
                genetic = 1 if gm_input == 'E' else 0
                break
            else: print("   ⚠️ Lütfen 'E' veya 'H' giriniz.")

    except KeyboardInterrupt:
        print("\n\n🚫 İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)

    return {
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Smoking": smoking,
        "CXR_Opacity": cxr,
        "Genetic_Marker": genetic
    }

def main():
    # 1. Initialize System
    context = create_ckks_context()
    model_data, scaler = load_resources()
    feature_names = model_data["feature_names"]

    while True:
        # 2. Get Data
        patient_data = get_user_input()
        
        # 3. Preprocess
        df_new = pd.DataFrame([patient_data], columns=feature_names)
        patient_data_norm = scaler.transform(df_new)[0]
        
        print("\n🔄 Veriler İşleniyor ve Şifreleniyor...")
        print(f"   -> Açık Veriler: Yaş={patient_data['Age']}, BMI={patient_data['BMI']}")
        print(f"   -> 🔒 Şifreli Veriler: Sigara, Röntgen, Genetik")
        
        # 4. Predict
        cost, risk = run_selective_he_prediction(context, patient_data_norm, model_data)
        
        # 5. Show Results
        print("\n" + "-"*50)
        print("📄 ANALİZ SONUCU")
        print("-" * 50)
        print(f"💰 Tahmini Yıllık Sağlık Harcaması: ${cost:,.2f}")
        print(f"⚠️ Hastalık Risk Olasılığı:        %{risk*100:.2f}")
        
        if risk > 0.5:
            print("\n🚨 SONUÇ: YÜKSEK RİSKLİ HASTA")
            print("   (Önleyici tedavi ve detaylı kontrol önerilir.)")
        else:
            print("\n✅ SONUÇ: DÜŞÜK RİSKLİ HASTA")
            print("   (Mevcut sağlık durumu stabil görünüyor.)")
        print("-" * 50)
        
        # 6. Loop
        cont = input("\nBaşka bir hasta için test yapmak ister misiniz? (E/H): ").strip().upper()
        if cont != 'E':
            print("\n👋 Programdan çıkılıyor. Sağlıklı günler!")
            break

if __name__ == "__main__":
    main()
