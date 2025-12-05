import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path('data') # Changed from data/covid_ct_cxr to data root
INPUT_CSV = DATA_DIR / 'multimodal.csv'
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    logger.info("ADIM 1: Veri Hazırlama Başlıyor...")
    
    # 1. Load Data
    if not INPUT_CSV.exists():
        logger.error(f"Veri dosyası bulunamadı: {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Veri yüklendi. Toplam kayıt: {len(df)}")
    
    # 2. Feature Selection
    # Updated to match synthetic data columns
    feature_cols = ['Age', 'Gender', 'BMI', 'Smoking', 'CXR_Opacity', 'Genetic_Marker']
    
    # Targets
    target_class = 'Risk_Class'
    target_reg = 'Treatment_Cost'
    
    # 3. Clean Data
    # Eksik verileri doldur (Mean imputation)
    for col in feature_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            logger.info(f"Eksik veriler dolduruldu: {col} (Ort: {mean_val:.2f})")
            
    X = df[feature_cols]
    y_class = df[target_class]
    y_reg = df[target_reg]
    
    # 4. Split Data (Train/Test)
    # Gelecek dataları simüle etmek için %20'yi ayırıyoruz.
    # random_state sabit tutuyoruz ki sonuçlar tekrarlanabilir olsun.
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    logger.info(f"Eğitim Seti: {len(X_train)} hasta")
    logger.info(f"Test Seti (Gelecek Data): {len(X_test)} hasta")
    
    # 5. Scaling (Standardization)
    # Lineer modeller ve HE için verilerin aynı ölçekte olması KRİTİKTİR.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Test setini eğitim setinin istatistikleriyle dönüştür
    
    # DataFrame'e geri çevir (kolaylık olsun diye)
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # 6. Save Artifacts
    # İşlenmiş verileri ve Scaler'ı kaydediyoruz.
    # Böylece "yeni hasta" geldiğinde aynı scaler'ı kullanabileceğiz.
    
    joblib.dump(scaler, PROCESSED_DIR / 'feature_scaler.joblib')
    
    X_train_df.to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    X_test_df.to_csv(PROCESSED_DIR / 'X_test.csv', index=False)
    
    y_class_train.to_csv(PROCESSED_DIR / 'y_class_train.csv', index=False)
    y_class_test.to_csv(PROCESSED_DIR / 'y_class_test.csv', index=False)
    
    y_reg_train.to_csv(PROCESSED_DIR / 'y_reg_train.csv', index=False)
    y_reg_test.to_csv(PROCESSED_DIR / 'y_reg_test.csv', index=False)
    
    logger.info("ADIM 1 Tamamlandı: Veriler işlendi, normalize edildi ve kaydedildi.")
    logger.info(f"Kayıt Yeri: {PROCESSED_DIR}")

if __name__ == "__main__":
    prepare_data()

