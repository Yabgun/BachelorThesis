#!/usr/bin/env python3
"""
ML Encrypted Data Preparation Script
Healthcare Data Encryption with Selective CKKS Strategy

This script implements the client-side encryption + ML model architecture
as specified in the research proposal form, focusing on selective encryption
of critical healthcare features while maintaining ML model compatibility.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib

# Pyfhel for CKKS encryption
from Pyfhel import Pyfhel, PyCtxt, PyPtxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EncryptionConfig:
    """Configuration for selective encryption strategy"""
    # Critical features to encrypt (high privacy risk)
    encrypt_features: List[str] = None
    # Plaintext features (low privacy risk)
    plaintext_features: List[str] = None
    # CKKS parameters
    n: int = 8192
    scale: int = 1099511627776  # 2^40
    qi_sizes: List[int] = None
    
    def __post_init__(self):
        if self.encrypt_features is None:
            self.encrypt_features = [
                'test_results_score',      # High sensitivity: medical test results
                'cxr_mean_intensity',      # High sensitivity: chest X-ray features
                'cxr_edge_density'         # High sensitivity: medical imaging features
            ]
        if self.plaintext_features is None:
            self.plaintext_features = [
                'age',                     # Low sensitivity: demographic
                'billing_amount_norm'       # Low sensitivity: financial
            ]
        if self.qi_sizes is None:
            self.qi_sizes = [60, 40, 40, 60]  # Optimal from previous analysis

class MLEncryptedDataPrep:
    """
    ML Encrypted Data Preparation Class
    
    Implements selective encryption strategy where:
    - Critical medical features (test_results, imaging) are encrypted with CKKS
    - Less sensitive features (age, billing) remain plaintext
    - Maintains ML model compatibility with mixed encrypted/plaintext features
    """
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.he = None
        self.encryption_times = []
        self.setup_encryption()
        
    def setup_encryption(self):
        """Initialize Pyfhel with CKKS parameters"""
        logger.info("Setting up CKKS encryption with parameters:")
        logger.info(f"n: {self.config.n}, scale: {self.config.scale}")
        logger.info(f"qi_sizes: {self.config.qi_sizes}")
        
        self.he = Pyfhel()
        self.he.contextGen(
            scheme='CKKS',
            n=self.config.n,
            scale=self.config.scale,
            qi_sizes=self.config.qi_sizes
        )
        self.he.keyGen()
        self.he.relinKeyGen()
        
    def load_multimodal_data(self, data_path: str) -> pd.DataFrame:
        """Load multimodal healthcare dataset"""
        logger.info(f"Loading multimodal data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} patient records with {len(df.columns)} features")
        
        # Validate required features exist
        required_features = self.config.encrypt_features + self.config.plaintext_features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        return df
    
    def create_risk_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create risk classification labels based on test results and imaging features
        Following medical risk stratification approach
        """
        logger.info("Creating risk classification labels")
        
        # Risk score based on test results (primary indicator)
        test_risk = df['test_results_score'].apply(
            lambda x: 'High' if x >= 0.8 else ('Medium' if x >= 0.3 else 'Low')
        )
        
        # Additional risk from imaging features
        # Higher entropy and edge density can indicate pathology
        imaging_risk = df.apply(lambda row: 
            'High' if (row['cxr_entropy'] > 7.2 and row['cxr_edge_density'] > 0.15) 
            else ('Medium' if (row['cxr_entropy'] > 6.8 or row['cxr_edge_density'] > 0.1) 
            else 'Low'), axis=1
        )
        
        # Combined risk assessment
        def combine_risk(test, imaging):
            if test == 'High' or imaging == 'High':
                return 'High'
            elif test == 'Medium' or imaging == 'Medium':
                return 'Medium'
            else:
                return 'Low'
        
        risk_labels = df.apply(
            lambda row: combine_risk(
                'High' if row['test_results_score'] >= 0.8 else ('Medium' if row['test_results_score'] >= 0.3 else 'Low'),
                'High' if (row['cxr_entropy'] > 7.2 and row['cxr_edge_density'] > 0.15) 
                else ('Medium' if (row['cxr_entropy'] > 6.8 or row['cxr_edge_density'] > 0.1) 
                else 'Low')
            ), axis=1
        )
        
        # Risk encoding: High=2, Medium=1, Low=0
        risk_encoded = risk_labels.map({'Low': 0, 'Medium': 1, 'High': 2})
        
        logger.info(f"Risk distribution:\n{risk_labels.value_counts()}")
        return risk_encoded
    
    def encrypt_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encrypt critical features using CKKS while preserving plaintext features
        Returns encrypted dataframe and encryption metadata
        """
        logger.info("Starting selective encryption process")
        
        encrypted_df = df.copy()
        encryption_metadata = {
            'encrypted_features': [],
            'plaintext_features': [],
            'encryption_times': {},
            'feature_stats': {}
        }
        
        # Process plaintext features (no encryption)
        for feature in self.config.plaintext_features:
            logger.info(f"Keeping {feature} as plaintext")
            encryption_metadata['plaintext_features'].append(feature)
            encryption_metadata['feature_stats'][feature] = {
                'type': 'plaintext',
                'mean': float(df[feature].mean()),
                'std': float(df[feature].std()),
                'min': float(df[feature].min()),
                'max': float(df[feature].max())
            }
        
        # Process encrypted features
        for feature in self.config.encrypt_features:
            logger.info(f"Encrypting {feature} with CKKS")
            start_time = time.time()
            
            # Convert to numpy array for encryption
            feature_values = df[feature].values.astype(np.float64)
            
            # Encrypt each value individually (for ML compatibility)
            encrypted_values = []
            for value in feature_values:
                # Encode and encrypt
                ptxt = self.he.encodeFrac(np.array([value], dtype=np.float64))
                ctxt = self.he.encryptPtxt(ptxt)
                encrypted_values.append(ctxt)
            
            encryption_time = time.time() - start_time
            self.encryption_times.append(encryption_time)
            
            # Store encrypted values (serialized for dataframe compatibility)
            encrypted_df[f"{feature}_encrypted"] = encrypted_values
            
            # Remove original plaintext feature
            encrypted_df = encrypted_df.drop(columns=[feature])
            
            encryption_metadata['encrypted_features'].append(feature)
            encryption_metadata['encryption_times'][feature] = encryption_time
            encryption_metadata['feature_stats'][f"{feature}_encrypted"] = {
                'type': 'encrypted',
                'original_mean': float(df[feature].mean()),
                'original_std': float(df[feature].std()),
                'encryption_time': encryption_time,
                'encryption_count': len(feature_values)
            }
            
            logger.info(f"Encrypted {len(feature_values)} values for {feature} in {encryption_time:.4f}s")
        
        logger.info(f"Selective encryption completed. Encrypted {len(self.config.encrypt_features)} features")
        return encrypted_df, encryption_metadata
    
    def prepare_ml_datasets(self, encrypted_df: pd.DataFrame, risk_labels: pd.Series, 
                          test_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare ML datasets with train/test split maintaining encryption structure
        """
        logger.info("Preparing ML datasets with encrypted features")
        
        # Separate features and target
        feature_columns = [col for col in encrypted_df.columns if col != 'patient_id']
        X = encrypted_df[feature_columns]
        y = risk_labels
        
        # Split data maintaining encryption structure
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create datasets dictionary
        datasets = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_columns,
            'n_features': len(feature_columns),
            'n_samples': len(encrypted_df),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_distribution': {
                'train': y_train.value_counts().to_dict(),
                'test': y_test.value_counts().to_dict()
            }
        }
        
        logger.info(f"Dataset prepared: {datasets['n_samples']} total samples")
        logger.info(f"Training set: {datasets['n_train']} samples")
        logger.info(f"Test set: {datasets['n_test']} samples")
        logger.info(f"Features: {datasets['n_features']} total")
        
        return datasets
    
    def save_prepared_data(self, encrypted_df: pd.DataFrame, encryption_metadata: Dict, 
                          datasets: Dict, output_dir: str):
        """Save prepared data and metadata for ML training"""
        logger.info(f"Saving prepared data to: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save encrypted dataframe
        encrypted_df_path = os.path.join(output_dir, 'encrypted_features.csv')
        # For encrypted features, save metadata instead of actual ciphertexts
        encrypted_df_meta = encrypted_df.copy()
        for col in encrypted_df.columns:
            if '_encrypted' in col:
                encrypted_df_meta[col] = encrypted_df_meta[col].apply(lambda x: f"encrypted_{type(x).__name__}")
        encrypted_df_meta.to_csv(encrypted_df_path, index=False)
        
        # Save encryption metadata
        metadata_path = os.path.join(output_dir, 'encryption_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(encryption_metadata, f, indent=2, default=str)
        
        # Save datasets (without encrypted objects for pickle compatibility)
        datasets_clean = datasets.copy()
        for key in ['X_train', 'X_test']:
            datasets_clean[key] = datasets_clean[key].to_dict('records')
        
        datasets_path = os.path.join(output_dir, 'ml_datasets.json')
        with open(datasets_path, 'w') as f:
            json.dump(datasets_clean, f, indent=2, default=str)
        
        # Save Pyfhel context for later decryption
        context_path = os.path.join(output_dir, 'pyfhel_context.pkl')
        joblib.dump(self.he, context_path)
        
        logger.info("Data preparation completed and saved successfully")
        
        # Generate summary report
        self.generate_summary_report(encryption_metadata, datasets, output_dir)
    
    def generate_summary_report(self, encryption_metadata: Dict, datasets: Dict, output_dir: str):
        """Generate comprehensive summary report"""
        report_path = os.path.join(output_dir, 'data_preparation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# ML Encrypted Data Preparation Report\n\n")
            f.write("## Selective Encryption Strategy\n\n")
            f.write("Following the research proposal's selective homomorphic encryption approach:\n\n")
            
            f.write("### Encrypted Features (High Privacy Risk)\n")
            for feature in encryption_metadata['encrypted_features']:
                f.write(f"- **{feature}**: Medical/clinical data with high sensitivity\n")
            
            f.write("\n### Plaintext Features (Low Privacy Risk)\n")
            for feature in encryption_metadata['plaintext_features']:
                f.write(f"- **{feature}**: Demographic/financial data with lower sensitivity\n")
            
            f.write(f"\n### Encryption Performance\n")
            total_encryption_time = sum(encryption_metadata['encryption_times'].values())
            f.write(f"- Total encryption time: {total_encryption_time:.4f} seconds\n")
            f.write(f"- Average per feature: {total_encryption_time/len(encryption_metadata['encrypted_features']):.4f} seconds\n")
            
            f.write(f"\n### Dataset Statistics\n")
            f.write(f"- Total samples: {datasets['n_samples']}\n")
            f.write(f"- Training samples: {datasets['n_train']}\n")
            f.write(f"- Test samples: {datasets['n_test']}\n")
            f.write(f"- Total features: {datasets['n_features']}\n")
            f.write(f"- Encrypted features: {len(encryption_metadata['encrypted_features'])}\n")
            f.write(f"- Plaintext features: {len(encryption_metadata['plaintext_features'])}\n")
            
            f.write(f"\n### Class Distribution\n")
            f.write("Training set:\n")
            for class_label, count in datasets['class_distribution']['train'].items():
                f.write(f"- Class {class_label}: {count} samples\n")
            
            f.write("\nTest set:\n")
            for class_label, count in datasets['class_distribution']['test'].items():
                f.write(f"- Class {class_label}: {count} samples\n")
            
            f.write(f"\n### Files Generated\n")
            f.write(f"- `encrypted_features.csv`: Encrypted feature matrix\n")
            f.write(f"- `encryption_metadata.json`: Encryption parameters and statistics\n")
            f.write(f"- `ml_datasets.json`: Training/test datasets\n")
            f.write(f"- `pyfhel_context.pkl`: Pyfhel context for decryption\n")
        
        logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main execution function"""
    logger.info("Starting ML Encrypted Data Preparation")
    
    # Configuration
    DATA_PATH = "data/covid_ct_cxr/multimodal.csv"
    OUTPUT_DIR = "data/ml_encrypted"
    
    try:
        # Initialize data preparation
        data_prep = MLEncryptedDataPrep()
        
        # Load multimodal data
        df = data_prep.load_multimodal_data(DATA_PATH)
        
        # Create risk classification labels
        risk_labels = data_prep.create_risk_labels(df)
        
        # Apply selective encryption
        encrypted_df, encryption_metadata = data_prep.encrypt_features(df)
        
        # Prepare ML datasets
        datasets = data_prep.prepare_ml_datasets(encrypted_df, risk_labels)
        
        # Save prepared data
        data_prep.save_prepared_data(encrypted_df, encryption_metadata, datasets, OUTPUT_DIR)
        
        logger.info("ML Encrypted Data Preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()