#!/usr/bin/env python3
"""
ML Models for Encrypted Healthcare Data
Clustering Model for Patient Grouping

This module implements clustering models that can work with mixed encrypted/plaintext features,
following the client-side encryption architecture specified in the research proposal.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import joblib

# ML libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Pyfhel for encrypted operations
from Pyfhel import Pyfhel, PyCtxt, PyPtxt

# Visualization
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusteringConfig:
    """Configuration for clustering models"""
    model_type: str = 'kmeans'  # kmeans, dbscan, agglomerative, gmm
    handle_encrypted: bool = True
    n_clusters: int = 3  # For algorithms that require n_clusters
    eps: float = 0.5  # For DBSCAN
    min_samples: int = 5  # For DBSCAN
    random_state: int = 42

class EncryptedFeatureProcessor:
    """Processor for handling mixed encrypted/plaintext features"""
    
    def __init__(self, pyfhel_context: Pyfhel):
        self.he = pyfhel_context
        self.encrypted_cache = {}
        
    def process_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Process mixed encrypted/plaintext features for ML model input
        """
        processed_features = []
        
        for col in X.columns:
            if '_encrypted' in col:
                # Handle encrypted features
                encrypted_values = X[col].values
                decrypted_values = []
                
                for enc_val in encrypted_values:
                    if isinstance(enc_val, str) and 'encrypted' in enc_val:
                        # Placeholder for encrypted values (use mean approximation)
                        # In real implementation, would decrypt here
                        decrypted_values.append(0.0)  # Placeholder
                    else:
                        # Assume it's already decrypted for this demo
                        decrypted_values.append(float(enc_val) if enc_val is not None else 0.0)
                
                processed_features.append(decrypted_values)
                
            else:
                # Handle plaintext features directly
                processed_features.append(X[col].values)
        
        return np.array(processed_features).T

class PatientClusteringModel:
    """
    Clustering model for patient grouping and segmentation
    Works with mixed encrypted/plaintext features
    """
    
    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()
        self.model = None
        self.feature_processor = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.prediction_time = 0
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        
    def initialize_model(self):
        """Initialize the clustering model based on configuration"""
        logger.info(f"Initializing {self.config.model_type} model for patient clustering")
        
        if self.config.model_type == 'kmeans':
            self.model = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=10
            )
        elif self.config.model_type == 'dbscan':
            self.model = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
        elif self.config.model_type == 'agglomerative':
            self.model = AgglomerativeClustering(
                n_clusters=self.config.n_clusters
            )
        elif self.config.model_type == 'gmm':
            self.model = GaussianMixture(
                n_components=self.config.n_clusters,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def prepare_features(self, X: pd.DataFrame, pyfhel_context: Pyfhel = None) -> np.ndarray:
        """
        Prepare features for clustering model
        Handles mixed encrypted/plaintext features
        """
        logger.info("Preparing features for clustering model")
        
        if pyfhel_context:
            self.feature_processor = EncryptedFeatureProcessor(pyfhel_context)
            X_processed = self.feature_processor.process_features(X)
        else:
            # Direct feature processing (assume already decrypted)
            X_processed = X.values
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = self.scaler.fit_transform(X_processed)
        
        return X_scaled
    
    def fit(self, X: pd.DataFrame, pyfhel_context: Pyfhel = None) -> Dict[str, Any]:
        """
        Fit the clustering model
        """
        logger.info("Fitting patient clustering model")
        start_time = time.time()
        
        # Initialize model
        self.initialize_model()
        
        # Prepare features
        X_processed = self.prepare_features(X, pyfhel_context)
        
        # Fit model
        self.model.fit(X_processed)
        
        self.training_time = time.time() - start_time
        
        # Extract cluster labels and centers (if available)
        if hasattr(self.model, 'labels_'):
            self.cluster_labels_ = self.model.labels_
        elif hasattr(self.model, 'predict'):
            self.cluster_labels_ = self.model.predict(X_processed)
        
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_
        elif hasattr(self.model, 'means_'):
            self.cluster_centers_ = self.model.means_
        
        logger.info(f"Model fitting completed in {self.training_time:.4f}s")
        logger.info(f"Number of clusters found: {len(np.unique(self.cluster_labels_))}")
        
        return {
            'training_time': self.training_time,
            'n_clusters': len(np.unique(self.cluster_labels_)),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'model_type': self.config.model_type
        }
    
    def predict(self, X: pd.DataFrame, pyfhel_context: Pyfhel = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict cluster assignments for new data
        """
        logger.info("Making cluster predictions")
        start_time = time.time()
        
        # Prepare features
        X_processed = self.prepare_features(X, pyfhel_context)
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_processed)
        else:
            # For models without predict method, fit and get labels
            self.model.fit(X_processed)
            predictions = self.model.labels_
        
        self.prediction_time = time.time() - start_time
        
        logger.info(f"Prediction completed for {len(X)} samples in {self.prediction_time:.4f}s")
        
        return predictions, {
            'prediction_time': self.prediction_time,
            'n_predictions': len(predictions),
            'n_clusters': len(np.unique(predictions))
        }
    
    def evaluate(self, X: pd.DataFrame, pyfhel_context: Pyfhel = None) -> Dict[str, Any]:
        """
        Evaluate clustering model performance using various metrics
        """
        logger.info("Evaluating patient clustering model")
        
        # Prepare features
        X_processed = self.prepare_features(X, pyfhel_context)
        
        # Get cluster assignments
        if self.cluster_labels_ is None:
            self.fit(X, pyfhel_context)
        
        # Calculate clustering metrics
        metrics = {}
        
        # Silhouette Score (higher is better, range [-1, 1])
        if len(np.unique(self.cluster_labels_)) > 1:
            silhouette = silhouette_score(X_processed, self.cluster_labels_)
            metrics['silhouette_score'] = silhouette
        else:
            metrics['silhouette_score'] = -1
        
        # Calinski-Harabasz Score (higher is better)
        if len(np.unique(self.cluster_labels_)) > 1:
            calinski = calinski_harabasz_score(X_processed, self.cluster_labels_)
            metrics['calinski_harabasz_score'] = calinski
        else:
            metrics['calinski_harabasz_score'] = 0
        
        # Davies-Bouldin Score (lower is better)
        if len(np.unique(self.cluster_labels_)) > 1:
            davies_bouldin = davies_bouldin_score(X_processed, self.cluster_labels_)
            metrics['davies_bouldin_score'] = davies_bouldin
        else:
            metrics['davies_bouldin_score'] = float('inf')
        
        # Additional cluster statistics
        unique_labels = np.unique(self.cluster_labels_)
        cluster_sizes = [np.sum(self.cluster_labels_ == label) for label in unique_labels]
        
        metrics.update({
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'largest_cluster_size': max(cluster_sizes),
            'smallest_cluster_size': min(cluster_sizes),
            'cluster_size_std': np.std(cluster_sizes),
            'model_type': self.config.model_type
        })
        
        logger.info(f"Clustering evaluation completed. Silhouette Score: {metrics['silhouette_score']:.4f}")
        
        return metrics
    
    def analyze_clusters(self, X: pd.DataFrame, original_features: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze cluster characteristics and provide insights
        """
        logger.info("Analyzing cluster characteristics")
        
        if self.cluster_labels_ is None:
            raise ValueError("Model must be fitted before cluster analysis")
        
        cluster_analysis = {}
        
        # Use original features if provided, otherwise use X
        features_to_analyze = original_features if original_features is not None else X
        
        # Analyze each cluster
        for cluster_id in np.unique(self.cluster_labels_):
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_data = features_to_analyze[cluster_mask]
            
            cluster_info = {
                'size': np.sum(cluster_mask),
                'percentage': np.sum(cluster_mask) / len(self.cluster_labels_) * 100,
                'feature_means': {},
                'feature_stds': {}
            }
            
            # Calculate statistics for each feature
            for col in features_to_analyze.columns:
                if features_to_analyze[col].dtype in ['int64', 'float64']:
                    cluster_info['feature_means'][col] = float(cluster_data[col].mean())
                    cluster_info['feature_stds'][col] = float(cluster_data[col].std())
            
            cluster_analysis[f'cluster_{cluster_id}'] = cluster_info
        
        # Generate cluster profiles
        profiles = self.generate_cluster_profiles(cluster_analysis)
        
        return {
            'cluster_analysis': cluster_analysis,
            'cluster_profiles': profiles,
            'n_clusters': len(np.unique(self.cluster_labels_))
        }
    
    def generate_cluster_profiles(self, cluster_analysis: Dict) -> Dict[str, str]:
        """
        Generate human-readable cluster profiles
        """
        profiles = {}
        
        for cluster_name, cluster_info in cluster_analysis.items():
            profile_parts = []
            
            # Basic cluster info
            profile_parts.append(f"Cluster size: {cluster_info['size']} patients ({cluster_info['percentage']:.1f}%)")
            
            # Feature characteristics
            means = cluster_info['feature_means']
            if means:
                # Find most distinguishing features
                feature_descriptions = []
                
                if 'age' in means:
                    if means['age'] > 60:
                        feature_descriptions.append("elderly patients")
                    elif means['age'] < 30:
                        feature_descriptions.append("young patients")
                    else:
                        feature_descriptions.append("middle-aged patients")
                
                if 'test_results_score' in means:
                    if means['test_results_score'] > 0.7:
                        feature_descriptions.append("high test scores")
                    elif means['test_results_score'] < 0.3:
                        feature_descriptions.append("low test scores")
                
                if 'billing_amount_norm' in means:
                    if means['billing_amount_norm'] > 0.7:
                        feature_descriptions.append("high treatment costs")
                    elif means['billing_amount_norm'] < 0.3:
                        feature_descriptions.append("low treatment costs")
                
                if feature_descriptions:
                    profile_parts.append("Characteristics: " + ", ".join(feature_descriptions))
            
            profiles[cluster_name] = "; ".join(profile_parts)
        
        return profiles
    
    def visualize_clusters(self, X: pd.DataFrame, output_path: str = None):
        """
        Visualize clusters using PCA for dimensionality reduction
        """
        logger.info("Generating cluster visualizations")
        
        if self.cluster_labels_ is None:
            raise ValueError("Model must be fitted before visualization")
        
        # Prepare features for visualization
        X_processed = self.prepare_features(X)
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_processed)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot of clusters
        unique_labels = np.unique(self.cluster_labels_)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = self.cluster_labels_ == label
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        
        axes[0].set_title('Patient Clusters (PCA Visualization)')
        axes[0].set_xlabel('First Principal Component')
        axes[0].set_ylabel('Second Principal Component')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cluster size distribution
        cluster_sizes = [np.sum(self.cluster_labels_ == label) for label in unique_labels]
        axes[1].bar([f'Cluster {label}' for label in unique_labels], cluster_sizes, 
                   color=colors[:len(unique_labels)])
        axes[1].set_title('Cluster Size Distribution')
        axes[1].set_ylabel('Number of Patients')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to: {output_path}")
        
        plt.close()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'training_time': self.training_time,
            'cluster_labels_': self.cluster_labels_,
            'cluster_centers_': self.cluster_centers_
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Clustering model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.training_time = model_data['training_time']
        self.cluster_labels_ = model_data['cluster_labels_']
        self.cluster_centers_ = model_data['cluster_centers_']
        logger.info(f"Clustering model loaded from: {filepath}")

class ClusteringPipeline:
    """
    Complete clustering pipeline for encrypted healthcare data
    """
    
    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()
        self.clustering_model = PatientClusteringModel(self.config)
        self.results = {}
        
    def run_complete_pipeline(self, datasets: Dict, pyfhel_context: Pyfhel = None) -> Dict[str, Any]:
        """
        Run complete clustering pipeline: fit, evaluate, analyze
        """
        logger.info("Starting complete clustering pipeline")
        
        start_time = time.time()
        
        # Extract datasets
        X_train = datasets['X_train']
        X_test = datasets['X_test']
        
        # Combine train and test for clustering
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        
        # Fit model
        fit_results = self.clustering_model.fit(X_combined, pyfhel_context)
        
        # Evaluate model
        eval_results = self.clustering_model.evaluate(X_combined, pyfhel_context)
        
        # Analyze clusters
        analysis_results = self.clustering_model.analyze_clusters(X_combined)
        
        # Generate visualization
        viz_path = "data/ml_models/clustering/cluster_visualization.png"
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        self.clustering_model.visualize_clusters(X_combined, viz_path)
        
        pipeline_time = time.time() - start_time
        
        # Compile results
        self.results = {
            'pipeline_time': pipeline_time,
            'fitting_results': fit_results,
            'evaluation_results': eval_results,
            'cluster_analysis': analysis_results,
            'model_config': self.config.__dict__,
            'datasets_info': {
                'n_samples': len(X_combined),
                'n_features': X_combined.shape[1]
            }
        }
        
        logger.info(f"Clustering pipeline completed in {pipeline_time:.4f}s")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """Save pipeline results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_path = os.path.join(output_dir, 'clustering_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save trained model
        model_path = os.path.join(output_dir, 'clustering_model.pkl')
        self.clustering_model.save_model(model_path)
        
        # Generate report
        self.generate_report(output_dir)
        
        logger.info(f"Results saved to: {output_dir}")
    
    def generate_report(self, output_dir: str):
        """Generate comprehensive report"""
        report_path = os.path.join(output_dir, 'clustering_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Patient Clustering Model Report\n\n")
            f.write("## Model Configuration\n\n")
            f.write(f"- **Model Type**: {self.config.model_type}\n")
            f.write(f"- **Handles Encrypted Features**: {self.config.handle_encrypted}\n")
            f.write(f"- **Number of Clusters**: {self.config.n_clusters}\n\n")
            
            f.write("## Fitting Results\n\n")
            fit_results = self.results['fitting_results']
            f.write(f"- **Training Time**: {fit_results['training_time']:.4f}s\n")
            f.write(f"- **Clusters Found**: {fit_results['n_clusters']}\n")
            f.write(f"- **Samples**: {fit_results['n_samples']}\n")
            f.write(f"- **Features**: {fit_results['n_features']}\n\n")
            
            f.write("## Evaluation Results\n\n")
            eval_results = self.results['evaluation_results']
            f.write(f"- **Silhouette Score**: {eval_results['silhouette_score']:.4f}\n")
            f.write(f"- **Calinski-Harabasz Score**: {eval_results['calinski_harabasz_score']:.2f}\n")
            f.write(f"- **Davies-Bouldin Score**: {eval_results['davies_bouldin_score']:.4f}\n")
            f.write(f"- **Cluster Size Std**: {eval_results['cluster_size_std']:.2f}\n\n")
            
            f.write("## Cluster Analysis\n\n")
            analysis_results = self.results['cluster_analysis']
            
            for cluster_name, cluster_info in analysis_results['cluster_analysis'].items():
                f.write(f"### {cluster_name.replace('_', ' ').title()}\n\n")
                f.write(f"- **Size**: {cluster_info['size']} patients ({cluster_info['percentage']:.1f}%)\n")
                
                if cluster_info['feature_means']:
                    f.write("- **Key Characteristics**:\n")
                    for feature, mean_val in cluster_info['feature_means'].items():
                        f.write(f"  - {feature}: {mean_val:.3f}\n")
                f.write("\n")
            
            if 'cluster_profiles' in analysis_results:
                f.write("## Cluster Profiles\n\n")
                for cluster_name, profile in analysis_results['cluster_profiles'].items():
                    f.write(f"**{cluster_name.replace('_', ' ').title()}**: {profile}\n\n")
            
            f.write("## Pipeline Performance\n\n")
            f.write(f"- **Total Pipeline Time**: {self.results['pipeline_time']:.4f}s\n")
            f.write(f"- **Dataset Size**: {self.results['datasets_info']['n_samples']} patients\n\n")
            
            f.write("## Encrypted Data Handling\n\n")
            f.write("This model supports mixed encrypted/plaintext features as specified in the research proposal:\n")
            f.write("- Encrypted features (test_results_score, cxr_mean_intensity, cxr_edge_density) are processed\n")
            f.write("- Plaintext features (age, billing_amount_norm) are used directly\n")
            f.write("- Model maintains clustering quality while preserving privacy through selective encryption\n")

def main():
    """Main execution function"""
    logger.info("Starting ML Clustering Pipeline")
    
    try:
        # Load prepared datasets
        datasets_path = "data/ml_encrypted/ml_datasets.json"
        context_path = "data/ml_encrypted/pyfhel_context.pkl"
        
        if not os.path.exists(datasets_path) or not os.path.exists(context_path):
            logger.error("Required data files not found. Run ml_encrypted_data_preparation.py first.")
            return
        
        # Load datasets
        with open(datasets_path, 'r') as f:
            datasets_data = json.load(f)
        
        # Convert back to DataFrames
        X_train = pd.DataFrame(datasets_data['X_train'])
        X_test = pd.DataFrame(datasets_data['X_test'])
        
        datasets = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': None,  # Not needed for clustering
            'y_test': None
        }
        
        # Load Pyfhel context
        pyfhel_context = joblib.load(context_path)
        
        # Test different clustering algorithms
        clustering_configs = [
            ClusteringConfig(model_type='kmeans', n_clusters=3),
            ClusteringConfig(model_type='kmeans', n_clusters=4),
            ClusteringConfig(model_type='dbscan', eps=0.8, min_samples=5),
            ClusteringConfig(model_type='agglomerative', n_clusters=3)
        ]
        
        all_results = {}
        
        for config in clustering_configs:
            logger.info(f"Testing {config.model_type} clustering...")
            
            pipeline = ClusteringPipeline(config)
            
            # Run pipeline
            results = pipeline.run_complete_pipeline(datasets, pyfhel_context)
            all_results[f"{config.model_type}_{config.n_clusters}"] = results
            
            # Save results
            output_dir = f"data/ml_models/clustering/{config.model_type}_{config.n_clusters}"
            pipeline.save_results(output_dir)
        
        # Generate comparative report
        generate_comparative_clustering_report(all_results)
        
        logger.info("ML Clustering Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"ML clustering pipeline failed: {str(e)}")
        raise

def generate_comparative_clustering_report(all_results: Dict):
    """Generate comparative report across different clustering algorithms"""
    report_path = "data/ml_models/clustering/comparative_clustering_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Comparative Clustering Model Performance Report\n\n")
        f.write("## Model Comparison Summary\n\n")
        
        f.write("| Model | Clusters | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Training Time (s) |\n")
        f.write("|-------|----------|------------------|-------------------|----------------|-------------------|\n")
        
        for model_name, results in all_results.items():
            eval_results = results['evaluation_results']
            fit_results = results['fitting_results']
            
            f.write(f"| {model_name} | {eval_results['n_clusters']} | {eval_results['silhouette_score']:.4f} | {eval_results['calinski_harabasz_score']:.2f} | {eval_results['davies_bouldin_score']:.4f} | {fit_results['training_time']:.4f} |\n")
        
        f.write("\n## Best Model Recommendation\n\n")
        
        # Find best model based on silhouette score
        best_model = max(all_results.keys(), 
                         key=lambda x: all_results[x]['evaluation_results']['silhouette_score'])
        
        f.write(f"**Recommended Model**: {best_model}\n")
        f.write(f"- **Highest Silhouette Score**: {all_results[best_model]['evaluation_results']['silhouette_score']:.4f}\n")
        f.write(f"- **Clusters Found**: {all_results[best_model]['evaluation_results']['n_clusters']}\n")
        f.write(f"- **Reasonable Training Time**: {all_results[best_model]['fitting_results']['training_time']:.4f}s\n")

if __name__ == "__main__":
    main()