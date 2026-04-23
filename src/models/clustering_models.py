"""Clustering models for country grouping and analysis"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd


class ClusteringModelManager:
    """Manage clustering models for country segmentation"""
    
    def __init__(self):
        """Initialize clustering model manager"""
        self.models = {}
        self.metrics = {}
    
    def train_kmeans(self, X, n_clusters=3):
        """Train KMeans clustering"""
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        self.models['kmeans'] = model
        self.metrics['kmeans'] = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'inertia': model.inertia_
        }
        
        print(f"✓ KMeans - Clusters: {n_clusters}, Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}")
        return model, labels
    
    def train_hierarchical(self, X, n_clusters=3):
        """Train Hierarchical Clustering"""
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = model.fit_predict(X)
        
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        self.models['hierarchical'] = model
        self.metrics['hierarchical'] = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin
        }
        
        print(f"✓ Hierarchical - Clusters: {n_clusters}, Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}")
        return model, labels
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        """Train DBSCAN clustering"""
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        # Filter out noise points (-1 label)
        mask = labels != -1
        
        if len(np.unique(labels[mask])) > 1:
            silhouette = silhouette_score(X[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        else:
            silhouette = -1
            davies_bouldin = -1
        
        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = list(labels).count(-1)
        
        self.models['dbscan'] = model
        self.metrics['dbscan'] = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'eps': eps,
            'min_samples': min_samples
        }
        
        print(f"✓ DBSCAN - Clusters: {n_clusters}, Noise Points: {n_noise}, Silhouette: {silhouette:.3f}")
        return model, labels
    
    def train_all_clustering(self, X, n_clusters=3):
        """Train all clustering models"""
        print("\n🎯 Training clustering models...\n")
        
        self.train_kmeans(X, n_clusters=n_clusters)
        self.train_hierarchical(X, n_clusters=n_clusters)
        self.train_dbscan(X, eps=0.5, min_samples=5)
        
        return self.models
    
    def save_models(self, save_dir='models/clustering'):
        """Save trained models to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = f'{save_dir}/{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {filepath}")
    
    def get_cluster_analysis(self, X, labels):
        """Analyze cluster distribution"""
        unique_clusters = np.unique(labels)
        cluster_info = {}
        
        for cluster in unique_clusters:
            cluster_mask = labels == cluster
            cluster_size = np.sum(cluster_mask)
            cluster_info[f'cluster_{cluster}'] = {
                'size': int(cluster_size),
                'percentage': float(100 * cluster_size / len(labels))
            }
        
        return cluster_info
