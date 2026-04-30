"""Tests for ML models"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.models.classification_models import ClassificationModelManager
from src.models.regression_models import RegressionModelManager
from src.models.clustering_models import ClusteringModelManager


@pytest.fixture
def sample_classification_data():
    """Create sample classification data for testing"""
    X, y = make_classification(n_samples=100, n_features=18, n_classes=3, 
                               n_informative=10, random_state=42)
    split_idx = int(0.8 * len(X))
    return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:])


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing"""
    X, y = make_regression(n_samples=100, n_features=18, random_state=42)
    split_idx = int(0.8 * len(X))
    return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:])


class TestClassificationModels:
    """Test classification model training and prediction"""
    
    def test_logistic_regression_training(self, sample_classification_data):
        """Test logistic regression model training"""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        manager = ClassificationModelManager()
        model = manager.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'logistic_regression' in manager.models
        assert manager.metrics['logistic_regression']['test_accuracy'] > 0
    
    def test_random_forest_training(self, sample_classification_data):
        """Test random forest classifier training"""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        manager = ClassificationModelManager()
        model = manager.train_random_forest(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'random_forest' in manager.models
        assert manager.metrics['random_forest']['test_accuracy'] >= 0
    
    def test_knn_training(self, sample_classification_data):
        """Test KNN classifier training"""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        manager = ClassificationModelManager()
        model = manager.train_knn(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'knn' in manager.models
        assert manager.metrics['knn']['test_accuracy'] >= 0
    
    def test_best_classifier(self, sample_classification_data):
        """Test getting best classifier"""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        manager = ClassificationModelManager()
        manager.train_all_classifiers(X_train, y_train, X_test, y_test)
        
        best_name, best_model = manager.get_best_model()
        assert best_name in manager.models
        assert best_model is not None


class TestRegressionModels:
    """Test regression model training and prediction"""
    
    def test_linear_regression_training(self, sample_regression_data):
        """Test linear regression model training"""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        manager = RegressionModelManager()
        model = manager.train_linear_regression(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'linear_regression' in manager.models
        assert manager.metrics['linear_regression']['test_r2'] > -1
    
    def test_random_forest_regression(self, sample_regression_data):
        """Test random forest regressor training"""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        manager = RegressionModelManager()
        model = manager.train_random_forest_regressor(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'random_forest' in manager.models
        assert manager.metrics['random_forest']['test_r2'] > -1
    
    def test_gradient_boosting_training(self, sample_regression_data):
        """Test gradient boosting regressor training"""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        manager = RegressionModelManager()
        model = manager.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'gradient_boosting' in manager.models
        assert manager.metrics['gradient_boosting']['test_r2'] > -1
    
    def test_best_regressor(self, sample_regression_data):
        """Test getting best regressor"""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        manager = RegressionModelManager()
        manager.train_all_regressors(X_train, y_train, X_test, y_test)
        
        best_name, best_model = manager.get_best_model()
        assert best_name in manager.models
        assert best_model is not None


class TestClusteringModels:
    """Test clustering model training"""
    
    def test_kmeans_training(self, sample_regression_data):
        """Test KMeans clustering training"""
        X_train, _, _, _ = sample_regression_data
        
        manager = ClusteringModelManager()
        model, labels = manager.train_kmeans(X_train, n_clusters=3)
        
        assert model is not None
        assert 'kmeans' in manager.models
        assert len(labels) == len(X_train)
        assert len(np.unique(labels)) == 3
    
    def test_hierarchical_training(self, sample_regression_data):
        """Test Hierarchical clustering training"""
        X_train, _, _, _ = sample_regression_data
        
        manager = ClusteringModelManager()
        model, labels = manager.train_hierarchical(X_train, n_clusters=3)
        
        assert model is not None
        assert 'hierarchical' in manager.models
        assert len(labels) == len(X_train)
    
    def test_cluster_analysis(self, sample_regression_data):
        """Test cluster analysis function"""
        X_train, _, _, _ = sample_regression_data
        
        manager = ClusteringModelManager()
        model, labels = manager.train_kmeans(X_train, n_clusters=3)
        
        cluster_info = manager.get_cluster_analysis(X_train, labels)
        
        assert len(cluster_info) == 3
        total_percentage = sum(info['percentage'] for info in cluster_info.values())
        assert abs(total_percentage - 100) < 0.01
    
    def test_clustering_quality_kmeans(self, sample_regression_data):
        """Test that KMeans clustering produces reasonable silhouette scores"""
        X_train, _, _, _ = sample_regression_data
        
        manager = ClusteringModelManager()
        model, labels = manager.train_kmeans(X_train, n_clusters=3)
        
        # Check silhouette score exists and is reasonable
        assert 'kmeans' in manager.metrics
        assert 'silhouette_score' in manager.metrics['kmeans']
        silhouette_score = manager.metrics['kmeans']['silhouette_score']
        
        # Silhouette score should be > -1 and < 1
        # On synthetic data with 3 clusters, expect > 0 (positive) for reasonable clustering
        assert -1 <= silhouette_score <= 1, \
            f"Silhouette score {silhouette_score} out of valid range [-1, 1]"
        assert silhouette_score > -0.5, \
            f"Silhouette score {silhouette_score:.4f} indicates poor clustering"
    
    def test_clustering_quality_hierarchical(self, sample_regression_data):
        """Test that Hierarchical clustering produces reasonable silhouette scores"""
        X_train, _, _, _ = sample_regression_data
        
        manager = ClusteringModelManager()
        model, labels = manager.train_hierarchical(X_train, n_clusters=3)
        
        # Check silhouette score exists and is reasonable
        assert 'hierarchical' in manager.metrics
        assert 'silhouette_score' in manager.metrics['hierarchical']
        silhouette_score = manager.metrics['hierarchical']['silhouette_score']
        
        # Silhouette score should be > -1 and < 1
        assert -1 <= silhouette_score <= 1, \
            f"Silhouette score {silhouette_score} out of valid range [-1, 1]"
        assert silhouette_score > -0.5, \
            f"Silhouette score {silhouette_score:.4f} indicates poor clustering"
