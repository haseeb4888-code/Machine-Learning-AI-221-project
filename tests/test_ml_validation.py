"""ML-specific validation tests"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from src.models.classification_models import ClassificationModelManager
from src.models.regression_models import RegressionModelManager


@pytest.fixture
def classification_data():
    """Create classification data for validation tests"""
    X, y = make_classification(n_samples=150, n_features=18, n_classes=3,
                               n_informative=12, random_state=42)
    split_idx = int(0.8 * len(X))
    return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:])


@pytest.fixture
def regression_data():
    """Create regression data for validation tests"""
    X, y = make_regression(n_samples=150, n_features=18, random_state=42)
    split_idx = int(0.8 * len(X))
    return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:])


class TestModelPerformanceThresholds:
    """Test that models meet minimum performance thresholds"""
    
    def test_classification_accuracy_threshold(self, classification_data):
        """Assert classification accuracy exceeds minimum threshold"""
        X_train, X_test, y_train, y_test = classification_data
        
        manager = ClassificationModelManager()
        manager.train_all_classifiers(X_train, y_train, X_test, y_test)
        
        # At least one model should have >70% accuracy (based on actual: 0.87 XGBoost)
        # Using synthetic data, conservative threshold is 0.70
        accuracies = [manager.metrics[m]['test_accuracy'] for m in manager.models]
        best_accuracy = max(accuracies)
        assert best_accuracy > 0.70, \
            f"Best accuracy {best_accuracy:.4f} must be > 0.70. Accuracies: {accuracies}"
    
    def test_regression_r2_positive(self, regression_data):
        """Assert regression models have good R² scores"""
        X_train, X_test, y_train, y_test = regression_data
        
        manager = RegressionModelManager()
        manager.train_all_regressors(X_train, y_train, X_test, y_test)
        
        # At least one model should have R² > 0.65 (based on actual: 0.95 XGBoost)
        # Using synthetic data, conservative threshold is 0.65
        r2_scores = [manager.metrics[m]['test_r2'] for m in manager.models]
        best_r2 = max(r2_scores)
        assert best_r2 > 0.65, \
            f"Best R² {best_r2:.4f} must be > 0.65. R² scores: {r2_scores}"


class TestModelConsistency:
    """Test model consistency and stability"""
    
    def test_random_forest_cv_consistency(self, classification_data):
        """Test Random Forest classifier consistency across folds"""
        X_train, _, y_train, _ = classification_data
        
        manager = ClassificationModelManager()
        manager.train_random_forest(X_train, y_train, X_train, y_train)
        
        # Cross-validation to check consistency
        model = manager.models['random_forest']
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        
        # Scores should be consistent (low standard deviation)
        std_dev = np.std(cv_scores)
        assert std_dev < 0.15, f"High variance in CV scores: {cv_scores}"
    
    def test_regression_cv_consistency(self, regression_data):
        """Test regression model consistency across folds"""
        X_train, _, y_train, _ = regression_data
        
        manager = RegressionModelManager()
        manager.train_random_forest_regressor(X_train, y_train, X_train, y_train)
        
        # Cross-validation for regression
        model = manager.models['random_forest']
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        
        # R² scores should be consistent
        std_dev = np.std(cv_scores)
        assert std_dev < 0.15, f"High variance in CV R² scores: {cv_scores}"


class TestPredictionValidity:
    """Test that predictions are in valid ranges"""
    
    def test_classification_output_valid(self, classification_data):
        """Verify classification predictions are valid class indices"""
        X_train, X_test, y_train, y_test = classification_data
        
        manager = ClassificationModelManager()
        manager.train_all_classifiers(X_train, y_train, X_test, y_test)
        
        for model_name, model in manager.models.items():
            predictions = model.predict(X_test)
            
            # All predictions should be in valid class range
            assert np.all(predictions >= 0), f"{model_name}: negative predictions"
            assert np.all(predictions < 3), f"{model_name}: predictions >= 3"
            assert predictions.dtype in [np.int32, np.int64], f"{model_name}: wrong dtype"
    
    def test_regression_output_bounds(self, regression_data):
        """Verify regression predictions are in reasonable bounds"""
        X_train, X_test, y_train, y_test = regression_data
        
        manager = RegressionModelManager()
        manager.train_all_regressors(X_train, y_train, X_test, y_test)
        
        for model_name, model in manager.models.items():
            predictions = model.predict(X_test)
            
            # Predictions should be within reasonable range of actual values
            y_min, y_max = y_test.min(), y_test.max()
            pred_min, pred_max = predictions.min(), predictions.max()
            
            # Allow 20% margin beyond actual range
            margin = (y_max - y_min) * 0.2
            assert pred_min >= (y_min - margin), f"{model_name}: predictions too low"
            assert pred_max <= (y_max + margin), f"{model_name}: predictions too high"
