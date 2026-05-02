"""Main training pipeline for all models"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.models.classification_models import ClassificationModelManager
from src.models.regression_models import RegressionModelManager
from src.models.hyperparameter_tuning import HyperparameterTuner


class TrainingPipeline:
    """Complete training pipeline for ML models"""
    
    def __init__(self, data_path='countries of the world.csv'):
        """Initialize training pipeline"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.y_train_clf = None
        self.y_test_clf = None
        self.le_gdp = None
        self.tuned_params: dict | None = None
    
    def load_and_preprocess(self):
        """Load and preprocess data"""
        print("=" * 60)
        print("📦 LOADING DATA")
        print("=" * 60)
        
        # Load data
        loader = DataLoader(self.data_path)
        self.df = loader.load_data()
        self.df = loader.clean_column_names(self.df)
        self.df = loader.convert_columns_to_float(self.df)
        loader.display_info(self.df)
        
        print("\n" + "=" * 60)
        print("🔧 PREPROCESSING DATA")
        print("=" * 60)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        self.df_processed, self.df = preprocessor.preprocess(self.df)
        
        return self.df, self.df_processed
    
    def engineer_features(self):
        """Engineer new features"""
        print("\n" + "=" * 60)
        print("🎨 FEATURE ENGINEERING")
        print("=" * 60)
        
        engineer = FeatureEngineer()
        self.df = engineer.engineer_all_features(self.df)
        
        return self.df
    
    def create_target_variable(self):
        """Create GDP categories for classification"""
        print("\n" + "=" * 60)
        print("🎯 CREATING TARGET VARIABLE")
        print("=" * 60)
        
        def classify_gdp(gdp_value):
            """Categorize GDP into 3 categories"""
            if gdp_value < 5000:
                return 0  # Low
            elif gdp_value <= 15000:
                return 1  # Medium
            else:
                return 2  # High
        
        self.df['GDP_Category'] = self.df['GDP ($ per capita)'].apply(classify_gdp)
        
        print("✓ GDP Categories created:")
        print(f"  - Low (< $5000): {(self.df['GDP_Category'] == 0).sum()} countries")
        print(f"  - Medium ($5000-$15000): {(self.df['GDP_Category'] == 1).sum()} countries")
        print(f"  - High (> $15000): {(self.df['GDP_Category'] == 2).sum()} countries")
        
        return self.df
    
    def prepare_train_test_split(self):
        """Prepare training and test sets"""
        print("\n" + "=" * 60)
        print("✂️  SPLITTING DATA")
        print("=" * 60)
        
        # Select features (exclude Country, Region, GDP_Category)
        feature_cols = [col for col in self.df.columns 
                       if col not in ['Country', 'Region', 'GDP_Category', 'GDP ($ per capita)']]
        
        X = self.df[feature_cols]
        y_reg = self.df['GDP ($ per capita)']  # For regression
        y_clf = self.df['GDP_Category']  # For classification
        
        # Train-test split
        self.X_train, self.X_test, self.y_train_reg, self.y_test_reg, self.y_train_clf, self.y_test_clf = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        
        print(f"✓ Training set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        print(f"✓ Features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train_reg, self.y_test_reg, self.y_train_clf, self.y_test_clf

    def tune_hyperparameters(self, n_trials: int = 50, timeout: int = 300, seed: int = 42) -> dict:
        """Sequential Optuna tuning BEFORE final training."""
        print("\n" + "=" * 60)
        print("🔧 HYPERPARAMETER TUNING (Optuna - Sequential)")
        print("=" * 60)

        tuner = HyperparameterTuner(n_trials=n_trials, timeout=timeout, seed=seed)

        # Classification tuning
        tuner.tune_logistic_regression(self.X_train, self.y_train_clf)
        tuner.tune_random_forest_classifier(self.X_train, self.y_train_clf)
        tuner.tune_xgboost_classifier(self.X_train, self.y_train_clf)

        # Regression tuning
        tuner.tune_random_forest_regressor(self.X_train, self.y_train_reg)
        tuner.tune_xgboost_regressor(self.X_train, self.y_train_reg)

        tuner.print_summary()

        self.tuned_params = tuner.get_all_best_params()
        # Save alongside models/results for later inspection.
        tuner.save_best_params("results/hyperparameters.json")

        return self.tuned_params
    
    def train_classification_models(self):
        """Train all classification models (optionally with tuned params)."""
        print("\n" + "=" * 60)
        print("🤖 TRAINING CLASSIFICATION MODELS")
        print("=" * 60)
        
        clf_manager = ClassificationModelManager()
        clf_manager.train_all_classifiers(
            self.X_train, self.y_train_clf, 
            self.X_test, self.y_test_clf,
            tuned_params=self.tuned_params,
        )
        clf_manager.save_models()
        
        return clf_manager
    
    def train_regression_models(self):
        """Train all regression models (optionally with tuned params)."""
        print("\n" + "=" * 60)
        print("📊 TRAINING REGRESSION MODELS")
        print("=" * 60)
        
        reg_manager = RegressionModelManager()
        reg_manager.train_all_regressors(
            self.X_train, self.y_train_reg,
            self.X_test, self.y_test_reg,
            tuned_params=self.tuned_params,
        )
        reg_manager.save_models()
        
        return reg_manager
    
    def train_classification_models_with_data(
        self,
        X_train,
        y_train_clf,
        X_test,
        y_test_clf,
        tuned_params: dict | None = None,
    ):
        """Train all classification models with provided data."""
        print("\n" + "=" * 60)
        print("🤖 TRAINING CLASSIFICATION MODELS")
        print("=" * 60)
        
        clf_manager = ClassificationModelManager()
        clf_manager.train_all_classifiers(
            X_train,
            y_train_clf,
            X_test,
            y_test_clf,
            tuned_params=tuned_params,
        )
        clf_manager.save_models()
        
        return clf_manager
    
    def train_regression_models_with_data(
        self,
        X_train,
        y_train_reg,
        X_test,
        y_test_reg,
        tuned_params: dict | None = None,
    ):
        """Train all regression models with provided data."""
        print("\n" + "=" * 60)
        print("📊 TRAINING REGRESSION MODELS")
        print("=" * 60)
        
        reg_manager = RegressionModelManager()
        reg_manager.train_all_regressors(
            X_train,
            y_train_reg,
            X_test,
            y_test_reg,
            tuned_params=tuned_params,
        )
        reg_manager.save_models()
        
        return reg_manager
    
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        print("\n\n")
        print("█" * 60)
        print("█  ECONOMIC GROWTH ANALYZER - TRAINING PIPELINE")
        print("█" * 60)
        
        # Step 1: Load and preprocess
        self.load_and_preprocess()
        
        # Step 2: Engineer features
        self.engineer_features()
        
        # Step 3: Create target variable
        self.create_target_variable()
        
        # Step 4: Split data
        self.prepare_train_test_split()

        # Step 5: NEW - Hyperparameter tuning BEFORE final training
        self.tune_hyperparameters(n_trials=50, timeout=300, seed=42)
        
        # Step 6: Train models (using tuned hyperparameters)
        clf_manager = self.train_classification_models()
        reg_manager = self.train_regression_models()
        clust_manager = self.train_clustering_models()
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE")
        print("=" * 60)
        print("\n🎉 All models trained and saved successfully!")
        
        return clf_manager, reg_manager, clust_manager
    def train_clustering_models(self):
        """Train all clustering models"""
        print("\n" + "=" * 60)
        print("🎯 TRAINING CLUSTERING MODELS")
        print("=" * 60)
        
        # Select features for clustering (same as classification/regression)
        feature_cols = [col for col in self.df.columns 
                    if col not in ['Country', 'Region', 'GDP_Category', 'GDP ($ per capita)']]
        
        X = self.df[feature_cols]
        
        # Import and train clustering
        from src.models.clustering_models import ClusteringModelManager
        
        clust_manager = ClusteringModelManager()
        clust_manager.train_all_clustering(X, n_clusters=3)
        clust_manager.save_models()
        
        return clust_manager

    def train_clustering_models_with_data(self, X):
        """Train clustering models with provided data"""
        print("\n" + "=" * 60)
        print("🎯 TRAINING CLUSTERING MODELS")
        print("=" * 60)
        
        from src.models.clustering_models import ClusteringModelManager
        
        clust_manager = ClusteringModelManager()
        clust_manager.train_all_clustering(X, n_clusters=3)
        clust_manager.save_models()
        
        return clust_manager


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    clf_manager, reg_manager, clust_manager = pipeline.run_full_pipeline()
