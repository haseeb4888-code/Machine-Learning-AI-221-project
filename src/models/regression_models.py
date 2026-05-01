"""Regression models for GDP prediction"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pandas as pd


class RegressionModelManager:
    """Manage regression models for GDP prediction"""
    
    def __init__(self):
        """Initialize regression model manager"""
        self.models = {}
        self.metrics = {}
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        self.models['linear_regression'] = model
        self.metrics['linear_regression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"✓ Linear Regression - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, RMSE: {rmse:.2f}")
        return model
    
    def train_random_forest_regressor(self, X_train, y_train, X_test, y_test, params=None):
        """Train Random Forest Regressor.

        params (optional): tuned hyperparameters from Optuna.
        """
        tuned = params or {}
        model = RandomForestRegressor(
            n_estimators=tuned.get("n_estimators", 100),
            max_depth=tuned.get("max_depth", 15),
            min_samples_split=tuned.get("min_samples_split", 2),
            min_samples_leaf=tuned.get("min_samples_leaf", 1),
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'hyperparameters': tuned,
        }
        
        print(f"✓ Random Forest - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, RMSE: {rmse:.2f}")
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting Regressor"""
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        self.models['gradient_boosting'] = model
        self.metrics['gradient_boosting'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"✓ Gradient Boosting - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, RMSE: {rmse:.2f}")
        return model
    
    def train_xgboost_regressor(self, X_train, y_train, X_test, y_test, params=None):
        """Train XGBoost Regressor.

        params (optional): tuned hyperparameters from Optuna.
        """
        tuned = params or {}
        model = xgb.XGBRegressor(
            n_estimators=tuned.get("n_estimators", 100),
            max_depth=tuned.get("max_depth", 6),
            learning_rate=tuned.get("learning_rate", 0.1),
            subsample=tuned.get("subsample", 1.0),
            colsample_bytree=tuned.get("colsample_bytree", 1.0),
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'hyperparameters': tuned,
        }
        
        print(f"✓ XGBoost - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, RMSE: {rmse:.2f}")
        return model
    
    def train_svm_regressor(self, X_train, y_train, X_test, y_test, params=None):
        """Train Support Vector Machine Regressor.
        
        params (optional): tuned hyperparameters from Optuna.
        Note: SVM requires feature scaling for optimal performance.
        """
        tuned = params or {}
        
        # Scale features for SVM (important for performance)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = SVR(
            C=tuned.get("C", 100.0),
            epsilon=tuned.get("epsilon", 0.1),
            kernel=tuned.get("kernel", "rbf"),
            gamma=tuned.get("gamma", "scale"),
            degree=tuned.get("degree", 3)
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        self.models['svm'] = model
        self.metrics['svm'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'hyperparameters': tuned,
        }
        
        print(f"✓ SVM - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, RMSE: {rmse:.2f}")
        return model
    
    def train_mlp_regressor(self, X_train, y_train, X_test, y_test):
        """Train Multi-Layer Perceptron Regressor"""
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        self.models['mlp'] = model
        self.metrics['mlp'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"✓ MLP Regressor - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, RMSE: {rmse:.2f}")
        return model
    
    def train_all_regressors(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        tuned_params: dict | None = None,
    ):
        """Train all regression models.

        tuned_params: optional Optuna best params dict. Keys come from
        src/models/hyperparameter_tuning.py, and are mapped to the models here.
        """
        print("\n📊 Training regression models...\n")

        tuned_params = tuned_params or {}
        self.train_linear_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest_regressor(
            X_train, y_train, X_test, y_test, params=tuned_params.get("random_forest_reg")
        )
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_xgboost_regressor(
            X_train, y_train, X_test, y_test, params=tuned_params.get("xgboost_reg")
        )
        self.train_svm_regressor(X_train, y_train, X_test, y_test, params=tuned_params.get("svm_reg"))
        self.train_mlp_regressor(X_train, y_train, X_test, y_test)

        return self.models
    
    def get_best_model(self):
        """Return best performing model by R²"""
        best_name = max(self.metrics, key=lambda x: self.metrics[x]['test_r2'])
        return best_name, self.models[best_name]
    
    def save_models(self, save_dir='models/regression'):
        """Save trained models to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = f'{save_dir}/{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {filepath}")
    
    def predict_gdp(self, features, model_name='random_forest'):
        """Make GDP prediction on features"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        prediction = model.predict(features)[0]
        
        return prediction
