"""Classification models for GDP category prediction"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pandas as pd


class ClassificationModelManager:
    """Manage classification models for GDP categorization"""
    
    def __init__(self):
        """Initialize model manager"""
        self.models = {}
        self.metrics = {}
        self.category_labels = ['Low', 'Medium', 'High']
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, params=None):
        """Train Logistic Regression classifier.

        params (optional): tuned hyperparameters from Optuna.
        """
        tuned = params or {}
        model = LogisticRegression(
            C=tuned.get("C", 1.0),
            max_iter=tuned.get("max_iter", 1000),
            solver="lbfgs",
            penalty="l2",
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'hyperparameters': tuned,
        }
        
        print(f"✓ Logistic Regression - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, params=None):
        """Train Random Forest classifier.

        params (optional): tuned hyperparameters from Optuna.
        """
        tuned = params or {}
        model = RandomForestClassifier(
            n_estimators=tuned.get("n_estimators", 100),
            max_depth=tuned.get("max_depth", 15),
            min_samples_split=tuned.get("min_samples_split", 2),
            min_samples_leaf=tuned.get("min_samples_leaf", 1),
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'hyperparameters': tuned,
        }
        
        print(f"✓ Random Forest - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """Train KNN classifier"""
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['knn'] = model
        self.metrics['knn'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
        
        print(f"✓ KNN - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_xgboost_classifier(self, X_train, y_train, X_test, y_test, params=None):
        """Train XGBoost classifier.

        params (optional): tuned hyperparameters from Optuna.
        """
        tuned = params or {}
        model = xgb.XGBClassifier(
            n_estimators=tuned.get("n_estimators", 100),
            max_depth=tuned.get("max_depth", 6),
            learning_rate=tuned.get("learning_rate", 0.1),
            subsample=tuned.get("subsample", 1.0),
            colsample_bytree=tuned.get("colsample_bytree", 1.0),
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['xgboost'] = model
        self.metrics['xgboost'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'hyperparameters': tuned,
        }
        
        print(f"✓ XGBoost - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train Support Vector Machine classifier"""
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['svm'] = model
        self.metrics['svm'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
        
        print(f"✓ SVM - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_gaussian_naive_bayes(self, X_train, y_train, X_test, y_test):
        """Train Gaussian Naive Bayes classifier"""
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['gaussian_naive_bayes'] = model
        self.metrics['gaussian_naive_bayes'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
        
        print(f"✓ Gaussian Naive Bayes - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_mlp_classifier(self, X_train, y_train, X_test, y_test):
        """Train Multi-Layer Perceptron classifier"""
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['mlp'] = model
        self.metrics['mlp'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
        
        print(f"✓ MLP Classifier - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_all_classifiers(self, X_train, y_train, X_test, y_test, tuned_params: dict | None = None):
        """Train all classification models.

        tuned_params: optional Optuna best params dict. Keys come from
        src/models/hyperparameter_tuning.py, and are mapped to the models here.
        """
        print("\n🤖 Training classification models...\n")

        tuned_params = tuned_params or {}
        self.train_logistic_regression(
            X_train, y_train, X_test, y_test, params=tuned_params.get("logistic_regression")
        )
        self.train_random_forest(
            X_train, y_train, X_test, y_test, params=tuned_params.get("random_forest_clf")
        )
        self.train_knn(X_train, y_train, X_test, y_test)
        self.train_xgboost_classifier(
            X_train, y_train, X_test, y_test, params=tuned_params.get("xgboost_clf")
        )
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_gaussian_naive_bayes(X_train, y_train, X_test, y_test)
        self.train_mlp_classifier(X_train, y_train, X_test, y_test)
        
        return self.models
    
    def get_best_model(self):
        """Return best performing model"""
        best_name = max(self.metrics, key=lambda x: self.metrics[x]['test_accuracy'])
        return best_name, self.models[best_name]
    
    def save_models(self, save_dir='models/classification'):
        """Save trained models to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = f'{save_dir}/{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {filepath}")
    
    def predict_category(self, features, model_name='random_forest'):
        """Make prediction on features"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        prediction = model.predict(features)[0]
        category = self.category_labels[prediction]
        
        return category, prediction
