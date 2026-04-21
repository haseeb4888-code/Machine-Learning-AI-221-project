"""Classification models for GDP category prediction"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


class ClassificationModelManager:
    """Manage classification models for GDP categorization"""
    
    def __init__(self):
        """Initialize model manager"""
        self.models = {}
        self.metrics = {}
        self.category_labels = ['Low', 'Medium', 'High']
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression classifier"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['logistic_regression'] = model
        self.metrics['logistic_regression'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
        
        print(f"✓ Logistic Regression - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest classifier"""
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
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
    
    def train_all_classifiers(self, X_train, y_train, X_test, y_test):
        """Train all classification models"""
        print("\n🤖 Training classification models...\n")
        
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_knn(X_train, y_train, X_test, y_test)
        
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
