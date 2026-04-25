"""PyTorch neural network for GDP value regression"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class RegressionNN(nn.Module):
    """Simple 3-layer neural network for regression"""
    
    def __init__(self, n_features=18):
        """
        Initialize regression neural network
        
        Args:
            n_features: Number of input features
        """
        super(RegressionNN, self).__init__()
        
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """Forward pass through network"""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PyTorchRegressor:
    """PyTorch-based GDP value regressor"""
    
    def __init__(self, n_features=18, device='cpu'):
        """Initialize PyTorch regressor"""
        self.n_features = n_features
        self.device = device
        self.model = RegressionNN(n_features).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, verbose=True):
        """
        Train the neural network for regression
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training target values (numpy array)
            X_val: Validation features (numpy array)
            y_val: Validation target values (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Print training progress
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print("\n📊 Training PyTorch Regression Neural Network...\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            self.history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"✓ Training complete. Best Val MSE: {best_val_loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            predictions: Predicted values (numpy array)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy().flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features (numpy array)
            y_test: Test target values (numpy array)
            
        Returns:
            metrics: Dict with RMSE, MAE, and R² score
        """
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def score(self, X_test, y_test):
        """Get R² score (sklearn-compatible)"""
        predictions = self.predict(X_test)
        return r2_score(y_test, predictions)
    
    def save_model(self, save_path='models/pytorch_regression.pth'):
        """Save model to disk"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"✓ Model saved to {save_path}")
    
    def load_model(self, save_path='models/pytorch_regression.pth'):
        """Load model from disk"""
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Model loaded from {save_path}")
