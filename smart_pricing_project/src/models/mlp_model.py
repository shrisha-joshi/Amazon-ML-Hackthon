# MIT License
# src/models/mlp_model.py
"""
Multi-layer perceptron model for combined embeddings.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.smape_loss import SMAPELoss, smape_numpy

log = get_logger("mlp_model")

class MLPModel(nn.Module, BaseModel):
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2, 
                 model_name: str = "mlp", save_dir: str = "outputs/models"):
        nn.Module.__init__(self)
        BaseModel.__init__(self, model_name, save_dir)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self.oof_predictions = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

    def train_model(self, X: np.ndarray, y: np.ndarray, num_folds: int = 5, 
                   epochs: int = 100, batch_size: int = 256, lr: float = 1e-3):
        log.info(f"[AI-Agent] Training MLP with {X.shape[0]} samples, {X.shape[1]} features")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[AI-Agent] Device chosen: {device}")
        self.to(device)
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            log.info(f"Training fold {fold + 1}/{num_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model for this fold
            self._reset_parameters()
            optimizer = optim.Adam(self.parameters(), lr=lr)
            criterion = SMAPELoss()
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                self.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    val_loss = criterion(val_outputs, torch.FloatTensor(y_val).to(device))
                    val_preds = val_outputs.cpu().numpy()
                    val_smape = smape_numpy(y_val, val_preds)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    oof[val_idx] = val_preds
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 20 == 0:
                    log.info(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, val_smape={val_smape:.4f}")
        
        self.oof_predictions = oof
        smape_score = smape_numpy(y, oof)
        log.info(f"[AI-Agent] MLP OOF SMAPE: {smape_score:.4f}%")
        return self

    def _reset_parameters(self):
        """Reset model parameters for each fold"""
        for layer in self.network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            predictions = self(X_tensor).cpu().numpy()
        
        return predictions

    def save(self):
        path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        torch.save(self.state_dict(), path)
        log.info(f"[AI-Agent] Saved MLP model to {path}")
        return path

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        self.load_state_dict(torch.load(path))
        log.info(f"[AI-Agent] Loaded MLP model from {path}")
        return self