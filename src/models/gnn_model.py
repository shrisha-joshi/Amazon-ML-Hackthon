# MIT License
# src/models/gnn_model.py
"""
Graph Neural Network model using GraphSAGE for product similarity graph.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.smape_loss import SMAPELoss, smape_numpy

log = get_logger("gnn_model")

# Simple GNN implementation without torch_geometric dependency
class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim * 2, output_dim)  # self + neighbor concat
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim), adj_matrix: (N, N)
        neighbor_agg = torch.mm(adj_matrix, x) / (adj_matrix.sum(dim=1, keepdim=True) + 1e-8)
        combined = torch.cat([x, neighbor_agg], dim=1)
        out = self.activation(self.linear(combined))
        return out

class GNNModel(nn.Module, BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, model_name: str = "gnn", save_dir: str = "outputs/models"):
        nn.Module.__init__(self)
        BaseModel.__init__(self, model_name, save_dir)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        
        self.oof_predictions = None

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj_matrix)
            x = self.dropout_layer(x)
        
        out = self.output(x).squeeze()
        return out

    def build_adjacency_matrix(self, embeddings: np.ndarray, k: int = 10) -> torch.Tensor:
        """Build k-NN adjacency matrix based on cosine similarity"""
        log.info(f"[AI-Agent] Building k-NN graph with k={k}")
        
        sim_matrix = cosine_similarity(embeddings)
        n = len(embeddings)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            # Get top-k similar nodes (excluding self)
            similarities = sim_matrix[i]
            top_k_indices = np.argsort(similarities)[-k-1:-1]  # Exclude self
            
            for j in top_k_indices:
                adj_matrix[i, j] = similarities[j]
        
        # Normalize adjacency matrix
        row_sums = adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        adj_matrix = adj_matrix / row_sums[:, np.newaxis]
        
        return torch.FloatTensor(adj_matrix)

    def train_model(self, X: np.ndarray, y: np.ndarray, num_folds: int = 5,
                   epochs: int = 200, lr: float = 1e-3, k: int = 10):
        log.info(f"[AI-Agent] Training GNN with {X.shape[0]} samples, {X.shape[1]} features")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[AI-Agent] Device chosen: {device}")
        self.to(device)
        
        # Build adjacency matrix
        adj_matrix = self.build_adjacency_matrix(X, k=k).to(device)
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            log.info(f"Training fold {fold + 1}/{num_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(device)  # Use full graph
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            
            # Initialize model for this fold
            self._reset_parameters()
            optimizer = optim.Adam(self.parameters(), lr=lr)
            criterion = SMAPELoss()
            
            best_val_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            for epoch in range(epochs):
                self.train()
                optimizer.zero_grad()
                
                # Forward pass on full graph, but loss only on training nodes
                outputs = self(X_tensor, adj_matrix)
                train_outputs = outputs[train_idx]
                loss = criterion(train_outputs, y_train_tensor)
                
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    self.eval()
                    with torch.no_grad():
                        val_outputs = outputs[val_idx].cpu().numpy()
                        val_smape = smape_numpy(y_val, val_outputs)
                        
                        if val_smape < best_val_loss:
                            best_val_loss = val_smape
                            patience_counter = 0
                            oof[val_idx] = val_outputs
                        else:
                            patience_counter += 1
                        
                        if epoch % 50 == 0:
                            log.info(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_smape={val_smape:.4f}")
                
                if patience_counter >= patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.oof_predictions = oof
        smape_score = smape_numpy(y, oof)
        log.info(f"[AI-Agent] GNN OOF SMAPE: {smape_score:.4f}%")
        return self

    def _reset_parameters(self):
        """Reset model parameters for each fold"""
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.output, 'reset_parameters'):
            self.output.reset_parameters()

    def predict(self, X: np.ndarray, k: int = 10) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        # Build adjacency matrix for prediction
        adj_matrix = self.build_adjacency_matrix(X, k=k).to(device)
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            predictions = self(X_tensor, adj_matrix).cpu().numpy()
        
        return predictions

    def save(self):
        path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        torch.save(self.state_dict(), path)
        log.info(f"[AI-Agent] Saved GNN model to {path}")
        return path

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        self.load_state_dict(torch.load(path))
        log.info(f"[AI-Agent] Loaded GNN model from {path}")
        return self