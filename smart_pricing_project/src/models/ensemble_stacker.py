# MIT License
# src/models/ensemble_stacker.py
"""
Meta-ensemble stacker to combine predictions from multiple base models.
"""

from __future__ import annotations
import os
import numpy as np
import joblib
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.smape_loss import smape_numpy

log = get_logger("ensemble_stacker")

class EnsembleStacker(BaseModel):
    def __init__(self, model_name: str = "ensemble_stacker", save_dir: str = "outputs/models"):
        super().__init__(model_name, save_dir)
        self.meta_model = ElasticNetCV(
            cv=5,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            random_state=42,
            max_iter=1000
        )
        self.oof_predictions = None

    def train(self, base_predictions: np.ndarray, y: np.ndarray, num_folds: int = 5):
        """
        Train the meta-model on base model predictions.
        
        Args:
            base_predictions: Shape (n_samples, n_base_models)
            y: Target values
            num_folds: Number of folds for meta-model training
        """
        log.info(f"[AI-Agent] Training ensemble stacker with {base_predictions.shape[0]} samples, {base_predictions.shape[1]} base models")
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(base_predictions)):
            log.info(f"Training meta-model fold {fold + 1}/{num_folds}")
            
            X_train, X_val = base_predictions[train_idx], base_predictions[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone the meta-model for this fold
            from sklearn.base import clone
            fold_meta_model = clone(self.meta_model)
            fold_meta_model.fit(X_train, y_train)
            
            # Predict on validation set
            val_preds = fold_meta_model.predict(X_val)
            oof[val_idx] = val_preds
            
            if fold == 0:  # Save the first fold model as the main model
                self.meta_model = fold_meta_model
        
        self.oof_predictions = oof
        smape_score = smape_numpy(y, oof)
        log.info(f"[AI-Agent] Ensemble Stacker OOF SMAPE: {smape_score:.4f}%")
        
        # Log model coefficients
        if hasattr(self.meta_model, 'coef_'):
            log.info(f"[AI-Agent] Meta-model coefficients: {self.meta_model.coef_}")
        
        return self

    def predict(self, base_predictions: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained meta-model.
        
        Args:
            base_predictions: Shape (n_samples, n_base_models)
        
        Returns:
            Final ensemble predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet. Call train() first.")
        
        return self.meta_model.predict(base_predictions)

    def save(self):
        if self.meta_model is None:
            raise ValueError("No meta-model to save. Train first.")
        
        path = os.path.join(self.save_dir, f"{self.model_name}.pkl")
        joblib.dump(self.meta_model, path)
        log.info(f"[AI-Agent] Saved ensemble stacker to {path}")
        return path

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.pkl")
        
        self.meta_model = joblib.load(path)
        log.info(f"[AI-Agent] Loaded ensemble stacker from {path}")
        return self

    def get_feature_importance(self) -> dict:
        """Get feature importance from the meta-model"""
        if self.meta_model is None or not hasattr(self.meta_model, 'coef_'):
            return {}
        
        return {
            f"base_model_{i}": abs(coef) 
            for i, coef in enumerate(self.meta_model.coef_)
        }