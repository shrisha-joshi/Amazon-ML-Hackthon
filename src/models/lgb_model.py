# MIT License
# src/models/lgb_model.py
"""
LightGBM model for tabular features and concatenated embeddings.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold
from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.smape_loss import smape_numpy

log = get_logger("lgb_model")

class LGBModel(BaseModel):
    def __init__(self, model_name: str = "lightgbm", save_dir: str = "outputs/models", params: dict = None):
        super().__init__(model_name, save_dir)
        self.params = params or {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.01,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "verbosity": -1,
            "seed": 42
        }
        self.model = None
        self.oof_predictions = None

    def train(self, X: np.ndarray, y: np.ndarray, num_folds: int = 5, num_boost_round: int = 2000):
        log.info(f"[AI-Agent] Training LightGBM with {X.shape[0]} samples, {X.shape[1]} features")
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            log.info(f"Training fold {fold + 1}/{num_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                self.params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=num_boost_round,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            
            if fold == 0:  # Save the first fold model as the main model
                self.model = model
        
        self.oof_predictions = oof
        smape_score = smape_numpy(y, oof)
        log.info(f"[AI-Agent] LightGBM OOF SMAPE: {smape_score:.4f}%")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self):
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        path = os.path.join(self.save_dir, f"{self.model_name}.pkl")
        joblib.dump(self.model, path)
        log.info(f"[AI-Agent] Saved LightGBM model to {path}")
        return path

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.pkl")
        self.model = joblib.load(path)
        log.info(f"[AI-Agent] Loaded LightGBM model from {path}")
        return self