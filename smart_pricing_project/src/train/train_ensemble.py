# MIT License
# src/train/train_ensemble.py
"""
Training script for ensemble stacker using base model predictions.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from src.config import Config
from src.data.feature_store import FeatureStore
from src.models.ensemble_stacker import EnsembleStacker
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("train_ensemble")
cfg = Config()

def load_base_predictions():
    """Load OOF predictions from base models"""
    
    fs = FeatureStore()
    train_df = fs.load_train()
    
    oof_dir = os.path.join(cfg.output_dir, "oof_predictions")
    base_predictions = []
    model_names = []
    
    # Load LightGBM predictions
    lgb_path = os.path.join(oof_dir, "lgb_oof.csv")
    if os.path.exists(lgb_path):
        lgb_df = pd.read_csv(lgb_path)
        base_predictions.append(lgb_df['lgb_pred'].values)
        model_names.append('LightGBM')
        log.info("Loaded LightGBM OOF predictions")
    
    # Load MLP predictions
    mlp_path = os.path.join(oof_dir, "mlp_oof.csv")
    if os.path.exists(mlp_path):
        mlp_df = pd.read_csv(mlp_path)
        base_predictions.append(mlp_df['mlp_pred'].values)
        model_names.append('MLP')
        log.info("Loaded MLP OOF predictions")
    
    # Load GNN predictions
    gnn_path = os.path.join(oof_dir, "gnn_oof.csv")
    if os.path.exists(gnn_path):
        gnn_df = pd.read_csv(gnn_path)
        base_predictions.append(gnn_df['gnn_pred'].values)
        model_names.append('GNN')
        log.info("Loaded GNN OOF predictions")
    
    # Load LoRA predictions
    lora_path = os.path.join(oof_dir, "lora_oof.csv")
    if os.path.exists(lora_path):
        lora_df = pd.read_csv(lora_path)
        base_predictions.append(lora_df['lora_pred'].values)
        model_names.append('LoRA')
        log.info("Loaded LoRA OOF predictions")
    
    if not base_predictions:
        raise ValueError("No base model OOF predictions found. Train base models first.")
    
    # Stack predictions
    X_ensemble = np.column_stack(base_predictions)
    y_train = train_df["price"].values
    
    log.info(f"Ensemble training data shape: {X_ensemble.shape}")
    log.info(f"Base models: {model_names}")
    
    return X_ensemble, y_train, model_names

def load_test_predictions():
    """Load test predictions from base models"""
    
    test_dir = os.path.join(cfg.output_dir, "test_predictions")
    base_predictions = []
    
    # Load predictions in the same order as training
    model_files = [
        ("lgb_test.csv", "lgb_pred"),
        ("mlp_test.csv", "mlp_pred"),
        ("gnn_test.csv", "gnn_pred"),
        ("lora_test.csv", "lora_pred")
    ]
    
    for filename, pred_col in model_files:
        filepath = os.path.join(test_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            base_predictions.append(df[pred_col].values)
            log.info(f"Loaded test predictions from {filename}")
    
    if not base_predictions:
        raise ValueError("No base model test predictions found.")
    
    return np.column_stack(base_predictions)

def main():
    set_seed(cfg.seed)
    log.info("[AI-Agent] Starting ensemble stacker training")
    
    # Load OOF predictions for training
    X_ensemble, y_train, model_names = load_base_predictions()
    
    # Initialize and train ensemble
    ensemble = EnsembleStacker(save_dir=cfg.model_dir)
    ensemble.train(X_ensemble, y_train, num_folds=cfg.num_folds)
    
    # Save ensemble model
    ensemble.save()
    
    # Generate final test predictions
    X_test_ensemble = load_test_predictions()
    final_test_preds = ensemble.predict(X_test_ensemble)
    
    # Load test sample IDs
    fs = FeatureStore()
    test_df = fs.load_test()
    
    # Save final predictions
    final_pred_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.maximum(final_test_preds, 0.01)  # Ensure positive prices
    })
    
    final_path = os.path.join(cfg.output_dir, "test_out.csv")
    final_pred_df.to_csv(final_path, index=False)
    
    # Save ensemble OOF predictions
    oof_ensemble_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'ensemble_pred': np.maximum(ensemble.oof_predictions, 0.01)
    })
    oof_ensemble_path = os.path.join(cfg.output_dir, "oof_predictions", "ensemble_oof.csv")
    oof_ensemble_df.to_csv(oof_ensemble_path, index=False)
    
    # Print feature importance
    importance = ensemble.get_feature_importance()
    log.info("[AI-Agent] Ensemble model weights:")
    for i, (model_name, weight) in enumerate(zip(model_names, ensemble.meta_model.coef_)):
        log.info(f"  {model_name}: {weight:.4f}")
    
    log.info(f"[AI-Agent] Finished ensemble training. Final predictions saved to {final_path}")

if __name__ == "__main__":
    # Need to load train_df for sample_id
    from src.data.feature_store import FeatureStore
    fs = FeatureStore()
    train_df = fs.load_train()
    main()