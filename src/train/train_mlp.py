# MIT License
# src/train/train_mlp.py
"""
Training script for MLP model.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from src.config import Config
from src.data.feature_store import FeatureStore
from src.models.mlp_model import MLPModel
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("train_mlp")
cfg = Config()

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Prepare features for MLP training - same as LGB"""
    
    # Tabular features
    feature_cols = [
        'text_len', 'word_count', 'num_digits', 'num_caps', 'has_image',
        'pack_qty', 'unit_qty', 'unit_base_qty', 'price_per_base'
    ]
    
    # Filter available columns
    available_cols = [col for col in feature_cols if col in train_df.columns]
    log.info(f"Using tabular features: {available_cols}")
    
    X_train_tabular = train_df[available_cols].fillna(0).values
    X_test_tabular = test_df[available_cols].fillna(0).values
    
    # Try to load embeddings
    text_emb_train_path = "outputs/feature_store/embeddings_text/train/text_embeddings.npy"
    text_emb_test_path = "outputs/feature_store/embeddings_text/test/text_embeddings.npy"
    
    features_train = [X_train_tabular]
    features_test = [X_test_tabular]
    
    # Add text embeddings if available
    if os.path.exists(text_emb_train_path) and os.path.exists(text_emb_test_path):
        log.info("Loading text embeddings")
        text_emb_train = np.load(text_emb_train_path)
        text_emb_test = np.load(text_emb_test_path)
        features_train.append(text_emb_train)
        features_test.append(text_emb_test)
    else:
        log.warning("Text embeddings not found, skipping")
    
    # Concatenate all features
    X_train = np.hstack(features_train)
    X_test = np.hstack(features_test)
    
    log.info(f"Final feature shape - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test

def main():
    set_seed(cfg.seed)
    log.info("[AI-Agent] Starting MLP training")
    
    # Load data
    fs = FeatureStore()
    train_df = fs.load_train()
    test_df = fs.load_test()
    
    # Prepare features
    X_train, X_test = prepare_features(train_df, test_df)
    
    # Target variable (log transform for better performance)
    y_train = np.log1p(train_df["price"].values)
    
    # Initialize and train model
    model = MLPModel(
        input_dim=X_train.shape[1], 
        hidden_dims=[256, 128, 64],
        model_name="mlp",
        save_dir=cfg.model_dir
    )
    
    model.train_model(X_train, y_train, num_folds=cfg.num_folds, epochs=50, batch_size=512)
    
    # Save model
    model.save()
    
    # Generate test predictions
    test_preds = np.expm1(model.predict(X_test))  # Transform back from log space
    
    # Save OOF predictions for ensemble
    oof_preds = np.expm1(model.oof_predictions)
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'mlp_pred': oof_preds
    })
    oof_path = os.path.join(cfg.output_dir, "oof_predictions", "mlp_oof.csv")
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    
    # Save test predictions
    test_df_pred = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'mlp_pred': test_preds
    })
    test_pred_path = os.path.join(cfg.output_dir, "test_predictions", "mlp_test.csv")
    os.makedirs(os.path.dirname(test_pred_path), exist_ok=True)
    test_df_pred.to_csv(test_pred_path, index=False)
    
    log.info(f"[AI-Agent] Finished MLP training. OOF saved to {oof_path}")
    log.info(f"[AI-Agent] Test predictions saved to {test_pred_path}")

if __name__ == "__main__":
    main()