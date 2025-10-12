# MIT License
# src/inference/predict.py
"""
Main inference pipeline for generating final predictions.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from src.config import Config
from src.data.feature_store import FeatureStore
from src.models.ensemble_stacker import EnsembleStacker
from src.postprocess.smoother import apply_post_processing
from src.utils.logger import get_logger

log = get_logger("inference")
cfg = Config()

def load_trained_ensemble():
    """Load the trained ensemble model"""
    ensemble = EnsembleStacker(save_dir=cfg.model_dir)
    ensemble.load()
    return ensemble

def load_base_test_predictions():
    """Load test predictions from all base models"""
    test_dir = os.path.join(cfg.output_dir, "test_predictions")
    base_predictions = []
    model_names = []
    
    # Define expected prediction files
    prediction_files = [
        ("lgb_test.csv", "lgb_pred", "LightGBM"),
        ("mlp_test.csv", "mlp_pred", "MLP"),
        ("gnn_test.csv", "gnn_pred", "GNN"),
        ("lora_test.csv", "lora_pred", "LoRA")
    ]
    
    for filename, pred_col, model_name in prediction_files:
        filepath = os.path.join(test_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            base_predictions.append(df[pred_col].values)
            model_names.append(model_name)
            log.info(f"Loaded {model_name} test predictions from {filename}")
        else:
            log.warning(f"Test predictions not found for {model_name}: {filepath}")
    
    if not base_predictions:
        raise ValueError("No base model test predictions found. Run base model training first.")
    
    return np.column_stack(base_predictions), model_names

def run_inference():
    """Run the complete inference pipeline"""
    log.info("[AI-Agent] Starting inference pipeline")
    
    # Load feature store
    fs = FeatureStore()
    train_df = fs.load_train()
    test_df = fs.load_test()
    
    # Load trained ensemble model
    try:
        ensemble = load_trained_ensemble()
        log.info("[AI-Agent] Loaded trained ensemble model")
    except Exception as e:
        log.error(f"Failed to load ensemble model: {e}")
        log.info("[AI-Agent] Falling back to simple average of base predictions")
        ensemble = None
    
    # Load base model test predictions
    base_test_preds, model_names = load_base_test_predictions()
    log.info(f"[AI-Agent] Loaded predictions from {len(model_names)} base models: {model_names}")
    
    # Generate ensemble predictions
    if ensemble is not None:
        log.info("[AI-Agent] Generating ensemble predictions")
        final_preds = ensemble.predict(base_test_preds)
    else:
        log.info("[AI-Agent] Using simple average as fallback")
        final_preds = np.mean(base_test_preds, axis=1)
    
    # Apply post-processing
    train_prices = train_df["price"].values
    final_preds = apply_post_processing(final_preds, train_prices)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_preds
    })
    
    # Ensure proper data types
    submission_df['sample_id'] = submission_df['sample_id'].astype(int)
    submission_df['price'] = submission_df['price'].astype(float)
    
    # Save final predictions
    output_path = os.path.join(cfg.output_dir, "test_out.csv")
    submission_df.to_csv(output_path, index=False)
    
    # Validation checks
    log.info("[AI-Agent] Running validation checks...")
    assert len(submission_df) == len(test_df), "Submission length mismatch"
    assert submission_df['price'].isnull().sum() == 0, "NaN values in predictions"
    assert (submission_df['price'] > 0).all(), "Non-positive prices found"
    
    # Summary statistics
    log.info(f"[AI-Agent] Prediction statistics:")
    log.info(f"  Count: {len(final_preds)}")
    log.info(f"  Mean: {np.mean(final_preds):.2f}")
    log.info(f"  Median: {np.median(final_preds):.2f}")
    log.info(f"  Std: {np.std(final_preds):.2f}")
    log.info(f"  Min: {np.min(final_preds):.2f}")
    log.info(f"  Max: {np.max(final_preds):.2f}")
    
    log.info(f"[AI-Agent] ✅ Inference completed. Final predictions saved to {output_path}")
    
    return submission_df

if __name__ == "__main__":
    run_inference()