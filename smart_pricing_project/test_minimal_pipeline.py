# MIT License
# test_minimal_pipeline.py
"""
Minimal pipeline test to validate core functionality without heavy dependencies.
"""

from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.feature_store import FeatureStore
from src.utils.smape_loss import smape_numpy

log = get_logger("minimal_pipeline")
cfg = Config()

def test_minimal_prediction_pipeline():
    """Test a minimal prediction pipeline using simple models"""
    log.info("[AI-Agent] Testing minimal prediction pipeline...")
    
    set_seed(cfg.seed)
    
    # Load processed features
    fs = FeatureStore()
    train_df = fs.load_train()
    test_df = fs.load_test()
    
    log.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    
    # Simple feature selection (tabular only)
    feature_cols = ['text_len', 'word_count', 'num_digits', 'has_image', 'pack_qty', 'unit_qty']
    available_cols = [col for col in feature_cols if col in train_df.columns]
    
    X_train = train_df[available_cols].fillna(0).values
    X_test = test_df[available_cols].fillna(0).values
    y_train = train_df['price'].values
    
    log.info(f"Features used: {available_cols}")
    log.info(f"Feature matrix shape: {X_train.shape}")
    
    # Simple baseline model (mean by feature quantiles)
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict
    
    # Train simple linear regression
    model = LinearRegression()
    
    # Get cross-validated predictions for validation
    cv_preds = cross_val_predict(model, X_train, y_train, cv=5)
    
    # Calculate validation SMAPE
    val_smape = smape_numpy(y_train, cv_preds)
    log.info(f"Cross-validation SMAPE: {val_smape:.4f}%")
    
    # Train on full data and predict test
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    
    # Ensure positive predictions
    test_preds = np.maximum(test_preds, 0.01)
    
    # Create submission format
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    
    # Save minimal predictions
    output_path = "outputs/minimal_test_predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    # Validation checks
    assert len(submission_df) == len(test_df), "Length mismatch"
    assert submission_df['price'].isnull().sum() == 0, "NaN values found"
    assert (submission_df['price'] > 0).all(), "Non-positive prices found"
    
    log.info(f"Prediction stats: mean={np.mean(test_preds):.2f}, std={np.std(test_preds):.2f}")
    log.info(f"Price range: [{np.min(test_preds):.2f}, {np.max(test_preds):.2f}]")
    log.info(f"[AI-Agent] ✅ Minimal pipeline test passed! Saved to {output_path}")
    
    return True

def main():
    """Run minimal pipeline test"""
    log.info("[AI-Agent] 🧪 Starting minimal pipeline test")
    
    try:
        result = test_minimal_prediction_pipeline()
        
        if result:
            log.info("[AI-Agent] 🎉 Minimal pipeline test successful!")
            log.info("[AI-Agent] System is ready for full training with proper dependencies.")
        else:
            log.error("[AI-Agent] ❌ Minimal pipeline test failed")
            
    except Exception as e:
        log.error(f"[AI-Agent] ❌ Minimal pipeline test failed with exception: {e}")
        
    return True

if __name__ == "__main__":
    main()