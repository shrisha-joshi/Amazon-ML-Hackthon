# MIT License  
# generate_final_predictions.py
"""
Generate final predictions using the best performing model.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.data.feature_store import FeatureStore
from src.utils.logger import get_logger

log = get_logger("final_predictions")
cfg = Config()

def main():
    log.info("[AI-Agent] Generating final predictions")
    
    # Load feature store
    fs = FeatureStore()
    test_df = fs.load_test()
    
    # Load LightGBM test predictions (best performing model)
    lgb_test_path = os.path.join(cfg.output_dir, "test_predictions", "lgb_test.csv")
    
    if os.path.exists(lgb_test_path):
        lgb_predictions = pd.read_csv(lgb_test_path)
        
        # Create final submission
        final_predictions = pd.DataFrame({
            'sample_id': lgb_predictions['sample_id'],
            'price': lgb_predictions['lgb_pred']
        })
        
        # Ensure positive prices and reasonable range
        final_predictions['price'] = np.clip(final_predictions['price'], 0.01, 10000.0)
        
        # Validation checks
        assert len(final_predictions) == len(test_df), "Length mismatch"
        assert final_predictions['price'].isnull().sum() == 0, "NaN values found"
        assert (final_predictions['price'] > 0).all(), "Non-positive prices found"
        
        # Save final submission
        output_path = os.path.join(cfg.output_dir, "test_out.csv")
        final_predictions.to_csv(output_path, index=False)
        
        # Summary statistics
        log.info(f"[AI-Agent] Final Prediction Statistics:")
        log.info(f"  Count: {len(final_predictions)}")
        log.info(f"  Mean: ${np.mean(final_predictions['price']):.2f}")
        log.info(f"  Median: ${np.median(final_predictions['price']):.2f}")
        log.info(f"  Std: ${np.std(final_predictions['price']):.2f}")
        log.info(f"  Min: ${np.min(final_predictions['price']):.2f}")
        log.info(f"  Max: ${np.max(final_predictions['price']):.2f}")
        
        log.info(f"[AI-Agent] ✅ Final predictions saved to {output_path}")
        
        # Verify against sample format
        sample_path = "dataset/sample_test_out.csv"
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            log.info(f"[AI-Agent] Format validation:")
            log.info(f"  Sample format columns: {sample_df.columns.tolist()}")
            log.info(f"  Our format columns: {final_predictions.columns.tolist()}")
            log.info(f"  Sample shape: {sample_df.shape}")
            log.info(f"  Our shape: {final_predictions.shape}")
            
            if list(sample_df.columns) == list(final_predictions.columns):
                log.info("[AI-Agent] ✅ Format matches sample submission!")
            else:
                log.warning("[AI-Agent] ⚠️ Format mismatch with sample submission")
        
        return final_predictions
        
    else:
        log.error(f"LightGBM predictions not found at {lgb_test_path}")
        return None

if __name__ == "__main__":
    main()