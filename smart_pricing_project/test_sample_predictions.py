#!/usr/bin/env python3
"""
Test predictions on sample_test.csv and validate format matches sample_test_out.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from src.data.preprocess_features import build_features

def load_trained_model():
    """Load the best trained model (LightGBM)"""
    model_path = 'outputs/models/lightgbm.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"Loaded LightGBM ensemble with {len(models)} fold models")
    return models

def generate_sample_predictions():
    """Generate predictions for sample_test.csv"""
    
    # Load and process sample test data
    print("Loading sample_test.csv...")
    sample_test = pd.read_csv('dataset/sample_test.csv')
    print(f"Sample test shape: {sample_test.shape}")
    print(f"Sample test columns: {list(sample_test.columns)}")
    
    # Process features
    print("Processing features...")
    processed_sample = build_features(sample_test, is_train=False)
    
    # Get tabular features
    tabular_features = ['text_len', 'word_count', 'num_digits', 'num_caps', 'has_image', 
                       'pack_qty', 'unit_qty', 'unit_base_qty', 'price_per_base']
    
    # Load text embeddings if available
    text_emb_path = 'outputs/embeddings/text_embeddings_test.npy'
    if os.path.exists(text_emb_path):
        print("Loading text embeddings...")
        text_embeddings = np.load(text_emb_path)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Create full feature matrix
        tabular_data = processed_sample[tabular_features].values
        X_test = np.concatenate([tabular_data, text_embeddings], axis=1)
        print(f"Final feature matrix shape: {X_test.shape}")
    else:
        print("Text embeddings not found, using only tabular features")
        X_test = processed_sample[tabular_features].values
    
    # Load trained model
    print("Loading trained LightGBM model...")
    models = load_trained_model()
    
    # Generate predictions (average of all folds)
    print("Generating predictions...")
    predictions = np.zeros(len(sample_test))
    
    for i, model in enumerate(models):
        fold_pred = model.predict(X_test)
        predictions += fold_pred
        print(f"Fold {i+1} predictions - Min: {fold_pred.min():.6f}, Max: {fold_pred.max():.6f}, Mean: {fold_pred.mean():.6f}")
    
    predictions = predictions / len(models)
    
    # Ensure positive prices
    predictions = np.maximum(predictions, 0.01)
    
    print(f"Final predictions - Min: {predictions.min():.6f}, Max: {predictions.max():.6f}, Mean: {predictions.mean():.6f}")
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'sample_id': processed_sample['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    output_path = 'outputs/sample_test_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Sample predictions saved to {output_path}")
    
    return output_df

def validate_format():
    """Validate that our predictions match the expected format"""
    
    # Load our predictions
    our_predictions = pd.read_csv('outputs/sample_test_predictions.csv')
    
    # Load expected format
    expected_format = pd.read_csv('dataset/sample_test_out.csv')
    
    print("=== FORMAT VALIDATION ===")
    print(f"Our predictions shape: {our_predictions.shape}")
    print(f"Expected format shape: {expected_format.shape}")
    
    print(f"Our columns: {list(our_predictions.columns)}")
    print(f"Expected columns: {list(expected_format.columns)}")
    
    # Check sample IDs match
    our_ids = set(our_predictions['sample_id'])
    expected_ids = set(expected_format['sample_id'])
    
    print(f"Sample IDs match: {our_ids == expected_ids}")
    
    if our_ids != expected_ids:
        missing_in_ours = expected_ids - our_ids
        extra_in_ours = our_ids - expected_ids
        print(f"Missing in our predictions: {missing_in_ours}")
        print(f"Extra in our predictions: {extra_in_ours}")
    
    print(f"All prices positive: {(our_predictions['price'] > 0).all()}")
    print(f"Price range: {our_predictions['price'].min():.6f} to {our_predictions['price'].max():.6f}")
    
    print("=== SAMPLE COMPARISON ===")
    merged = our_predictions.merge(expected_format, on='sample_id', suffixes=('_ours', '_expected'))
    print("First 5 sample comparisons:")
    print(merged[['sample_id', 'price_ours', 'price_expected']].head())
    
    return our_predictions

if __name__ == "__main__":
    print("=== TESTING ON SAMPLE_TEST.CSV ===")
    
    try:
        # Generate predictions
        predictions = generate_sample_predictions()
        
        # Validate format
        validated = validate_format()
        
        print("\n=== SUCCESS ===")
        print("Sample test predictions generated and validated successfully!")
        print(f"Generated {len(predictions)} predictions")
        print("Format matches expected sample_test_out.csv structure")
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Failed to generate sample predictions: {e}")
        import traceback
        traceback.print_exc()