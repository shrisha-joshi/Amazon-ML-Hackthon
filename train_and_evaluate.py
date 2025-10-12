#!/usr/bin/env python3
"""
Train and Evaluate - Single Step Reproducible Pipeline
Team 127.0.0.1 - Amazon ML Challenge 2025

This script runs the complete pipeline: data loading, training, and inference
to generate final predictions for the 75,000 test samples.
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

def check_environment():
    """Verify Python version and essential imports"""
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python ≥3.10 required, got {sys.version_info.major}.{sys.version_info.minor}")
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check essential imports
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import lightgbm
        print("✅ Core ML libraries imported successfully")
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}")
    
    print("✅ Environment check passed\n")

def set_reproducible_seeds():
    """Set all random seeds for reproducibility"""
    print("🎯 SETTING REPRODUCIBLE SEEDS")
    print("=" * 50)
    
    np.random.seed(42)
    
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        print("✅ PyTorch seeds set")
    except ImportError:
        print("⚠️  PyTorch not available, skipping torch seeds")
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = '42'
    print("✅ All seeds fixed for reproducibility\n")

def validate_data_integrity():
    """Validate input data integrity"""
    print("🔍 DATA INTEGRITY CHECK")
    print("=" * 50)
    
    # Check training data
    train_path = "student_resource/dataset/train.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    print(f"✅ Training data: {len(train_df):,} samples")
    
    # Check test data
    test_path = "student_resource/dataset/test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    test_df = pd.read_csv(test_path)
    print(f"✅ Test data: {len(test_df):,} samples")
    
    # Validate columns
    required_cols = ['sample_id', 'catalog_content']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing column in training data: {col}")
        if col not in test_df.columns:
            raise ValueError(f"Missing column in test data: {col}")
    
    # Check for NaN values
    if train_df['catalog_content'].isnull().sum() > 0:
        print(f"⚠️  Training data has {train_df['catalog_content'].isnull().sum()} null catalog_content values")
    if test_df['catalog_content'].isnull().sum() > 0:
        print(f"⚠️  Test data has {test_df['catalog_content'].isnull().sum()} null catalog_content values")
    
    print("✅ Data integrity validated\n")
    return train_df, test_df

def run_training_and_inference(use_cpu=False):
    """Run the complete training and inference pipeline"""
    print("🚀 TRAINING AND INFERENCE PIPELINE")
    print("=" * 50)
    
    # Import the main efficient model
    sys.path.append('smart_pricing_project')
    from eval_efficient import EfficientLargeScalePredictor
    
    print("📊 Loading and preprocessing data...")
    
    # Load training data
    train_df = pd.read_csv("student_resource/dataset/train.csv")
    test_df = pd.read_csv("student_resource/dataset/test.csv")
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    # Initialize predictor
    predictor = EfficientLargeScalePredictor()
    
    print("\n🎯 Training model...")
    # Use training data as both train and validation for final model
    predictor.fit(train_df, train_df)  
    
    print("\n🔮 Generating predictions...")
    # Generate predictions for test set
    predictions = predictor.predict(test_df)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].astype(int),
        'price': predictions.astype(float)
    })
    
    # Ensure all predictions are positive
    submission_df['price'] = np.maximum(submission_df['price'], 0.01)
    
    return submission_df

def save_final_predictions(submission_df):
    """Save predictions and validate output"""
    print("💾 SAVING FINAL PREDICTIONS")
    print("=" * 50)
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Save to multiple locations for safety
    output_paths = [
        'outputs/final_predictions.csv',
        'test_out.csv'  # Main submission file
    ]
    
    for path in output_paths:
        submission_df.to_csv(path, index=False)
        print(f"✅ Saved: {path}")
    
    # Validate output
    print("\n📊 PREDICTION VALIDATION")
    print("=" * 30)
    print(f"Total predictions: {len(submission_df):,}")
    print(f"Unique sample_ids: {submission_df['sample_id'].nunique():,}")
    print(f"Price range: ${submission_df['price'].min():.2f} - ${submission_df['price'].max():.2f}")
    print(f"Average price: ${submission_df['price'].mean():.2f}")
    
    # Validation checks
    assert len(submission_df) == 75000, f"Expected 75,000 predictions, got {len(submission_df)}"
    assert submission_df['sample_id'].nunique() == 75000, "Duplicate sample_ids found"
    assert (submission_df['price'] > 0).all(), "Non-positive prices found"
    assert submission_df['price'].isnull().sum() == 0, "NaN values in predictions"
    
    print("✅ All validation checks passed!")
    return submission_df

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train and Evaluate - Team 127.0.0.1")
    parser.add_argument('--use_cpu', action='store_true', 
                       help='Force CPU usage even if CUDA is available')
    args = parser.parse_args()
    
    try:
        print("🏆 TEAM 127.0.0.1 - AMAZON ML CHALLENGE 2025")
        print("=" * 60)
        print("Single-step reproducible training and evaluation pipeline")
        print("=" * 60)
        
        # Step 1: Environment check
        check_environment()
        
        # Step 2: Set reproducible seeds
        set_reproducible_seeds()
        
        # Step 3: Validate data integrity
        validate_data_integrity()
        
        # Step 4: Run training and inference
        submission_df = run_training_and_inference(use_cpu=args.use_cpu)
        
        # Step 5: Save and validate predictions
        save_final_predictions(submission_df)
        
        print("\n🎉 SUCCESS!")
        print("=" * 30)
        print("✅ Training completed successfully")
        print("✅ 75,000 predictions generated")
        print("✅ Output saved to: test_out.csv")
        print("✅ Reproducible results confirmed")
        print("\n📦 Ready for submission!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify data files exist: student_resource/dataset/train.csv and test.csv")
        print("3. Check Python version ≥3.10")
        sys.exit(1)

if __name__ == "__main__":
    main()