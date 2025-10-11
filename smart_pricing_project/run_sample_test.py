#!/usr/bin/env python3
"""
Run the corrected model on sample_test.csv
This uses our best performing simple model with reliable features only.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os

# Add src to path
sys.path.append('src')
from src.data.preprocess_features import build_features

def main():
    print('🚀 RUNNING CORRECTED MODEL ON SAMPLE_TEST.CSV')
    print('=' * 60)

    # Load and process sample test data
    sample_test = pd.read_csv('dataset/sample_test.csv')
    print(f'Sample test data loaded: {sample_test.shape}')

    # Process features using the same pipeline
    processed_sample = build_features(sample_test, is_train=False)

    # Use only reliable features (the ones that worked well)
    reliable_features = ['text_len', 'word_count', 'num_digits', 'num_caps', 'has_image']
    X_sample = processed_sample[reliable_features].fillna(0)

    print(f'Using {len(reliable_features)} reliable features')
    print(f'Features: {reliable_features}')

    # Recreate best model using training data
    print('\nRecreating best model using training data...')

    # Load training sample for model training
    train_df = pd.read_csv('dataset/train.csv', nrows=5000)  # Use subset for speed
    processed_train = build_features(train_df, is_train=True)

    X_train = processed_train[reliable_features].fillna(0)
    y_train = train_df['price'].values

    # Train Random Forest (our best model)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    # Generate predictions for sample test
    sample_predictions = model.predict(X_sample)

    # Ensure positive predictions
    sample_predictions = np.maximum(sample_predictions, 0.1)

    print(f'\n📊 SAMPLE PREDICTIONS GENERATED:')
    print(f'Predictions range: ${sample_predictions.min():.2f} to ${sample_predictions.max():.2f}')
    print(f'Mean prediction: ${sample_predictions.mean():.2f}')

    # Create output DataFrame
    sample_output = pd.DataFrame({
        'sample_id': processed_sample['sample_id'],
        'price': sample_predictions
    })

    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Save predictions
    sample_output.to_csv('outputs/final_sample_predictions.csv', index=False)
    print(f'\n💾 Predictions saved to outputs/final_sample_predictions.csv')

    # Compare with ground truth if available
    try:
        ground_truth = pd.read_csv('dataset/sample_test_out.csv')
        comparison = sample_output.merge(ground_truth, on='sample_id', suffixes=('_pred', '_actual'))
        
        mae = mean_absolute_error(comparison['price_actual'], comparison['price_pred'])
        r2 = r2_score(comparison['price_actual'], comparison['price_pred'])
        
        # Calculate SMAPE
        smape = np.mean(200 * np.abs(comparison['price_actual'] - comparison['price_pred']) / 
                       (np.abs(comparison['price_actual']) + np.abs(comparison['price_pred'])))
        
        print(f'\n🎯 ACCURACY RESULTS:')
        print(f'MAE: ${mae:.2f}')
        print(f'R² Score: {r2:.4f}')
        print(f'SMAPE: {smape:.1f}%')
        
        # Show best and worst predictions
        comparison['abs_error'] = np.abs(comparison['price_actual'] - comparison['price_pred'])
        comparison['rel_error'] = comparison['abs_error'] / comparison['price_actual'] * 100
        
        print(f'\n🏆 BEST 5 PREDICTIONS:')
        best = comparison.nsmallest(5, 'rel_error')
        for _, row in best.iterrows():
            print(f'ID: {int(row["sample_id"]):6d} | Actual: ${row["price_actual"]:6.2f} | Pred: ${row["price_pred"]:6.2f} | Error: {row["rel_error"]:5.1f}%')
        
        print(f'\n⚠️ WORST 5 PREDICTIONS:')
        worst = comparison.nlargest(5, 'rel_error')
        for _, row in worst.iterrows():
            print(f'ID: {int(row["sample_id"]):6d} | Actual: ${row["price_actual"]:6.2f} | Pred: ${row["price_pred"]:6.2f} | Error: {row["rel_error"]:5.1f}%')
            
    except Exception as e:
        print(f'\n⚠️ Could not load ground truth: {e}')

    print(f'\n✅ SAMPLE_TEST.CSV PROCESSING COMPLETE!')
    
    # Display first few predictions
    print(f'\n📋 FIRST 10 PREDICTIONS:')
    print(sample_output.head(10).to_string(index=False))

if __name__ == '__main__':
    main()