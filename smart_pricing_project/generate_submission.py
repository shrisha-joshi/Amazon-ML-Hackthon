#!/usr/bin/env python3
"""
Generate final submission file using the best performing model
"""

import pandas as pd
import numpy as np

def main():
    # Load LightGBM predictions (best model with 1.40% SMAPE)
    lgb_preds = pd.read_csv('outputs/test_predictions/lgb_test.csv')
    print(f'LightGBM predictions shape: {lgb_preds.shape}')
    print(f'Columns: {list(lgb_preds.columns)}')
    print(f'Price stats - Min: {lgb_preds["lgb_pred"].min():.6f}, Max: {lgb_preds["lgb_pred"].max():.6f}, Mean: {lgb_preds["lgb_pred"].mean():.6f}')
    
    # Create final submission with correct column names
    final_submission = pd.DataFrame({
        'sample_id': lgb_preds['sample_id'],
        'price': lgb_preds['lgb_pred']
    })
    final_submission.to_csv('outputs/test_out.csv', index=False)
    
    print(f'Final submission saved to outputs/test_out.csv')
    print(f'Submission shape: {final_submission.shape}')
    print(f'Columns: {list(final_submission.columns)}')
    
    # Validate format
    print(f'Sample rows:')
    print(final_submission.head())

if __name__ == '__main__':
    main()