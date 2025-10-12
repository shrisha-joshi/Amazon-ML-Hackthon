#!/usr/bin/env python3

import pandas as pd

def show_predictions():
    """Display all sample predictions in clean format"""
    
    # Load predictions
    df = pd.read_csv('outputs/sample_test_pred_final.csv')
    df = df.sort_values('sample_id')
    
    print('🎯 COMPLETE SAMPLE_TEST.CSV PREDICTIONS')
    print('=' * 45)
    print('Sample ID    | Predicted Price')
    print('-------------|----------------')
    
    for _, row in df.iterrows():
        sample_id = int(row['sample_id'])
        price = float(row['price'])
        print(f'{sample_id:>12} | ${price:>13.2f}')
    
    print('-------------|----------------')
    print(f'Total: {len(df)} predictions generated')
    print('=' * 45)
    
    return df

if __name__ == "__main__":
    predictions_df = show_predictions()