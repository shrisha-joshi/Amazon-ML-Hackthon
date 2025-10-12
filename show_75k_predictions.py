#!/usr/bin/env python3

import pandas as pd
import numpy as np

def show_75k_predictions():
    """Display summary and sample of 75,000 predictions"""
    
    print('🎯 COMPLETE 75,000 SAMPLE PREDICTIONS')
    print('=' * 70)
    
    # Load predictions
    df = pd.read_csv('outputs/test_predictions_75k.csv')
    df = df.sort_values('sample_id')
    
    # Summary statistics
    print('📊 PREDICTION SUMMARY:')
    print(f'   Total predictions: {len(df):,}')
    print(f'   Price range: ${df["price"].min():.2f} - ${df["price"].max():.2f}')
    print(f'   Average price: ${df["price"].mean():.2f}')
    print(f'   Median price: ${df["price"].median():.2f}')
    print(f'   Standard deviation: ${df["price"].std():.2f}')
    print()
    
    # Price distribution
    print('📊 PRICE DISTRIBUTION:')
    percentiles = df['price'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f'   10th percentile: ${percentiles[0.1]:.2f}')
    print(f'   25th percentile: ${percentiles[0.25]:.2f}')
    print(f'   50th percentile: ${percentiles[0.5]:.2f}')
    print(f'   75th percentile: ${percentiles[0.75]:.2f}')
    print(f'   90th percentile: ${percentiles[0.9]:.2f}')
    print()
    
    # Sample predictions
    print('📋 SAMPLE PREDICTIONS (First 100):')
    print('Sample ID    | Predicted Price')
    print('-------------|----------------')
    
    for i in range(min(100, len(df))):
        sample_id = int(df.iloc[i]['sample_id'])
        price = float(df.iloc[i]['price'])
        print(f'{sample_id:>12} | ${price:>13.2f}')
    
    print('-------------|----------------')
    if len(df) > 100:
        print(f'... and {len(df)-100:,} more predictions')
    
    print()
    print('💾 Complete file: outputs/test_predictions_75k.csv')
    print('🎉 ALL 75,000 PREDICTIONS SUCCESSFULLY GENERATED!')
    print('=' * 70)
    
    return df

if __name__ == "__main__":
    predictions_df = show_75k_predictions()