#!/usr/bin/env python3
"""
Create properly formatted test_out.csv for submission
Team: 127.0.0.1
"""

import pandas as pd

def create_submission_file():
    print('📋 Creating properly formatted test_out.csv for team 127.0.0.1...')
    
    # Load our 75k predictions
    df = pd.read_csv('outputs/test_predictions_75k.csv')
    
    # Sort by sample_id (competition requirement)
    df = df.sort_values('sample_id')
    
    # Ensure all prices are positive (competition requirement)
    negative_count = len(df[df['price'] <= 0])
    if negative_count > 0:
        print(f'⚠️  Found {negative_count} non-positive prices, fixing...')
        df.loc[df['price'] <= 0, 'price'] = 1.0
    
    # Verify we have exactly 75,000 samples
    assert len(df) == 75000, f"Expected 75,000 samples, got {len(df)}"
    
    # Save with exact formatting matching sample_test_out.csv
    output_path = 'submission/team_127.0.0.1/test_out.csv'
    df.to_csv(output_path, index=False)
    
    # Validation
    print(f'✅ Created test_out.csv with {len(df):,} predictions')
    print(f'   Sample ID range: {df["sample_id"].min()} to {df["sample_id"].max()}')
    print(f'   Price range: ${df["price"].min():.2f} to ${df["price"].max():.2f}')
    print(f'   All prices positive: {(df["price"] > 0).all()}')
    print(f'💾 Saved to: {output_path}')
    
    # Verify file format matches sample
    print('\n🔍 Verifying format matches sample_test_out.csv...')
    sample_format = pd.read_csv('student_resource/dataset/sample_test_out.csv')
    
    if list(df.columns) == list(sample_format.columns):
        print('✅ Column names match perfectly')
    else:
        print('❌ Column name mismatch!')
        
    print('🎉 test_out.csv ready for submission!')
    return df

if __name__ == "__main__":
    result = create_submission_file()