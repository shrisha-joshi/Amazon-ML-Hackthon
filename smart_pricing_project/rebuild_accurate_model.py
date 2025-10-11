#!/usr/bin/env python3
"""
Deep analysis and rebuilding of the pricing model for maximum accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def analyze_data():
    """Deep analysis of the training data"""
    print("=" * 60)
    print("COMPREHENSIVE DATA ANALYSIS")
    print("=" * 60)
    
    # Load training data
    train = pd.read_csv('dataset/train.csv')
    print(f"Training data shape: {train.shape}")
    print(f"Columns: {list(train.columns)}")
    
    # Price distribution analysis
    print("\nPRICE DISTRIBUTION:")
    print("-" * 40)
    prices = train['price']
    print(f"Min price: ${prices.min():.2f}")
    print(f"Max price: ${prices.max():.2f}")
    print(f"Mean price: ${prices.mean():.2f}")
    print(f"Median price: ${prices.median():.2f}")
    print(f"Std price: ${prices.std():.2f}")
    
    # Percentile analysis
    print("\nPrice percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(prices, p)
        print(f"{p:2d}th percentile: ${val:8.2f}")
    
    # Price range distribution
    print("\nPrice range distribution:")
    ranges = [
        (0, 1, '$0-1'),
        (1, 5, '$1-5'), 
        (5, 10, '$5-10'),
        (10, 25, '$10-25'),
        (25, 50, '$25-50'),
        (50, 100, '$50-100'),
        (100, 200, '$100-200'),
        (200, float('inf'), '$200+')
    ]
    
    for min_p, max_p, label in ranges:
        if max_p == float('inf'):
            count = len(train[train['price'] >= min_p])
        else:
            count = len(train[(train['price'] >= min_p) & (train['price'] < max_p)])
        pct = count / len(train) * 100
        print(f"{label:8}: {count:6,} samples ({pct:5.1f}%)")
    
    return train

def analyze_catalog_content(train):
    """Analyze catalog content for better feature engineering"""
    print("\n" + "=" * 60)
    print("CATALOG CONTENT ANALYSIS")
    print("=" * 60)
    
    # Sample catalog content analysis
    print("Sample catalog content (first 3):")
    for i in range(min(3, len(train))):
        content = train.iloc[i]['catalog_content']
        price = train.iloc[i]['price']
        print(f"\nSample {i+1} (Price: ${price:.2f}):")
        print(f"Content length: {len(str(content))} characters")
        print(f"Content preview: {str(content)[:200]}...")
    
    # Analyze content length vs price correlation
    train['content_length'] = train['catalog_content'].astype(str).str.len()
    train['word_count'] = train['catalog_content'].astype(str).str.split().str.len()
    
    print(f"\nContent length vs price correlation: {train['content_length'].corr(train['price']):.4f}")
    print(f"Word count vs price correlation: {train['word_count'].corr(train['price']):.4f}")
    
    return train

def extract_advanced_features(df):
    """Extract more sophisticated features for better price prediction"""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    # Ensure catalog_content is string
    df['catalog_content'] = df['catalog_content'].astype(str)
    
    # Basic text features
    df['text_length'] = df['catalog_content'].str.len()
    df['word_count'] = df['catalog_content'].str.split().str.len()
    df['sentence_count'] = df['catalog_content'].str.count('\.') + 1
    df['char_per_word'] = df['text_length'] / (df['word_count'] + 1)
    
    # Numerical features from text
    df['digit_count'] = df['catalog_content'].str.count(r'\d')
    df['number_count'] = df['catalog_content'].str.count(r'\d+')
    df['capital_count'] = df['catalog_content'].str.count(r'[A-Z]')
    df['special_char_count'] = df['catalog_content'].str.count(r'[!@#$%^&*(),.?":{}|<>]')
    
    # Price-related keywords
    price_keywords = ['premium', 'luxury', 'professional', 'deluxe', 'gold', 'platinum', 
                     'pro', 'advanced', 'supreme', 'elite', 'executive', 'ultimate']
    budget_keywords = ['basic', 'standard', 'economy', 'budget', 'value', 'affordable',
                      'cheap', 'simple', 'essential', 'starter']
    
    df['premium_keywords'] = df['catalog_content'].str.lower().str.count('|'.join(price_keywords))
    df['budget_keywords'] = df['catalog_content'].str.lower().str.count('|'.join(budget_keywords))
    
    # Brand indicators (common high-value brands)
    brand_keywords = ['apple', 'samsung', 'sony', 'nike', 'adidas', 'canon', 'hp', 'dell',
                     'microsoft', 'google', 'amazon', 'brand', 'original']
    df['brand_mentions'] = df['catalog_content'].str.lower().str.count('|'.join(brand_keywords))
    
    # Quantity and size indicators
    df['pack_indicators'] = df['catalog_content'].str.count(r'\d+\s*(pack|pcs|pieces|count|ct)')
    df['size_indicators'] = df['catalog_content'].str.count(r'\d+\s*(ml|l|oz|lb|kg|g|inch|cm|mm)')
    df['volume_numbers'] = df['catalog_content'].str.extractall(r'(\d+\.?\d*)\s*(ml|l|oz)').groupby(level=0).size()
    df['volume_numbers'] = df['volume_numbers'].fillna(0)
    
    # Weight indicators
    df['weight_numbers'] = df['catalog_content'].str.extractall(r'(\d+\.?\d*)\s*(lb|kg|g|gram)').groupby(level=0).size()
    df['weight_numbers'] = df['weight_numbers'].fillna(0)
    
    # Image link features
    df['has_image'] = (~df['image_link'].isna()).astype(int)
    df['amazon_image'] = df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
    
    print(f"Created {len([col for col in df.columns if col not in ['sample_id', 'catalog_content', 'image_link', 'price']])} features")
    
    return df

def build_optimized_model(train_df):
    """Build a highly optimized model for price prediction"""
    print("\n" + "=" * 60)
    print("BUILDING OPTIMIZED MODEL")
    print("=" * 60)
    
    # Feature columns (exclude target and identifiers)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['sample_id', 'catalog_content', 'image_link', 'price']]
    
    X = train_df[feature_cols].fillna(0)
    y = train_df['price']
    
    print(f"Features used: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    # Optimized LightGBM parameters for price prediction
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1,
        'seed': 42
    }
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/5...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )
        
        # Predict on validation set
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred
        models.append(model)
        
        # Calculate fold metrics
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # SMAPE calculation
        smape = np.mean(200 * np.abs(y_val - val_pred) / (np.abs(y_val) + np.abs(val_pred)))
        
        print(f"Fold {fold + 1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape:.4f}%")
    
    # Overall metrics
    overall_mae = mean_absolute_error(y, oof_predictions)
    overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    overall_smape = np.mean(200 * np.abs(y - oof_predictions) / (np.abs(y) + np.abs(oof_predictions)))
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"MAE: {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"SMAPE: {overall_smape:.4f}%")
    
    return models, feature_cols, oof_predictions

def analyze_errors(train_df, predictions):
    """Analyze prediction errors to identify improvement areas"""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    actual = train_df['price']
    errors = predictions - actual
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / actual * 100
    
    # Error statistics
    print(f"Mean Absolute Error: ${np.mean(abs_errors):.2f}")
    print(f"Median Absolute Error: ${np.median(abs_errors):.2f}")
    print(f"Mean Relative Error: {np.mean(rel_errors):.1f}%")
    print(f"Median Relative Error: {np.median(rel_errors):.1f}%")
    
    # Error by price range
    print("\nError by price range:")
    ranges = [
        (0, 5, '$0-5'),
        (5, 10, '$5-10'),
        (10, 25, '$10-25'),
        (25, 50, '$25-50'),
        (50, 100, '$50-100'),
        (100, float('inf'), '$100+')
    ]
    
    for min_p, max_p, label in ranges:
        if max_p == float('inf'):
            mask = actual >= min_p
        else:
            mask = (actual >= min_p) & (actual < max_p)
        
        if mask.sum() > 0:
            range_mae = np.mean(abs_errors[mask])
            range_rel = np.mean(rel_errors[mask])
            count = mask.sum()
            print(f"{label:8}: MAE=${range_mae:6.2f}, Rel={range_rel:5.1f}%, n={count:,}")
    
    # Worst predictions
    worst_idx = np.argsort(rel_errors)[-10:]
    print(f"\nWorst 10 predictions:")
    print("Sample_ID | Actual  | Predicted | Error   | Rel_Error")
    print("-" * 55)
    for idx in worst_idx:
        sample_id = train_df.iloc[idx]['sample_id']
        actual_price = actual.iloc[idx]
        pred_price = predictions[idx]
        error = pred_price - actual_price
        rel_error = abs(error) / actual_price * 100
        print(f"{sample_id:9d} | ${actual_price:7.2f} | ${pred_price:9.2f} | ${error:7.2f} | {rel_error:7.1f}%")

def main():
    """Main execution function"""
    print("REBUILDING SMART PRODUCT PRICING MODEL FOR MAXIMUM ACCURACY")
    print("=" * 80)
    
    # Step 1: Analyze data
    train = analyze_data()
    
    # Step 2: Analyze catalog content
    train = analyze_catalog_content(train)
    
    # Step 3: Advanced feature engineering
    train = extract_advanced_features(train)
    
    # Step 4: Build optimized model
    models, feature_cols, predictions = build_optimized_model(train)
    
    # Step 5: Error analysis
    analyze_errors(train, predictions)
    
    print("\n" + "=" * 80)
    print("MODEL REBUILDING COMPLETE")
    print("=" * 80)
    
    return models, feature_cols

if __name__ == "__main__":
    models, features = main()