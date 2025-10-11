#!/usr/bin/env python3
"""
Final Ultra-Precise Price Predictor
Focus on handling the extreme price ranges better
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_price_aware_features(df):
    """Create features specifically designed for accurate price prediction"""
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Processing: {idx}/{len(df)}")
            
        content = str(row['catalog_content']).lower()
        features = {}
        
        # Extract key price indicators with better precision
        
        # 1. Pack/quantity analysis (crucial for unit pricing)
        pack_patterns = [
            (r'pack of (\d+)', 1), (r'(\d+) pack', 1), (r'(\d+) count', 1),
            (r'(\d+) ct\b', 1), (r'(\d+) pcs', 1), (r'(\d+) pieces', 1),
            (r'(\d+) bottles', 1), (r'(\d+) cans', 1), (r'(\d+) bars', 1)
        ]
        
        pack_count = 1
        for pattern, multiplier in pack_patterns:
            matches = re.findall(pattern, content)
            if matches:
                pack_count = max(pack_count, max([int(m) * multiplier for m in matches]))
        
        features['pack_count'] = pack_count
        features['log_pack_count'] = np.log1p(pack_count)
        
        # 2. Size/Volume with standardized units
        volume_oz = 0
        volume_patterns = [
            (r'(\d+\.?\d*)\s*fl\s*oz', 1.0), (r'(\d+\.?\d*)\s*ounce', 1.0),
            (r'(\d+\.?\d*)\s*oz\b', 1.0), (r'(\d+\.?\d*)\s*ml', 0.033814),
            (r'(\d+\.?\d*)\s*liter', 33.814), (r'(\d+\.?\d*)\s*l\b', 33.814)
        ]
        
        for pattern, multiplier in volume_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                volume_oz += float(match) * multiplier
        
        features['total_volume_oz'] = volume_oz
        features['log_volume'] = np.log1p(volume_oz)
        features['volume_per_pack'] = volume_oz / pack_count if pack_count > 0 else 0
        
        # 3. Weight analysis
        weight_oz = 0
        weight_patterns = [
            (r'(\d+\.?\d*)\s*lb', 16.0), (r'(\d+\.?\d*)\s*pound', 16.0),
            (r'(\d+\.?\d*)\s*kg', 35.274), (r'(\d+\.?\d*)\s*gram', 0.035274),
            (r'(\d+\.?\d*)\s*g\b', 0.035274)
        ]
        
        for pattern, multiplier in weight_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                weight_oz += float(match) * multiplier
        
        features['total_weight_oz'] = weight_oz
        features['log_weight'] = np.log1p(weight_oz)
        features['weight_per_pack'] = weight_oz / pack_count if pack_count > 0 else 0
        
        # 4. Brand quality indicators
        premium_brands = ['apple', 'samsung', 'sony', 'nike', 'canon', 'hp', 'microsoft']
        budget_brands = ['generic', 'store brand', 'no name', 'basic']
        amazon_brands = ['amazon basics', 'amazon', 'kindle']
        
        features['premium_brand'] = 1 if any(brand in content for brand in premium_brands) else 0
        features['budget_brand'] = 1 if any(brand in content for brand in budget_brands) else 0
        features['amazon_brand'] = 1 if any(brand in content for brand in amazon_brands) else 0
        
        # 5. Quality keywords with weights
        premium_words = ['premium', 'luxury', 'professional', 'deluxe', 'elite', 'platinum', 'gold']
        budget_words = ['basic', 'standard', 'economy', 'budget', 'value', 'cheap']
        
        features['premium_score'] = sum(2 if word in content else 0 for word in premium_words)
        features['budget_score'] = sum(1 if word in content else 0 for word in budget_words)
        features['quality_ratio'] = features['premium_score'] / (features['budget_score'] + 1)
        
        # 6. Category-specific pricing (different categories have different price ranges)
        category_price_multipliers = {
            'electronics': ['electronic', 'digital', 'smart', 'wireless', 'bluetooth'],
            'food': ['food', 'snack', 'drink', 'beverage', 'coffee', 'tea'],
            'health': ['vitamin', 'supplement', 'medicine', 'health', 'organic'],
            'beauty': ['beauty', 'cosmetic', 'skincare', 'shampoo', 'lotion'],
            'home': ['kitchen', 'home', 'cleaning', 'laundry', 'appliance'],
            'books': ['book', 'novel', 'magazine', 'paperback', 'hardcover']
        }
        
        for category, keywords in category_price_multipliers.items():
            features[f'is_{category}'] = 1 if any(kw in content for kw in keywords) else 0
        
        # 7. Text complexity (correlates with product sophistication)
        features['char_count'] = len(content)
        features['word_count'] = len(content.split())
        features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1)
        features['bullet_points'] = content.count('bullet point')
        features['has_description'] = 1 if 'product description' in content else 0
        
        # 8. Special low-price indicators (to handle very cheap items)
        cheap_indicators = ['sample', 'trial', 'mini', 'travel size', 'single use', 'disposable']
        features['cheap_indicators'] = sum(1 for ind in cheap_indicators if ind in content)
        
        # 9. High-value indicators
        expensive_indicators = ['professional', 'commercial', 'industrial', 'enterprise', 'premium', 'luxury']
        features['expensive_indicators'] = sum(1 for ind in expensive_indicators if ind in content)
        
        # 10. Price range hints from content
        if any(word in content for word in ['under', 'less than', 'below', 'affordable', 'budget']):
            features['price_hint_low'] = 1
        else:
            features['price_hint_low'] = 0
            
        if any(word in content for word in ['premium', 'luxury', 'expensive', 'high-end', 'top-tier']):
            features['price_hint_high'] = 1  
        else:
            features['price_hint_high'] = 0
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def train_price_aware_model():
    """Train model with price-aware techniques"""
    print("=" * 80)
    print("TRAINING PRICE-AWARE ULTRA-ACCURATE MODEL")
    print("=" * 80)
    
    # Load data
    train_df = pd.read_csv('dataset/train.csv')
    print(f"Training data: {train_df.shape}")
    
    # Analyze price distribution
    prices = train_df['price']
    print(f"Price distribution:")
    print(f"  Min: ${prices.min():.2f}, Max: ${prices.max():.2f}")
    print(f"  Mean: ${prices.mean():.2f}, Median: ${prices.median():.2f}")
    
    # Create price-aware features
    print("Creating price-aware features...")
    features_df = create_price_aware_features(train_df)
    
    # Add image features
    features_df['has_image'] = (~train_df['image_link'].isna()).astype(int)
    features_df['amazon_image'] = train_df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
    
    # Create focused embeddings (shorter, more focused text)
    print("Creating focused text embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract just the product name for embedding (most price-relevant info)
    product_names = []
    for content in train_df['catalog_content']:
        content = str(content)
        if 'Item Name:' in content:
            name = content.split('Item Name:')[1].split('\n')[0].strip()
        else:
            name = content.split('\n')[0][:100]  # First line, first 100 chars
        product_names.append(name)
    
    embeddings = embedder.encode(product_names, batch_size=128, show_progress_bar=True)
    
    # Use only first 128 embedding dimensions to avoid overfitting
    embedding_df = pd.DataFrame(embeddings[:, :128], columns=[f'emb_{i}' for i in range(128)])
    
    # Combine features
    X = pd.concat([features_df, embedding_df], axis=1).fillna(0)
    y = train_df['price'].values
    
    print(f"Final features shape: {X.shape}")
    
    # Price-aware model parameters (optimized for handling wide price ranges)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 256,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 30,  # Higher to avoid overfitting on outliers
        'min_child_weight': 0.01,
        'reg_alpha': 0.3,  # Higher regularization
        'reg_lambda': 0.3,
        'max_depth': 7,  # Slightly lower depth
        'verbosity': -1,
        'seed': 42
    }
    
    # Cross-validation with special focus on low-price accuracy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/5...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create sample weights (give more importance to getting low prices right)
        sample_weights = np.ones(len(y_train))
        # Increase weight for items under $5 (these are often predicted poorly)
        low_price_mask = y_train < 5
        sample_weights[low_price_mask] *= 2.0
        
        # Very low prices get even more weight
        very_low_mask = y_train < 2
        sample_weights[very_low_mask] *= 3.0
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
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
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        # Post-process predictions to handle very low prices better
        val_pred = np.maximum(val_pred, 0.1)  # Minimum price
        
        oof_predictions[val_idx] = val_pred
        models.append(model)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, val_pred)
        smape = np.mean(200 * np.abs(y_val - val_pred) / (np.abs(y_val) + np.abs(val_pred)))
        
        # Special metric for low-price accuracy
        low_price_mask_val = y_val < 5
        if low_price_mask_val.sum() > 0:
            low_price_mae = mean_absolute_error(y_val[low_price_mask_val], val_pred[low_price_mask_val])
            low_price_smape = np.mean(200 * np.abs(y_val[low_price_mask_val] - val_pred[low_price_mask_val]) / 
                                    (np.abs(y_val[low_price_mask_val]) + np.abs(val_pred[low_price_mask_val])))
            print(f"Fold {fold + 1} - Overall MAE: {mae:.4f}, SMAPE: {smape:.4f}%")
            print(f"           Low-price MAE: {low_price_mae:.4f}, SMAPE: {low_price_smape:.4f}%")
        else:
            print(f"Fold {fold + 1} - MAE: {mae:.4f}, SMAPE: {smape:.4f}%")
    
    # Overall performance
    overall_mae = mean_absolute_error(y, oof_predictions)
    overall_smape = np.mean(200 * np.abs(y - oof_predictions) / (np.abs(y) + np.abs(oof_predictions)))
    
    # Low-price performance
    low_mask = y < 5
    if low_mask.sum() > 0:
        low_mae = mean_absolute_error(y[low_mask], oof_predictions[low_mask])
        low_smape = np.mean(200 * np.abs(y[low_mask] - oof_predictions[low_mask]) / 
                           (np.abs(y[low_mask]) + np.abs(oof_predictions[low_mask])))
        
        print(f"\n🎯 FINAL PERFORMANCE:")
        print(f"Overall - MAE: {overall_mae:.4f}, SMAPE: {overall_smape:.4f}%")
        print(f"Low-price (<$5) - MAE: {low_mae:.4f}, SMAPE: {low_smape:.4f}%")
    else:
        print(f"\n🎯 FINAL PERFORMANCE:")
        print(f"MAE: {overall_mae:.4f}, SMAPE: {overall_smape:.4f}%")
    
    # Save model
    model_data = {
        'models': models,
        'embedder': embedder,
        'feature_columns': X.columns.tolist()
    }
    
    with open('outputs/models/price_aware_final_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def predict_with_price_aware_model(test_df, model_data):
    """Generate predictions with price-aware post-processing"""
    models = model_data['models']
    embedder = model_data['embedder']
    feature_columns = model_data['feature_columns']
    
    # Create features
    features_df = create_price_aware_features(test_df)
    features_df['has_image'] = (~test_df['image_link'].isna()).astype(int)
    features_df['amazon_image'] = test_df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
    
    # Create embeddings
    product_names = []
    for content in test_df['catalog_content']:
        content = str(content)
        if 'Item Name:' in content:
            name = content.split('Item Name:')[1].split('\n')[0].strip()
        else:
            name = content.split('\n')[0][:100]
        product_names.append(name)
    
    embeddings = embedder.encode(product_names, batch_size=128, show_progress_bar=True)
    embedding_df = pd.DataFrame(embeddings[:, :128], columns=[f'emb_{i}' for i in range(128)])
    
    X = pd.concat([features_df, embedding_df], axis=1).fillna(0)
    
    # Ensure same columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    
    # Average predictions
    predictions = np.zeros(len(test_df))
    for model in models:
        pred = model.predict(X, num_iteration=model.best_iteration)
        predictions += pred
    
    predictions = predictions / len(models)
    
    # Smart post-processing based on content features
    # If item has cheap indicators, cap the prediction
    cheap_mask = features_df['cheap_indicators'] > 0
    predictions[cheap_mask] = np.minimum(predictions[cheap_mask], 10.0)
    
    # If very small pack/volume, likely cheap
    small_mask = (features_df['pack_count'] == 1) & (features_df['total_volume_oz'] < 2)
    predictions[small_mask] = np.minimum(predictions[small_mask], 15.0)
    
    # Ensure reasonable minimum
    predictions = np.maximum(predictions, 0.1)
    
    return predictions

if __name__ == "__main__":
    # Train the model
    model_data = train_price_aware_model()
    
    # Test on sample
    print("\n" + "=" * 60)
    print("TESTING ON SAMPLE DATA")
    print("=" * 60)
    
    sample_test = pd.read_csv('dataset/sample_test.csv')
    sample_predictions = predict_with_price_aware_model(sample_test, model_data)
    
    sample_output = pd.DataFrame({
        'sample_id': sample_test['sample_id'],
        'price': sample_predictions
    })
    sample_output.to_csv('outputs/price_aware_sample_predictions.csv', index=False)
    
    # Compare with expected
    expected = pd.read_csv('dataset/sample_test_out.csv')
    comparison = sample_output.merge(expected, on='sample_id', suffixes=('_pred', '_actual'))
    
    mae = mean_absolute_error(comparison['price_actual'], comparison['price_pred'])
    smape = np.mean(200 * np.abs(comparison['price_actual'] - comparison['price_pred']) / 
                   (np.abs(comparison['price_actual']) + np.abs(comparison['price_pred'])))
    
    print(f"Sample Test Performance:")
    print(f"MAE: {mae:.4f}, SMAPE: {smape:.4f}%")
    
    # Generate final predictions
    print("\n" + "=" * 60)
    print("GENERATING FINAL PREDICTIONS")
    print("=" * 60)
    
    test_df = pd.read_csv('dataset/test.csv')
    final_predictions = predict_with_price_aware_model(test_df, model_data)
    
    final_output = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    final_output.to_csv('test_out_price_aware_final.csv', index=False)
    
    print(f"🎯 FINAL SUBMISSION READY!")
    print(f"Generated {len(final_predictions)} predictions")
    print(f"Price range: ${final_predictions.min():.2f} to ${final_predictions.max():.2f}")
    print(f"Mean price: ${final_predictions.mean():.2f}")
    print(f"File saved: test_out_price_aware_final.csv")