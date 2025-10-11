#!/usr/bin/env python3
"""
Maximum Accuracy Price Predictor - Final Version
Focus on extracting the strongest price signals from text
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

def extract_price_indicators(text):
    """Extract the most important price indicators from catalog content"""
    text = str(text).lower()
    features = {}
    
    # Size/Volume indicators (strong price predictors)
    # Extract all numbers with units
    volume_matches = re.findall(r'(\d+\.?\d*)\s*(fl oz|fluid ounce|ounce|oz|ml|liter|l)\b', text)
    weight_matches = re.findall(r'(\d+\.?\d*)\s*(pound|lb|kg|kilogram|gram|g)\b', text)
    dimension_matches = re.findall(r'(\d+\.?\d*)\s*(inch|in|cm|mm|foot|ft)\b', text)
    
    # Convert to standardized units and sum
    total_volume = 0
    for num, unit in volume_matches:
        multiplier = {'fl oz': 1, 'fluid ounce': 1, 'ounce': 1, 'oz': 1, 
                     'ml': 0.033814, 'liter': 33.814, 'l': 33.814}
        total_volume += float(num) * multiplier.get(unit, 1)
    
    total_weight = 0
    for num, unit in weight_matches:
        multiplier = {'pound': 16, 'lb': 16, 'kg': 35.274, 'kilogram': 35.274,
                     'gram': 0.035274, 'g': 0.035274}
        total_weight += float(num) * multiplier.get(unit, 1)
    
    total_dimension = 0
    for num, unit in dimension_matches:
        multiplier = {'inch': 1, 'in': 1, 'cm': 0.393701, 'mm': 0.0393701,
                     'foot': 12, 'ft': 12}
        total_dimension += float(num) * multiplier.get(unit, 1)
    
    features.update({
        'total_volume': total_volume,
        'total_weight': total_weight, 
        'total_dimension': total_dimension,
        'has_volume': 1 if total_volume > 0 else 0,
        'has_weight': 1 if total_weight > 0 else 0,
        'has_dimension': 1 if total_dimension > 0 else 0
    })
    
    # Pack/Count indicators (affects per-unit pricing)
    pack_patterns = [
        r'pack of (\d+)', r'(\d+) pack', r'(\d+) count', r'(\d+) ct\b',
        r'(\d+) pcs', r'(\d+) pieces', r'(\d+) units', r'(\d+) bottles',
        r'(\d+) cans', r'(\d+) bars', r'(\d+) bags'
    ]
    
    pack_count = 1
    for pattern in pack_patterns:
        matches = re.findall(pattern, text)
        if matches:
            pack_count = max(pack_count, max([int(m) for m in matches]))
    
    features['pack_count'] = pack_count
    features['is_multipack'] = 1 if pack_count > 1 else 0
    
    # Brand/Quality indicators
    premium_brands = [
        'apple', 'samsung', 'sony', 'lg', 'panasonic', 'canon', 'nikon',
        'bose', 'beats', 'jbl', 'nike', 'adidas', 'under armour',
        'kraft', 'nestle', 'coca cola', 'pepsi', 'kellogg', 'general mills',
        'procter & gamble', 'unilever', 'johnson & johnson', 'pfizer'
    ]
    
    amazon_brands = ['amazon basics', 'amazon', 'kindle', 'echo', 'fire tv']
    
    features['premium_brand'] = 1 if any(brand in text for brand in premium_brands) else 0
    features['amazon_brand'] = 1 if any(brand in text for brand in amazon_brands) else 0
    
    # Quality/Premium keywords
    premium_keywords = [
        'premium', 'luxury', 'deluxe', 'professional', 'pro', 'elite',
        'ultimate', 'supreme', 'platinum', 'gold', 'advanced', 'premium quality',
        'high quality', 'top quality', 'best quality', 'superior'
    ]
    
    budget_keywords = [
        'basic', 'standard', 'economy', 'budget', 'value', 'generic',
        'discount', 'cheap', 'affordable', 'low cost', 'economical'
    ]
    
    features['premium_keywords'] = sum(1 for kw in premium_keywords if kw in text)
    features['budget_keywords'] = sum(1 for kw in budget_keywords if kw in text)
    
    # Product category detection (different categories have different price ranges)
    categories = {
        'electronics': ['electronic', 'digital', 'smart', 'wireless', 'bluetooth', 'usb', 'hdmi', 'wifi'],
        'food_beverage': ['food', 'snack', 'drink', 'beverage', 'coffee', 'tea', 'juice', 'soda'],
        'health_beauty': ['vitamin', 'supplement', 'medicine', 'health', 'beauty', 'cosmetic', 'skincare'],
        'home_kitchen': ['kitchen', 'home', 'cleaning', 'laundry', 'bathroom', 'cookware', 'appliance'],
        'clothing': ['shirt', 'pants', 'dress', 'shoes', 'jacket', 'clothing', 'apparel', 'fashion'],
        'toys_games': ['toy', 'game', 'puzzle', 'doll', 'action figure', 'board game', 'video game'],
        'books_media': ['book', 'novel', 'magazine', 'dvd', 'cd', 'bluray', 'music', 'movie'],
        'tools_hardware': ['tool', 'hardware', 'screw', 'nail', 'hammer', 'wrench', 'drill']
    }
    
    for category, keywords in categories.items():
        features[f'cat_{category}'] = 1 if any(kw in text for kw in keywords) else 0
    
    # Text complexity and length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['bullet_points'] = text.count('bullet point')
    features['has_description'] = 1 if 'product description' in text else 0
    features['description_length'] = len(text.split('product description:')[-1]) if 'product description:' in text else 0
    
    # Special features for better price prediction
    features['price_per_unit'] = 0  # Will calculate later
    if features['pack_count'] > 1 and features['total_volume'] > 0:
        features['volume_per_pack'] = features['total_volume'] / features['pack_count']
    else:
        features['volume_per_pack'] = features['total_volume']
    
    # Extract any numerical values (could be specifications)
    all_numbers = re.findall(r'\d+\.?\d*', text)
    if all_numbers:
        numbers = [float(n) for n in all_numbers]
        features['max_number'] = max(numbers)
        features['avg_number'] = np.mean(numbers)
        features['number_count'] = len(numbers)
    else:
        features['max_number'] = 0
        features['avg_number'] = 0  
        features['number_count'] = 0
    
    return features

def create_price_bins(prices, n_bins=10):
    """Create price bins for stratified sampling"""
    return pd.cut(prices, bins=n_bins, labels=False)

def train_ultra_model():
    """Train the most accurate model possible"""
    print("=" * 80)
    print("TRAINING MAXIMUM ACCURACY PRICE PREDICTION MODEL")
    print("=" * 80)
    
    # Load data
    train_df = pd.read_csv('dataset/train.csv')
    print(f"Training data: {train_df.shape}")
    
    # Extract features for all training samples
    print("Extracting price indicators...")
    features_list = []
    for idx, content in enumerate(train_df['catalog_content']):
        if idx % 10000 == 0:
            print(f"Processing: {idx}/{len(train_df)}")
        features = extract_price_indicators(content)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    print(f"Created {features_df.shape[1]} engineered features")
    
    # Add image features
    features_df['has_image'] = (~train_df['image_link'].isna()).astype(int)
    features_df['amazon_image'] = train_df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
    
    # Generate focused text embeddings
    print("Generating text embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract key text for embedding (item name + key descriptors)
    focused_text = []
    for content in train_df['catalog_content']:
        content = str(content)
        # Get item name
        if 'Item Name:' in content:
            item_name = content.split('Item Name:')[1].split('\n')[0]
        else:
            item_name = content[:100]
        
        # Get first bullet point (often most important)
        if 'Bullet Point 1:' in content:
            bullet1 = content.split('Bullet Point 1:')[1].split('\n')[0]
        else:
            bullet1 = ''
        
        focused = f"{item_name} {bullet1}".strip()[:200]  # Limit to 200 chars
        focused_text.append(focused)
    
    embeddings = embedder.encode(focused_text, batch_size=64, show_progress_bar=True)
    embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    
    # Combine all features
    X = pd.concat([features_df, embedding_df], axis=1).fillna(0)
    y = train_df['price'].values
    
    print(f"Final feature matrix: {X.shape}")
    print(f"Price range: ${y.min():.2f} to ${y.max():.2f}")
    
    # Create price bins for stratified sampling
    price_bins = create_price_bins(y)
    
    # Use stratified k-fold to ensure balanced price distribution in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Optimized LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae', 
        'boosting_type': 'gbdt',
        'num_leaves': 512,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': 10,
        'verbosity': -1,
        'seed': 42,
        'force_col_wise': True
    }
    
    oof_predictions = np.zeros(len(y))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, price_bins)):
        print(f"\nTraining fold {fold + 1}/5...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train price range: ${y_train.min():.2f} to ${y_train.max():.2f}")
        print(f"Val price range: ${y_val.min():.2f} to ${y_val.max():.2f}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(500)
            ]
        )
        
        # Predict
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred
        models.append(model)
        
        # Metrics
        mae = mean_absolute_error(y_val, val_pred)
        smape = np.mean(200 * np.abs(y_val - val_pred) / (np.abs(y_val) + np.abs(val_pred)))
        
        print(f"Fold {fold + 1} - MAE: {mae:.4f}, SMAPE: {smape:.4f}%")
    
    # Overall performance
    overall_mae = mean_absolute_error(y, oof_predictions)
    overall_smape = np.mean(200 * np.abs(y - oof_predictions) / (np.abs(y) + np.abs(oof_predictions)))
    
    print(f"\n🎯 FINAL MODEL PERFORMANCE:")
    print(f"MAE: {overall_mae:.4f}")
    print(f"SMAPE: {overall_smape:.4f}%")
    
    # Save model
    model_data = {
        'models': models,
        'embedder': embedder,
        'feature_columns': X.columns.tolist()
    }
    
    with open('outputs/models/maximum_accuracy_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved!")
    
    # Test on sample data
    print("\n" + "=" * 60)
    print("TESTING ON SAMPLE DATA")
    print("=" * 60)
    
    sample_test = pd.read_csv('dataset/sample_test.csv')
    sample_predictions = predict_with_model(sample_test, model_data)
    
    # Save sample predictions
    sample_output = pd.DataFrame({
        'sample_id': sample_test['sample_id'],
        'price': sample_predictions
    })
    sample_output.to_csv('outputs/max_accuracy_sample_predictions.csv', index=False)
    
    # Compare with expected
    expected = pd.read_csv('dataset/sample_test_out.csv')
    comparison = sample_output.merge(expected, on='sample_id', suffixes=('_pred', '_actual'))
    
    sample_mae = mean_absolute_error(comparison['price_actual'], comparison['price_pred'])
    sample_smape = np.mean(200 * np.abs(comparison['price_actual'] - comparison['price_pred']) / 
                          (np.abs(comparison['price_actual']) + np.abs(comparison['price_pred'])))
    
    print(f"Sample Test Performance:")
    print(f"MAE: {sample_mae:.4f}")
    print(f"SMAPE: {sample_smape:.4f}%")
    
    # Show examples
    comparison['abs_error'] = np.abs(comparison['price_actual'] - comparison['price_pred'])
    comparison['rel_error'] = comparison['abs_error'] / comparison['price_actual'] * 100
    
    print(f"\nBest 5 predictions:")
    best = comparison.nsmallest(5, 'rel_error')
    for _, row in best.iterrows():
        print(f"ID: {int(row['sample_id']):6d} | Actual: ${row['price_actual']:6.2f} | Pred: ${row['price_pred']:6.2f} | Error: {row['rel_error']:5.1f}%")
    
    print(f"\nWorst 5 predictions:")
    worst = comparison.nlargest(5, 'rel_error')
    for _, row in worst.iterrows():
        print(f"ID: {int(row['sample_id']):6d} | Actual: ${row['price_actual']:6.2f} | Pred: ${row['price_pred']:6.2f} | Error: {row['rel_error']:5.1f}%")
    
    return model_data

def predict_with_model(test_df, model_data):
    """Generate predictions using trained model"""
    models = model_data['models']
    embedder = model_data['embedder'] 
    feature_columns = model_data['feature_columns']
    
    # Extract features
    features_list = []
    for content in test_df['catalog_content']:
        features = extract_price_indicators(content)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Add image features
    features_df['has_image'] = (~test_df['image_link'].isna()).astype(int)
    features_df['amazon_image'] = test_df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
    
    # Generate embeddings
    focused_text = []
    for content in test_df['catalog_content']:
        content = str(content)
        if 'Item Name:' in content:
            item_name = content.split('Item Name:')[1].split('\n')[0]
        else:
            item_name = content[:100]
        
        if 'Bullet Point 1:' in content:
            bullet1 = content.split('Bullet Point 1:')[1].split('\n')[0]
        else:
            bullet1 = ''
        
        focused = f"{item_name} {bullet1}".strip()[:200]
        focused_text.append(focused)
    
    embeddings = embedder.encode(focused_text, batch_size=64, show_progress_bar=True)
    embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    
    # Combine features
    X = pd.concat([features_df, embedding_df], axis=1).fillna(0)
    
    # Ensure same columns as training
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    
    # Average predictions from all models
    predictions = np.zeros(len(test_df))
    for model in models:
        pred = model.predict(X, num_iteration=model.best_iteration)
        predictions += pred
    
    predictions = predictions / len(models)
    predictions = np.maximum(predictions, 0.1)  # Ensure positive
    
    return predictions

if __name__ == "__main__":
    model_data = train_ultra_model()
    
    # Generate final test predictions
    print("\n" + "=" * 60)
    print("GENERATING FINAL TEST PREDICTIONS")
    print("=" * 60)
    
    test_df = pd.read_csv('dataset/test.csv')
    final_predictions = predict_with_model(test_df, model_data)
    
    final_output = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    final_output.to_csv('test_out_maximum_accuracy.csv', index=False)
    
    print(f"Final predictions saved!")
    print(f"Generated {len(final_predictions)} predictions")
    print(f"Price range: ${final_predictions.min():.2f} to ${final_predictions.max():.2f}")
    print(f"Mean price: ${final_predictions.mean():.2f}")
    print(f"Ready for competition submission!")