#!/usr/bin/env python3
"""
Ultra-High Accuracy Price Prediction Model
Combines advanced text embeddings with sophisticated feature engineering
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class UltraAccuratePricePredictor:
    def __init__(self):
        self.model = None
        self.embedder = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def extract_price_signals(self, catalog_content):
        """Extract strong price indicators from catalog content"""
        content = str(catalog_content).lower()
        
        features = {}
        
        # Volume/Size extraction with proper unit conversion
        volume_patterns = [
            (r'(\d+\.?\d*)\s*fl\s*oz', 1.0),  # fluid ounces
            (r'(\d+\.?\d*)\s*ounce', 1.0),    # ounces
            (r'(\d+\.?\d*)\s*oz', 1.0),       # oz
            (r'(\d+\.?\d*)\s*ml', 0.033814),  # ml to fl oz
            (r'(\d+\.?\d*)\s*liter', 33.814), # liter to fl oz
            (r'(\d+\.?\d*)\s*l\b', 33.814),   # l to fl oz
        ]
        
        total_volume = 0
        for pattern, multiplier in volume_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                total_volume += float(match) * multiplier
        
        features['total_volume_oz'] = total_volume
        
        # Weight extraction
        weight_patterns = [
            (r'(\d+\.?\d*)\s*lb', 16.0),      # pounds to oz
            (r'(\d+\.?\d*)\s*pound', 16.0),   # pounds to oz
            (r'(\d+\.?\d*)\s*kg', 35.274),    # kg to oz
            (r'(\d+\.?\d*)\s*gram', 0.035274), # gram to oz
            (r'(\d+\.?\d*)\s*g\b', 0.035274), # g to oz
        ]
        
        total_weight = 0
        for pattern, multiplier in weight_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                total_weight += float(match) * multiplier
                
        features['total_weight_oz'] = total_weight
        
        # Pack count extraction
        pack_patterns = [
            r'pack\s*of\s*(\d+)',
            r'(\d+)\s*pack',
            r'(\d+)\s*count',
            r'(\d+)\s*ct\b',
            r'(\d+)\s*pcs',
            r'(\d+)\s*pieces'
        ]
        
        pack_count = 1
        for pattern in pack_patterns:
            matches = re.findall(pattern, content)
            if matches:
                pack_count = max(pack_count, max([int(m) for m in matches]))
                
        features['pack_count'] = pack_count
        
        # Brand indicators (premium brands typically cost more)
        premium_brands = [
            'apple', 'samsung', 'sony', 'nike', 'adidas', 'canon', 'nikon',
            'hp', 'dell', 'microsoft', 'google', 'amazon basics', 'duracell',
            'energizer', 'coca cola', 'pepsi', 'kraft', 'nestle', 'unilever',
            'procter', 'johnson', 'kimberly', 'gillette', 'oral-b'
        ]
        
        features['premium_brand'] = any(brand in content for brand in premium_brands)
        
        # Premium keywords
        premium_words = [
            'premium', 'luxury', 'professional', 'deluxe', 'elite', 'ultimate',
            'supreme', 'executive', 'platinum', 'gold', 'pro', 'advanced',
            'organic', 'natural', 'gourmet', 'artisan', 'handcrafted'
        ]
        
        features['premium_word_count'] = sum(1 for word in premium_words if word in content)
        
        # Budget indicators
        budget_words = ['basic', 'standard', 'economy', 'budget', 'value', 'generic']
        features['budget_word_count'] = sum(1 for word in budget_words if word in content)
        
        # Category indicators (different categories have different price ranges)
        categories = {
            'electronics': ['electronic', 'digital', 'smart', 'wireless', 'bluetooth', 'usb'],
            'food': ['food', 'snack', 'drink', 'beverage', 'coffee', 'tea', 'chocolate'],
            'health': ['vitamin', 'supplement', 'medicine', 'health', 'wellness', 'organic'],
            'beauty': ['beauty', 'cosmetic', 'skincare', 'makeup', 'shampoo', 'lotion'],
            'home': ['home', 'kitchen', 'cleaning', 'laundry', 'bathroom', 'decor'],
            'clothing': ['shirt', 'pants', 'dress', 'shoes', 'jacket', 'clothing']
        }
        
        for category, keywords in categories.items():
            features[f'category_{category}'] = any(keyword in content for keyword in keywords)
            
        # Text complexity indicators
        features['word_count'] = len(content.split())
        features['char_count'] = len(content)
        features['bullet_points'] = content.count('bullet point')
        features['description_length'] = len(content.split('product description:')[-1]) if 'product description:' in content else 0
        
        # Price per unit calculation
        if features['pack_count'] > 1:
            if features['total_volume_oz'] > 0:
                features['volume_per_unit'] = features['total_volume_oz'] / features['pack_count']
            if features['total_weight_oz'] > 0:
                features['weight_per_unit'] = features['total_weight_oz'] / features['pack_count']
        else:
            features['volume_per_unit'] = features['total_volume_oz']
            features['weight_per_unit'] = features['total_weight_oz']
            
        return features
    
    def create_features(self, df):
        """Create comprehensive feature set from catalog content"""
        print("Creating advanced features...")
        
        # Extract price signals
        price_signals = []
        for idx, content in enumerate(df['catalog_content']):
            if idx % 10000 == 0:
                print(f"Processing price signals: {idx}/{len(df)}")
            signals = self.extract_price_signals(content)
            price_signals.append(signals)
        
        # Convert to DataFrame
        price_df = pd.DataFrame(price_signals)
        
        # Create text embeddings
        print("Generating text embeddings...")
        if self.embedder is None:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use only the product name and description for embeddings (more focused)
        text_for_embedding = []
        for content in df['catalog_content']:
            content = str(content)
            # Extract item name and product description
            if 'Item Name:' in content:
                item_name = content.split('Item Name:')[1].split('\n')[0] if 'Item Name:' in content else ''
            else:
                item_name = content[:100]  # First 100 chars
                
            if 'Product Description:' in content:
                description = content.split('Product Description:')[1][:500]  # First 500 chars of description
            else:
                description = ''
                
            combined_text = f"{item_name} {description}".strip()
            text_for_embedding.append(combined_text)
        
        embeddings = self.embedder.encode(text_for_embedding, batch_size=32, show_progress_bar=True)
        
        # Combine all features
        feature_df = pd.concat([
            df[['sample_id']].reset_index(drop=True),
            price_df.reset_index(drop=True),
            pd.DataFrame(embeddings, columns=[f'embed_{i}' for i in range(embeddings.shape[1])])
        ], axis=1)
        
        # Add image features
        feature_df['has_image'] = (~df['image_link'].isna()).astype(int)
        feature_df['amazon_image'] = df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
        
        print(f"Created {feature_df.shape[1]-1} features total")
        return feature_df
    
    def train(self, train_df):
        """Train the ultra-accurate price prediction model"""
        print("Training Ultra-Accurate Price Prediction Model...")
        
        # Create features
        feature_df = self.create_features(train_df)
        
        # Prepare training data
        self.feature_columns = [col for col in feature_df.columns if col != 'sample_id']
        X = feature_df[self.feature_columns].fillna(0)
        y = train_df['price'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Training with {X_scaled.shape[1]} features on {len(y)} samples")
        
        # Optimized parameters for price prediction
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 256,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': 8,
            'verbosity': -1,
            'seed': 42
        }
        
        # Cross-validation training
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(y))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            print(f"Training fold {fold + 1}/5...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=3000,
                callbacks=[
                    lgb.early_stopping(150),
                    lgb.log_evaluation(0)
                ]
            )
            
            # Predict on validation set
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = val_pred
            models.append(model)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, val_pred)
            smape = np.mean(200 * np.abs(y_val - val_pred) / (np.abs(y_val) + np.abs(val_pred)))
            
            print(f"Fold {fold + 1} - MAE: {mae:.4f}, SMAPE: {smape:.4f}%")
        
        # Overall performance
        overall_mae = mean_absolute_error(y, oof_predictions)
        overall_smape = np.mean(200 * np.abs(y - oof_predictions) / (np.abs(y) + np.abs(oof_predictions)))
        
        print(f"\nFINAL PERFORMANCE:")
        print(f"MAE: {overall_mae:.4f}")
        print(f"SMAPE: {overall_smape:.4f}%")
        
        self.models = models
        return oof_predictions
    
    def predict(self, test_df):
        """Generate predictions for test data"""
        print("Generating predictions...")
        
        # Create features
        feature_df = self.create_features(test_df)
        X = feature_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Average predictions from all fold models
        predictions = np.zeros(len(test_df))
        for model in self.models:
            fold_pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions += fold_pred
        
        predictions = predictions / len(self.models)
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.1)
        
        return predictions
    
    def save(self, path):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'embedder': self.embedder
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load a trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.embedder = model_data['embedder']
        print(f"Model loaded from {path}")

def main():
    """Main training and prediction pipeline"""
    print("=" * 80)
    print("ULTRA-ACCURATE SMART PRODUCT PRICING MODEL")
    print("=" * 80)
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('dataset/train.csv')
    print(f"Training data: {train_df.shape}")
    
    # Initialize and train model
    predictor = UltraAccuratePricePredictor()
    oof_predictions = predictor.train(train_df)
    
    # Save model
    predictor.save('outputs/models/ultra_accurate_model.pkl')
    
    # Test on sample data
    print("\nTesting on sample_test.csv...")
    sample_test = pd.read_csv('dataset/sample_test.csv')
    sample_predictions = predictor.predict(sample_test)
    
    # Create sample output
    sample_output = pd.DataFrame({
        'sample_id': sample_test['sample_id'],
        'price': sample_predictions
    })
    sample_output.to_csv('outputs/ultra_accurate_sample_predictions.csv', index=False)
    
    # Compare with expected output
    expected = pd.read_csv('dataset/sample_test_out.csv')
    comparison = sample_output.merge(expected, on='sample_id', suffixes=('_pred', '_actual'))
    
    mae = mean_absolute_error(comparison['price_actual'], comparison['price_pred'])
    smape = np.mean(200 * np.abs(comparison['price_actual'] - comparison['price_pred']) / 
                   (np.abs(comparison['price_actual']) + np.abs(comparison['price_pred'])))
    
    print(f"\nSample Test Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"SMAPE: {smape:.4f}%")
    
    # Show best and worst predictions
    comparison['abs_error'] = np.abs(comparison['price_actual'] - comparison['price_pred'])
    comparison['rel_error'] = comparison['abs_error'] / comparison['price_actual'] * 100
    
    print("\nBest 5 predictions:")
    best = comparison.nsmallest(5, 'rel_error')
    for _, row in best.iterrows():
        print(f"ID: {row['sample_id']:6d} | Actual: ${row['price_actual']:6.2f} | Pred: ${row['price_pred']:6.2f} | Error: {row['rel_error']:5.1f}%")
    
    print("\nWorst 5 predictions:")
    worst = comparison.nlargest(5, 'rel_error')
    for _, row in worst.iterrows():
        print(f"ID: {row['sample_id']:6d} | Actual: ${row['price_actual']:6.2f} | Pred: ${row['price_pred']:6.2f} | Error: {row['rel_error']:5.1f}%")
    
    # Generate final test predictions
    print("\nGenerating final test predictions...")
    test_df = pd.read_csv('dataset/test.csv')
    final_predictions = predictor.predict(test_df)
    
    final_output = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    final_output.to_csv('test_out_ultra_accurate.csv', index=False)
    
    print(f"Final predictions saved to test_out_ultra_accurate.csv")
    print(f"Generated {len(final_predictions)} predictions")
    print(f"Price range: ${final_predictions.min():.2f} to ${final_predictions.max():.2f}")
    print(f"Mean price: ${final_predictions.mean():.2f}")

if __name__ == "__main__":
    main()