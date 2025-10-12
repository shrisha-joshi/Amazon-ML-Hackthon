#!/usr/bin/env python3
"""
Ultra-Precision Model for Full Test Dataset (75,000 samples)
Optimized version of eval_final.py for large-scale processing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gc
from datetime import datetime

class LargeScaleUltraPrecisionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_cache = {}
        
    def extract_optimized_features(self, df, batch_size=5000):
        """Extract features in batches for memory efficiency"""
        print(f"🔧 Extracting features for {len(df):,} samples...")
        
        all_features = []
        
        # Process in batches to manage memory
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].copy()
            batch_features = self._extract_batch_features(batch)
            all_features.append(batch_features)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {min(i+batch_size, len(df)):,}/{len(df):,} samples...")
                gc.collect()  # Force garbage collection
        
        # Combine all batches
        final_features = pd.concat(all_features, ignore_index=True)
        print(f"✅ Feature extraction complete: {final_features.shape[1]} features")
        return final_features
    
    def _extract_batch_features(self, batch):
        """Extract features for a single batch"""
        features = pd.DataFrame(index=batch.index)
        
        # Basic text features
        features['text_length'] = batch['catalog_content'].str.len().fillna(0)
        features['word_count'] = batch['catalog_content'].str.split().str.len().fillna(0)
        features['sentence_count'] = batch['catalog_content'].str.count(r'[.!?]').fillna(0)
        
        # Price indicators from text
        features['has_price_mention'] = batch['catalog_content'].str.contains(r'\$|\bprice\b|\bcost\b', case=False, na=False).astype(int)
        features['has_discount'] = batch['catalog_content'].str.contains(r'discount|sale|off|deal', case=False, na=False).astype(int)
        features['has_premium'] = batch['catalog_content'].str.contains(r'premium|luxury|high.?quality|professional', case=False, na=False).astype(int)
        features['has_budget'] = batch['catalog_content'].str.contains(r'budget|cheap|affordable|economy', case=False, na=False).astype(int)
        
        # Brand/quality indicators
        features['has_brand'] = batch['catalog_content'].str.contains(r'\b[A-Z][a-z]+\s*[A-Z][a-z]+\b', na=False).astype(int)
        features['caps_ratio'] = batch['catalog_content'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
        
        # Product category hints
        features['is_electronics'] = batch['catalog_content'].str.contains(r'electronic|digital|tech|device|gadget', case=False, na=False).astype(int)
        features['is_clothing'] = batch['catalog_content'].str.contains(r'shirt|dress|clothing|apparel|wear', case=False, na=False).astype(int)
        features['is_home'] = batch['catalog_content'].str.contains(r'home|kitchen|furniture|decor', case=False, na=False).astype(int)
        features['is_books'] = batch['catalog_content'].str.contains(r'book|novel|guide|manual', case=False, na=False).astype(int)
        
        # Image link features
        features['image_id_numeric'] = batch['image_link'].str.extract(r'/([0-9]+)').astype(float).fillna(0)
        features['has_multiple_images'] = batch['image_link'].str.contains(r'[,;]', na=False).astype(int)
        
        # Sample ID patterns (learned from training)
        features['sample_id_mod_100'] = batch['sample_id'] % 100
        features['sample_id_mod_1000'] = batch['sample_id'] % 1000
        features['sample_id_log'] = np.log1p(batch['sample_id'])
        
        return features
    
    def create_price_clusters(self, train_df):
        """Create price-based clusters for specialized models"""
        print("🎯 Creating price-based clusters...")
        
        # Define price ranges based on training data distribution
        price_percentiles = train_df['price'].quantile([0.0, 0.33, 0.66, 1.0]).values
        
        def get_price_cluster(price):
            if price <= price_percentiles[1]:
                return 'budget'
            elif price <= price_percentiles[2]:
                return 'mid_tier'
            else:
                return 'premium'
        
        train_df['price_cluster'] = train_df['price'].apply(get_price_cluster)
        return price_percentiles
    
    def train_models(self):
        """Train the ultra-precision model ensemble"""
        print("🚀 Training Ultra-Precision Model Ensemble...")
        
        # Load training data
        train_df = pd.read_csv('student_resource/dataset/train.csv')
        print(f"📊 Training data: {len(train_df):,} samples")
        
        # Extract features for training
        train_features = self.extract_optimized_features(train_df)
        
        # Create price clusters
        price_percentiles = self.create_price_clusters(train_df)
        self.price_percentiles = price_percentiles
        
        # Prepare target
        y = train_df['price'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(train_features)
        
        # Train multiple models for ensemble
        print("🎯 Training ensemble models...")
        
        # Primary model - RandomForest (best for our use case)
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Secondary models for ensemble
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['elastic'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        
        # Train all models
        for name, model in self.models.items():
            print(f"   Training {name}...")
            model.fit(X_scaled, y)
        
        # Train cluster-specific models
        self._train_cluster_models(train_df, train_features, X_scaled)
        
        print("✅ Model training complete!")
        
    def _train_cluster_models(self, train_df, train_features, X_scaled):
        """Train specialized models for each price cluster"""
        print("🎯 Training cluster-specific models...")
        
        self.cluster_models = {}
        
        for cluster in ['budget', 'mid_tier', 'premium']:
            cluster_mask = train_df['price_cluster'] == cluster
            if cluster_mask.sum() > 100:  # Minimum samples for training
                X_cluster = X_scaled[cluster_mask]
                y_cluster = train_df.loc[cluster_mask, 'price'].values
                
                self.cluster_models[cluster] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
                self.cluster_models[cluster].fit(X_cluster, y_cluster)
                print(f"   {cluster}: {len(y_cluster):,} samples")
    
    def predict_large_scale(self, test_df, batch_size=2000):
        """Predict prices for large test dataset in batches"""
        print(f"🔮 Generating predictions for {len(test_df):,} samples...")
        
        all_predictions = []
        
        # Process in smaller batches for memory efficiency
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i+batch_size].copy()
            
            # Extract features for this batch
            batch_features = self.extract_optimized_features(batch, batch_size=len(batch))
            
            # Scale features
            X_scaled = self.scaler.transform(batch_features)
            
            # Get ensemble predictions
            ensemble_preds = []
            
            # Primary models
            for name, model in self.models.items():
                pred = model.predict(X_scaled)
                weight = 0.4 if name == 'rf' else 0.2  # RF gets higher weight
                ensemble_preds.append(pred * weight)
            
            # Combine ensemble
            base_pred = np.sum(ensemble_preds, axis=0)
            
            # Apply cluster-specific corrections
            final_pred = self._apply_cluster_corrections(batch, base_pred)
            
            all_predictions.extend(final_pred)
            
            # Progress update
            processed = min(i + batch_size, len(test_df))
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {processed:,}/{len(test_df):,} samples ({100*processed/len(test_df):.1f}%)")
            
            gc.collect()  # Memory management
        
        print("✅ Predictions complete!")
        return np.array(all_predictions)
    
    def _apply_cluster_corrections(self, batch, base_pred):
        """Apply cluster-specific model corrections"""
        corrected_pred = base_pred.copy()
        
        # Estimate price clusters for test samples
        for i, pred in enumerate(base_pred):
            if pred <= self.price_percentiles[1]:
                cluster = 'budget'
            elif pred <= self.price_percentiles[2]:
                cluster = 'mid_tier'
            else:
                cluster = 'premium'
            
            # Apply cluster model if available
            if cluster in self.cluster_models:
                # Re-extract features for single sample (simplified)
                sample_features = self.extract_optimized_features(
                    batch.iloc[i:i+1], batch_size=1
                )
                X_sample = self.scaler.transform(sample_features)
                cluster_pred = self.cluster_models[cluster].predict(X_sample)[0]
                
                # Blend with base prediction
                corrected_pred[i] = 0.7 * base_pred[i] + 0.3 * cluster_pred
        
        return corrected_pred

def run_large_scale_prediction():
    """Main function to run predictions on full test dataset"""
    print("🚀 ULTRA-PRECISION MODEL - LARGE SCALE PREDICTION")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize predictor
    predictor = LargeScaleUltraPrecisionPredictor()
    
    # Train models
    predictor.train_models()
    
    # Load test data
    print("\n📥 Loading main test dataset...")
    test_df = pd.read_csv('student_resource/dataset/test.csv')
    print(f"✅ Loaded {len(test_df):,} test samples")
    
    # Generate predictions
    print("\n🔮 Running predictions...")
    predictions = predictor.predict_large_scale(test_df)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': predictions
    })
    
    # Save results
    output_path = 'outputs/test_predictions_final.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"📊 Generated {len(output_df):,} predictions")
    print(f"💰 Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"📈 Average price: ${predictions.mean():.2f}")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("🎉 LARGE SCALE PREDICTION COMPLETE!")
    
    return output_df

if __name__ == "__main__":
    result_df = run_large_scale_prediction()