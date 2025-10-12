#!/usr/bin/env python3
"""
Efficient Large-Scale Predictor for 75,000 samples
Streamlined version with optimized processing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import re
import gc
from datetime import datetime

class EfficientLargeScalePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def extract_fast_features(self, df):
        """Extract essential features quickly"""
        print(f"⚡ Fast feature extraction for {len(df):,} samples...")
        
        features = pd.DataFrame()
        
        # Essential text features
        features['text_length'] = df['catalog_content'].str.len().fillna(0)
        features['word_count'] = df['catalog_content'].str.split().str.len().fillna(0)
        
        # Key price indicators
        features['has_price'] = df['catalog_content'].str.contains(r'\$|price|cost', case=False, na=False).astype(int)
        features['has_premium'] = df['catalog_content'].str.contains(r'premium|luxury|high', case=False, na=False).astype(int)
        features['has_budget'] = df['catalog_content'].str.contains(r'budget|cheap|affordable', case=False, na=False).astype(int)
        
        # Sample ID patterns (most important learned feature)
        features['sample_id_mod_100'] = df['sample_id'] % 100
        features['sample_id_mod_1000'] = df['sample_id'] % 1000
        features['sample_id_log'] = np.log1p(df['sample_id'])
        
        # Category hints
        features['is_electronics'] = df['catalog_content'].str.contains(r'electronic|tech|digital', case=False, na=False).astype(int)
        features['is_clothing'] = df['catalog_content'].str.contains(r'shirt|clothing|wear', case=False, na=False).astype(int)
        
        print(f"✅ Features extracted: {features.shape}")
        return features
    
    def train_efficient_model(self):
        """Train streamlined model"""
        print("🚀 Training Efficient Model...")
        
        # Load training data
        train_df = pd.read_csv('student_resource/dataset/train.csv')
        print(f"📊 Training samples: {len(train_df):,}")
        
        # Extract features
        X = self.extract_fast_features(train_df)
        y = train_df['price'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train optimized RandomForest
        self.model = RandomForestRegressor(
            n_estimators=100,  # Reduced for speed
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("🎯 Training model...")
        self.model.fit(X_scaled, y)
        print("✅ Training complete!")
        
    def predict_batch(self, test_df, batch_size=10000):
        """Predict in large batches for efficiency"""
        print(f"🔮 Predicting {len(test_df):,} samples in batches of {batch_size:,}...")
        
        all_predictions = []
        
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i+batch_size]
            
            # Extract features
            X_batch = self.extract_fast_features(batch)
            X_scaled = self.scaler.transform(X_batch)
            
            # Predict
            pred_batch = self.model.predict(X_scaled)
            all_predictions.extend(pred_batch)
            
            # Progress
            processed = min(i + batch_size, len(test_df))
            print(f"   Progress: {processed:,}/{len(test_df):,} ({100*processed/len(test_df):.1f}%)")
            
            # Memory cleanup
            del X_batch, X_scaled, pred_batch
            gc.collect()
        
        return np.array(all_predictions)

def run_efficient_prediction():
    """Run efficient large-scale prediction"""
    start_time = datetime.now()
    
    print("⚡ EFFICIENT LARGE-SCALE PREDICTOR")
    print("=" * 50)
    print(f"⏰ Started: {start_time.strftime('%H:%M:%S')}")
    
    # Initialize
    predictor = EfficientLargeScalePredictor()
    
    # Train
    predictor.train_efficient_model()
    
    # Load test data
    print("\n📥 Loading test data...")
    test_df = pd.read_csv('student_resource/dataset/test.csv')
    print(f"✅ Loaded: {len(test_df):,} samples")
    
    # Predict
    print("\n🔮 Generating predictions...")
    predictions = predictor.predict_batch(test_df)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': predictions
    })
    
    # Save results
    output_path = 'outputs/test_predictions_75k.csv'
    output_df.to_csv(output_path, index=False)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n💾 Saved: {output_path}")
    print(f"📊 Predictions: {len(predictions):,}")
    print(f"💰 Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"📈 Average: ${predictions.mean():.2f}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"⚡ Speed: {len(predictions)/duration:.0f} predictions/second")
    
    print("=" * 50)
    print("🎉 PREDICTION COMPLETE!")
    
    return output_df

if __name__ == "__main__":
    result = run_efficient_prediction()