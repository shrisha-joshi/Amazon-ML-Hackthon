#!/usr/bin/env python3
"""
Deep Analysis: Why 1.12% SMAPE became 71.976% SMAPE?
Innovative approach to achieve <30% SMAPE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import Counter

def deep_analysis():
    """Analyze the massive performance gap and identify solutions"""
    print("🔬 DEEP ANALYSIS: Model Performance Investigation")
    print("=" * 70)
    
    # Load data
    test_out = pd.read_csv('test_out.csv')
    sample_out = pd.read_csv('student_resource/dataset/sample_test_out.csv')
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    
    print("\n📊 DISTRIBUTION MISMATCH ANALYSIS:")
    print("-" * 40)
    
    # Training data analysis
    print(f"Training data price distribution:")
    print(f"   Range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    print(f"   Mean: ${train_df['price'].mean():.2f}")
    print(f"   Median: ${train_df['price'].median():.2f}")
    print(f"   Std: ${train_df['price'].std():.2f}")
    
    # Sample validation 
    print(f"\nSample validation (100 samples):")
    print(f"   Range: ${sample_out['price'].min():.2f} - ${sample_out['price'].max():.2f}")
    print(f"   Mean: ${sample_out['price'].mean():.2f}")
    print(f"   Median: ${sample_out['price'].median():.2f}")
    
    # Our predictions
    print(f"\nOur 75K predictions:")
    print(f"   Range: ${test_out['price'].min():.2f} - ${test_out['price'].max():.2f}")
    print(f"   Mean: ${test_out['price'].mean():.2f}")
    print(f"   Median: ${test_out['price'].median():.2f}")
    
    # Price distribution analysis
    print(f"\n🎯 PRICE DISTRIBUTION INSIGHTS:")
    print("-" * 40)
    
    # Define price tiers
    tiers = [
        (0, 15, "Budget"),
        (15, 35, "Mid-Range"), 
        (35, 75, "Premium"),
        (75, 200, "Luxury"),
        (200, 1000, "Ultra-Premium")
    ]
    
    print("Training data distribution:")
    for low, high, name in tiers:
        count = ((train_df['price'] >= low) & (train_df['price'] < high)).sum()
        pct = 100 * count / len(train_df)
        print(f"   {name} (${low}-{high}): {count:,} samples ({pct:.1f}%)")
    
    print("\nOur prediction distribution:")
    for low, high, name in tiers:
        count = ((test_out['price'] >= low) & (test_out['price'] < high)).sum()
        pct = 100 * count / len(test_out)
        print(f"   {name} (${low}-{high}): {count:,} samples ({pct:.1f}%)")
    
    return analyze_text_patterns(train_df)

def analyze_text_patterns(train_df):
    """Deep text analysis to find price-predictive patterns"""
    print(f"\n🔍 TEXT PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Extract key patterns from training data
    patterns = {
        'pack_quantities': [],
        'sizes': [],
        'brands': [],
        'materials': [],
        'quality_indicators': []
    }
    
    # Analyze sample of training data
    sample_size = min(1000, len(train_df))
    sample_df = train_df.sample(sample_size, random_state=42)
    
    pack_pattern = re.compile(r'(\d+)\s*(pack|ct|count|pcs|pieces)', re.I)
    size_pattern = re.compile(r'(\d+\.?\d*)\s*(oz|ml|l|liter|gram|g|kg|lb)', re.I) 
    
    for _, row in sample_df.iterrows():
        text = str(row['catalog_content']).lower()
        
        # Find pack quantities
        pack_matches = pack_pattern.findall(text)
        if pack_matches:
            patterns['pack_quantities'].extend([int(m[0]) for m in pack_matches])
        
        # Find sizes
        size_matches = size_pattern.findall(text)  
        if size_matches:
            patterns['sizes'].extend([float(m[0]) for m in size_matches])
    
    # Analyze patterns
    if patterns['pack_quantities']:
        pack_counts = Counter(patterns['pack_quantities'])
        print("Top pack quantities in training data:")
        for pack, count in pack_counts.most_common(5):
            print(f"   {pack} pack: {count} occurrences")
    
    if patterns['sizes']:
        print(f"Size range: {min(patterns['sizes']):.1f} - {max(patterns['sizes']):.1f}")
        
    return identify_key_improvements()

def identify_key_improvements():
    """Identify specific improvements for <30% SMAPE"""
    print(f"\n🚀 KEY IMPROVEMENTS FOR <30% SMAPE:")
    print("-" * 50)
    
    improvements = [
        "1. PRICE-TIER SPECIALIZED MODELS",
        "   → Budget model: Focus on pack quantities, bulk indicators",
        "   → Premium model: Focus on brand, material, quality words",
        "   → Luxury model: Focus on exclusivity, craftsmanship terms",
        "",
        "2. ADVANCED TEXT INTELLIGENCE", 
        "   → Extract numerical features (size, quantity, weight)",
        "   → Brand classification with tier mapping",
        "   → Material/quality detection (premium, deluxe, pro)",
        "",
        "3. SYSTEMATIC BIAS CORRECTION",
        "   → Price range calibration using isotonic regression", 
        "   → Distribution alignment with training data",
        "   → Prediction smoothing and outlier handling",
        "",
        "4. ENSEMBLE WITH CROSS-VALIDATION",
        "   → Multiple models with different random seeds",
        "   → Bayesian model averaging", 
        "   → Uncertainty quantification"
    ]
    
    for improvement in improvements:
        print(improvement)
    
    print(f"\n💡 INNOVATION HYPOTHESIS:")
    print("Current model treats all products the same way.")
    print("SOLUTION: Different algorithms for different product categories!")
    print("Expected improvement: 71.976% → <30% SMAPE")
    
    return True

if __name__ == "__main__":
    deep_analysis()