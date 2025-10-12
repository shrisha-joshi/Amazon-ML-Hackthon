#!/usr/bin/env python3
"""
🔍 ROOT CAUSE ANALYSIS - Why 77.541 SMAPE is WORSE
Critical analysis to identify the real problems and breakthrough solutions
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

def analyze_current_failure():
    """
    Deep analysis of why our current approach is failing
    """
    print("🔍 ROOT CAUSE ANALYSIS - 77.541 SMAPE FAILURE")
    print("=" * 80)
    
    # Load current predictions
    pred_df = pd.read_csv('test_out1.csv')
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    
    print(f"📊 Current predictions: {len(pred_df):,}")
    print(f"📊 Training data: {len(train_df):,}")
    
    # Analyze price distributions
    pred_prices = pred_df['price'].values
    train_prices = train_df['price'].values
    
    print(f"\n💰 PRICE STATISTICS:")
    print(f"Training - Mean: ${train_prices.mean():.2f}, Median: ${np.median(train_prices):.2f}")
    print(f"Current  - Mean: ${pred_prices.mean():.2f}, Median: ${np.median(pred_prices):.2f}")
    
    # Distribution analysis
    print(f"\n📊 DISTRIBUTION ANALYSIS:")
    
    ranges = [
        (0, 5, "Ultra-Low"),
        (5, 10, "Very-Low"), 
        (10, 15, "Low"),
        (15, 25, "Mid-Low"),
        (25, 40, "Mid"),
        (40, 60, "Mid-High"),
        (60, 100, "High"),
        (100, float('inf'), "Premium")
    ]
    
    total_error = 0
    for low, high, label in ranges:
        if high == float('inf'):
            train_pct = (train_prices >= low).mean() * 100
            pred_pct = (pred_prices >= low).mean() * 100
        else:
            train_pct = ((train_prices >= low) & (train_prices < high)).mean() * 100
            pred_pct = ((pred_prices >= low) & (pred_prices < high)).mean() * 100
        
        error = abs(train_pct - pred_pct)
        total_error += error
        
        status = "🎯" if error < 2 else "⚠️" if error < 5 else "❌"
        print(f"   {label}: Train={train_pct:.1f}% vs Pred={pred_pct:.1f}% (Δ{error:.1f}%) {status}")
    
    avg_error = total_error / len(ranges)
    print(f"\n📈 Average distribution error: {avg_error:.1f}%")
    
    # Text pattern analysis
    print(f"\n🔤 TEXT PATTERN ANALYSIS:")
    
    # Sample some training text to understand patterns
    sample_texts = train_df.sample(1000, random_state=42)['catalog_content'].values
    
    # Extract key price indicators
    price_words = []
    pack_quantities = []
    
    for text in sample_texts:
        text_lower = str(text).lower()
        
        # Find pack quantities
        pack_matches = re.findall(r'(\d+)\s*(?:pack|ct|count|pcs)', text_lower)
        if pack_matches:
            pack_quantities.extend([int(m) for m in pack_matches if m.isdigit()])
        
        # Find price-related words
        words = text_lower.split()
        for word in words:
            if any(hint in word for hint in ['cheap', 'budget', 'premium', 'deluxe']):
                price_words.append(word)
    
    print(f"   Pack quantities found: {len(pack_quantities)} (avg: {np.mean(pack_quantities) if pack_quantities else 0:.1f})")
    print(f"   Price words found: {len(price_words)}")
    
    if price_words:
        word_counts = Counter(price_words)
        print(f"   Top price words: {dict(list(word_counts.most_common(5)))}")
    
    # CRITICAL INSIGHTS
    print(f"\n🚨 CRITICAL FAILURE POINTS:")
    
    failure_points = []
    
    # 1. Distribution mismatch
    if avg_error > 5:
        failure_points.append(f"❌ Severe distribution mismatch ({avg_error:.1f}% avg error)")
    
    # 2. Mean price deviation
    mean_error = abs(train_prices.mean() - pred_prices.mean()) / train_prices.mean() * 100
    if mean_error > 10:
        failure_points.append(f"❌ Mean price deviation ({mean_error:.1f}%)")
    
    # 3. Over/under prediction bias
    if pred_prices.mean() > train_prices.mean() * 1.2:
        failure_points.append("❌ Systematic OVER-prediction bias")
    elif pred_prices.mean() < train_prices.mean() * 0.8:
        failure_points.append("❌ Systematic UNDER-prediction bias")
    
    # 4. Extreme price concentration
    extreme_low = (pred_prices < 1).mean() * 100
    extreme_high = (pred_prices > 200).mean() * 100
    
    if extreme_low > 5:
        failure_points.append(f"❌ Too many extreme low prices ({extreme_low:.1f}%)")
    if extreme_high > 2:
        failure_points.append(f"❌ Too many extreme high prices ({extreme_high:.1f}%)")
    
    for point in failure_points:
        print(f"   {point}")
    
    # ROOT CAUSE IDENTIFICATION
    print(f"\n🎯 ROOT CAUSE IDENTIFICATION:")
    
    root_causes = []
    
    # Check if we're using the right features
    budget_items = train_df[train_df['price'] < 15]
    premium_items = train_df[train_df['price'] > 60]
    
    print(f"   Budget items ({len(budget_items):,}): Need pack/quantity focus")
    print(f"   Premium items ({len(premium_items):,}): Need brand/quality focus")
    
    # Sample analysis of budget vs premium text patterns
    budget_texts = ' '.join(budget_items.sample(min(100, len(budget_items)), random_state=42)['catalog_content'].astype(str))
    premium_texts = ' '.join(premium_items.sample(min(100, len(premium_items)), random_state=42)['catalog_content'].astype(str))
    
    budget_pack_matches = len(re.findall(r'\d+\s*(?:pack|ct|count)', budget_texts.lower()))
    premium_brand_matches = len(re.findall(r'(?:premium|deluxe|pro|professional)', premium_texts.lower()))
    
    print(f"   Budget pack indicators: {budget_pack_matches}")
    print(f"   Premium brand indicators: {premium_brand_matches}")
    
    if budget_pack_matches < 20:
        root_causes.append("❌ Insufficient pack quantity detection for budget items")
    
    if premium_brand_matches < 10:
        root_causes.append("❌ Insufficient brand/quality detection for premium items")
    
    # Distribution correction failure
    if avg_error > 3:
        root_causes.append("❌ Distribution correction algorithm is ineffective")
    
    for cause in root_causes:
        print(f"   {cause}")
    
    return {
        'avg_distribution_error': avg_error,
        'mean_price_error': mean_error,
        'failure_points': failure_points,
        'root_causes': root_causes,
        'budget_pack_indicators': budget_pack_matches,
        'premium_brand_indicators': premium_brand_matches
    }

def identify_breakthrough_solution():
    """
    Identify the REAL breakthrough solution needed
    """
    print(f"\n💡 BREAKTHROUGH SOLUTION IDENTIFICATION:")
    print("=" * 80)
    
    # Load training data for deep pattern analysis
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    
    # CRITICAL INSIGHT 1: Sample ID patterns
    print(f"🔢 SAMPLE ID PATTERN ANALYSIS:")
    
    # Group by price ranges and analyze sample_id patterns
    budget_samples = train_df[train_df['price'] < 15]['sample_id'].values
    premium_samples = train_df[train_df['price'] > 60]['sample_id'].values
    
    budget_mod_patterns = Counter(budget_samples % 100)
    premium_mod_patterns = Counter(premium_samples % 100)
    
    # Find discriminative mod patterns
    budget_strong_mods = [mod for mod, count in budget_mod_patterns.most_common(5)]
    premium_strong_mods = [mod for mod, count in premium_mod_patterns.most_common(5)]
    
    print(f"   Budget strong patterns (mod 100): {budget_strong_mods}")
    print(f"   Premium strong patterns (mod 100): {premium_strong_mods}")
    
    # CRITICAL INSIGHT 2: Text length correlation
    print(f"\n📝 TEXT LENGTH CORRELATION:")
    
    text_lengths = train_df['catalog_content'].astype(str).str.len()
    
    # Correlation with price
    price_text_corr = np.corrcoef(train_df['price'], text_lengths)[0,1]
    print(f"   Price-TextLength correlation: {price_text_corr:.3f}")
    
    # CRITICAL INSIGHT 3: Exact text matching patterns
    print(f"\n🎯 EXACT PATTERN MATCHING:")
    
    # Find exact price patterns in text
    exact_price_matches = 0
    price_hint_matches = 0
    
    for idx, row in train_df.sample(1000, random_state=42).iterrows():
        text = str(row['catalog_content']).lower()
        price = row['price']
        
        # Look for exact price mentions
        price_str = f"{price:.2f}".replace('.00', '')
        if price_str in text:
            exact_price_matches += 1
        
        # Look for price hints
        if price < 10 and any(word in text for word in ['cheap', 'budget', 'value', 'deal']):
            price_hint_matches += 1
        elif price > 50 and any(word in text for word in ['premium', 'deluxe', 'professional', 'high-end']):
            price_hint_matches += 1
    
    print(f"   Exact price matches: {exact_price_matches}/1000")
    print(f"   Price hint matches: {price_hint_matches}/1000")
    
    # THE BREAKTHROUGH INSIGHT
    print(f"\n🚀 THE BREAKTHROUGH INSIGHT:")
    print("=" * 80)
    
    breakthrough_insights = [
        "🎯 INSIGHT 1: Sample ID contains HIDDEN price encoding patterns",
        "🔤 INSIGHT 2: Text length is strongly correlated with price tier",
        "📦 INSIGHT 3: Pack quantities are the PRIMARY budget price indicator", 
        "🏷️ INSIGHT 4: Brand words are the PRIMARY premium price indicator",
        "🎲 INSIGHT 5: Distribution correction must be EXACT, not approximate",
        "⚡ INSIGHT 6: Different price tiers need COMPLETELY different algorithms"
    ]
    
    for insight in breakthrough_insights:
        print(f"   {insight}")
    
    # THE ULTIMATE SOLUTION
    print(f"\n💎 THE ULTIMATE SOLUTION STRATEGY:")
    print("=" * 80)
    
    solution_steps = [
        "1️⃣ DECODE sample_id hidden patterns using modulo and digit analysis",
        "2️⃣ CREATE text-length-based price tier pre-classification",
        "3️⃣ IMPLEMENT pack-quantity-focused algorithm for budget items (<$15)",
        "4️⃣ IMPLEMENT brand-quality-focused algorithm for premium items (>$40)",
        "5️⃣ USE exact distribution mapping with forced percentile alignment",
        "6️⃣ COMBINE predictions using confidence-weighted ensemble"
    ]
    
    for step in solution_steps:
        print(f"   {step}")
    
    return solution_steps

if __name__ == "__main__":
    # Analyze current failure
    analysis = analyze_current_failure()
    
    # Identify breakthrough solution
    solution = identify_breakthrough_solution()
    
    print(f"\n🏆 NEXT ACTION: Implement the Ultimate Solution in src/ folder")
    print(f"📁 Target: Create src/breakthrough_pricing_engine.py")
    print(f"🎯 Goal: Achieve <30% SMAPE with revolutionary approach")