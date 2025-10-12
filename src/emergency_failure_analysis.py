#!/usr/bin/env python3
"""
🚨 EMERGENCY ANALYSIS: 63.983% SMAPE FAILURE INVESTIGATION
Critical gap between expected 15-30% and actual 63.983% SMAPE

EMERGENCY PRIORITIES:
1. Root cause analysis of Ultra-Precision Engine v2.0 failure
2. Identify specific failure patterns in 63.983% result
3. Design emergency breakthrough system targeting <40% SMAPE
4. Implement radical new approach based on failure analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EmergencyFailureAnalysis:
    """
    🚨 EMERGENCY ANALYSIS SYSTEM
    
    Critical mission: Understand why 63.983% SMAPE occurred
    when Ultra-Precision Engine v2.0 expected 15-30%
    """
    
    def __init__(self):
        self.train_df = None
        self.predicted_prices = None
        self.actual_prices = None
        self.failure_insights = {}
        
        print("🚨 EMERGENCY FAILURE ANALYSIS ACTIVATED")
        print("🎯 Target: Understand 63.983% SMAPE failure")
        print("🔍 Mission: Find breakthrough path to <40% SMAPE")
    
    def calculate_smape(self, actual, predicted):
        """Calculate SMAPE score"""
        return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
    
    def load_data_for_analysis(self):
        """Load training data for comparison analysis"""
        print("\n📊 Loading training data for failure analysis...")
        
        self.train_df = pd.read_csv('student_resource/dataset/train.csv')
        
        # Load our predictions
        pred_df = pd.read_csv('test_out2.csv')
        self.predicted_prices = pred_df['price'].values
        
        print(f"✅ Training data: {len(self.train_df):,} samples")
        print(f"✅ Predictions: {len(self.predicted_prices):,} samples")
        print(f"💰 Training price stats:")
        print(f"   Mean: ${self.train_df['price'].mean():.2f}")
        print(f"   Median: ${self.train_df['price'].median():.2f}")
        print(f"   Std: ${self.train_df['price'].std():.2f}")
        print(f"   Range: ${self.train_df['price'].min():.2f} - ${self.train_df['price'].max():.2f}")
        
        print(f"🎯 Prediction price stats:")
        print(f"   Mean: ${np.mean(self.predicted_prices):.2f}")
        print(f"   Median: ${np.median(self.predicted_prices):.2f}")
        print(f"   Std: ${np.std(self.predicted_prices):.2f}")
        print(f"   Range: ${np.min(self.predicted_prices):.2f} - ${np.max(self.predicted_prices):.2f}")
    
    def analyze_distribution_mismatch(self):
        """Analyze distribution differences that might cause SMAPE failure"""
        print("\n🔍 DISTRIBUTION MISMATCH ANALYSIS")
        print("=" * 80)
        
        # Define price ranges for analysis
        ranges = [
            (0, 5, "Ultra-Budget"),
            (5, 10, "Budget"),
            (10, 15, "Low-Mid"),
            (15, 20, "Mid-Low"),
            (20, 25, "Mid"),
            (25, 35, "Mid-High"),
            (35, 50, "High"),
            (50, 100, "Premium"),
            (100, float('inf'), "Luxury")
        ]
        
        print(f"{'Range':<15} {'Training %':<12} {'Predicted %':<12} {'Difference':<12}")
        print("-" * 55)
        
        total_mismatch = 0
        
        for low, high, label in ranges:
            if high == float('inf'):
                train_count = (self.train_df['price'] >= low).sum()
                pred_count = (self.predicted_prices >= low).sum()
            else:
                train_count = ((self.train_df['price'] >= low) & (self.train_df['price'] < high)).sum()
                pred_count = ((self.predicted_prices >= low) & (self.predicted_prices < high)).sum()
            
            train_pct = 100 * train_count / len(self.train_df)
            pred_pct = 100 * pred_count / len(self.predicted_prices)
            diff = pred_pct - train_pct
            total_mismatch += abs(diff)
            
            print(f"{label:<15} {train_pct:<11.1f}% {pred_pct:<11.1f}% {diff:+.1f}%")
        
        print(f"\n📊 Total Distribution Mismatch: {total_mismatch:.1f}%")
        self.failure_insights['distribution_mismatch'] = total_mismatch
        
        if total_mismatch > 50:
            print("🚨 CRITICAL: Massive distribution mismatch detected!")
        elif total_mismatch > 20:
            print("⚠️ WARNING: Significant distribution mismatch")
        else:
            print("✅ Distribution relatively aligned")
    
    def analyze_prediction_patterns(self):
        """Analyze prediction patterns that might explain SMAPE failure"""
        print("\n🔍 PREDICTION PATTERN ANALYSIS")
        print("=" * 80)
        
        # Extreme predictions analysis
        extreme_low = (self.predicted_prices < 1).sum()
        extreme_high = (self.predicted_prices > 200).sum()
        
        print(f"Extreme predictions analysis:")
        print(f"   Too low (<$1): {extreme_low:,} ({100*extreme_low/len(self.predicted_prices):.1f}%)")
        print(f"   Too high (>$200): {extreme_high:,} ({100*extreme_high/len(self.predicted_prices):.1f}%)")
        
        # Variance analysis
        pred_variance = np.var(self.predicted_prices)
        train_variance = np.var(self.train_df['price'])
        variance_ratio = pred_variance / train_variance
        
        print(f"\nVariance analysis:")
        print(f"   Training variance: {train_variance:.2f}")
        print(f"   Prediction variance: {pred_variance:.2f}")
        print(f"   Variance ratio: {variance_ratio:.2f}")
        
        self.failure_insights['extreme_predictions'] = extreme_low + extreme_high
        self.failure_insights['variance_ratio'] = variance_ratio
        
        if variance_ratio > 3:
            print("🚨 CRITICAL: Predictions too volatile!")
        elif variance_ratio < 0.3:
            print("⚠️ WARNING: Predictions too conservative")
        else:
            print("✅ Variance reasonably aligned")
    
    def identify_critical_failure_points(self):
        """Identify the most critical failure points causing high SMAPE"""
        print("\n🚨 CRITICAL FAILURE POINT IDENTIFICATION")
        print("=" * 80)
        
        # Since we don't have actual test labels, we'll simulate based on training patterns
        # and identify likely problem areas
        
        # Analyze prediction concentrations
        pred_counts, pred_bins = np.histogram(self.predicted_prices, bins=50)
        train_counts, train_bins = np.histogram(self.train_df['price'], bins=50)
        
        # Find bins with largest differences
        bin_centers = (pred_bins[:-1] + pred_bins[1:]) / 2
        pred_density = pred_counts / len(self.predicted_prices)
        train_density = train_counts / len(self.train_df)
        
        density_diff = np.abs(pred_density - train_density)
        worst_bins = np.argsort(density_diff)[-5:]  # 5 worst bins
        
        print("Top 5 problematic price ranges:")
        for i, bin_idx in enumerate(worst_bins[::-1]):
            bin_start = pred_bins[bin_idx]
            bin_end = pred_bins[bin_idx + 1]
            diff = density_diff[bin_idx]
            print(f"   {i+1}. ${bin_start:.1f}-${bin_end:.1f}: {diff:.3f} density difference")
        
        # Identify over-prediction vs under-prediction tendency
        mean_diff = np.mean(self.predicted_prices) - self.train_df['price'].mean()
        
        if mean_diff > 5:
            print(f"\n🚨 SYSTEMATIC OVER-PREDICTION: +${mean_diff:.2f} average")
            self.failure_insights['bias'] = 'over_prediction'
        elif mean_diff < -5:
            print(f"\n🚨 SYSTEMATIC UNDER-PREDICTION: ${mean_diff:.2f} average")
            self.failure_insights['bias'] = 'under_prediction'
        else:
            print(f"\n✅ Mean bias acceptable: ${mean_diff:.2f}")
            self.failure_insights['bias'] = 'acceptable'
    
    def generate_emergency_recommendations(self):
        """Generate emergency recommendations for breakthrough improvement"""
        print("\n🚀 EMERGENCY BREAKTHROUGH RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = []
        
        # Based on distribution mismatch
        if self.failure_insights.get('distribution_mismatch', 0) > 30:
            recommendations.append("🔧 CRITICAL: Implement aggressive distribution alignment")
            recommendations.append("   → Use quantile mapping with training distribution")
            recommendations.append("   → Apply post-processing distribution correction")
        
        # Based on variance issues
        if self.failure_insights.get('variance_ratio', 1) > 2:
            recommendations.append("🔧 CRITICAL: Reduce prediction volatility")
            recommendations.append("   → Apply robust scaling and outlier clipping")
            recommendations.append("   → Use conservative ensemble weighting")
        
        # Based on extreme predictions
        if self.failure_insights.get('extreme_predictions', 0) > 1000:
            recommendations.append("🔧 CRITICAL: Fix extreme prediction issues")
            recommendations.append("   → Implement strict bounds: $0.50 - $500.00")
            recommendations.append("   → Add prediction sanity checks")
        
        # Based on bias
        if self.failure_insights.get('bias') == 'over_prediction':
            recommendations.append("🔧 CRITICAL: Correct systematic over-prediction")
            recommendations.append("   → Apply multiplicative correction factor")
            recommendations.append("   → Increase conservative model weights")
        elif self.failure_insights.get('bias') == 'under_prediction':
            recommendations.append("🔧 CRITICAL: Correct systematic under-prediction")
            recommendations.append("   → Apply multiplicative boost factor")
            recommendations.append("   → Increase aggressive model weights")
        
        # Always recommend emergency system
        recommendations.append("🚀 EMERGENCY SYSTEM: Build Emergency SMAPE Rescue System")
        recommendations.append("   → Simple, robust approach focused on SMAPE optimization")
        recommendations.append("   → Conservative feature set avoiding over-engineering")
        recommendations.append("   → Direct SMAPE loss minimization")
        
        print("PRIORITY ACTIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
    
    def run_complete_analysis(self):
        """Run complete emergency failure analysis"""
        print("🚨" * 40)
        print("EMERGENCY FAILURE ANALYSIS - 63.983% SMAPE INVESTIGATION")
        print("🚨" * 40)
        
        self.load_data_for_analysis()
        self.analyze_distribution_mismatch()
        self.analyze_prediction_patterns()
        self.identify_critical_failure_points()
        self.generate_emergency_recommendations()
        
        print(f"\n📋 FAILURE ANALYSIS SUMMARY:")
        print("=" * 80)
        print(f"Actual SMAPE: 63.983%")
        print(f"Expected SMAPE: 15-30%")
        print(f"Performance Gap: {63.983 - 22.5:.1f} percentage points")
        print(f"Distribution Mismatch: {self.failure_insights.get('distribution_mismatch', 0):.1f}%")
        print(f"Variance Ratio: {self.failure_insights.get('variance_ratio', 1):.2f}")
        print(f"Extreme Predictions: {self.failure_insights.get('extreme_predictions', 0):,}")
        print(f"Prediction Bias: {self.failure_insights.get('bias', 'unknown')}")
        
        print(f"\n🎯 EMERGENCY MISSION:")
        print("Build Emergency SMAPE Rescue System targeting <40% SMAPE")
        print("Focus: Simple, robust, SMAPE-optimized approach")
        print("Abandon complex features, focus on SMAPE minimization")

def main():
    """Execute emergency failure analysis"""
    analyzer = EmergencyFailureAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()