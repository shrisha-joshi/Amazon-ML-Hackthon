# MIT License
# src/postprocess/smoother.py
"""
Post-processing utilities for smoothing and optimizing predictions.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage
from src.utils.logger import get_logger

log = get_logger("smoother")

def smooth_predictions(preds: np.ndarray, method: str = "clip") -> np.ndarray:
    """
    Smooth extreme predictions using various methods.
    
    Args:
        preds: Raw predictions
        method: Smoothing method ('clip', 'median', 'winsorize')
    
    Returns:
        Smoothed predictions
    """
    preds = np.array(preds, dtype=float)
    
    if method == "clip":
        # Clip extreme values to 99th percentile
        upper_bound = np.percentile(preds, 99)
        lower_bound = np.percentile(preds, 1)
        preds_smooth = np.clip(preds, lower_bound, upper_bound)
        log.info(f"[AI-Agent] Clipped predictions to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
    elif method == "median":
        # Apply median filter to smooth outliers
        preds_smooth = ndimage.median_filter(preds, size=3)
        log.info("[AI-Agent] Applied median filter smoothing")
        
    elif method == "winsorize":
        # Winsorize at 5% and 95% percentiles
        from scipy.stats import mstats
        preds_smooth = mstats.winsorize(preds, limits=[0.05, 0.05])
        log.info("[AI-Agent] Applied winsorization at 5% and 95% percentiles")
        
    else:
        log.warning(f"Unknown smoothing method: {method}, using clip")
        preds_smooth = smooth_predictions(preds, method="clip")
    
    # Ensure all predictions are positive
    preds_smooth = np.maximum(preds_smooth, 0.01)
    
    return preds_smooth

def calibrate_predictions(preds: np.ndarray, target_mean: float = None) -> np.ndarray:
    """
    Calibrate predictions to match target statistics.
    
    Args:
        preds: Raw predictions
        target_mean: Target mean to calibrate to
    
    Returns:
        Calibrated predictions
    """
    preds = np.array(preds, dtype=float)
    
    if target_mean is not None:
        current_mean = np.mean(preds)
        scale_factor = target_mean / current_mean
        preds_calibrated = preds * scale_factor
        log.info(f"[AI-Agent] Calibrated predictions: {current_mean:.2f} -> {target_mean:.2f} (scale: {scale_factor:.3f})")
    else:
        preds_calibrated = preds
    
    return preds_calibrated

def apply_post_processing(preds: np.ndarray, train_prices: np.ndarray = None) -> np.ndarray:
    """
    Apply comprehensive post-processing pipeline.
    
    Args:
        preds: Raw predictions
        train_prices: Training prices for calibration (optional)
    
    Returns:
        Post-processed predictions
    """
    log.info("[AI-Agent] Starting post-processing pipeline")
    
    # Step 1: Smooth outliers
    preds = smooth_predictions(preds, method="clip")
    
    # Step 2: Calibrate to training distribution if available
    if train_prices is not None:
        target_mean = np.mean(train_prices)
        preds = calibrate_predictions(preds, target_mean)
    
    # Step 3: Final clipping to ensure reasonable range
    preds = np.clip(preds, 0.01, 10000.0)  # Reasonable price range
    
    log.info(f"[AI-Agent] Post-processing complete. Final range: [{np.min(preds):.2f}, {np.max(preds):.2f}]")
    
    return preds