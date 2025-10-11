# MIT License
# download_images.py
"""
Download images for the Smart Product Pricing dataset.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import urllib.request
import requests
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils.logger import get_logger

log = get_logger("download_images")
cfg = Config()

def download_image(image_link, save_folder, sample_id):
    """Download a single image"""
    if pd.isna(image_link) or not isinstance(image_link, str):
        return False
        
    try:
        # Use sample_id as filename to ensure consistency
        filename = f"{sample_id}.jpg"
        image_save_path = os.path.join(save_folder, filename)
        
        if not os.path.exists(image_save_path):
            # Try urllib first
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
                return True
            except:
                # Fallback to requests
                response = requests.get(image_link, timeout=10)
                if response.status_code == 200:
                    with open(image_save_path, 'wb') as f:
                        f.write(response.content)
                    return True
                return False
        return True  # Already exists
        
    except Exception as e:
        log.warning(f'Warning: Not able to download image for sample {sample_id}: {e}')
        return False

def download_images_for_dataset():
    """Download images for both train and test datasets"""
    
    # Create images directory
    images_dir = os.path.join("dataset", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load datasets
    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test.csv")
    
    log.info(f"[AI-Agent] Starting image download for {len(train_df)} train + {len(test_df)} test samples")
    
    # Download train images
    train_success = 0
    log.info("Downloading train images...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train images"):
        if download_image(row['image_link'], images_dir, row['sample_id']):
            train_success += 1
        if idx > 0 and idx % 1000 == 0:  # Progress update every 1000 images
            log.info(f"Downloaded {train_success}/{idx+1} train images so far")
    
    # Download test images
    test_success = 0
    log.info("Downloading test images...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test images"):
        if download_image(row['image_link'], images_dir, row['sample_id']):
            test_success += 1
        if idx > 0 and idx % 1000 == 0:  # Progress update every 1000 images
            log.info(f"Downloaded {test_success}/{idx+1} test images so far")
    
    log.info(f"[AI-Agent] Image download completed:")
    log.info(f"  Train images: {train_success}/{len(train_df)} ({train_success/len(train_df)*100:.1f}%)")
    log.info(f"  Test images: {test_success}/{len(test_df)} ({test_success/len(test_df)*100:.1f}%)")
    log.info(f"  Total images: {train_success + test_success}")
    
    return train_success, test_success

if __name__ == "__main__":
    download_images_for_dataset()