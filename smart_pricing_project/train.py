# MIT License
# train.py
"""
Main training orchestrator for the Smart Product Pricing system.
Runs the complete pipeline: preprocessing → embeddings → training → ensemble → inference
"""

from __future__ import annotations
import os
import sys
import argparse
from src.config import Config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("TrainPipeline")
cfg = Config()

def run_preprocessing():
    """Run data preprocessing"""
    log.info("[AI-Agent] Starting data preprocessing...")
    
    from src.data.preprocess_features import main as preprocess_main
    preprocess_main(
        train_csv="dataset/train.csv",
        test_csv="dataset/test.csv", 
        out_dir="outputs/feature_store"
    )
    
    log.info("[AI-Agent] ✅ Data preprocessing completed")

def run_embedding_generation():
    """Generate text and image embeddings"""
    log.info("[AI-Agent] Starting embedding generation...")
    
    # Generate text embeddings (fast mode)
    try:
        from src.data.embeddings_text_lora import TextEmbedder
        from src.data.feature_store import FeatureStore
        import numpy as np
        
        fs = FeatureStore()
        train = fs.load_train()
        test = fs.load_test()
        
        te = TextEmbedder(use_lora=False)
        
        # Create directories
        os.makedirs("outputs/feature_store/embeddings_text/train", exist_ok=True)
        os.makedirs("outputs/feature_store/embeddings_text/test", exist_ok=True)
        
        # Generate embeddings
        train_emb = te.embed(train["catalog_clean"].tolist(), mode="fast")
        test_emb = te.embed(test["catalog_clean"].tolist(), mode="fast")
        
        # Save embeddings
        np.save("outputs/feature_store/embeddings_text/train/text_embeddings.npy", train_emb)
        np.save("outputs/feature_store/embeddings_text/test/text_embeddings.npy", test_emb)
        
        log.info("[AI-Agent] ✅ Text embeddings generated successfully")
        
    except Exception as e:
        log.error(f"Text embedding generation failed: {e}")
        log.info("Continuing without text embeddings...")
    
    # Generate image embeddings (if images available)
    try:
        from src.data.embeddings_image_clip import build_image_embeddings
        from src.data.feature_store import FeatureStore
        
        fs = FeatureStore()
        train = fs.load_train()
        test = fs.load_test()
        
        # Check if image directory exists
        if os.path.exists("dataset/images"):
            os.makedirs("outputs/feature_store/embeddings_image/train", exist_ok=True)
            os.makedirs("outputs/feature_store/embeddings_image/test", exist_ok=True)
            
            build_image_embeddings(
                "dataset/images", 
                train["sample_id"].astype(str).tolist(), 
                "outputs/feature_store/embeddings_image/train", 
                mode="clip"
            )
            build_image_embeddings(
                "dataset/images", 
                test["sample_id"].astype(str).tolist(), 
                "outputs/feature_store/embeddings_image/test", 
                mode="clip"
            )
            
            log.info("[AI-Agent] ✅ Image embeddings generated successfully")
        else:
            log.warning("Image directory not found, skipping image embeddings")
            
    except Exception as e:
        log.error(f"Image embedding generation failed: {e}")
        log.info("Continuing without image embeddings...")

def run_model_training():
    """Train all base models"""
    log.info("[AI-Agent] Starting model training...")
    
    # Train LightGBM
    try:
        from src.train.train_lgb import main as train_lgb_main
        train_lgb_main()
        log.info("[AI-Agent] ✅ LightGBM training completed")
    except Exception as e:
        log.error(f"LightGBM training failed: {e}")
    
    # Additional models can be trained here
    # For now, we'll focus on LightGBM as the primary model
    
    log.info("[AI-Agent] ✅ Model training completed")

def run_ensemble_training():
    """Train the meta-ensemble"""
    log.info("[AI-Agent] Starting ensemble training...")
    
    try:
        from src.train.train_ensemble import main as train_ensemble_main
        train_ensemble_main()
        log.info("[AI-Agent] ✅ Ensemble training completed")
    except Exception as e:
        log.error(f"Ensemble training failed: {e}")
        log.info("Ensemble training failed, inference will use fallback methods")

def run_inference():
    """Generate final predictions"""
    log.info("[AI-Agent] Starting inference...")
    
    try:
        from src.inference.predict import run_inference
        submission_df = run_inference()
        log.info("[AI-Agent] ✅ Inference completed successfully")
        return submission_df
    except Exception as e:
        log.error(f"Inference failed: {e}")
        return None

def main():
    """Run the complete training pipeline"""
    set_seed(cfg.seed)
    log.info("[AI-Agent] 🚀 Starting Smart Product Pricing training pipeline")
    
    try:
        # Step 1: Preprocessing
        run_preprocessing()
        
        # Step 2: Embedding generation
        run_embedding_generation()
        
        # Step 3: Model training
        run_model_training()
        
        # Step 4: Ensemble training
        run_ensemble_training()
        
        # Step 5: Final inference
        submission_df = run_inference()
        
        if submission_df is not None:
            log.info("[AI-Agent] 🎉 Training pipeline completed successfully!")
            log.info(f"[AI-Agent] Final submission shape: {submission_df.shape}")
            log.info(f"[AI-Agent] Price range: [{submission_df['price'].min():.2f}, {submission_df['price'].max():.2f}]")
        else:
            log.error("[AI-Agent] ❌ Pipeline completed with errors")
            
    except Exception as e:
        log.error(f"[AI-Agent] ❌ Pipeline failed: {e}")
        raise
    
    log.info("[AI-Agent] ✅ Project build complete. Ready to train and infer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Product Pricing Training Pipeline")
    parser.add_argument("--step", choices=["preprocess", "embeddings", "train", "ensemble", "inference", "all"], 
                        default="all", help="Pipeline step to run")
    
    args = parser.parse_args()
    
    if args.step == "all":
        main()
    elif args.step == "preprocess":
        run_preprocessing()
    elif args.step == "embeddings":
        run_embedding_generation()
    elif args.step == "train":
        run_model_training()
    elif args.step == "ensemble":
        run_ensemble_training()
    elif args.step == "inference":
        run_inference()