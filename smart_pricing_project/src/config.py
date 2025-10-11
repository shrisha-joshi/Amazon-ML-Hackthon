# MIT License
# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "dataset"
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"
    text_model_name: str = "distilbert-base-uncased"
    image_model_name: str = "openai/clip-vit-base-patch32"  # fallback to resnet if CLIP not found
    num_folds: int = 5
    device: str = "cuda"  # agent should detect availability
    max_text_length: int = 128
    tfidf_max_features: int = 30000
    svd_components: int = 200
    knn_k: int = 10