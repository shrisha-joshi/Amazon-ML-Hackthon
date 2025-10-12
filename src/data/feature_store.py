# MIT License
# src/data/feature_store.py
from __future__ import annotations
import os
import pandas as pd
from src.config import Config
from src.utils.logger import get_logger

log = get_logger("feature_store")
cfg = Config()

class FeatureStore:
    def __init__(self, base_out: str = None):
        self.base_out = base_out or "outputs/feature_store"
        os.makedirs(self.base_out, exist_ok=True)

    def train_path(self) -> str:
        return os.path.join(self.base_out, "train_features.csv")

    def test_path(self) -> str:
        return os.path.join(self.base_out, "test_features.csv")

    def load_train(self) -> pd.DataFrame:
        path = self.train_path()
        log.info(f"Loading train features from {path}")
        return pd.read_csv(path)

    def load_test(self) -> pd.DataFrame:
        path = self.test_path()
        log.info(f"Loading test features from {path}")
        return pd.read_csv(path)

    def save_feature(self, df: pd.DataFrame, name: str):
        path = os.path.join(self.base_out, name)
        log.info(f"Saving feature to {path}")
        df.to_csv(path, index=False)
        return path