# MIT License
# src/data/preprocess_features.py
"""
Preprocess raw dataset:
- load train/test
- clean catalog_content
- extract IPQ (pack_qty, unit_qty, unit)
- create base tabular features: text_len, word_count, num_digits, has_image
- compute price_per_unit (train-only)
- persist cleaned CSVs and a small metadata file
"""

from __future__ import annotations
import os
import re
import argparse
import pandas as pd
import numpy as np
from src.config import Config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("preprocess_features")
cfg = Config()

# IPQ parsing function (robust)
UNIT_MAP = {
    "l": 1000.0, "litre": 1000.0, "liter": 1000.0,
    "ml": 1.0,
    "kg": 1000.0, "g": 1.0,
    "oz": 28.3495
}

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[^0-9A-Za-z\.\,\-\_\s\(\)]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def parse_ipq(s: str):
    s = (s or "").lower()
    # pattern 1: "12x500 ml" or "12 x 500ml" or "12×500 ml"
    m = re.search(r'(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(ml|l|g|kg|oz)?', s)
    if m:
        pack = float(m.group(1))
        unit_qty = float(m.group(2))
        unit = (m.group(3) or "").lower()
        base_qty = UNIT_MAP.get(unit, 1.0) * unit_qty if unit else unit_qty
        return pack, unit_qty, unit, base_qty

    # pattern 2: "pack of 6" or "6 pack" or "6 pcs"
    m = re.search(r'(?:pack of|pack|pk|pcs|pieces|count|ct|ct\.)\D*(\d+)', s)
    if m:
        pack = float(m.group(1))
        return pack, 1.0, "unit", 1.0

    # pattern 3: standalone units "500 ml", "1.5 l"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(ml|l|g|kg|oz)\b', s)
    if m:
        unit_qty = float(m.group(1)); unit = m.group(2)
        base_qty = UNIT_MAP.get(unit, 1.0) * unit_qty
        return 1.0, unit_qty, unit, base_qty

    # fallback
    return 1.0, 1.0, "unit", 1.0

def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    # normalize catalog
    df["catalog_clean"] = df["catalog_content"].fillna("").astype(str).map(clean_text)
    # basic text stats
    df["text_len"] = df["catalog_clean"].str.len().fillna(0).astype(int)
    df["word_count"] = df["catalog_clean"].apply(lambda x: len(x.split()))
    df["num_digits"] = df["catalog_clean"].str.count(r'\d').fillna(0).astype(int)
    df["num_caps"] = df["catalog_content"].fillna("").apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    # image presence
    df["has_image"] = df["image_link"].notnull().astype(int)
    # IPQ parsing
    parsed = df["catalog_content"].fillna("").apply(parse_ipq)
    df["pack_qty"] = parsed.apply(lambda x: x[0])
    df["unit_qty"] = parsed.apply(lambda x: x[1])
    df["unit"] = parsed.apply(lambda x: x[2])
    df["unit_base_qty"] = parsed.apply(lambda x: x[3])
    # price per base unit (only for training)
    if is_train and "price" in df.columns:
        df["price_per_base"] = df["price"] / (df["pack_qty"] * df["unit_base_qty"] + 1e-9)
    else:
        df["price_per_base"] = np.nan
    # fill nan
    df.fillna({"pack_qty":1.0, "unit_qty":1.0, "unit":"unit", "unit_base_qty":1.0}, inplace=True)
    return df

def main(train_csv: str, test_csv: str, out_dir: str):
    set_seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Loading train: {train_csv}")
    df_train = pd.read_csv(train_csv)
    log.info(f"Loading test: {test_csv}")
    df_test = pd.read_csv(test_csv)
    log.info("Building train features")
    train_feat = build_features(df_train, is_train=True)
    log.info("Building test features")
    test_feat = build_features(df_test, is_train=False)
    train_out = os.path.join(out_dir, "train_features.csv")
    test_out = os.path.join(out_dir, "test_features.csv")
    train_feat.to_csv(train_out, index=False)
    test_feat.to_csv(test_out, index=False)
    log.info(f"[AI-Agent] Saved train features to {train_out}")
    log.info(f"[AI-Agent] Saved test features to {test_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--test_csv", default="dataset/test.csv")
    parser.add_argument("--out_dir", default="outputs/feature_store")
    args = parser.parse_args()
    main(args.train_csv, args.test_csv, args.out_dir)