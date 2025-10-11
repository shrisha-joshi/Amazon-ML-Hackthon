# MIT License
# src/data/tests_quick.py
"""
Run quick checks that feature CSVs and embeddings exist and shapes align.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import os
from src.data.feature_store import FeatureStore
from src.utils.logger import get_logger

log = get_logger("tests_quick")

def run_checks():
    fs = FeatureStore()
    train = fs.load_train()
    test = fs.load_test()
    log.info(f"train rows: {len(train)}, test rows: {len(test)}")
    # check embeddings
    emb_base = "outputs/feature_store/embeddings_text"
    emb_img_base = "outputs/feature_store/embeddings_image"
    for folder in ["train","test"]:
        txt_p = os.path.join(emb_base, folder, "text_embeddings.npy")
        img_p = os.path.join(emb_img_base, folder, "image_embeddings.npy")
        if os.path.exists(txt_p):
            te = np.load(txt_p)
            log.info(f"{folder} text emb shape: {te.shape}")
            assert te.shape[0] == (len(train) if folder=="train" else len(test))
        else:
            log.warning(f"{txt_p} missing")
        if os.path.exists(img_p):
            ie = np.load(img_p)
            log.info(f"{folder} image emb shape: {ie.shape}")
            assert ie.shape[0] == (len(train) if folder=="train" else len(test))
        else:
            log.warning(f"{img_p} missing")

if __name__ == "__main__":
    run_checks()