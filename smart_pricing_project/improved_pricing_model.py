#!/usr/bin/env python3
"""
Improved pricing pipeline (single-file, no external price lookups).

Highlights:
- Robust IPQ/unit parsing (pack_qty, volume_ml, weight_g)
- Text features: TF-IDF + SVD (SBERT optional if installed)
- kNN price feature (median neighbor price) via FAISS if available or sklearn fallback
- Train on log1p(price) with LightGBM, 5-fold OOF, early stopping
- Postprocess: clipping to train percentiles, cluster smoothing, quantile residual correction
- Reproducible, CLI args, saves artifacts and outputs

Outputs:
- outputs/oof_preds.npy, outputs/test_preds.npy
- outputs/test_out_improved.csv
- outputs/models_lgb.pkl (models, scaler, vectorizers, optional SBERT)

Usage examples (PowerShell):
  python smart_pricing_project/improved_pricing_model.py `
    --train_csv student_resource/dataset/train.csv `
    --test_csv student_resource/dataset/test.csv `
    --sample_csv student_resource/dataset/sample_test.csv `
    --sample_out student_resource/dataset/sample_test_out.csv `
    --out_dir outputs

For a quick smoke test (use fewer rows):
  python smart_pricing_project/improved_pricing_model.py `
    --train_csv student_resource/dataset/train.csv `
    --test_csv student_resource/dataset/test.csv `
    --debug_nrows 5000 `
    --out_dir outputs
"""

from __future__ import annotations
import os
import re
import math
import json
import pickle
import argparse
import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error

# LightGBM
import lightgbm as lgb

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ------------------------
# Utility functions
# ------------------------
def set_seed(seed: int = SEED) -> None:
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def smape(y_true, y_pred, eps=1e-8) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, eps, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def safe_expm1(x):
    x = np.array(x, dtype=float)
    return np.expm1(x)


# ------------------------
# IPQ/Unit parsing
# ------------------------
UNIT_MAP = {
    # normalize to base volume_ml or weight_g
    "ml": ("volume_ml", 1.0),
    "milliliter": ("volume_ml", 1.0),
    "millilitre": ("volume_ml", 1.0),
    "l": ("volume_ml", 1000.0),
    "liter": ("volume_ml", 1000.0),
    "litre": ("volume_ml", 1000.0),
    "gal": ("volume_ml", 3785.41),
    "gallon": ("volume_ml", 3785.41),
    "qt": ("volume_ml", 946.353),
    "quart": ("volume_ml", 946.353),
    "pt": ("volume_ml", 473.176),
    "pint": ("volume_ml", 473.176),
    "oz": ("volume_ml", 29.5735),
    "fl oz": ("volume_ml", 29.5735),
    "ounce": ("volume_ml", 29.5735),
    "g": ("weight_g", 1.0),
    "gram": ("weight_g", 1.0),
    "kg": ("weight_g", 1000.0),
    "kilogram": ("weight_g", 1000.0),
    "lb": ("weight_g", 453.592),
    "pound": ("weight_g", 453.592),
}

IPQ_PATTERNS = [
    r"(?P<pack>\d+)\s*[x×]\s*(?P<unitqty>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Zµ\.\s]+)",
    r"(?P<unitqty>\d+(?:\.\d+)?)\s*(?P<unit>ml|l|liter|litre|gal|gallon|qt|quart|pt|pint|g|kg|oz|ounce|pound|lb)\b",
    r"pack of (?P<pack>\d+)\b",
    r"(?P<pack>\d+)\s*(?:pack|pk|pcs|pieces|count|ct|counts)\b",
    r"(?P<pack>dozen)\b",
]


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_ipq(text: str) -> dict:
    text = (text or "").lower()
    # Use max of detected single-unit quantities rather than sum to avoid double-counting
    res = {"pack_qty": 1.0, "volume_ml": 0.0, "weight_g": 0.0}
    for pat in IPQ_PATTERNS:
        for m in re.finditer(pat, text):
            g = m.groupdict()
            if g.get("pack"):
                try:
                    if str(g["pack"]).lower() == "dozen":
                        res["pack_qty"] = max(res["pack_qty"], 12.0)
                    else:
                        res["pack_qty"] = max(res["pack_qty"], float(g["pack"]))
                except Exception:
                    pass
            if g.get("unitqty") and g.get("unit"):
                try:
                    u = g["unit"].strip().replace(".", "")
                    qty = float(g["unitqty"])
                    # match unit
                    matched = False
                    for key, (kind, mult) in UNIT_MAP.items():
                        if u.startswith(key):
                            val = qty * mult
                            if kind == "volume_ml":
                                res["volume_ml"] = max(res["volume_ml"], val)
                            elif kind == "weight_g":
                                res["weight_g"] = max(res["weight_g"], val)
                            matched = True
                            break
                    if not matched:
                        # try compact units (e.g., 500ml)
                        pass
                except Exception:
                    pass
    # compact attached units like "500ml", "1.5l", "10kg"
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ml|l|gal|gallon|qt|quart|pt|pint|g|kg|oz|lb)\b", text):
        unit = unit.lower()
        qty = float(num)
        for key, (kind, mult) in UNIT_MAP.items():
            if unit.startswith(key):
                val = qty * mult
                if kind == "volume_ml":
                    res["volume_ml"] = max(res["volume_ml"], val)
                else:
                    res["weight_g"] = max(res["weight_g"], val)
                break
    return res


def build_features(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    rows = []
    cat_col = df.get("catalog_content")
    if cat_col is None:
        # Try a fallback if different column name exists
        raise ValueError("Expected 'catalog_content' column in input DataFrame")
    for text in tqdm(cat_col.astype(str).fillna(""), desc="parse IPQ"):
        ipq = parse_ipq(text)
        t = str(text)
        char_count = len(t)
        word_count = len(t.split())
        digits = len(re.findall(r"\d+", t))
        upper_count = sum(1 for c in t if c.isupper())
        rows.append({
            "pack_qty": ipq["pack_qty"],
            "volume_ml": ipq["volume_ml"],
            "weight_g": ipq["weight_g"],
            "char_count": char_count,
            "word_count": word_count,
            "digits": digits,
            "upper_count": upper_count,
        })
    feat = pd.DataFrame(rows)
    feat["has_image"] = (~df.get("image_link").isna()).astype(int) if "image_link" in df.columns else 0
    feat["contains_pack_word"] = df["catalog_content"].astype(str).str.contains(r"\bpack\b|\bpcs\b|\bpack of\b", case=False, regex=True, na=False).astype(int)
    if is_train and "price" in df.columns:
        price = df["price"].fillna(0).values.astype(float)
        base_unit = (feat["pack_qty"].replace(0, 1.0) * (feat["volume_ml"].replace(0, 1.0)))
        base_unit = base_unit.replace(0, 1.0)
        feat["price_per_unit"] = price / base_unit
    else:
        feat["price_per_unit"] = np.nan
    feat.fillna(0, inplace=True)
    return feat


# ------------------------
# Text features
# ------------------------
def build_text_features(train_texts, test_texts, tfidf_max=20000, svd_n=150):
    print("[info] building TF-IDF...")
    tf = TfidfVectorizer(max_features=tfidf_max, ngram_range=(1, 2), min_df=3)
    all_texts = list(train_texts) + list(test_texts)
    tfidf = tf.fit_transform(all_texts)
    n_train = len(train_texts)
    X_tfidf_train = tfidf[:n_train]
    X_tfidf_test = tfidf[n_train:]
    print("[info] applying SVD...")
    svd = TruncatedSVD(n_components=svd_n, random_state=SEED)
    X_svd_train = svd.fit_transform(X_tfidf_train)
    X_svd_test = svd.transform(X_tfidf_test)
    return X_svd_train, X_svd_test, tf, svd


def build_sbert_embeddings(train_texts, test_texts, model_name='all-MiniLM-L6-v2', batch_size=64):
    if not _HAS_SBERT:
        print("[warn] sentence-transformers not installed; skipping SBERT.")
        return None, None, None
    print("[info] building SBERT embeddings...")
    model = SentenceTransformer(model_name)
    X_train = model.encode(train_texts, batch_size=batch_size, show_progress_bar=True)
    X_test = model.encode(test_texts, batch_size=batch_size, show_progress_bar=True)
    return X_train, X_test, model


# ------------------------
# kNN price feature
# ------------------------
def build_nn_price(train_emb: np.ndarray, train_prices: np.ndarray, test_emb: np.ndarray, k: int = 10) -> np.ndarray:
    if train_emb is None or test_emb is None:
        return np.full(len(test_emb), np.median(train_prices))
    if _HAS_FAISS:
        # cosine via normalized inner product
        tr = train_emb.astype('float32').copy()
        te = test_emb.astype('float32').copy()
        faiss.normalize_L2(tr)
        faiss.normalize_L2(te)
        index = faiss.IndexFlatIP(tr.shape[1])
        index.add(tr)
        D, I = index.search(te, k)
        return np.median(train_prices[I], axis=1)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=8).fit(train_emb)
        dists, inds = nn.kneighbors(test_emb)
        return np.median(train_prices[inds], axis=1)


def build_nn_price_train(train_emb: np.ndarray, train_prices: np.ndarray, k: int = 10) -> np.ndarray:
    if _HAS_FAISS:
        tr = train_emb.astype('float32').copy()
        faiss.normalize_L2(tr)
        index = faiss.IndexFlatIP(tr.shape[1])
        index.add(tr)
        D, I = index.search(tr, k + 1)
        med = []
        for i, neighbors in enumerate(I):
            neighbors = neighbors[neighbors != i][:k]
            if len(neighbors) == 0:
                med.append(np.median(train_prices))
            else:
                med.append(np.median(train_prices[neighbors]))
        return np.array(med)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=8).fit(train_emb)
        dists, inds = nn.kneighbors(train_emb)
        med = []
        for i, row in enumerate(inds):
            row = [j for j in row if j != i][:k]
            if len(row) == 0:
                med.append(np.median(train_prices))
            else:
                med.append(np.median(train_prices[row]))
        return np.array(med)


# ------------------------
# Main training/prediction
# ------------------------
def focused_text_series(df: pd.DataFrame) -> list[str]:
    """Extract a compact, information-dense text string per item.
    Heuristics:
    - Prefer explicit fields like 'item name:' and 'brand:' if present
    - Include first bullet
    - Add likely model tokens (alnum with digits)
    - Fallback: first 20 words of first line
    """
    out = []
    for s in df['catalog_content'].astype(str).fillna(""):
        s_low = s.lower()
        # Fields
        def extract_field(label: str, default: str = ""):
            if label in s_low:
                try:
                    return s[s_low.index(label) + len(label):].split("\n")[0].strip()
                except Exception:
                    return default
            return default

        name = extract_field("item name:", "").strip()
        brand = extract_field("brand:", "").strip()
        bullet = extract_field("bullet point 1:", "").strip()
        if not name:
            name = s.split("\n")[0]
            name = " ".join(name.split()[:20])
        # Model-like tokens: include up to 2 alnum tokens containing digits
        model_tokens = re.findall(r"\b[\w-]*\d[\w-]*\b", s)
        model_tokens = [t for t in model_tokens if len(t) <= 20][:2]
        pieces = [brand, name, bullet] + model_tokens
        compact = " ".join([p for p in pieces if p]).strip()
        out.append(compact[:200])
    return out


def train_and_predict(
    train_csv: str,
    test_csv: str,
    out_dir: str = "outputs",
    tfidf_max: int = 20000,
    svd_n: int = 150,
    use_sbert: bool = False,
    n_splits: int = 5,
    debug_nrows: Optional[int] = None,
    nn_k: int = 10,
    num_boost_round: int = 6000,
    early_stopping_rounds: int = 300,
    blend_main_weight: float = 0.7,
):
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[start] loading data: train_csv={train_csv}, test_csv={test_csv}")
    train_df = pd.read_csv(train_csv, nrows=debug_nrows)
    test_df = pd.read_csv(test_csv, nrows=debug_nrows)
    print(f"[info] train rows: {len(train_df)}, test rows: {len(test_df)}")

    # Build tabular features
    print("[step] building tabular features...")
    train_feat = build_features(train_df, is_train=True)
    test_feat = build_features(test_df, is_train=False)

    # Focused text
    print("[step] preparing text features...")
    train_texts = focused_text_series(train_df)
    test_texts = focused_text_series(test_df)

    # TF-IDF + SVD
    X_svd_train, X_svd_test, tfidf_obj, svd_obj = build_text_features(train_texts, test_texts, tfidf_max=tfidf_max, svd_n=svd_n)

    # Optional SBERT
    X_sbert_train = X_sbert_test = sbert_model = None
    if use_sbert and _HAS_SBERT:
        X_sbert_train, X_sbert_test, sbert_model = build_sbert_embeddings(train_texts, test_texts)
    elif use_sbert and not _HAS_SBERT:
        print("[warn] SBERT requested but not available; continuing without it.")

    # Assemble feature matrices
    print("[step] assembling features...")
    X_tab_train = train_feat.values
    X_tab_test = test_feat.values
    parts_train = [X_tab_train, X_svd_train]
    parts_test = [X_tab_test, X_svd_test]
    if X_sbert_train is not None:
        parts_train.append(X_sbert_train)
        parts_test.append(X_sbert_test)
    X_train_full = np.hstack(parts_train)
    X_test_full = np.hstack(parts_test)

    # kNN price features
    print("[step] computing kNN median price features...")
    train_prices = train_df['price'].values.astype(float)
    emb_for_knn_train = X_sbert_train if X_sbert_train is not None else X_svd_train
    emb_for_knn_test = X_sbert_test if X_sbert_test is not None else X_svd_test
    nn_price_train = build_nn_price_train(emb_for_knn_train, train_prices, k=nn_k)
    nn_price_test = build_nn_price(emb_for_knn_train, train_prices, emb_for_knn_test, k=nn_k)

    X_train_full = np.hstack([X_train_full, nn_price_train.reshape(-1, 1)])
    X_test_full = np.hstack([X_test_full, nn_price_test.reshape(-1, 1)])

    # Prepare target
    y = np.log1p(train_prices)

    # Compute unit_base for price-per-unit head
    def compute_unit_base(feat_df: pd.DataFrame) -> np.ndarray:
        pack = feat_df["pack_qty"].values.astype(float)
        vol = feat_df["volume_ml"].values.astype(float)
        wt = feat_df["weight_g"].values.astype(float)
        base = np.where(vol > 0, vol, np.where(wt > 0, wt, 1.0))
        # winsorize base to avoid extreme scales from text noise
        return pack * np.clip(base, 1.0, None)

    unit_base_train = compute_unit_base(train_feat)
    # cap unit base using p99 of training (avoid huge reconstructed prices)
    ub_cap = np.percentile(unit_base_train, 99)
    unit_base_train = np.clip(unit_base_train, 1.0, max(ub_cap, 10.0))
    unit_base_test = compute_unit_base(test_feat)
    unit_base_test = np.clip(unit_base_test, 1.0, max(ub_cap, 10.0))

    # Scale
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    # LightGBM OOF (main head)
    print("[step] training LightGBM (log1p target) with KFold...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(y), dtype=float)
    test_preds_folds = []
    models = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled), start=1):
        print(f"[fold {fold}/{n_splits}] train={len(tr_idx)} val={len(val_idx)}")
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': 0.03,
            'num_leaves': 128,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.5,
            'lambda_l2': 1.0,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': SEED + fold,
        }
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        models.append(model)
        pred_val_log = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = np.expm1(pred_val_log)
        pred_test_log = model.predict(X_test_scaled, num_iteration=model.best_iteration)
        test_preds_folds.append(np.expm1(pred_test_log))

        val_smape = smape(np.expm1(y_val), np.expm1(pred_val_log))
        val_mae = mean_absolute_error(np.expm1(y_val), np.expm1(pred_val_log))
        print(f"[fold {fold}] val SMAPE: {val_smape:.4f} | MAE: {val_mae:.4f}")

    test_preds_agg = np.median(np.vstack(test_preds_folds), axis=0)
    total_smape = smape(train_prices, oof_preds)
    total_mae = mean_absolute_error(train_prices, oof_preds)
    print(f"[OOF] SMAPE: {total_smape:.4f} | MAE: {total_mae:.4f}")

    # Quantile regression head (upper-tail guard)
    print("[step] training LightGBM quantile head (q=0.80)...")
    q_alpha = 0.80
    params_q = {
        'objective': 'quantile',
        'alpha': q_alpha,
        'metric': 'quantile',
        'learning_rate': 0.03,
        'num_leaves': 96,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 0.5,
        'min_data_in_leaf': 20,
        'verbose': -1,
    }
    models_q80 = []
    test_q80_folds = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled), start=1):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr_q, y_val_q = train_prices[tr_idx], train_prices[val_idx]
        dtr_q = lgb.Dataset(X_tr, label=y_tr_q)
        dval_q = lgb.Dataset(X_val, label=y_val_q, reference=dtr_q)
        m_q = lgb.train(
            params_q,
            dtr_q,
            num_boost_round=max(2000, num_boost_round // 2),
            valid_sets=[dval_q],
            callbacks=[
                lgb.early_stopping(max(100, early_stopping_rounds // 2), verbose=False),
                lgb.log_evaluation(300),
            ],
        )
        models_q80.append(m_q)
        test_q = m_q.predict(X_test_scaled, num_iteration=m_q.best_iteration)
        test_q80_folds.append(test_q)
    test_q80_agg = np.median(np.vstack(test_q80_folds), axis=0)

    # Secondary head: price-per-unit (PPU), trained on log1p(price_per_unit)
    print("[step] training secondary head: price-per-unit (PPU)...")
    # price per unit with clipping to stabilize target
    ppu_train = train_prices / unit_base_train
    ppu_cap = np.percentile(ppu_train, 99)
    ppu_train = np.clip(ppu_train, 0.0, max(ppu_cap, 1.0))
    y_ppu = np.log1p(ppu_train)
    oof_ppu = np.zeros(len(y), dtype=float)
    test_ppu_folds = []
    models_ppu = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled), start=1):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_ppu[tr_idx], y_ppu[val_idx]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
        params_ppu = {
            'objective': 'regression',
            'metric': 'l2',
            'learning_rate': 0.03,
            'num_leaves': 96,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'lambda_l1': 0.2,
            'lambda_l2': 0.8,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': SEED + 100 + fold,
        }
        m = lgb.train(
            params_ppu,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(300),
            ],
        )
        models_ppu.append(m)
    oof_val_ppu = np.expm1(m.predict(X_val, num_iteration=m.best_iteration)) * unit_base_train[val_idx]
    # clip to train price percentiles before OOF blend accounting
    oof_val_ppu = np.clip(oof_val_ppu, np.percentile(train_prices, 1), np.percentile(train_prices, 99))
    oof_ppu[val_idx] = oof_val_ppu
    test_ppu = np.expm1(m.predict(X_test_scaled, num_iteration=m.best_iteration)) * unit_base_test
    test_ppu = np.clip(test_ppu, np.percentile(train_prices, 1), np.percentile(train_prices, 99))
    test_ppu_folds.append(test_ppu)

    test_ppu_agg = np.median(np.vstack(test_ppu_folds), axis=0)
    oof_blend = blend_main_weight * oof_preds + (1.0 - blend_main_weight) * oof_ppu
    total_mae_blend = mean_absolute_error(train_prices, oof_blend)
    total_smape_blend = smape(train_prices, oof_blend)
    print(f"[OOF blend] SMAPE: {total_smape_blend:.4f} | MAE: {total_mae_blend:.4f}")

    # Save artifacts
    print("[step] saving artifacts...")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "models_lgb.pkl"), "wb") as f:
        pickle.dump({
            "models": models,
            "scaler": scaler,
            "tfidf": tfidf_obj,
            "svd": svd_obj,
            # Do not pickle SBERT model object; store name + flag instead
            "use_sbert": bool(use_sbert and _HAS_SBERT),
            "sbert_model_name": 'all-MiniLM-L6-v2',
            # store for kNN price feature at inference time
            "emb_for_knn_train": emb_for_knn_train,
            "train_prices": train_prices,
            "models_ppu": models_ppu,
            "models_q80": models_q80,
            "blend_main_weight": blend_main_weight,
            "nn_k": nn_k,
        }, f)
    np.save(os.path.join(out_dir, "oof_preds.npy"), oof_preds)
    np.save(os.path.join(out_dir, "test_preds_folds.npy"), np.vstack(test_preds_folds))
    np.save(os.path.join(out_dir, "test_preds.npy"), test_preds_agg)

    # Postprocessing + Blending with PPU head
    print("[step] postprocessing predictions...")
    # Initial blend before postprocessing
    preblend = blend_main_weight * test_preds_agg + (1.0 - blend_main_weight) * test_ppu_agg
    p_low, p_high = np.percentile(train_prices, [1, 99])
    test_preds_clip = np.clip(preblend, p_low, p_high)

    # cluster smoothing on embedding space
    n_clusters = min(400, max(10, len(train_df) // 200))
    clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=4096)
    clustering.fit(emb_for_knn_train)
    train_clusters = clustering.predict(emb_for_knn_train)
    cluster_medians = {}
    for c in range(n_clusters):
        idxs = np.where(train_clusters == c)[0]
        cluster_medians[c] = np.median(train_prices[idxs]) if len(idxs) else np.median(train_prices)
    test_clusters = clustering.predict(emb_for_knn_test)
    cluster_median_test = np.array([cluster_medians[c] for c in test_clusters])
    alpha = 0.85
    test_preds_smooth = alpha * test_preds_clip + (1 - alpha) * cluster_median_test

    # quantile residual correction using OOF
    oof_pred_values = oof_preds
    q_bins = 10
    bins = pd.qcut(oof_pred_values, q_bins, labels=False, duplicates='drop')
    df_oof = pd.DataFrame({"pred": oof_pred_values, "actual": train_prices, "bin": bins})
    bin_median_resid = df_oof.groupby("bin").apply(lambda x: np.median(x["actual"] - x["pred"]))
    bin_median_resid = bin_median_resid.to_dict()

    quantile_edges = np.quantile(oof_pred_values, np.linspace(0, 1, q_bins + 1))
    test_bins = np.digitize(test_preds_smooth, quantile_edges) - 1
    test_bins = np.clip(test_bins, 0, q_bins - 1)
    test_bias = np.array([bin_median_resid.get(int(b), 0.0) for b in test_bins])
    final_preds = np.clip(test_preds_smooth + test_bias, 0.01, None)

    # Upward guard with quantile head (ensures we don't miss high-priced tail)
    if 'test_q80_agg' in locals():
        guard = 0.85 * test_q80_agg + 0.15 * test_preds_clip
        final_preds = np.maximum(final_preds, guard)

    # Save final CSV
    out_csv = os.path.join(out_dir, "test_out_improved.csv")
    pd.DataFrame({"sample_id": test_df["sample_id"], "price": final_preds}).to_csv(out_csv, index=False)
    print(f"[done] saved predictions to {out_csv}")
    print(f"Pred range: {final_preds.min():.2f} - {final_preds.max():.2f} | mean: {final_preds.mean():.2f}")

    return {
        "oof_smape": total_smape,
        "oof_mae": total_mae,
        "oof_smape_blend": total_smape_blend,
        "oof_mae_blend": total_mae_blend,
        "out_csv": out_csv,
    }


def eval_on_sample(sample_csv: str, sample_out: str, model_pkl: str, scaler: StandardScaler, tfidf, svd, sbert_model=None, emb_for_knn_train: Optional[np.ndarray]=None, train_prices: Optional[np.ndarray]=None, blend_main_weight: float = 0.7, nn_k: int = 10, use_sbert_flag: bool = False, sbert_model_name: str = 'all-MiniLM-L6-v2'):
    if not (os.path.exists(sample_csv) and os.path.exists(sample_out)):
        print("[info] sample files not found; skipping sample evaluation.")
        return None
    print("[info] evaluating on sample test set...")
    df = pd.read_csv(sample_csv)
    feat = build_features(df, is_train=False)
    texts = focused_text_series(df)
    tfidf_mat = tfidf.transform(texts)
    X_svd = svd.transform(tfidf_mat)
    parts = [feat.values, X_svd]
    X_sbert = None
    # Instantiate SBERT if requested and not provided
    if (sbert_model is not None) or use_sbert_flag:
        try:
            if sbert_model is None and use_sbert_flag and _HAS_SBERT:
                sbert_model = SentenceTransformer(sbert_model_name)
            if sbert_model is not None:
                X_sbert = sbert_model.encode(texts, show_progress_bar=False)
        except Exception as _e:
            print(f"[warn] SBERT inference failed, continuing without: {_e}")
        parts.append(X_sbert)
    # compute kNN price feature for sample to match training features
    if emb_for_knn_train is not None and train_prices is not None:
        emb_for_knn_sample = X_sbert if X_sbert is not None else X_svd
        nn_price_sample = build_nn_price(emb_for_knn_train, train_prices.astype(float), emb_for_knn_sample, k=nn_k)
        parts.append(nn_price_sample.reshape(-1, 1))
    X = np.hstack(parts)
    X_scaled = scaler.transform(X)

    with open(model_pkl, "rb") as f:
        data = pickle.load(f)
    models = data["models"]
    models_ppu = data.get("models_ppu", [])
    preds_main = np.zeros(len(X_scaled))
    for m in models:
        preds_main += np.expm1(m.predict(X_scaled, num_iteration=getattr(m, "best_iteration", None)))
    preds_main /= max(1, len(models))
    # PPU head
    preds_ppu = None
    if models_ppu:
        preds_ppu_log = np.zeros(len(X_scaled))
        for m in models_ppu:
            preds_ppu_log += m.predict(X_scaled, num_iteration=getattr(m, "best_iteration", None))
        preds_ppu_log /= max(1, len(models_ppu))
        # reconstruct unit_base for sample
        feat_df = feat  # already computed
        pack = feat_df["pack_qty"].values.astype(float)
        vol = feat_df["volume_ml"].values.astype(float)
        wt = feat_df["weight_g"].values.astype(float)
        base = np.where(vol > 0, vol, np.where(wt > 0, wt, 1.0))
        unit_base = pack * np.clip(base, 1e-6, None)
        preds_ppu = np.expm1(preds_ppu_log) * unit_base
    # blend
    if preds_ppu is not None:
        preds = blend_main_weight * preds_main + (1.0 - blend_main_weight) * preds_ppu
    else:
        preds = preds_main
    preds = np.clip(preds, 0.01, None)

    truth = pd.read_csv(sample_out)
    merged = pd.DataFrame({"sample_id": df["sample_id"], "pred": preds}).merge(truth, on="sample_id")
    mae = mean_absolute_error(merged["price"], merged["pred"])
    s = smape(merged["price"], merged["pred"])
    print(f"[sample] MAE: {mae:.4f} | SMAPE: {s:.2f}%")
    return {"mae": mae, "smape": s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--test_csv", default="dataset/test.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--use_sbert", action="store_true")
    parser.add_argument("--tfidf_max", type=int, default=20000)
    parser.add_argument("--svd_n", type=int, default=150)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--debug_nrows", type=int, default=None)
    parser.add_argument("--nn_k", type=int, default=10)
    parser.add_argument("--num_boost_round", type=int, default=6000)
    parser.add_argument("--early_stopping_rounds", type=int, default=300)
    parser.add_argument("--blend_main_weight", type=float, default=0.7)
    parser.add_argument("--sample_csv", default="dataset/sample_test.csv")
    parser.add_argument("--sample_out", default="dataset/sample_test_out.csv")
    args = parser.parse_args()

    res = train_and_predict(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        tfidf_max=args.tfidf_max,
        svd_n=args.svd_n,
        use_sbert=args.use_sbert,
        n_splits=args.n_splits,
        debug_nrows=args.debug_nrows,
        nn_k=args.nn_k,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        blend_main_weight=args.blend_main_weight,
    )

    # Sample eval if artifacts available
    model_pkl = os.path.join(args.out_dir, "models_lgb.pkl")
    if os.path.exists(model_pkl):
        try:
            with open(model_pkl, "rb") as f:
                data = pickle.load(f)
            eval_on_sample(
                sample_csv=args.sample_csv,
                sample_out=args.sample_out,
                model_pkl=model_pkl,
                scaler=data["scaler"],
                tfidf=data["tfidf"],
                svd=data["svd"],
                sbert_model=None,  # re-instantiate if needed
                emb_for_knn_train=data.get("emb_for_knn_train"),
                train_prices=data.get("train_prices"),
                blend_main_weight=data.get("blend_main_weight", 0.7),
                nn_k=int(data.get("nn_k", 10)),
                use_sbert_flag=bool(data.get("use_sbert", False)),
                sbert_model_name=str(data.get("sbert_model_name", 'all-MiniLM-L6-v2')),
            )
        except Exception as e:
            print(f"[warn] sample evaluation failed: {e}")

    print(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
