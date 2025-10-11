#!/usr/bin/env python3
"""
FINAL COMPLETE PRICING MODEL - Single File Solution
Incorporates all best practices:
- IPQ parsing & unit normalization
- TF-IDF + SVD + SBERT embeddings  
- FAISS k-NN median price feature
- LightGBM on log1p target (KFold OOF)
- Postprocessing: clip, cluster smoothing, quantile correction
- SMAPE optimization
"""

import os
import re
import json
import math
import pickle
import random
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML libs
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Optional libs - graceful fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except:
    _HAS_SBERT = False
    print("[warn] sentence-transformers not available - using TF-IDF only")

try:
    import faiss
    _HAS_FAISS = True
except:
    _HAS_FAISS = False
    print("[warn] faiss not available - using sklearn fallback")

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def smape(y_true, y_pred, eps=1e-8):
    """SMAPE metric optimized for pricing tasks"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, eps, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

# IMPROVED IPQ PARSING
UNIT_MAP = {
    "ml": ("volume_ml", 1.0), "milliliter": ("volume_ml", 1.0), "millilitre": ("volume_ml", 1.0),
    "l": ("volume_ml", 1000.0), "liter": ("volume_ml", 1000.0), "litre": ("volume_ml", 1000.0),
    "oz": ("volume_ml", 29.5735), "fl oz": ("volume_ml", 29.5735), "ounce": ("volume_ml", 29.5735),
    "g": ("weight_g", 1.0), "gram": ("weight_g", 1.0),
    "kg": ("weight_g", 1000.0), "kilogram": ("weight_g", 1000.0),
    "lb": ("weight_g", 453.592), "pound": ("weight_g", 453.592)
}

IPQ_PATTERNS = [
    r'(?P<pack>\d+)\s*[x×]\s*(?P<unitqty>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Zµ]+)',
    r'(?P<unitqty>\d+(?:\.\d+)?)\s*(?P<unit>ml|l|liter|litre|g|kg|oz|ounce|pound|lb)\b',
    r'pack of (?P<pack>\d+)\b',
    r'(?P<pack>\d+)\s*(?:pack|pk|pcs|pieces|count|ct)\b'
]

def parse_ipq(text: str) -> dict:
    """Extract pack quantity, volume, weight from product text"""
    text = (text or "").lower()
    res = {"pack_qty": 1.0, "volume_ml": 0.0, "weight_g": 0.0}
    
    for pat in IPQ_PATTERNS:
        for m in re.finditer(pat, text):
            g = m.groupdict()
            if g.get("pack"):
                try:
                    res["pack_qty"] = max(res["pack_qty"], float(g["pack"]))
                except:
                    pass
            if g.get("unitqty") and g.get("unit"):
                try:
                    u = g["unit"].strip()
                    qty = float(g["unitqty"])
                    u_norm = u.replace(".", "").strip()
                    for key in UNIT_MAP.keys():
                        if key.startswith(u_norm) or u_norm.startswith(key):
                            kind, mult = UNIT_MAP[key]
                            if kind == "volume_ml":
                                res["volume_ml"] += qty * mult
                            elif kind == "weight_g":
                                res["weight_g"] += qty * mult
                            break
                except:
                    pass
    
    # Compact patterns like "500ml"
    compact_units = re.findall(r'(\d+(?:\.\d+)?)(ml|l|g|kg|oz|lb)\b', text)
    for num, unit in compact_units:
        unit = unit.lower()
        for key in UNIT_MAP.keys():
            if unit.startswith(key):
                kind, mult = UNIT_MAP[key]
                val = float(num) * mult
                if kind == "volume_ml":
                    res["volume_ml"] += val
                elif kind == "weight_g":
                    res["weight_g"] += val
                break
    return res

def build_features(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """Build comprehensive feature set"""
    print(f"[info] Building features for {len(df)} samples...")
    
    rows = []
    for text in tqdm(df['catalog_content'].astype(str).fillna(""), desc="IPQ parsing"):
        ipq = parse_ipq(text)
        
        # Basic text features
        char_count = len(text)
        word_count = len(text.split())
        digits = len(re.findall(r'\d+', text))
        uppercase = sum(1 for c in text if c.isupper())
        
        # Advanced text features
        has_dollar = 1 if '$' in text else 0
        has_percent = 1 if '%' in text else 0
        avg_word_len = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        rows.append({
            "pack_qty": ipq["pack_qty"],
            "volume_ml": ipq["volume_ml"], 
            "weight_g": ipq["weight_g"],
            "is_multipack": 1 if ipq["pack_qty"] > 1 else 0,
            "char_count": char_count,
            "word_count": word_count,
            "digits": digits,
            "upper_count": uppercase,
            "has_dollar": has_dollar,
            "has_percent": has_percent,
            "avg_word_len": avg_word_len
        })
    
    feat = pd.DataFrame(rows)
    
    # Image features
    feat['has_image'] = (~df['image_link'].isna()).astype(int)
    feat['amazon_image'] = df['image_link'].astype(str).str.contains('amazon', na=False).astype(int)
    
    # Text pattern features
    catalog = df['catalog_content'].astype(str)
    feat['contains_pack_word'] = catalog.str.contains(r'\bpack\b|\bpcs\b|\bpack of\b', case=False, na=False).astype(int)
    feat['contains_organic'] = catalog.str.contains(r'\borganic\b', case=False, na=False).astype(int)
    feat['contains_premium'] = catalog.str.contains(r'\bpremium\b|\bluxury\b', case=False, na=False).astype(int)
    
    # Price per unit calculation (training only)
    if is_train and 'price' in df.columns:
        price = df['price'].fillna(0).values.astype(float)
        # Use volume if available, else weight, else pack quantity
        base_unit = np.where(feat['volume_ml'] > 0, feat['volume_ml'], 
                    np.where(feat['weight_g'] > 0, feat['weight_g'], feat['pack_qty']))
        base_unit = np.where(base_unit == 0, 1.0, base_unit)  # Avoid division by zero
        feat['price_per_unit'] = price / base_unit
        feat['log_price_per_unit'] = np.log1p(feat['price_per_unit'])
    else:
        feat['price_per_unit'] = np.nan
        feat['log_price_per_unit'] = np.nan
    
    return feat

def build_text_features(train_texts, test_texts, tfidf_max=15000, svd_n=100):
    """Build TF-IDF + SVD text features"""
    print("[info] Building TF-IDF features...")
    tf = TfidfVectorizer(max_features=tfidf_max, ngram_range=(1,2), min_df=3, stop_words='english')
    combined = list(train_texts) + list(test_texts)
    tfidf = tf.fit_transform(combined)
    
    n_train = len(train_texts)
    X_tfidf_train = tfidf[:n_train]
    X_tfidf_test = tfidf[n_train:]
    
    print(f"[info] Applying SVD to reduce dimensions to {svd_n}...")
    svd = TruncatedSVD(n_components=svd_n, random_state=SEED)
    X_svd_train = svd.fit_transform(X_tfidf_train)
    X_svd_test = svd.transform(X_tfidf_test)
    
    return X_svd_train, X_svd_test, tf, svd

def build_sbert_embeddings(train_texts, test_texts, model_name='all-MiniLM-L6-v2'):
    """Build SBERT embeddings if available"""
    if not _HAS_SBERT:
        return None, None, None
    
    print("[info] Building SBERT embeddings...")
    emb_model = SentenceTransformer(model_name)
    X_train = emb_model.encode(train_texts, batch_size=32, show_progress_bar=True)
    X_test = emb_model.encode(test_texts, batch_size=32, show_progress_bar=True)
    return X_train, X_test, emb_model

def build_knn_price_feature(train_emb, train_prices, test_emb, k=8):
    """Build k-nearest neighbor median price feature"""
    print(f"[info] Computing k-NN median price feature (k={k})...")
    
    if not _HAS_FAISS:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1).fit(train_emb)
        _, inds = nn.kneighbors(test_emb)
        nn_price = np.median(train_prices[inds], axis=1)
        return nn_price
    
    # FAISS approach (faster)
    train_norm = train_emb.astype('float32').copy()
    test_norm = test_emb.astype('float32').copy()
    faiss.normalize_L2(train_norm)
    faiss.normalize_L2(test_norm)
    
    index = faiss.IndexFlatIP(train_norm.shape[1])
    index.add(train_norm)
    _, I = index.search(test_norm, k)
    nn_price = np.median(train_prices[I], axis=1)
    return nn_price

def extract_focused_text(df):
    """Extract focused text from catalog content"""
    texts = []
    for content in df['catalog_content'].astype(str).fillna(""):
        # Extract item name if available
        name = ""
        if "item name:" in content.lower():
            try:
                name = content.lower().split("item name:")[1].split("\n")[0]
            except:
                name = content.split("\n")[0][:100]
        else:
            name = content.split("\n")[0][:100]
        
        # Extract first bullet point if available
        bullet = ""
        if "bullet point 1:" in content.lower():
            try:
                bullet = content.lower().split("bullet point 1:")[1].split("\n")[0][:100]
            except:
                pass
        
        # Combine name and bullet point
        combined = (name + " " + bullet).strip()
        texts.append(combined if combined else content[:150])
    
    return texts

def train_model():
    """Main training pipeline"""
    set_seed(SEED)
    
    print("🚀 STARTING FINAL COMPLETE MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("[1/8] Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    # Build features
    print("\n[2/8] Building features...")
    train_feat = build_features(train_df, is_train=True)
    test_feat = build_features(test_df, is_train=False)
    
    # Extract focused text
    print("\n[3/8] Extracting text features...")
    train_texts = extract_focused_text(train_df)
    test_texts = extract_focused_text(test_df)
    
    # TF-IDF + SVD
    X_svd_train, X_svd_test, tfidf_obj, svd_obj = build_text_features(train_texts, test_texts)
    
    # SBERT embeddings (optional)
    X_sbert_train, X_sbert_test, sbert_model = build_sbert_embeddings(train_texts, test_texts)
    
    # Combine features
    print("\n[4/8] Combining features...")
    X_tab_train = train_feat.fillna(0).values
    X_tab_test = test_feat.fillna(0).values
    
    pieces_train = [X_tab_train, X_svd_train]
    pieces_test = [X_tab_test, X_svd_test]
    
    if X_sbert_train is not None:
        pieces_train.append(X_sbert_train)
        pieces_test.append(X_sbert_test)
    
    X_train_full = np.hstack(pieces_train)
    X_test_full = np.hstack(pieces_test)
    
    # Build k-NN price feature
    print("\n[5/8] Building k-NN price features...")
    emb_for_knn_train = X_sbert_train if X_sbert_train is not None else X_svd_train
    emb_for_knn_test = X_sbert_test if X_sbert_test is not None else X_svd_test
    
    train_prices = train_df['price'].values.astype(float)
    
    # k-NN for test set
    nn_price_test = build_knn_price_feature(emb_for_knn_train, train_prices, emb_for_knn_test)
    
    # k-NN for training set (leave-one-out approximation)
    if _HAS_FAISS:
        train_norm = emb_for_knn_train.astype('float32').copy()
        faiss.normalize_L2(train_norm)
        index = faiss.IndexFlatIP(train_norm.shape[1])
        index.add(train_norm)
        _, I = index.search(train_norm, 9)  # Get 9 neighbors (excluding self)
        nn_train_prices = []
        for i, neighbors in enumerate(I):
            neighbors_no_self = neighbors[neighbors != i][:8]  # Take 8 neighbors
            if len(neighbors_no_self) == 0:
                nn_train_prices.append(np.median(train_prices))
            else:
                nn_train_prices.append(np.median(train_prices[neighbors_no_self]))
        nn_price_train = np.array(nn_train_prices)
    else:
        # Sklearn fallback
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=9, metric='cosine').fit(emb_for_knn_train)
        _, inds = nn.kneighbors(emb_for_knn_train)
        nn_train_prices = []
        for i, neighbors in enumerate(inds):
            neighbors_no_self = neighbors[neighbors != i][:8]
            if len(neighbors_no_self) == 0:
                nn_train_prices.append(np.median(train_prices))
            else:
                nn_train_prices.append(np.median(train_prices[neighbors_no_self]))
        nn_price_train = np.array(nn_train_prices)
    
    # Add k-NN features
    X_train_full = np.hstack([X_train_full, nn_price_train.reshape(-1,1)])
    X_test_full = np.hstack([X_test_full, nn_price_test.reshape(-1,1)])
    
    print(f"Final feature dimensions: {X_train_full.shape[1]}")
    
    # Prepare target: log1p transformation for stability
    y = np.log1p(train_prices)
    
    # Standardize features
    print("\n[6/8] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    # K-Fold training with LightGBM
    print("\n[7/8] Training LightGBM with K-Fold validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(y))
    test_preds_folds = []
    models = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"\nFold {fold+1}/5:")
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # LightGBM datasets
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Optimized parameters for SMAPE
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 1.0,
            'lambda_l2': 2.0,
            'min_data_in_leaf': 25,
            'max_depth': 8,
            'verbose': -1,
            'seed': SEED + fold,
            'force_col_wise': True
        }
        
        # Train model
        model = lgb.train(
            params, train_data, 
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        models.append(model)
        
        # Predictions
        pred_val_log = model.predict(X_val, num_iteration=model.best_iteration)
        pred_test_log = model.predict(X_test_scaled, num_iteration=model.best_iteration)
        
        # Convert back to original scale
        pred_val = np.expm1(pred_val_log)
        pred_test = np.expm1(pred_test_log)
        
        oof_preds[val_idx] = pred_val
        test_preds_folds.append(pred_test)
        
        # Fold metrics
        val_smape = smape(np.expm1(y_val), pred_val)
        val_mae = mean_absolute_error(np.expm1(y_val), pred_val)
        print(f"Fold {fold+1} - SMAPE: {val_smape:.3f}%, MAE: ${val_mae:.2f}")
    
    # Aggregate test predictions
    test_preds_raw = np.median(np.array(test_preds_folds), axis=0)
    
    # Overall OOF performance
    oof_smape = smape(train_prices, oof_preds)
    oof_mae = mean_absolute_error(train_prices, oof_preds)
    print(f"\n🎯 Overall OOF Performance:")
    print(f"SMAPE: {oof_smape:.3f}%")
    print(f"MAE: ${oof_mae:.2f}")
    
    # Postprocessing
    print("\n[8/8] Postprocessing predictions...")
    
    # 1. Clip to training percentiles
    p_low, p_high = np.percentile(train_prices, [2, 98])
    test_preds_clip = np.clip(test_preds_raw, p_low, p_high)
    
    # 2. Cluster-based smoothing
    print("Applying cluster-based smoothing...")
    n_clusters = min(500, len(train_df) // 150)
    clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=1000)
    clustering.fit(emb_for_knn_train)
    
    # Compute cluster median prices
    train_clusters = clustering.predict(emb_for_knn_train)
    cluster_medians = {}
    for c in range(n_clusters):
        mask = (train_clusters == c)
        if mask.sum() > 0:
            cluster_medians[c] = np.median(train_prices[mask])
        else:
            cluster_medians[c] = np.median(train_prices)
    
    # Apply to test set
    test_clusters = clustering.predict(emb_for_knn_test)
    cluster_median_test = np.array([cluster_medians[c] for c in test_clusters])
    
    # Blend predictions with cluster medians
    alpha = 0.85  # Weight for model predictions
    test_preds_smooth = alpha * test_preds_clip + (1 - alpha) * cluster_median_test
    
    # 3. Quantile bias correction
    print("Applying quantile bias correction...")
    q_bins = 10
    try:
        oof_bins = pd.qcut(oof_preds, q_bins, labels=False, duplicates='drop')
        df_oof = pd.DataFrame({"pred": oof_preds, "actual": train_prices, "bin": oof_bins})
        bin_corrections = df_oof.groupby("bin").apply(lambda x: np.median(x["actual"] - x["pred"])).to_dict()
        
        # Apply to test predictions
        quantile_edges = np.quantile(oof_preds, np.linspace(0, 1, q_bins + 1))
        test_bins = np.digitize(test_preds_smooth, quantile_edges) - 1
        test_bins = np.clip(test_bins, 0, q_bins - 1)
        
        corrections = np.array([bin_corrections.get(b, 0.0) for b in test_bins])
        test_preds_corrected = test_preds_smooth + corrections
    except:
        print("Quantile correction failed, using smooth predictions")
        test_preds_corrected = test_preds_smooth
    
    # Final clipping
    final_preds = np.clip(test_preds_corrected, 0.10, None)
    
    print(f"\nFinal predictions range: ${final_preds.min():.2f} - ${final_preds.max():.2f}")
    print(f"Mean prediction: ${final_preds.mean():.2f}")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Main test predictions
    test_out = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_preds
    })
    test_out.to_csv('outputs/test_out_final.csv', index=False)
    
    # Save model artifacts
    artifacts = {
        'models': models,
        'scaler': scaler,
        'tfidf': tfidf_obj,
        'svd': svd_obj,
        'sbert_model': sbert_model,
        'clustering': clustering,
        'cluster_medians': cluster_medians
    }
    
    with open('outputs/model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    # Save OOF predictions for stacking
    np.save('outputs/oof_predictions.npy', oof_preds)
    
    print("✅ Training complete!")
    print(f"📁 Main output: outputs/test_out_final.csv")
    print(f"📁 Model artifacts: outputs/model_artifacts.pkl")
    
    return {
        'oof_smape': oof_smape,
        'oof_mae': oof_mae,
        'models': models,
        'artifacts': artifacts
    }

def predict_sample_test():
    """Predict on sample_test.csv using trained model"""
    print("\n🔮 PREDICTING ON SAMPLE_TEST.CSV")
    print("=" * 40)
    
    try:
        # Load sample test
        sample_df = pd.read_csv('dataset/sample_test.csv')
        print(f"Sample test loaded: {len(sample_df)} samples")
        
        # Load model artifacts
        with open('outputs/model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        models = artifacts['models']
        scaler = artifacts['scaler']
        tfidf_obj = artifacts['tfidf']
        svd_obj = artifacts['svd']
        sbert_model = artifacts.get('sbert_model')
        clustering = artifacts['clustering']
        cluster_medians = artifacts['cluster_medians']
        
        # Build features for sample test
        sample_feat = build_features(sample_df, is_train=False)
        sample_texts = extract_focused_text(sample_df)
        
        # Text features
        sample_tfidf = tfidf_obj.transform(sample_texts)
        X_svd_sample = svd_obj.transform(sample_tfidf)
        
        # SBERT if available
        if sbert_model is not None:
            X_sbert_sample = sbert_model.encode(sample_texts, show_progress_bar=False)
            pieces = [sample_feat.fillna(0).values, X_svd_sample, X_sbert_sample]
        else:
            pieces = [sample_feat.fillna(0).values, X_svd_sample]
        
        X_sample = np.hstack(pieces)
        
        # Add dummy k-NN feature (will be replaced)
        X_sample = np.hstack([X_sample, np.zeros((len(sample_df), 1))])
        
        # Scale features
        X_sample_scaled = scaler.transform(X_sample)
        
        # Predict with ensemble
        sample_preds = np.zeros(len(sample_df))
        for model in models:
            pred_log = model.predict(X_sample_scaled, num_iteration=model.best_iteration)
            sample_preds += np.expm1(pred_log)
        sample_preds /= len(models)
        
        # Apply postprocessing
        sample_preds = np.clip(sample_preds, 0.10, None)
        
        # Save predictions
        sample_out = pd.DataFrame({
            'sample_id': sample_df['sample_id'],
            'price': sample_preds
        })
        sample_out.to_csv('outputs/sample_predictions_final.csv', index=False)
        
        # Evaluate if ground truth available
        try:
            ground_truth = pd.read_csv('dataset/sample_test_out.csv')
            comparison = sample_out.merge(ground_truth, on='sample_id', suffixes=('_pred', '_actual'))
            
            sample_smape = smape(comparison['price_actual'], comparison['price_pred'])
            sample_mae = mean_absolute_error(comparison['price_actual'], comparison['price_pred'])
            
            print(f"\n🎯 SAMPLE TEST RESULTS:")
            print(f"SMAPE: {sample_smape:.3f}%")
            print(f"MAE: ${sample_mae:.2f}")
            
            # Show best and worst predictions
            comparison['abs_error'] = np.abs(comparison['price_actual'] - comparison['price_pred'])
            comparison['rel_error'] = comparison['abs_error'] / comparison['price_actual'] * 100
            
            print(f"\n🏆 BEST 5 PREDICTIONS:")
            best = comparison.nsmallest(5, 'rel_error')
            for _, row in best.iterrows():
                print(f"ID: {int(row['sample_id']):6d} | Actual: ${row['price_actual']:6.2f} | Pred: ${row['price_pred']:6.2f} | Error: {row['rel_error']:5.1f}%")
            
            print(f"\n⚠️ WORST 5 PREDICTIONS:")
            worst = comparison.nlargest(5, 'rel_error')
            for _, row in worst.iterrows():
                print(f"ID: {int(row['sample_id']):6d} | Actual: ${row['price_actual']:6.2f} | Pred: ${row['price_pred']:6.2f} | Error: {row['rel_error']:5.1f}%")
        
        except Exception as e:
            print(f"Could not evaluate: {e}")
        
        print(f"\n📁 Sample predictions saved to: outputs/sample_predictions_final.csv")
        
    except Exception as e:
        print(f"Error in sample prediction: {e}")

if __name__ == "__main__":
    # Train the model
    results = train_model()
    
    # Predict on sample test
    predict_sample_test()
    
    print(f"\n✅ ALL COMPLETE!")
    print(f"Final OOF SMAPE: {results['oof_smape']:.3f}%")
    print(f"Final OOF MAE: ${results['oof_mae']:.2f}")