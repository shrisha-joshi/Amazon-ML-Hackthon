# 🧠 Smart Product Pricing ML System

A multimodal ensemble machine learning system for the Smart Product Pricing Challenge 2025, combining text, image, and graph neural networks with advanced LoRA fine-tuning and meta-ensemble techniques.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing
python src/data/preprocess_features.py --train_csv dataset/train.csv --test_csv dataset/test.csv --out_dir outputs/feature_store

# 3. Generate embeddings (fast mode)
python -c "
from src.data.embeddings_text_lora import TextEmbedder
from src.data.feature_store import FeatureStore
import numpy as np, os
fs = FeatureStore()
train = fs.load_train(); test = fs.load_test()
te = TextEmbedder(use_lora=False)
os.makedirs('outputs/feature_store/embeddings_text/train', exist_ok=True)
os.makedirs('outputs/feature_store/embeddings_text/test', exist_ok=True)
np.save('outputs/feature_store/embeddings_text/train/text_embeddings.npy', te.embed(train['catalog_clean'].tolist(), mode='fast'))
np.save('outputs/feature_store/embeddings_text/test/text_embeddings.npy', te.embed(test['catalog_clean'].tolist(), mode='fast'))
print('[AI-Agent] Saved text embeddings (fast) for train/test.')
"

# 4. Train models
python src/train/train_lgb.py

# 5. Generate final predictions
python src/inference/predict.py
```

## 🏗️ Architecture

### Hybrid Multimodal Ensemble:

1. **LoRA-DistilBERT** (Text Processing)
   - Fine-tuned with LoRA adapters for parameter efficiency
   - Processes product catalog descriptions
   - Extracts semantic and quantity information

2. **CLIP/ResNet** (Image Processing)
   - CLIP for aligned text-image embeddings (preferred)
   - ResNet50 fallback for visual feature extraction
   - Handles product images and packaging information

3. **GraphSAGE** (Graph Neural Network)
   - Builds k-NN similarity graph from combined embeddings
   - Captures product relationship and market structure
   - Learns from product similarity patterns

4. **LightGBM** (Tabular Learning)
   - Processes engineered tabular features
   - Handles IPQ (Individual Pack Quantity) parsing
   - Combines with embedding features

5. **ElasticNet Meta-Ensemble**
   - Stacks predictions from all base models
   - SMAPE-optimized blending weights
   - Provides final robust predictions

## 📁 Project Structure

```
smart_pricing_project/
├── dataset/                    # Training and test data
├── src/
│   ├── utils/                 # Core utilities (logger, SMAPE, seeds)
│   ├── data/                  # Data processing and embeddings
│   ├── models/                # Model architectures
│   ├── train/                 # Training scripts
│   ├── inference/             # Prediction pipeline
│   └── postprocess/           # Post-processing utilities
├── outputs/
│   ├── feature_store/         # Processed features and embeddings
│   ├── models/                # Trained model artifacts
│   ├── oof_predictions/       # Out-of-fold predictions
│   └── test_out.csv          # Final submission
├── docs/                      # Documentation
└── notebooks/                 # Analysis notebooks
```

## 🔬 Key Features

- **Advanced Feature Engineering**: IPQ parsing, text statistics, multimodal embeddings
- **Parameter Efficient Fine-tuning**: LoRA adapters for large language models
- **Graph-based Learning**: Product similarity networks with GraphSAGE
- **Robust Ensemble**: Meta-learning with cross-validation and SMAPE optimization
- **Production Ready**: Clean OOP design, comprehensive logging, error handling

## 📊 Model Performance

The system uses Symmetric Mean Absolute Percentage Error (SMAPE) as the primary metric:

```
SMAPE = (1/n) * Σ(|y_true - y_pred| / (|y_true| + |y_pred|)) * 100%
```

Each base model contributes through cross-validated out-of-fold predictions, combined by the meta-ensemble for optimal performance.

## ⚙️ Configuration

Key parameters in `src/config.py`:
- `seed`: Random seed for reproducibility (42)
- `num_folds`: Cross-validation folds (5)
- `device`: Computation device ("cuda" if available)
- `max_text_length`: Maximum text sequence length (128)
- `knn_k`: k-NN graph neighbors (10)

## 🔐 Fair Play Compliance

- ✅ No external price data or web scraping
- ✅ Model parameters < 8B (LoRA + GNN easily satisfied)
- ✅ MIT License, open source
- ✅ Uses only provided dataset contents
- ✅ Reproducible with fixed random seeds

## 🚀 Advanced Usage

### Custom Model Training
```bash
# Train individual models
python src/train/train_lgb.py      # LightGBM
python src/train/train_mlp.py      # Neural Network
python src/train/train_gnn.py      # Graph Neural Network
python src/train/train_lora.py     # LoRA Text Model

# Train meta-ensemble
python src/train/train_ensemble.py
```

### Embedding Generation
```bash
# Generate text embeddings with LoRA (if transformers available)
python -c "
from src.data.embeddings_text_lora import TextEmbedder
# LoRA fine-tuning and extraction code here
"

# Generate image embeddings with CLIP
python -c "
from src.data.embeddings_image_clip import build_image_embeddings
# CLIP embedding extraction code here
"
```

## 📈 Performance Monitoring

The system provides comprehensive logging:
- `[AI-Agent]` prefixed messages for key pipeline steps
- Model-specific SMAPE scores during training
- Feature importance and ensemble weights
- Validation checks and data quality metrics

## 🔧 Troubleshooting

**GPU Issues**: System gracefully falls back to CPU with appropriate logging
**Missing Dependencies**: Each module checks imports and provides clear error messages  
**Memory Issues**: Batch processing with configurable batch sizes
**Missing Images**: Placeholder generation for consistent array shapes

## 📄 License

MIT License - See full license header in each source file.

## 🏆 Competition Compliance

This system fully complies with Smart Product Pricing Challenge 2025 rules:
- Uses only provided dataset
- No external data sources
- Model size constraints satisfied
- Open source with proper licensing
- Reproducible results with seed fixing

---

**Team**: Built with advanced prompt engineering and systematic implementation following competition best practices.