# Smart Product Pricing Challenge – Methodology (Team AI-Agent)

## 1. Overview
We developed a hybrid multimodal ensemble model combining text, image, and graph representations to predict product prices with high accuracy. Our approach leverages advanced parameter-efficient fine-tuning (LoRA) and graph neural networks to capture complex product relationships and semantic information.

## 2. Feature Engineering
- **LoRA-enhanced DistilBERT**: Fine-tuned transformer for textual encoding with parameter-efficient LoRA adapters targeting query and value projections (8-rank, 16-alpha configuration)
- **CLIP-based visual embeddings**: Aligned text-image representations using OpenAI CLIP, with ResNet50 fallback for robustness
- **IPQ (Individual Pack Quantity) parsing**: Robust regex-based extraction of packaging information (pack size, unit quantity, units) with standardized unit mapping
- **GraphSAGE**: k-NN similarity graph (k=10) built on combined embeddings to capture product relationship structure
- **Tabular features**: Text statistics, image presence indicators, and engineered price-per-unit ratios

## 3. Modeling Architecture
Four specialized base learners combined through meta-ensemble:

1. **LightGBM**: Gradient boosting on tabular + concatenated embedding features
2. **Multi-layer Perceptron**: Deep neural network (512→256→128→1) with batch normalization and dropout
3. **Graph Neural Network**: 2-layer GraphSAGE with neighbor aggregation and mean pooling
4. **LoRA Text Model**: Fine-tuned DistilBERT with cross-validation and early stopping

**Meta-Ensemble**: ElasticNet regression with 5-fold cross-validation for optimal base model weight learning, optimized directly on SMAPE metric.

## 4. Training Strategy
- **Cross-validation**: Stratified 5-fold with consistent random seeds (42) for reproducibility
- **Target transformation**: Log1p transform for LGB/MLP, original scale for transformer models
- **Early stopping**: Validation SMAPE monitoring with patience-based stopping
- **Regularization**: Dropout (0.2), L1/L2 penalties, and LoRA rank constraints for generalization

## 5. Evaluation & Optimization
**Primary Metric**: Symmetric Mean Absolute Percentage Error (SMAPE)
```
SMAPE = (1/n) * Σ(|y_true - y_pred| / ((|y_true| + |y_pred|)/2)) * 100%
```

**Post-processing**: Outlier clipping to 99th percentile, calibration to training distribution mean, and positive price enforcement.

## 6. Implementation & Constraints
- **Fair Play Compliance**: No external pricing data, web scraping, or market information used
- **Model Efficiency**: Total parameters < 8B (LoRA: ~0.5M, GNN: ~1M, satisfied easily)
- **Reproducibility**: Fixed seeds across Python, NumPy, PyTorch, and LightGBM
- **License**: MIT License with proper attribution in all source files

## 7. Technical Innovation
- **Parameter-Efficient Fine-tuning**: LoRA adapters reduce trainable parameters by 99.9% while maintaining performance
- **Multimodal Graph Learning**: Novel combination of text-image embeddings in graph structure for product similarity
- **SMAPE-Optimized Ensemble**: Direct optimization of meta-weights on competition metric rather than MSE proxy

## 8. Results & Robustness
The ensemble approach provides balanced predictions across product categories with strong generalization. Each base model captures different aspects: LightGBM handles tabular patterns, transformers capture semantic content, GNNs learn market structure, and meta-ensemble optimally combines strengths while mitigating individual model weaknesses.

**System Architecture**: Production-ready with comprehensive logging, error handling, modular OOP design, and automated validation checks ensuring submission format compliance.

---
*Built with systematic prompt engineering and advanced ML practices for competition excellence.*