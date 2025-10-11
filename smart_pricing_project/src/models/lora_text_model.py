# MIT License
# src/models/lora_text_model.py
"""
LoRA-enhanced transformer model for text-based price prediction.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.smape_loss import SMAPELoss, smape_numpy

log = get_logger("lora_text_model")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

class LoraTextModel(BaseModel):
    def __init__(self, model_name: str = "lora_text", save_dir: str = "outputs/models",
                 base_model_name: str = "distilbert-base-uncased", max_length: int = 128):
        super().__init__(model_name, save_dir)
        
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers and peft libraries required for LoRA model")
        
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.oof_predictions = None

    def _build_model(self):
        """Build the LoRA model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model for regression
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, 
            num_labels=1,
            problem_type="regression"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"] if "distilbert" in self.base_model_name else ["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        return self.model

    def train(self, texts: list[str], y: np.ndarray, num_folds: int = 5, 
             epochs: int = 3, batch_size: int = 16, lr: float = 3e-4):
        log.info(f"[AI-Agent] Training LoRA model with {len(texts)} samples")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[AI-Agent] Device chosen: {device}")
        
        # Build model
        self._build_model()
        self.model.to(device)
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
            log.info(f"Training fold {fold + 1}/{num_folds}")
            
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Tokenize
            train_encodings = self.tokenizer(
                train_texts, 
                truncation=True, 
                padding=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            val_encodings = self.tokenizer(
                val_texts, 
                truncation=True, 
                padding=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            # Move to device
            train_encodings = {k: v.to(device) for k, v in train_encodings.items()}
            val_encodings = {k: v.to(device) for k, v in val_encodings.items()}
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            
            # Initialize model for this fold
            self._build_model()
            self.model.to(device)
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            criterion = SMAPELoss()
            
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                
                # Training step
                optimizer.zero_grad()
                outputs = self.model(**train_encodings)
                predictions = outputs.logits.squeeze()
                loss = criterion(predictions, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                # Validation step
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(**val_encodings)
                    val_predictions = val_outputs.logits.squeeze().cpu().numpy()
                    val_smape = smape_numpy(y_val, val_predictions)
                
                if val_smape < best_val_loss:
                    best_val_loss = val_smape
                    patience_counter = 0
                    oof[val_idx] = val_predictions
                else:
                    patience_counter += 1
                
                log.info(f"Epoch {epoch + 1}: train_loss={loss.item():.4f}, val_smape={val_smape:.4f}")
                
                if patience_counter >= patience:
                    log.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.oof_predictions = oof
        smape_score = smape_numpy(y, oof)
        log.info(f"[AI-Agent] LoRA Text Model OOF SMAPE: {smape_score:.4f}%")
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        # Tokenize
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = outputs.logits.squeeze().cpu().numpy()
        
        return predictions

    def save(self):
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        model_path = os.path.join(self.save_dir, f"{self.model_name}")
        os.makedirs(model_path, exist_ok=True)
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        log.info(f"[AI-Agent] Saved LoRA model to {model_path}")
        return model_path

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}")
        
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, 
            num_labels=1,
            problem_type="regression"
        )
        self.model = PeftModel.from_pretrained(base_model, path)
        
        log.info(f"[AI-Agent] Loaded LoRA model from {path}")
        return self