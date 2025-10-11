# MIT License
# src/data/embeddings_text_lora.py
"""
Train/Load LoRA-tuned DistilBERT and extract text embeddings.
Two modes:
- fast: use sentence-transformers (all-MiniLM-L6-v2)
- lora: fine-tune DistilBERT with PEFT (LoRA) for small epochs and extract CLS/mean embeddings
"""

from __future__ import annotations
import os
import numpy as np
import torch
from dataclasses import dataclass
from src.config import Config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("embeddings_text_lora")
cfg = Config()

try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import datasets
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

HAS_SBERT = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    HAS_SBERT = False

@dataclass
class LoRAConfig:
    model_name: str = cfg.text_model_name
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    num_epochs: int = 3
    batch_size: int = 16
    max_length: int = cfg.max_text_length
    device: str = cfg.device

class TextEmbedder:
    def __init__(self, use_lora: bool = False, model_dir: str = None):
        set_seed(cfg.seed)
        self.use_lora = use_lora and _HAS_TRANSFORMERS
        self.model_dir = model_dir or cfg.model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.lora_config = LoRAConfig()

    def fast_embed(self, texts: list[str], batch_size: int = 256):
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed for fast embedding.")
        log.info("Using sentence-transformers (fast) to build embeddings.")
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        embs = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
        return embs

    def fine_tune_lora(self, train_texts: list[str], train_targets: np.ndarray, output_dir: str = None):
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers/peft not available for LoRA training.")
        output_dir = output_dir or os.path.join(self.model_dir, "lora_text")
        os.makedirs(output_dir, exist_ok=True)
        # build datasets
        ds = datasets.Dataset.from_dict({"text": train_texts, "label": train_targets.tolist()})
        tokenizer = AutoTokenizer.from_pretrained(self.lora_config.model_name)
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=self.lora_config.max_length)
        ds = ds.map(tokenize, batched=True)
        ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

        model = AutoModelForSequenceClassification.from_pretrained(self.lora_config.model_name, problem_type="regression", num_labels=1)
        # prepare LoRA
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=self.lora_config.dropout,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, peft_config)

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.lora_config.batch_size,
            num_train_epochs=self.lora_config.num_epochs,
            learning_rate=3e-4,
            logging_steps=100,
            save_strategy="epoch",
            evaluation_strategy="no",
            fp16=torch.cuda.is_available(),
            save_total_limit=2
        )
        data_collator = DataCollatorWithPadding(tokenizer)
        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
        log.info("Starting LoRA fine-tune (this may take time if dataset large).")
        trainer.train()
        trainer.save_model(output_dir)
        log.info(f"[AI-Agent] LoRA model saved to {output_dir}")
        return output_dir

    def extract_from_lora(self, model_dir: str, texts: list[str], batch_size: int = 64):
        # load model and tokenizer and do mean pooling on last_hidden_state
        from transformers import AutoTokenizer, AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir).to(device)
        model.eval()
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                toks = tokenizer(batch, padding=True, truncation=True, max_length=self.lora_config.max_length, return_tensors="pt")
                toks = {k:v.to(device) for k,v in toks.items()}
                out = model(**toks)
                # mean pooling
                last = out.last_hidden_state  # (B, L, D)
                mask = toks["attention_mask"].unsqueeze(-1).expand(last.size()).float()
                summed = (last * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                pooled = (summed / counts).cpu().numpy()
                embs.append(pooled)
        embs = np.vstack(embs)
        return embs

    def embed(self, texts: list[str], mode: str = "fast", **kwargs):
        if mode == "fast":
            return self.fast_embed(texts, **kwargs)
        elif mode == "lora":
            # expects that fine-tune was run and model stored at model_dir/lora_text
            model_dir = os.path.join(self.model_dir, "lora_text")
            if not os.path.exists(model_dir):
                raise RuntimeError("LoRA model not found. Run fine_tune_lora first.")
            return self.extract_from_lora(model_dir, texts, **kwargs)
        else:
            raise ValueError("mode must be 'fast' or 'lora'")