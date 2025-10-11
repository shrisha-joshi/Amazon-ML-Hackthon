# MIT License
# src/data/embeddings_image_clip.py
"""
Compute image embeddings using CLIP (preferred) or ResNet50 (fallback).
Saves embeddings into outputs/feature_store/embeddings_image/
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from src.config import Config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("embeddings_image")
cfg = Config()
set_seed(cfg.seed)

# try CLIP
_HAS_CLIP = True
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except Exception:
    _HAS_CLIP = False

# fallback: torchvision resnet
_HAS_RESNET = True
try:
    import torch
    from torchvision import models, transforms
except Exception:
    _HAS_RESNET = False

def _ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _read_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def embed_with_clip(image_paths: List[str], batch_size: int = 64, device: str = "cuda"):
    assert _HAS_CLIP, "CLIP not installed"
    model = CLIPModel.from_pretrained(cfg.image_model_name).to(device)
    proc = CLIPProcessor.from_pretrained(cfg.image_model_name)
    model.eval()
    embs=[]
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP batches"):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = _read_image(p)
            if img is None:
                imgs.append(Image.new("RGB", (224,224), color=(255,255,255)))
            else:
                imgs.append(img)
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            out = out.cpu().numpy()
        embs.append(out)
    return np.vstack(embs)

def embed_with_resnet(image_paths: List[str], batch_size:int=64, device: str="cuda"):
    assert _HAS_RESNET, "ResNet not available"
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    embs=[]
    for i in tqdm(range(0, len(image_paths), batch_size), desc="ResNet batches"):
        batch_paths = image_paths[i:i+batch_size]
        tensors=[]
        for p in batch_paths:
            img = _read_image(p)
            if img is None:
                tensors.append(torch.zeros(3,224,224))
            else:
                tensors.append(preprocess(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch).squeeze(-1).squeeze(-1)
            embs.append(out.cpu().numpy())
    return np.vstack(embs)

def build_image_embeddings(image_dir: str, ids: List[str], out_dir: str, mode: str = "clip"):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = [os.path.join(image_dir, f"{i}.jpg") for i in ids]
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if mode == "clip" and _HAS_CLIP:
        log.info("[AI-Agent] Using CLIP for image embeddings")
        embs = embed_with_clip(image_paths, device=device)
    else:
        log.info("[AI-Agent] Using ResNet50 for image embeddings")
        embs = embed_with_resnet(image_paths, device=device)
    out_path = os.path.join(out_dir, "image_embeddings.npy")
    np.save(out_path, embs)
    log.info(f"[AI-Agent] Saved image embeddings to {out_path}")
    return out_path