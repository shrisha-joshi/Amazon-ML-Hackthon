# MIT License
# src/models/base_model.py
from abc import ABC, abstractmethod
import os
import joblib
import torch

class BaseModel(ABC):
    def __init__(self, model_name: str, save_dir: str):
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @abstractmethod
    def train(self, *args, **kwargs): 
        pass

    @abstractmethod
    def predict(self, *args, **kwargs): 
        pass

    def save(self):
        path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        if hasattr(self, 'state_dict'):
            torch.save(self.state_dict(), path)
        return path

    def load(self, path=None):
        if not path:
            path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        if hasattr(self, 'load_state_dict'):
            self.load_state_dict(torch.load(path))