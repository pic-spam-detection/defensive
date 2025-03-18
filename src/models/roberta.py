from transformers import pipeline
import torch
from typing import List, Optional

from src.models.base_model import BaseModel


class Roberta(BaseModel):
    def __init__(self):
        super().__init__()
        self.pipe = pipeline("text-classification", model="mshenoda/roberta-spam")

    def classify(self, mail: List[str]) -> int:
        return self.pipe(inputs=mail)["label"] == "LABEL_1"
    
    def predict(self, X: List[str]) -> List[int]:
        X = list(X)
        results = self.pipe(
        inputs=X, 
        batch_size=256,
        truncation=True,
        padding=True,
        max_length=512  
        )   
        return torch.Tensor([result["label"] == "LABEL_1" for result in results]).int()

    def train(self, dataset: List[str], save_path: Optional[str] = None):
        return None
    
    def get_model(self):
        return self.pipe

