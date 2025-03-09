from typing import Any, List, Literal, Optional, Union

import joblib
import torch
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from src.utils.const import DEVICE


class Vectorizer:
    def __init__(
        self,
        type: Literal["sklearn", "bert"],
        checkpoint_path: Optional[str] = None,
    ):
        self.type = type
        self.vectorizer = None
        self.model = None

        if self.type == "sklearn":
            self._init_sklearn(checkpoint_path)
        elif self.type == "bert":
            self._init_bert()
        else:
            raise ValueError("Invalid vectorizer type. Choose 'sklearn' or 'bert'")

    def __call__(self, X: Union[List[str], Any]) -> Any:
        return self._vectorize(X)

    def _init_sklearn(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path:
            self.vectorizer = joblib.load(checkpoint_path + "vectorizer_sklearn.joblib")
        else:
            self.vectorizer = CountVectorizer()

    def _init_bert(self):
        self.vectorizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

    def _vectorize(self, X: Union[List[str], Any]) -> Any:
        if self.type == "sklearn":
            return self.vectorizer.transform(X)

        if not isinstance(X, list):
            X = X.tolist()

        embeddings = []
        for email in tqdm(X):
            inputs = self.vectorizer(
                email,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            word_embeddings = outputs.last_hidden_state
            embeddings.append(word_embeddings)

        return embeddings
