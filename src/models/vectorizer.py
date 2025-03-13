import os
from typing import Any, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
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

    def __call__(
        self,
        X_train: Union[List[str], Any],
        X_test: Union[List[str], Any],
        save_folder: Optional[str] = None,
    ) -> Any:
        train_embeddings, test_embeddings = self._vectorize(X_train, X_test)

        if save_folder is not None:
            self._save_embeddings(
                os.path.join(save_folder, "train.pt"), train_embeddings
            )
            self._save_embeddings(os.path.join(save_folder, "test.pt"), test_embeddings)

        return train_embeddings, test_embeddings

    def _init_sklearn(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path:
            self.vectorizer = joblib.load(checkpoint_path + "vectorizer_sklearn.joblib")
        else:
            self.vectorizer = CountVectorizer(max_features=5000)

    def _init_bert(self):
        self.vectorizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

    def _vectorize(
        self, X_train: Union[List[str], Any], X_test: Union[List[str], Any]
    ) -> Tuple[Any, Any]:
        if self.type == "sklearn":
            train_embeddings = self.vectorizer.fit_transform(X_train)
            test_embeddings = self.vectorizer.transform(X_test)

            train_embeddings = torch.tensor(
                train_embeddings.toarray(), dtype=torch.float32
            )
            test_embeddings = torch.tensor(
                test_embeddings.toarray(), dtype=torch.float32
            )
        else:
            train_embeddings = self._run_bert(X_train)
            test_embeddings = self._run_bert(X_test)

        return train_embeddings, test_embeddings

    def _save_embeddings(
        self,
        save_path: str,
        X: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Save embeddings to disk"""
        torch.save(X, save_path)

    def _run_bert(self, X: Union[List[str], Any]) -> torch.Tensor:
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
            cls_embeddings = word_embeddings[:, 0, :]  # use CLS token as the embedding
            embeddings.append(cls_embeddings.cpu())

        return torch.stack(embeddings, dim=0).squeeze()

    def get_vocab_size(self) -> int:
        if self.type == "sklearn":
            return len(self.vectorizer.vocabulary_)
        return self.vectorizer.vocab_size
