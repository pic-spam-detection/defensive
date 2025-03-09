from typing import Any, List, Literal, Optional, Union

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer


class Vectorizer:
    def __init__(
        self,
        type: Literal["sklearn", "bert"],
        checkpoint_path: Optional[str] = None,
    ):
        self.type = type
        self.vectorizer = None

        if self.type == "sklearn":
            self._init_sklearn(checkpoint_path)
        elif self.type == "bert":
            self._init_bert()
        else:
            raise ValueError("Invalid vectorizer type. Choose 'sklearn' or 'bert'")

    def __call__(self, X: Union[List[str], Any]) -> Any:
        return self.vectorize(X)

    def _init_sklearn(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path:
            self.vectorizer = joblib.load(checkpoint_path + "vectorizer_sklearn.joblib")
        else:
            self.vectorizer = CountVectorizer()

    def _init_bert(self):
        self.vectorizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def vectorize(self, X: Union[List[str], Any]) -> Any:
        if self.type == "sklearn":
            return self.vectorizer.transform(X)

        elif self.type == "bert":
            if not isinstance(X, list):
                X = X.tolist()
            inputs = self.vectorizer(
                X,
                return_tensors="pt",
                truncation=True,
                max_length=10000,
                padding="max_length",
            )
            return inputs["input_ids"].reshape(inputs["input_ids"].shape[0], -1)

        else:
            raise ValueError("Invalid vectorizer type. Choose 'sklearn' or 'bert'")
