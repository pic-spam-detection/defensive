from src.models.base_model import BaseModel
from src.models.classical_ml_classifier import ClassicalMLClassifier
from typing import Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch

from src.utils.const import CLASSICAL_ML_MODELS


class Vote(BaseModel):
    def __init__(
        self,
        models_type: List[BaseModel] = CLASSICAL_ML_MODELS,
        checkpoint_paths: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.models = [
            ClassicalMLClassifier(
                classifier_type=model,
                checkpoint_path=(
                    checkpoint_paths[index] if checkpoint_paths is not None else None
                ),
            )
            for index, model in enumerate(models_type)
        ]
        self.threshold = threshold

    def vote(self, mail: List[str], models: List[BaseModel]):
        """
        Main function used to detect spam mails

        :param list(string, string, string, string, string, int) mail: list containing the mail information
        :param model: model used to detect spam mails
        :return: 1 if the mail is a spam, 0 otherwise and -1 if there was an error
        """
        if models is None:
            return -1
        results = [model.classify(mail).reshape(1, -1)[0] for model in models]
        spam_votes = np.sum(results)
        return 1 if spam_votes / len(models) > self.threshold else 0

    def votes(self, mail, models: List[BaseModel]):
        """
        Main function used to detect spam mails

        :param tensor mail: tensor containing the mails already vectorized
        :param model: model used to detect spam mails
        :return:
        """
        if models is None:
            return -1
        results = [model.predict(mail) for model in models]
        spam_votes = np.sum(results, axis=0)
        return torch.Tensor(spam_votes / len(models) > self.threshold).int()

    def classify(self, mail: List[str]) -> int:
        return self.vote(mail, self.models)

    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict labels for a list of texts"""
        return self.votes(X, self.models)

    def train(
        self, dataset: List[Dict[str, str]], save_path: Optional[str] = None
    ) -> Dict[str, BaseModel]:
        """Train the model on the provided dataset"""
        for model in self.models:
            model.train(dataset, save_path)

    def get_model(self):
        return super().get_model()
