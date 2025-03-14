from src.models.base_model import BaseModel
from src.models.classical_ml_classifier import ClassicalMLClassifier
from typing import Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from src.utils.const import CLASSICAL_ML_MODELS


class Vote(BaseModel):
    def __init__(
        self,
        models_type: List[BaseModel] = CLASSICAL_ML_MODELS,
        checkpoint_paths: Optional[List[str]] = None,
        threshold: float = 0.5,
        use_meta_features: bool = False,
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
        self.use_meta_features = use_meta_features
        self.meta_classifier = None

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
        if not self.use_meta_features:
            return self.votes(X, self.models)
        else:
            meta_features = []
            for model in self.models:
                model_preds = model.predict(X)
                if isinstance(model_preds, torch.Tensor):
                    model_preds = model_preds.numpy()
                meta_features.append(model_preds.reshape(-1, 1))

            X_meta = np.hstack(meta_features)

            predictions = self.meta_classifier.predict(X_meta)
            return torch.tensor(predictions, dtype=torch.int)

    def train(
        self, dataloader, save_path: Optional[str] = None
    ) -> Dict[str, BaseModel]:
        """Train the model on the provided dataset"""
        # Train each base model
        for model in self.models:
            model.train(dataloader, save_path)

        if self.use_meta_features:
            # Extract training data from dataloader batches
            X_train_list, y_train_list = [], []
            for X_batch, y_batch in dataloader:
                X_train_list.append(X_batch)
                y_train_list.append(y_batch)

            # Convert lists to tensors
            X_data = torch.cat(X_train_list)
            y_data = torch.cat(y_train_list)

            # Generate meta-features (predictions from base models)
            meta_features = []
            for model in self.models:
                # Get predictions from this model
                model_preds = model.predict(X_data)
                if isinstance(model_preds, torch.Tensor):
                    model_preds = model_preds.numpy()

                # Ensure we have a 2D array with shape (n_samples, 1)
                if len(model_preds.shape) == 1:
                    model_preds = model_preds.reshape(-1, 1)

                meta_features.append(model_preds)

            # Combine all model predictions into a feature matrix
            X_meta = np.hstack(meta_features)
            y_meta = y_data.numpy()  # Don't repeat labels

            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_meta, y_meta, test_size=0.2, random_state=42
            )

            # Normalize meta-features
            # Train meta-classifier, penalize false positives more
            self.meta_classifier = LogisticRegression(C=1.0, class_weight="balanced")
            self.meta_classifier.fit(X_train, y_train)

            # Save meta-classifier if requested
            if save_path:
                joblib.dump(self.meta_classifier, save_path + "meta_classifier.joblib")
                joblib.dump(self.scaler, save_path + "meta_scaler.joblib")

            # Evaluate meta-classifier
            y_pred = self.meta_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Meta-classifier accuracy: {accuracy}")

    def get_model(self):
        return super().get_model()
