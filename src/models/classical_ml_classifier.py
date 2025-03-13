from typing import Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel


def _get_classifier(
    classifier_type: Literal["naive_bayes", "logistic_regression", "svm"],
) -> BaseEstimator:
    match classifier_type:
        case "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        case "naive_bayes":
            return MultinomialNB()
        case "svm":
            return SVC(kernel="linear", random_state=42)
        case _:
            raise ValueError(
                "Invalid classifier type. Choose 'logistic_regression' or 'naive_bayes' or 'svm."
            )


class ClassicalMLClassifier(BaseModel):
    def __init__(
        self,
        classifier_type: Literal[
            "naive_bayes", "logistic_regression", "svm"
        ] = "naive_bayes",
        checkpoint_path: Optional[str] = None,
    ):
        # init classifier
        self.classifier_type = classifier_type
        self.classifier = _get_classifier(classifier_type)

        # load checkpoint if provided
        if checkpoint_path:
            self.classifier = joblib.load(checkpoint_path)

    def train(
        self,
        dataloader: DataLoader,
        save_path: Optional[str] = None,
    ):
        """Train the classifier using pre-vectorized data"""

        # extrain training data
        X_train_list, y_train_list = [], []
        for X_batch, y_batch in dataloader:
            X_train_list.append(X_batch)
            y_train_list.append(y_batch)

        # convert lists to tensors
        X_train = torch.cat(X_train_list).numpy()
        y_train = torch.cat(y_train_list).numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # normalize
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # train
        self.classifier.fit(X_train, y_train)

        if save_path:
            joblib.dump(
                self.classifier, save_path + f"classifier_{self.classifier_type}.joblib"
            )

        # evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

    def classify(self, mail: Dict[str, str]) -> Tuple[int, bool]:
        """
        Classify a single email
        Args:
            mail: dict
                mail information (address, domain, domain_extension, subject, body, ground_truth)
        """
        required_keys = {
            "address",
            "domain",
            "domain_extension",
            "subject",
            "body",
            "ground_truth",
        }
        if not required_keys.issubset(mail.keys()):
            raise ValueError(
                "Mail dictionary must contain keys: 'address', 'domain', 'domain_extension', 'subject', 'body', 'ground_truth'"
            )

        # preproces text
        text = mail["subject"] + " " + mail["body"]
        X = self.vectorizer([text])

        # predict
        pred = self.classifier.predict(X)[0]
        ground_truth = mail["ground_truth"]

        return pred, pred == ground_truth

    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict labels for a list of texts"""
        return self.classifier.predict(X)

    def get_model(self) -> BaseEstimator:
        """Returns model itself"""
        return self.classifier
