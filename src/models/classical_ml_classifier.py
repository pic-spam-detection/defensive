import os
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
from sklearn.svm import SVC

from src.models.base_model import BaseModel
from src.models.vectorizer import Vectorizer


def _get_classifier(
    classifier_type: Literal["naive_bayes", "logistic_regression"],
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
                "Invalid classifier type. Choose 'logistic_regression' or 'naive_bayes'."
            )


class ClassicalMLClassifier(BaseModel):
    def __init__(
        self,
        classifier_type: Literal[
            "naive_bayes", "logistic_regression", "svm"
        ] = "naive_bayes",
        vectorizer_type: Literal["sklearn", "bert"] = "sklearn",
        checkpoint_path: Optional[str] = None,
        vectorizer_checkpoint_path: Optional[str] = None,
    ):
        # init classifier
        self.classifier_type = classifier_type
        self.classifier = _get_classifier(classifier_type)

        # load checkpoint if provided
        if checkpoint_path:
            try:
                self.classifier = joblib.load(
                    checkpoint_path + f"classifier_{classifier_type}.joblib"
                )
            except FileNotFoundError:
                print(
                    f"No checkpoint found for {classifier_type}. Initializing a new classifier"
                )

        # init vectorizer
        self.vectorizer_type = vectorizer_type
        self.vectorizer = Vectorizer(
            type=vectorizer_type, checkpoint_path=vectorizer_checkpoint_path
        )

    def train(
        self,
        embeddings_path: Optional[str] = None,
        dataset: Optional[List[Dict[str, str]]] = None,
        save_path: Optional[str] = None,
    ):
        """Train the classifier"""

        if embeddings_path is not None:
            # load precomputed embeddings
            X_train, X_test, y_train, y_test = self._load_embeddings(embeddings_path)

        elif dataset is not None:
            # preprocess dataset
            mails_df = pd.DataFrame(dataset)
            mails_df = mails_df.dropna(subset=["subject", "body"])
            mails_df["text"] = mails_df["subject"] + " " + mails_df["body"]

            # prepare training data
            y = mails_df["ground_truth"]
            X_train, X_test, y_train, y_test = train_test_split(
                mails_df["text"], y, test_size=0.2, random_state=42
            )

            # vectorize
            if self.vectorizer_type == "sklearn":
                X_train = self.vectorizer.vectorizer.fit_transform(X_train)
            else:
                X_train = self.vectorizer(X_train)

            X_test = self.vectorizer(X_test)

            # save embeddings
            if save_path:
                if self.vectorizer_type == "sklearn":
                    joblib.dump(
                        self.classifier,
                        save_path + f"classifier_{self.classifier_type}.joblib",
                    )
                    joblib.dump(
                        self.vectorizer, save_path + "vectorizer_sklearn.joblib"
                    )
                else:
                    self._save_embeddings(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("Provide either 'embeddings' or 'dataset'")

        # train
        self.classifier.fit(X_train, y_train)

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
        X = self.vectorizer(X)
        return self.classifier.predict(X)

    def get_model(self) -> BaseEstimator:
        """Returns model itself"""
        return self.classifier

    def _save_embeddings(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        X_test: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        y_test: Union[np.ndarray, torch.Tensor],
        save_path: str,
    ) -> None:
        """Save embeddings to disk"""
        torch.save(X_train, save_path + "X_train_embeddings.pt")
        torch.save(X_test, save_path + "X_test_embeddings.pt")
        torch.save(y_train, save_path + "y_train_embeddings.pt")
        torch.save(y_test, save_path + "y_test_embeddings.pt")

    def _load_embeddings(self, path: str) -> Tuple[
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
    ]:
        X_train = torch.load(os.path.join(path, "X_train_embeddings.pt"))
        X_test = torch.load(os.path.join(path, "X_test_embeddings.pt"))
        y_train = torch.load(os.path.join(path, "y_train_embeddings.pt"))
        y_test = torch.load(os.path.join(path, "y_test_embeddings.pt"))

        return X_train, X_test, y_train, y_test
