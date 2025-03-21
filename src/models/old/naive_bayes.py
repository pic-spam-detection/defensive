from typing import Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB

from models.base_model import BaseModel
from models.vectorizer import Vectorizer


class NaiveBayes(BaseModel):
    def __init__(
        self,
        dataset: Optional[List[Dict[str, str]]] = None,
        checkpoint_path: Optional[str] = None,
        vectorizer_type: Literal["sklearn", "bert"] = "sklearn",
        vectorizer_checkpoint_path: Optional[str] = None,
    ):
        self.vectorizer_type = vectorizer_type
        self.vectorizer = Vectorizer(
            type=vectorizer_type, checkpoint_path=vectorizer_checkpoint_path
        )

        # load or init the classifier
        if checkpoint_path:
            try:
                self.classifier = joblib.load(
                    checkpoint_path + "classifier_bayes.joblib"
                )
            except FileNotFoundError:
                print("No classifier checkpoint found. Initializing a new classifier.")
                self.classifier = MultinomialNB()
        else:
            self.classifier = MultinomialNB()

        # train the model if dataset is provided
        if dataset is not None:
            self.train_model(dataset)

    def train(
        self, dataset: List[Dict[str, str]], save_path: Optional[str] = None
    ) -> None:
        """Train the NaiveBayes classifier"""
        # preprocess dataset
        # TODO: preprocessing should be done outside of the model
        mails_df = pd.DataFrame(dataset).dropna()
        mails_df["text"] = mails_df["subject"] + " " + mails_df["body"]

        # prepare training data
        X_train = mails_df["text"]
        y_train = mails_df["ground_truth"]

        # vectorize
        if self.vectorizer_type == "sklearn":
            self.vectorizer.vectorizer.fit_transform(X_train)
        X_train = self.vectorizer(X_train)

        # train
        self.classifier.fit(X_train, y_train)

        # save checkpoint and vectorizer
        if self.vectorizer_type == "sklearn" and save_path is not None:
            joblib.dump(self.classifier, save_path + "classifier_bayes.joblib")
            joblib.dump(self.vectorizer, save_path + "vectorizer_sklearn.joblib")

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

        # preprocess text for classification
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
