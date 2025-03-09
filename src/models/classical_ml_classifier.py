from typing import Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
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
        self, dataset: List[Dict[str, str]], save_path: Optional[str] = None
    ) -> float:
        """Train the classifier"""
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
            X_test = self.vectorizer(X_test)
        else:
            # for best -> just transform the data
            X_train = self.vectorizer(X_train)
            X_test = self.vectorizer(X_test)

        # train
        self.classifier.fit(X_train, y_train)

        # save checkpoint and vectorizer
        if self.vectorizer_type == "sklearn" and save_path is not None:
            joblib.dump(
                self.classifier, save_path + f"classifier_{self.classifier_type}.joblib"
            )
            joblib.dump(self.vectorizer, save_path + "vectorizer_sklearn.joblib")

        # evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        return accuracy

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
