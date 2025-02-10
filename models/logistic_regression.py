from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from models.vectorizer import Vectorizer


class LogisticRegressionMail:

    def __init__(self, dataset, type="sklearn", checkpoint_path=None):
        if checkpoint_path is not None:
            try:
                self.classifier = joblib.load(
                    checkpoint_path + "classifier_logistic_regression.joblib"
                )
            except:
                print("No classifier checkpoint found.")
        else:
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)

        self.vectorizer = Vectorizer(type=type, checkpoint_path=checkpoint_path)
        self.type = type

    def train(self, dataset, save_path=None):

        mails_df = pd.DataFrame(dataset)
        mails_df["text"] = mails_df["subject"] + " " + mails_df["body"]

        y = mails_df["ground_truth"]

        X_train, X_test, y_train, y_test = train_test_split(
            mails_df["text"], y, test_size=0.2, random_state=47
        )

        X_train = self.vectorizer(X_train)
        X_test = self.vectorizer(X_test)

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        if self.type == "sklearn" and save_path is not None:
            joblib.dump(
                self.classifier, save_path + "classifier_logistic_regression.joblib"
            )
            joblib.dump(self.vectorizer, save_path + "vectorizer_sklearn.joblib")

        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy} {accuracy_score(y_test, y_pred)}")
        return accuracy

    def classify(self, mail):
        """
        args:
            mail: dict
                mail information (address, domain, domain_extension, subject, body, ground_truth)

        returns:
            (boolean) prediction 0 if ham, 1 if spam, (boolean) prediction == ground_truth
        """
        try:
            adress = mail["address"]
            domain = mail["domain"]
            domain_extension = mail["domain_extension"]
            subject = mail["subject"]
            body = mail["body"]
            ground_truth = mail["ground_truth"]
        except:
            raise ValueError(
                "Mail should be dictionary like 'address', 'domain', 'domain_extension', 'subject', 'body' and 'ground_truth'"
            )

        text = mail["subject"] + " " + mail["body"]

        X = self.vectorizer(text)
        pred = self.classifier.predict(X)

        return pred, pred == ground_truth
