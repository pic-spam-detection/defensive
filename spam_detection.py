# from spam_classifier import SpamClassifier
import argparse
from typing import List, Optional

from src.models.base_model import BaseModel
from src.models.classical_ml_classifier import ClassicalMLClassifier
from dataset.dataset import get_dataset

"""
Mail formatting :

[address, domain, domain_extension, subject, body, ground_truth]
[string, string, string, string, string, int]

Example : ["alice.bob", "gmail", "com", "Hello", "I am a spam mail", 1]

"""


MODELS = ["naive_bayes", "logistic_regression", "svm", "keywords"]


def is_spam(mail: List[str], model: Optional[BaseModel]):
    """
    Main function used to detect spam mails

    :param list(string, string, string, string, string, int) mail: list containing the mail information
    :param model: model used to detect spam mails
    :return: 1 if the mail is a spam, 0 otherwise and -1 if there was an error
    """
    if model is None:
        return -1

    response = model.classify(mail)
    x = "Spam" if response else "Ham"
    print("Response : ", x)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple command line spam email detector."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS,
        help="Models available: " + ", ".join(MODELS),
    )

    parser.add_argument(
        "--subject", type=str, required=True, help="The subject of the email."
    )

    parser.add_argument(
        "--body", type=str, required=True, help="The body of the email."
    )

    parser.add_argument(
        "--address",
        type=str,
        required=False,
        help="The address of the email.",
        default="",
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=False,
        help="The domain of the email.",
        default="",
    )

    parser.add_argument(
        "--domain_extension",
        type=str,
        required=False,
        help="The domain extension of the email.",
        default="",
    )

    parser.add_argument(
        "--ground_truth",
        type=int,
        required=False,
        help="The ground truth of the email. 1 if spam, 0 otherwise.",
        default=0,
    )

    parser.add_argument(
        "--train", type=bool, required=False, help="Train the model.", default=True
    )
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        parser.error(str(e))

    mail = {
        "address": args.address,
        "domain": args.domain,
        "domain_extension": args.domain_extension,
        "subject": args.subject,
        "body": args.body,
        "ground_truth": 0,
    }

    if args.model in ["naive_bayes", "logistic_regression", "keywords"]:
        model = ClassicalMLClassifier(args.model)
    else:
        raise ValueError("NN based models are not yet available")

    if args.train:
        model.train(dataset=get_dataset()["train"])

    # # Ask for a spam and a model to the user in the terminal
    # mail = input("Enter the mail to check : ")

    print(is_spam(mail, model))
