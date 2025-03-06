# from spam_classifier import SpamClassifier
from models.naive_bayes import NaiveBayes
from utils.dataset import get_dataset
import argparse

"""
Mail formatting :

[address, domain, domain_extension, subject, body, ground_truth]
[string, string, string, string, string, int]

Example : ["alice.bob", "gmail", "com", "Hello", "I am a spam mail", 1]

"""

MODELS = {
    # "spam_classifier": SpamClassifier,
    "naive_bayes" : NaiveBayes
}


def is_spam(mail, model=None):
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
        choices=MODELS.keys(),
        help="Models available: " + ", ".join(MODELS.keys()),
    )

    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="The subject of the email."
    )

    parser.add_argument(
        "--body",
        type=str,
        required=True,
        help="The body of the email."
    )

    parser.add_argument(
        "--address",
        type=str,
        required=False,
        help="The address of the email.",
        default=""
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=False,
        help="The domain of the email.",
        default=""
    )

    parser.add_argument(
        "--domain_extension",
        type=str,
        required=False,
        help="The domain extension of the email.",
        default=""
    )

    parser.add_argument(
        "--ground_truth",
        type=int,
        required=False,
        help="The ground truth of the email. 1 if spam, 0 otherwise.",
        default=0
    )

    parser.add_argument(
        "--train",
        type=bool,
        required=False,
        help="Train the model.",
        default=False
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
        "ground_truth": 0
    }

    if args.model == "naive_bayes":
        model = MODELS[args.model](dataset=get_dataset()["train"])
    else:
        model = MODELS[args.model]()

    # # Ask for a spam and a model to the user in the terminal
    # mail = input("Enter the mail to check : ")

    print(is_spam(mail, model))
