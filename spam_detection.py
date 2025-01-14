from spam_classifier import SpamClassifier

"""
Mail formatting :

[address, domain, domain_extension, subject, body, ground_truth]
[string, string, string, string, string, int]

Example : ["alice.bob", "gmail", "com", "Hello", "I am a spam mail", 1]

"""


def is_spam(mail, model=SpamClassifier()):
    """
    Main function used to detect spam mails

    :param list(string, string, string, string, string, int) mail: list containing the mail information
    :param model: model used to detect spam mails
    :return: 1 if the mail is a spam, 0 otherwise and -1 if there was an error
    """

    if model.model is None:
        return -1

    response = model.classify(mail)
    predicted_label = response["labels"][0]
    is_spam = 1 if predicted_label == "spam" else 0

    return is_spam


# Ask for a spam and a model to the user in the terminal

mail = input("Enter the mail to check : ")

print(is_spam(mail))
