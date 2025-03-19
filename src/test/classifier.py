from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base_model import BaseModel
from src.models.keywords_classifier import KeywordsClassifier
from src.models.roberta import Roberta
from src.utils.results import Results
from src.utils.const import DEVICE


def is_deep_learning_model(classifier: BaseModel) -> bool:
    return isinstance(classifier.get_model(), torch.nn.Module)


def test_classifier(
    classifier: BaseModel,
    dataloader: Union[DataLoader, Tuple],
    verbose: bool = True,
) -> Results:
    """Test a classifier model on the given dataset loader"""

    results = Results()

    if isinstance(classifier, KeywordsClassifier):
        mails, labels = zip(*dataloader)

        outputs = classifier.predict(mails)
        outputs = torch.tensor(outputs)

        labels = torch.tensor(labels)
        labels = labels.expand_as(outputs)

        results.add_predictions(outputs, labels)

    elif isinstance(classifier, Roberta):
        mails, labels = zip(*dataloader)
        print("Predicting... (may take some time)")
        outputs = classifier.predict(mails)
        print("Predictions done.")
        labels = torch.tensor(labels)

        results.add_predictions(outputs, labels)

    elif not is_deep_learning_model(classifier):
        # classical ML

        for emails, labels in tqdm(
            dataloader, desc="Testing classifier", disable=not verbose
        ):
            outputs = classifier.predict(emails)
            labels = torch.tensor(labels)

            results.add_predictions(outputs, labels)

    else:
        # NN

        classifier.eval()
        classifier.to(DEVICE)

        with torch.no_grad():
            for emails, labels in tqdm(
                dataloader, desc="Testing classifier", disable=not verbose
            ):
                emails, labels = emails.to(DEVICE), labels.to(DEVICE)

                outputs = classifier(emails).squeeze()
                results.add_predictions(outputs, labels)

    results.compute_metrics()

    return results
