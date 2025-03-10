import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base_model import BaseModel
from src.utils.results import Results


def is_deep_learning_model(classifier: BaseModel) -> bool:
    return isinstance(classifier.get_model(), torch.nn.Module)


def test_classifier(
    classifier: BaseModel, dataloader: DataLoader, device: str, verbose: bool = True
):
    """Test a classifier model on the given dataset loader"""

    results = Results()

    if not is_deep_learning_model(classifier):
        # classical ML

        for emails, labels in tqdm(
            dataloader, desc="Testing classifier", disable=not verbose
        ):
            outputs = classifier.predict(emails)

            results.add_predictions(outputs, labels)

    else:
        # NN

        classifier.eval()
        classifier.to(device)

        with torch.no_grad():
            for emails, labels in tqdm(
                dataloader, desc="Testing classifier", disable=not verbose
            ):
                emails, labels = emails.to(device), labels.to(device)

                outputs = classifier(emails).squeeze()
                results.add_predictions(outputs, labels)

    results.compute_metrics()

    return results
