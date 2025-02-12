import torch
from tqdm import tqdm

from defensive.utils.results import Results


def test_classifier(classifier, dataloader, device, verbose=True):
    """Test a classifier model on the given dataset loader"""

    classifier.eval()
    classifier.to(device)

    results = Results()

    with torch.no_grad():
        for emails, labels in tqdm(
            dataloader, desc="Testing classifier", disable=not verbose
        ):
            emails, labels = emails.to(device), labels.to(device)

            outputs = classifier(emails).squeeze()
            results.add_predictions(outputs, labels)

    results.compute_metrics()

    return results
