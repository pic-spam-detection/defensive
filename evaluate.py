from typing import Literal, Optional

import click
from torch.utils.data import DataLoader

from src.models.classical_ml_classifier import ClassicalMLClassifier
from src.test.classifier import test_classifier
from src.utils.cli import (batch_size, checkpoint_path, classifier, device,
                           save_results, vectorizer,
                           vectorizer_checkpoint_path)
from src.utils.dataset import get_dataset
from src.utils.results import Results
from src.utils.spam_dataset import SpamDataset


@click.group()
def main():
    pass


@main.command()
@classifier
@vectorizer
@batch_size
@device
@checkpoint_path
@vectorizer_checkpoint_path
@save_results
def test(
    classifier: Literal["naive_bayes", "logistic_regression"],
    vectorizer: Literal["sklearn", "bert"],
    batch_size: int,
    device: str,
    save_results: Optional[str],
    checkpoint_path: str,
    vectorizer_checkpoint_path: Optional[str],
):
    """Test classifiers."""

    # Load the dataset
    dataset = get_dataset()
    test_dataset = dataset["test"]

    test_dataset = SpamDataset(test_dataset)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = ClassicalMLClassifier(
        classifier, vectorizer, checkpoint_path, vectorizer_checkpoint_path
    )

    if checkpoint_path is None:
        model.train(dataset["train"])

    results = test_classifier(model, dataloader, device)

    print(results)

    if save_results is not None:
        with open(save_results, "w") as f:
            f.write(results.to_json())


@main.command()
@click.option(
    "--path",
    help="The path to the logs",
    type=str,
)
def logs(path: str):
    """Plot training logs."""

    training: list[Results] = []
    testing: list[Results] = []

    # Read the log file
    with open(path, "r") as f:
        for line in f:
            result = Results.from_json(line)
            if result.mode == "train":
                training.append(result)
            else:
                testing.append(result)

    Results.plot_loss(training)
    Results.plot_metrics(training, testing)


if __name__ == "__main__":
    main()
