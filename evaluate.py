import click
import torch
from torch.utils.data import DataLoader

from test.classifier import test_classifier

from models.naive_bayes import NaiveBayes

from utils.dataset import get_dataset
from utils.spam_dataset import SpamDataset
from utils.cli import (
    batch_size,
    checkpoint,
    classifier,
    device,
    save_results,
)
from utils.results import Results


@click.group()
def main():
    pass


@main.command()
@classifier
@batch_size
@checkpoint
@save_results
@device
def test(
    classifier: str,
    checkpoint: str,
    batch_size: int,
    device: str,
    save_results: str | None,
):
    """Test classifiers."""

    # Load the dataset
    dataset = get_dataset()
    train_dataset = dataset["test"]
    test_dataset = dataset["test"]

    test_dataset = SpamDataset(test_dataset)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    match classifier:
        case "naive_bayes":
            model = NaiveBayes(train_dataset)
        case _:
            raise ValueError(f"Unknown classifier variant: {classifier}")

    if checkpoint:
        model.load_state_dict(
            torch.load(checkpoint, weights_only=True, map_location=device)
        )

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
