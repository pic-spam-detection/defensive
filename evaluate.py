from typing import Literal, Optional

import click
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.classical_ml_classifier import ClassicalMLClassifier
from src.models.vectorizer import Vectorizer
from src.test.classifier import test_classifier
from src.utils.cli import (
    batch_size,
    checkpoint_path,
    classifier,
    device,
    save_results,
    test_embeddings_path,
    train_embeddings_path,
    vectorizer,
)
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
@save_results
@checkpoint_path
@train_embeddings_path
@test_embeddings_path
def test(
    classifier: Literal["naive_bayes", "logistic_regression"],
    vectorizer: Literal["sklearn", "bert"],
    batch_size: int,
    device: str,
    save_results: Optional[str],
    checkpoint_path: Optional[str],
    train_embeddings_path: Optional[str],
    test_embeddings_path: Optional[str],
):
    """Test classifiers."""
    vectorizer = Vectorizer(type=vectorizer)

    train_dataset = SpamDataset(split="train")
    test_dataset = SpamDataset(split="test")

    if train_embeddings_path is not None and test_embeddings_path is not None:
        train_embeddings = torch.load(train_embeddings_path)
        test_embeddings = torch.load(test_embeddings_path)
    else:
        train_embeddings, test_embeddings = vectorizer(
            [sample["text"] for sample in train_dataset],
            [sample["text"] for sample in test_dataset],
            save_folder="embeddings",
        )

    train_tensor = train_embeddings.clone().detach().float()
    train_labels = torch.tensor(
        [sample["label"] for sample in train_dataset], dtype=torch.long
    )
    train_dataset = TensorDataset(train_tensor, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_tensor = test_embeddings.clone().detach().float()
    test_labels = torch.tensor(
        [sample["label"] for sample in test_dataset], dtype=torch.long
    )
    test_dataset = TensorDataset(test_tensor, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = ClassicalMLClassifier(classifier, checkpoint_path)
    if checkpoint_path is None:
        model.train(train_dataloader)

    results = test_classifier(model, test_dataloader, device)

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
