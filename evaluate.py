from typing import Literal, Optional

import click
import torch

from src.dataset.dataloader import get_dataloader
from src.dataset.spam_dataset import SpamDataset
from src.models.classical_ml_classifier import ClassicalMLClassifier
from src.models.keywords_classifier import KeywordsClassifier
from src.models.ltsm_classifier import LSTM_classifier
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
    classifier: str,
    vectorizer: Literal["sklearn", "bert"],
    batch_size: int,
    device: str,
    save_results: Optional[str],
    checkpoint_path: Optional[str],
    train_embeddings_path: Optional[str],
    test_embeddings_path: Optional[str],
):
    """Test classifiers."""

    train_dataset = SpamDataset(split="train")
    test_dataset = SpamDataset(split="test")

    if classifier == "keywords":
        model = KeywordsClassifier()

        test_texts = [sample["text"] for sample in test_dataset]
        test_labels = [sample["label"] for sample in test_dataset]

        test_dataloader = zip(test_texts, test_labels)
    else:
        vectorizer = Vectorizer(type=vectorizer)

        if train_embeddings_path is not None and test_embeddings_path is not None:
            train_embeddings = torch.load(train_embeddings_path, weights_only=False)
            test_embeddings = torch.load(test_embeddings_path, weights_only=False)

        else:
            train_embeddings, test_embeddings = vectorizer(
                [sample["text"] for sample in train_dataset],
                [sample["text"] for sample in test_dataset],
                save_folder="embeddings",
            )

            train_labels = [sample["label"] for sample in train_dataset]
            train_dataloader = get_dataloader(
                train_embeddings, train_labels, batch_size
            )

            test_labels = [sample["label"] for sample in test_dataset]
            test_dataloader = get_dataloader(test_embeddings, test_labels, batch_size)

            if classifier == "ltsm":
                input_dim = vectorizer.get_vocab_size() + 1
                model = LSTM_classifier(input_dim, checkpoint_path)

            else:
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
