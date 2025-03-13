import json
import os
from typing import Literal, Optional

import click
import pandas as pd
import torch

from src.dataset.dataloader import get_dataloader
from src.dataset.spam_dataset import SpamDataset
from src.models.classical_ml_classifier import ClassicalMLClassifier
from src.models.keywords_classifier import KeywordsClassifier
from src.models.ltsm_classifier import LSTM_classifier
from src.models.vectorizer import Vectorizer
from src.models.vote import Vote
from src.models.weight_manager import weight_manager
from src.test.classifier import test_classifier
from src.utils.cli import (
    batch_size,
    checkpoint_path,
    classifier,
    device,
    file_path,
    save_checkpoint,
    save_results,
    test_embeddings_path,
    train_embeddings_path,
    vectorizer,
    vote_threshold,
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
@vote_threshold
@save_checkpoint
def test(
    classifier: str,
    vectorizer: Literal["sklearn", "bert"],
    batch_size: int,
    device: str,
    save_results: Optional[str],
    checkpoint_path: Optional[str],
    train_embeddings_path: Optional[str],
    test_embeddings_path: Optional[str],
    vote_threshold: float,
    save_checkpoint: bool = True,
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
        vectorizer_manager = Vectorizer(type=vectorizer)

        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        if train_embeddings_path is not None and test_embeddings_path is not None:
            train_embeddings = torch.load(train_embeddings_path, weights_only=False)
            test_embeddings = torch.load(test_embeddings_path, weights_only=False)

        else:
            train_embeddings, test_embeddings = vectorizer_manager(
                [sample["text"] for sample in train_dataset],
                [sample["text"] for sample in test_dataset],
                save_folder="embeddings",
            )

        test_labels = [sample["label"] for sample in test_dataset]
        test_dataloader = get_dataloader(test_embeddings, test_labels, batch_size)

        if classifier == "ltsm":
            input_dim = vectorizer_manager.get_vocab_size() + 1
            model = LSTM_classifier(input_dim, checkpoint_path)

        elif classifier == "vote":
            model = Vote(checkpoint_path=checkpoint_path, threshold=vote_threshold)

        else:
            model = ClassicalMLClassifier(classifier, checkpoint_path)

        if checkpoint_path is None:
            train_labels = [sample["label"] for sample in train_dataset]
            train_dataloader = get_dataloader(
                train_embeddings, train_labels, batch_size
            )
            model.train(train_dataloader)

    results = test_classifier(model, test_dataloader, device)

    print(results)

    if save_results is not None:
        with open(os.path.join("evaluation/", save_results), "w") as f:
            f.write(results.to_json())

    if save_checkpoint:
        weight_manager.save_model(model, f"{classifier}_{vectorizer}")


@main.command()
@file_path
@batch_size
@save_results
def evaluate_dataset(
    file_path: str,
    batch_size: int,
    save_results: str = "evaluation.csv",
):
    """Run evaluation for a given file of email over all classifiers."""
    dataset = SpamDataset(csv_file=file_path)

    vectorizer = Vectorizer(type="bert")

    embeddings = vectorizer.get_embeddings([sample["text"] for sample in dataset])
    labels = [sample["label"] for sample in dataset]
    dataloader = get_dataloader(embeddings, labels, batch_size)

    total_result = []
    for model_name in weight_manager.models():
        print(f"\nProcessing {model_name}...")

        model = weight_manager.load(model_name)
        results = test_classifier(model, dataloader)
        results_dict = json.loads(results.to_json())

        results_df = pd.DataFrame([results_dict])
        results_df = results_df[["accuracy", "recall", "precision", "f1"]]
        results_df.insert(0, "model", model_name)

        total_result.append(results_df)

    final_df = pd.concat(total_result, ignore_index=True)
    final_df.to_csv(save_results, index=False)


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
