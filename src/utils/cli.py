import click
import torch


def batch_size(func):
    return click.option(
        "--batch-size",
        default=32,
        help="The batch size to use for training/testing",
        type=int,
    )(func)


def device(func):
    return click.option(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=click.Choice(["cpu", "cuda"]),
        help="The device to use. Defaults to 'cuda' if available, otherwise 'cpu'",
    )(func)


def classifier(func):
    return click.option(
        "--classifier",
        help="The classifier to use",
        type=click.Choice(["naive_bayes", "logistic_regression", "svm", "keywords"]),
    )(func)


def checkpoint_path(func, required=False):
    return click.option(
        "--checkpoint-path",
        help="The path to the model checkpoint",
        required=required,
        type=str,
    )(func)


def save_results(func):
    return click.option(
        "--save-results",
        default=None,
        help="The path to save the results. If not provided, results will not be saved.",
        type=str,
    )(func)


def vectorizer(func):
    return click.option(
        "--vectorizer",
        help="The vectorizer to use",
        type=click.Choice(["sklearn", "bert"]),
    )(func)


def train_embeddings_path(func):
    return click.option(
        "--train-embeddings-path",
        help="The path to saved embeddings.",
        type=str,
    )(func)


def test_embeddings_path(func):
    return click.option(
        "--test-embeddings-path",
        help="The path to saved embeddings.",
        type=str,
    )(func)
