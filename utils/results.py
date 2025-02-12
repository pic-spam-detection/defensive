import json
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch import Tensor


@dataclass
class Results:
    """Classification results"""

    # Training metadata
    iteration = None
    loss = None
    mode = None

    total = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    def add_predictions(self, outputs: Tensor, labels: Tensor):
        """Add predictions to the results.

        Args:
            outputs: Raw logits from a classifier
            labels: Ground truth labels
        """
        predicted = torch.sigmoid(outputs) > 0.5

        self.total += labels.size(0)
        self.true_positives += int((predicted & labels).sum().item())
        self.true_negatives += int(((~predicted) & (~labels)).sum().item())
        self.false_positives += int((predicted & (~labels)).sum().item())
        self.false_negatives += int(((~predicted) & labels).sum().item())

    def compute_metrics(self):
        """Compute the metrics from the results"""
        self.accuracy = (self.true_positives + self.true_negatives) / self.total
        self.precision = self.true_positives / (
            self.true_positives + self.false_positives
        )
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __str__(self):
        parts = []

        if self.iteration is not None:
            parts.append(f"Iteration: {self.iteration}\n")
        if self.mode is not None:
            parts.append(f"Mode: {self.mode}\n")
        if self.loss is not None:
            parts.append(f"Loss: {self.loss:.4f}\n")

        parts.append(f"Total: {self.total}\n")
        parts.append(f"True positives: {self.true_positives}\n")
        parts.append(f"True negatives: {self.true_negatives}\n")
        parts.append(f"False positives: {self.false_positives}\n")
        parts.append(f"False negatives: {self.false_negatives}\n")
        parts.append(f"Accuracy: {self.accuracy:.4f}\n")
        parts.append(f"Precision: {self.precision:.4f}\n")
        parts.append(f"Recall: {self.recall:.4f}\n")
        parts.append(f"F1: {self.f1:.4f}")

        return "".join(parts)

    @classmethod
    def from_json(cls, data):
        """Load results from a JSON string"""
        return cls(**json.loads(data))

    def to_json(self):
        """Dump results to a JSON string"""
        return json.dumps(filter_null_attrs(self.__dict__), indent=2)

    def append_log(self, path):
        """Append the results to a log file"""
        with open(path, "a") as f:
            f.write(json.dumps(filter_null_attrs(self.__dict__)) + "\n")

    @staticmethod
    def plot_loss(results):
        """Plot the loss from a list of training results"""

        losses = [result.loss for result in results if result.loss is not None]

        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()
        plt.show()

    @staticmethod
    def plot_metrics(training, testing):
        """Plot the metrics from training and testing results"""

        metrics = {
            "Accuracy": {
                "train": [result.accuracy for result in training],
                "test": [result.accuracy for result in testing],
            },
            "Precision": {
                "train": [result.precision for result in training],
                "test": [result.precision for result in testing],
            },
            "Recall": {
                "train": [result.recall for result in training],
                "test": [result.recall for result in testing],
            },
            "F1": {
                "name": [result.f1 for result in training],
                "test": [result.f1 for result in testing],
            },
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        fig.suptitle("Training Metrics")

        for ax, (metric_name, values) in zip(axes.flatten(), metrics.items()):
            ax.plot(values.get("train"), label="Train")
            ax.plot(values.get("test"), label="Test")
            ax.set_title(metric_name)

            ax.grid()
            ax.legend()

        plt.show()

    @staticmethod
    def plot_evaluations(results: list["Results"], models: list[str]):
        """Plot evaluation metrics from the testing results"""

        metrics = {
            "Accuracy": [result.accuracy for result in results],
            "Precision": [result.precision for result in results],
            "Recall": [result.recall for result in results],
            "F1": [result.f1 for result in results],
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        fig.suptitle("Testing Metrics")

        viridis = plt.cm.get_cmap("viridis", len(models))
        colors = [viridis(i) for i in range(len(models))]

        for ax, (name, values) in zip(axes.flatten(), metrics.items()):
            sorted_pairs = sorted(zip(values, models), reverse=True)
            sorted_values, sorted_models = zip(*sorted_pairs)

            ax.bar(sorted_models, sorted_values, color=colors)
            ax.set_ylim(0.9, 1.0)
            ax.set_title(name)
            ax.grid()

        plt.show()


def filter_null_attrs(d: dict) -> dict:
    """Filter out null attributes from a dictionary"""
    return {k: v for k, v in d.items() if v is not None}
