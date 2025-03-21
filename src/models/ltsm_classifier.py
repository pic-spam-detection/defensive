from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel
from src.models.ltsm import LSTM
from src.utils.const import DEVICE


class LSTM_classifier(BaseModel):
    # ltsm params
    hidden_dim = 256
    output_dim = 1
    droupout = 0.5

    # training params
    lr = 0.0001
    n_epochs = 20
    decay_factor = 1.00004

    def __init__(
        self,
        input_dim: int,
        checkpoint_path: Optional[str] = None,
    ):
        super(LSTM_classifier, self).__init__()

        self.model = LSTM(
            input_dim, self.hidden_dim, self.output_dim, droupout=self.droupout
        )

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    def forward(self, input_sequence):
        """Forward pass of the LSTM classifier"""
        return self.model(input_sequence)

    def train_model(
        self,
        dataloader: DataLoader,
        save_path: Optional[str] = "lstm_checkpoint.pth",
    ):
        self.model = self.model.to(DEVICE)
        bce = nn.BCELoss().to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.decay_factor
        )

        losses = []

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

                # forward
                output_y = self.model(batch_X)  # (batch_size, 1)
                output_y = output_y.squeeze(1)  # (batch_size,)

                # loss
                loss = bce(output_y, batch_y.float())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add loss value
                epoch_loss += loss.item()

            scheduler.step()

            epoch_loss /= len(dataloader)
            losses.append(epoch_loss)
            print(f"Loss: {epoch_loss:.4f}")

        checkpoint = {
            "epoch": self.n_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses[-1],
        }
        torch.save(checkpoint, save_path)

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, self.n_epochs + 1), losses, marker="o", linestyle="-", color="b"
        )
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig("lstm_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict labels for a list of texts"""
        pass

    def classify(self, mail: Dict[str, str]) -> Tuple[int, bool]:
        """
        Classify a single email
        Args:
            mail: dict
                mail information (address, domain, domain_extension, subject, body, ground_truth)
        """
        pass

    def get_model(self) -> LSTM:
        return self.model
