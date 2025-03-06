import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, dataframe):
        self.data = pd.DataFrame(dataframe).dropna()
        self.text = self.data["body"].dropna()
        self.labels = self.data["ground_truth"].dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        email = self.text.iloc[idx]
        label = self.labels.iloc[idx]

        return email, torch.tensor(label, dtype=torch.long)
