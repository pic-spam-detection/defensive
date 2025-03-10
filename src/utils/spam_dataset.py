from typing import Any, Dict, Literal

import pandas as pd
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str = "data/enron_spam_data.csv",
        split: Literal["train", "test"] = "train",
        test_split_ratio: int = 0.1,
        max_n_rows: int = 100000,
        content_max_length: int = 200,
    ):
        self.content_max_length = content_max_length
        data_frame = pd.read_csv(csv_file).dropna()
        data_frame = data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data_frame = data_frame[: min(len(data_frame), max_n_rows)]

        n_rows = len(self.data_frame)

        if split == "train":
            self.data_frame = self.data_frame.iloc[
                0 : int(n_rows * (1 - test_split_ratio))
            ]
        else:
            self.data_frame = self.data_frame.iloc[
                int(n_rows * (1 - test_split_ratio)) :
            ]

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = 0 if self.data_frame.iloc[idx]["Spam/Ham"] == "ham" else 1
        text = self.data_frame.iloc[idx]["Message"]
        title = self.data_frame.iloc[idx]["Subject"]

        return {"label": label, "text": text, "title": title}
