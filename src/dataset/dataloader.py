from typing import Any, List

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(
    embeddings: Any,
    labels: List[str],
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    embeddings_tensor = embeddings.clone().detach().float()
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    tensor_dataset = TensorDataset(embeddings_tensor, labels_tensor)

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
