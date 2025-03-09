from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def train(
        self, dataset: List[Dict[str, str]], save_path: Optional[str] = None
    ) -> Dict[str, nn.Module]:
        """Train the model on the provided dataset"""
        pass

    @abstractmethod
    def classify(self, mail: List[str]) -> Tuple[int, bool]:
        """Classify a single email"""
        pass

    @abstractmethod
    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict labels for a list of texts"""
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """Returns model itself"""
        pass
