import os
import pickle
from typing import Dict, List

import torch

from src.models.base_model import BaseModel
from src.models.classical_ml_classifier import ClassicalMLClassifier
from src.models.ltsm_classifier import LSTM_classifier
from src.models.vote import Vote
from src.utils.const import CLASSICAL_ML_MODELS
from src.models.roberta import Roberta


class WeightManager:
    folder: str = "weights"
    models_weights_dict: Dict[str, str] = {
        "lstm": "ltsm_bert.pth",
        "svm": "svm_bert.pkl",
        "logistic_regression": "logistic_regression_bert.pkl",
        "naive_bayes": "naive_bayes_bert.pkl",
    }

    def __init__(self):
        os.makedirs(self.folder, exist_ok=True)

    def models(self) -> List[str]:
        return [*CLASSICAL_ML_MODELS, "vote", "roberta", "lstm"]

    def load(self, model_type: str, vectorizer_manager=None):
        """Load a model based on its type."""

        if model_type == "vote":
            checkpoints = [
                os.path.join(self.folder, self.models_weights_dict.get(name))
                for name in CLASSICAL_ML_MODELS
            ]
            model = Vote(models_type=CLASSICAL_ML_MODELS, checkpoint_paths=checkpoints)

        elif model_type == "roberta":
            model = Roberta()

        else:
            load_path: str = os.path.join(
                self.folder, self.models_weights_dict.get(model_type)
            )

            if model_type == "lstm":
                model = LSTM_classifier(input_dim=768, checkpoint_path=load_path)
                model.get_model().load_state_dict(torch.load(load_path))
            else:
                model = ClassicalMLClassifier(model_type, load_path)

        return model

    def save_model(self, model: BaseModel, save_name: str):
        """Save a trained model."""

        if isinstance(model, ClassicalMLClassifier):
            save_path = os.path.join(self.folder, f"{save_name}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(model.classifier, f)
        else:
            save_path = os.path.join(self.folder, f"{save_name}.pth")
            torch.save(model.get_model().state_dict(), save_path)

        print(f"Model saved to {save_path}")


weight_manager = WeightManager()
