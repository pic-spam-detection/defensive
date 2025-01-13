import torch
from transformers import pipeline


class SpamClassifier:
    def __init__(self, model="facebook/bart-large-mnli"):
        device = 0 if torch.cuda.is_available() else -1  # 0 for gpu, -1 for cpu
        self.model = pipeline("zero-shot-classification", model=model, device=device)

    def classify(self, sequence_to_classify):
        candidate_labels = ["spam", "not spam"]
        response = self.model(sequence_to_classify, candidate_labels)
        return response

