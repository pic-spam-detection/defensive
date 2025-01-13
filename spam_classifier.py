from transformers import pipeline


class SpamClassifier:
    def __init__(self, model="facebook/bart-large-mnli"):
        self.model = pipeline("zero-shot-classification", model=model)

    def classify(self, sequence_to_classify):
        candidate_labels = ["spam", "not spam"]
        response = self.model(sequence_to_classify, candidate_labels)
        return response

