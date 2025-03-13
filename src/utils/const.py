import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSICAL_ML_MODELS = ["logistic_regression", "naive_bayes", "svm"]
