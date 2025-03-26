import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données depuis les fichiers JSON
data = {}
files = [f for f in os.listdir("evaluation")]

for file_path in files:
    with open(os.path.join("evaluation", file_path), "r") as f:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data[file_name] = json.load(f)

# Sélectionner 4 modèles
selected_models = ["logistic_regression_sklearn", "naive_bayes_sklearn", "svm_sklearn", "vote_sklearn_33"]

# Vérifier que l'on a bien 4 modèles
if len(selected_models) != 4:
    raise ValueError("Veuillez sélectionner exactement 4 modèles.")

# Préparer les matrices de confusion
conf_matrices = []
for model in selected_models:
    results = data[model]
    conf_matrix = np.array([
        [results["true_negatives"], results["false_positives"]],
        [results["false_negatives"], results["true_positives"]]
    ])
    conf_matrices.append((model, conf_matrix))

labels = ["Negative", "Positive"]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Afficher les matrices de confusion
for i, (model, matrix) in enumerate(conf_matrices):
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[i])
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("True Label")
    axes[i].set_title(f"Confusion Matrix ({model})")

plt.tight_layout()
plt.show()
plt.savefig("confusion_matrices.png", dpi=300, bbox_inches='tight')