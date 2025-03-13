import json
import os

import matplotlib.pyplot as plt
import pandas as pd

data = {}

files = [f for f in os.listdir("evaluation")]

for file_path in files:
    with open(os.path.join("evaluation", file_path), "r") as f:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data[file_name] = json.load(f)

df = pd.DataFrame(data).T
df = df.sort_values(by=["accuracy"])

metrics = ["accuracy", "precision", "recall", "f1"]

plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(df.index, df[metric], marker="o", label=metric)

plt.ylabel("Score")
plt.title("Model Performance Metrics")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
plt.savefig("comparison.png", dpi=300)
