import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

directory = "results"

dataframes = []

files = [
    "microsoft-Phi-3.5-mini-instruct_Zero-shot.csv",
    "Qwen2.5-3B-Instruct-GPTQ-Int8_Zero-shot.csv",
    "GPT3_Zero-shot.csv",
    "microsoft-Phi-3.5-mini-instruct_Few-shot.csv",
    "Qwen2.5-3B-Instruct-GPTQ-Int8_Few-shot.csv",
    "GPT3_Few-shot.csv",
]

for filename in files:
    print(filename)
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)

        dataset_name = filename.split(".csv")[0]
        df["dataset"] = dataset_name

        dataframes.append(df)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()

colors = ['g','c', 'b','r']
markers = ['o', 's', 's', 'o']

for i, df in enumerate(dataframes):
    dataset_name = df["dataset"].iloc[0]
    df = df.sort_values(by="f1", ascending=True)

    for j, (metric, style) in enumerate(zip(
        ["recall", "accuracy", "precision", "f1"],
        ["-", "--", "-", "-"]
    )):
        sns.lineplot(
            x="model",
            y=metric,
            data=df,
            label=metric,
            ax=axes[i],
            marker=markers[j],
            linestyle=style,
            color=colors[j]
        )

    axes[i].set_title(dataset_name, fontsize=14)
    axes[i].set_xlabel("Model", fontsize=12)
    axes[i].set_ylabel("Score", fontsize=12)
    axes[i].grid(True, linestyle="--", alpha=0.6)
    axes[i].get_legend().remove()

for j in range(len(dataframes), len(axes)):
    axes[j].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=4,
    fontsize=12,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("dataset_evaluation.png", dpi=300)
