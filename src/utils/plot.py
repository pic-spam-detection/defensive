import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {}

files = [f for f in os.listdir("evaluation")]

for file_path in files:
    with open(os.path.join("evaluation", file_path), "r") as f:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data[file_name] = json.load(f)

df = pd.DataFrame(data).T
df = df.sort_values(by=["accuracy"])

# Set the style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette("viridis", 4)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

# Plot metrics with different styles
metrics = ["accuracy", "precision", "recall", "f1"]
markers = ['o', 's', '^', 'D']
line_styles = ['-', '--', '-.', ':']

for i, metric in enumerate(metrics):
    plt.plot(np.arange(len(df)), df[metric], 
             marker=markers[i], markersize=8, alpha=0.8,
             linestyle=line_styles[i], linewidth=2.5, 
             label=metric, color=colors[i])

# Enhance the appearance
plt.ylabel("Score", fontsize=14)
plt.xlabel("Models", fontsize=14)
plt.title("Model Performance Metrics", fontsize=18, fontweight='bold', pad=15)

# Add a light background to the legend and position it outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), 
           frameon=True, fontsize=12, facecolor='white', framealpha=0.8)

# Set the x-ticks and labels with better spacing
plt.xticks(np.arange(len(df)), df.index, rotation=45, ha='right', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=10)

# Add a light grid
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust the layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Add subtle annotation showing the ranking
for i, (idx, row) in enumerate(df.iterrows()):
    plt.annotate(f"#{len(df)-(i)}", (i, min(row[metrics])-0.05), 
                ha='center', va='top', fontsize=8, alpha=0.7)

# Save with high resolution
plt.savefig("comparison.png", dpi=300, bbox_inches='tight')