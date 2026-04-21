"""
summary_table.py

Generates a summary table image from transfer_results.json.

How to run:
    python cifar100/src/summary_table.py
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
results_path = os.path.join(project_root, "cifar100", "outputs", "transfer_results.json")
output_path  = os.path.join(project_root, "cifar100", "outputs", "summary_table.png")

with open(results_path) as f:
    r = json.load(f)

rows = [
    ["Model",           r["model"]],
    ["Dataset",         "CIFAR-100 (100 classes)"],
    ["Test Accuracy",   f"{r['test_accuracy']*100:.2f}%"],
    ["Test Loss",       f"{r['test_loss']:.4f}"],
    ["Phase 1 Epochs",  str(r["epochs_phase1"])],
    ["Phase 2 Epochs",  str(r["epochs_phase2"])],
    ["Label Smoothing", str(r["label_smoothing"])],
    ["Input Size",      "224 × 224"],
    ["Optimizer",       "Adam + Cosine Decay"],
    ["Fine-tuned Layers", "Top 50 of EfficientNetB3"],
]

fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=["Metric", "Value"],
    cellLoc="left",
    loc="center",
    colWidths=[0.45, 0.55],
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# Style header
for col in range(2):
    cell = table[0, col]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Style alternating rows
for row in range(1, len(rows) + 1):
    for col in range(2):
        cell = table[row, col]
        cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "white")
        cell.set_text_props(color="#2c3e50")
        if col == 1 and row == 3:  # highlight accuracy
            cell.set_text_props(color="#27ae60", fontweight="bold")

ax.set_title("EfficientNetB3 — CIFAR-100 Training Summary",
             fontsize=13, fontweight="bold", pad=16, color="#2c3e50")

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.show()
