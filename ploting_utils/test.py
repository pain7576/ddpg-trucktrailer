import matplotlib.pyplot as plt

# Data
categories = ["cont+no_back", "cont+back", "Peak", "Peak+smooth"]
accuracy = [37.5, 27.1, 57.9, 61.2]

# Use XKCD style for sketch-like effect
plt.xkcd()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, accuracy, color="lightcoral", edgecolor="black", hatch='//')

# Labels and title
ax.set_ylabel("Accuracy")
ax.set_xlabel("Diff. Reward model ")
ax.set_title("Results reward function space")

# Y-axis limits
ax.set_ylim(0, 65)

# Add value labels on top of bars
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
