import numpy as np
import matplotlib.pyplot as plt

# Define characteristics
labels = [
    "Latency", 
    "Budget", 
    "Cost/Accuracy", 
    "Compute", 
    "Scalability", 
    "Maintenance", 
    "Security",
    "Factual Fit"
]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Scores (1 to 5 scale, 5 being best/easiest)
cutoff = [5, 5, 5, 5, 5, 3, 5, 4] + [5]
mmr    = [4, 4, 4, 3, 3, 3, 5, 3] + [4]
rerank = [2, 3, 2, 1, 2, 5, 5, 5] + [2]
agent  = [1, 1, 1, 1, 1, 2, 2, 5] + [1]

# BuPu_r inspired palette colors
colors = ['#4d004b', '#810f7c', '#8c96c6', '#8c6bb1'] 

# Create 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))
fig.tight_layout(pad=6.0)

methods = [
    ("Cutoff Threshold", cutoff, colors[0]),
    ("MMR", mmr, colors[1]),
    ("Two-Stage Reranking", rerank, colors[2]),
    ("Agentic / Adaptive", agent, colors[3])
]

for i, (title, scores, color) in enumerate(methods):
    ax = axs[i // 2, i % 2]
    
    # Draw radar
    ax.plot(angles, scores, linewidth=2, linestyle='solid', color=color)
    ax.fill(angles, scores, color=color, alpha=0.3)
    
    # Grid lines and ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='black', size=8)
    ax.set_rlabel_position(0)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="grey", size=7)
    ax.set_ylim(0, 5)
    
    ax.set_title(title, size=12, color='black', y=1.2)

plt.suptitle("Performance Octagons by Retrieval Method\n(Higher scores are more favorable)", size=16, y=1.02)
plt.show()
