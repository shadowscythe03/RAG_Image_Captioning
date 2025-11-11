import csv
import matplotlib.pyplot as plt
import numpy as np
import os

csv_path = os.path.join(os.path.dirname(__file__), 'evaluation_results_20251105_222803.csv')
out_path = os.path.join(os.path.dirname(__file__), 'plot_columns_2_3.png')

metrics = []
col2 = []
col3 = []

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        metrics.append(row[0])
        def tof(x):
            try:
                return float(x)
            except:
                return 0.0
        col2.append(tof(row[2]))
        col3.append(tof(row[3]))

# grouped bar chart
N = len(metrics)
ind = np.arange(N)
width = 0.35

fig, ax = plt.subplots(figsize=(10,5))
bars1 = ax.bar(ind - width/2, col2, width, label=header[2].strip('"'))
bars2 = ax.bar(ind + width/2, col3, width, label=header[3].strip('"'))

ax.set_xticks(ind)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Evaluation: Proposed RAG variants (columns 2 & 3)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)

def autolabel(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.3g}',
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

autolabel(bars1)
autolabel(bars2)

fig.tight_layout()
plt.savefig(out_path, dpi=200)
print('Saved grouped bar chart to', out_path)
