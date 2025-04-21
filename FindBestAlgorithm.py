import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Get report as dictionary
report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()

# Filter out 'accuracy', 'macro avg', 'weighted avg'
df = df[df.index.str.isdigit()]  # keep only classes

# Plot
df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Precision, Recall, and F1-score per Trust Level')
plt.xlabel('Trust Level')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

