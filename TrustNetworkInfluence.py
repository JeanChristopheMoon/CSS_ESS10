import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

trust_columns = [
    'trstprl',  # Trust in country's parliament
    'trstlgl',  # Trust in the legal system
    'trstplc',  # Trust in the police
    'trstplt',  # Trust in politicians
    'trstprt',  # Trust in political parties
    'trstep',   # Trust in the European Parliament
    'trstun'    # Trust in the United Nations
]

# Subset the relevant data
trust_data = data[trust_columns].dropna()

# Create an empty matrix to store importance scores
importance_matrix = pd.DataFrame(
    np.zeros((len(trust_columns), len(trust_columns))),
    index=trust_columns,
    columns=trust_columns
)

# Loop through each trust variable as the target
for target_col in trust_columns:
    X = trust_data.drop(columns=[target_col])
    y = trust_data[target_col]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get importances and fill in the matrix
    importances = rf.feature_importances_
    importance_matrix.loc[target_col, X.columns] = importances



plt.figure(figsize=(10, 7))
sns.set(style="whitegrid")

sns.heatmap(
    importance_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Feature Importance'}
)

plt.title("Trust Variable Influence Heatmap", fontsize=16)
plt.xlabel("Feature (used to predict)", fontsize=12)
plt.ylabel("Target (being predicted)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
