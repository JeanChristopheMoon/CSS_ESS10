import pandas as pd

# Load your dataset
df = pd.read_csv("/Users/farshad/Desktop/LLmScript/ESS11.csv")  # replace with your actual file name

# List of columns to extract
trust_columns = [
    'trstprl',  # Trust in country's parliament
    'trstlgl',  # Trust in the legal system
    'trstplc',  # Trust in the police
    'trstplt',  # Trust in politicians
    'trstprt',  # Trust in political parties
    'trstep',   # Trust in the European Parliament
    'trstun'    # Trust in the United Nations
]

# Extract the selected columns
df_trust = df[trust_columns]

# Save the extracted columns to a new CSV file
df_trust.to_csv('trust_variables.csv', index=False)

###############Phase2


# Load the CSV file that already contains 'fpo_vote'
df_fpo = pd.read_csv('output_fpo_vote.csv')

# Load the original dataset (or the one that contains the trust columns)
df_original = pd.read_csv('trust_variables.csv')  # Replace with your actual original file

# List of trust columns to extract
trust_columns = [
    'trstprl',  # Trust in country's parliament
    'trstlgl',  # Trust in the legal system
    'trstplc',  # Trust in the police
    'trstplt',  # Trust in politicians
    'trstprt',  # Trust in political parties
    'trstep',   # Trust in the European Parliament
    'trstun'    # Trust in the United Nations
]

# Extract trust columns
df_trust = df_original[trust_columns]

# Add the trust columns to df_fpo (aligning by index)
df_merged = pd.concat([df_fpo, df_trust], axis=1)

# Save the final merged dataframe to a new CSV
df_merged.to_csv('output_fpo_vote_with_trust.csv', index=False)



# remove prtvtdat and save updated CSV


# Load the merged dataset
df = pd.read_csv('output_fpo_vote_with_trust.csv')

# Drop the 'prtvtdat' column
df = df.drop(columns=['prtvtdat'])

# Save the updated dataframe
df.to_csv('output_fpo_vote_with_trust_clean.csv', index=False)


##########Second Phase#############


list(df.columns)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load your cleaned dataset
df = pd.read_csv('output_fpo_vote_with_trust_clean.csv')

# Step 2: Drop rows with missing values (you can handle them differently if needed)
df = df.dropna()

# Step 3: Define X (features) and y (target)
X = df.drop(columns=['fpo_vote'])    # all columns except the target
y = df['fpo_vote']                 # target variable

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances:\n", importances.sort_values(ascending=False))



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 0: Load the data
df = pd.read_csv("output_fpo_vote_with_trust_clean.csv")

# Show column names
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Step 1: Fill NaNs in target column BEFORE splitting
df['fpo_vote'] = df['fpo_vote'].fillna(0)

# Step 2: Separate features and target
X = df.drop(columns=['fpo_vote'])
y = df['fpo_vote']

print(f"Target name: {y.name}")
print(f"Number of NaNs in target after filling: {y.isnull().sum()}")

# Step 3: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Step 4: Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # leave numerical columns unchanged
)

# Step 5: Split the data (now clean)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Build the pipeline with preprocessing + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

#We have imbalance data proble (Fix it with ML too)
