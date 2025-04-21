import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib as mt
import matplotlib.pyplot as plt
import lpd

data = pd.read_csv("prt_trst.csv")

#Show me colmuns names
for i, col in enumerate(data.columns):
    print(f"{i}: {col}")

#Step 1: Load and Split the Data


df = pd.read_csv("prt_trst.csv")
X = df.drop(columns=['trstplt'])
y = df['trstplt']

#Step 2: Understand Data Types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()



# Step 3: Preprocessing
#Use ColumnTransformer to encode categorical features and keep numeric ones untouched.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Simple transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Leave numerical columns as they are
)




#Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Step 5: Combine Preprocessing + Model into a Pipeline

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


#Train Model

model.fit(X_train, y_train)



#Result

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



