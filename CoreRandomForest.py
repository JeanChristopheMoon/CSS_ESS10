import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib as mt
import matplotlib.pyplot as plt
import lpd

Data = pd.read_csv("/Users/farshad/Desktop/LLmScript/RandomForest/ESS11.csv")

#Error : This is a common warning from pandas when a column has mixed data types (e.g., some rows are numbers, others are strings).
#Tell pandas to process the file in chunks, so it can better infer the types

data = pd.read_csv("/Users/farshad/Desktop/LLmScript/RandomForest/ESS11.csv", low_memory=False)

#Show me colmuns names

for i, col in enumerate(data.columns):
    print(f"{i}: {col}")

#Extracting
#First step is to save thoese ranges
cols_to_extract = list(range(21, 28)) + list(range(65, 90))

#Second stage is to extract them
selected_columns = data.iloc[:, cols_to_extract]

#Save it
selected_columns.to_csv("prt_trst.csv", index=False)

# Show me the names of column
# Load the CSV file into a DataFrame
data = pd.read_csv("prt_trst.csv")

#Now Print names of the columns
for i, col in enumerate(data.columns):
    print(f"{i}: {col}")





