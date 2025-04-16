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




