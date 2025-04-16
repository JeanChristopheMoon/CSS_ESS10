import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as mt
import matplotlib.pyplot as plt



# Load the CSV file (use full path if needed)
data = pd.read_csv("prt_trst.csv")

#Shows columns
for i, col in enumerate(data.columns):
    print(f"{i}: {col}")


# Select the first 10 columns (columns 0 to 9)
selected_data = data.iloc[:, 0:7]


#Color of column
colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'gold', 
          'turquoise', 'coral', 'lightpink', 'khaki', 'lightslategray']

# Define descriptive names for each variable
column_names = [
    'Trust in country\'s parliament',
    'Trust in the legal system',
    'Trust in the police',
    'Trust in politicians',
    'Trust in political parties',
    'Trust in the European Parliament',
    'Trust in the United Nations'
]

#Calculate the average for each column (variable) from 0 to 9
column_averages = selected_data.mean()

#Plot the averages in a single bar chart
plt.figure(figsize=(12, 6))
column_averages.plot(kind='bar', color=colors)

# Set chart title and labels
plt.title("Comparison of Averages for Variables 0 to 9")
plt.xlabel("Variables (Columns 0 to 9)")
plt.ylabel("Average Value")
plt.xticks(range(len(column_names)), column_names, rotation=45)
plt.tight_layout()

plt.show()

