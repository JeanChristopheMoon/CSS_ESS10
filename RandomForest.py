import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib as mt
import matplotlib.pyplot as plt
import lpd

data = pd.read_csv('/Users/x/Desktop/DicorceRandomForest/ESS10.csv', header = 0)
print(data)

#give me the name of the colmun 
print(data.columns)
#give me all name of columns without deleting it
print(data.columns.values)
#give me indexes of columns
print(data.columns.get_loc('trstprl')) #24
print(data.columns.get_loc('trstsci')) #31

#separate colmun indexed 24 till 31
trust = data.iloc[:,24:31]
print(trust)

 #give me the data in a new csv file saved in directory
 trust.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS11.csv')
 print(trust)


#add again the data 
data = pd.read_csv('/Users/x/Desktop/DicorceRandomForest/ESS10.csv', header = 0)


#Just extract column 
party = data.iloc[:,50]
print(party)
party.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS12.csv')


#add party to the trust
print(trust)
print(party)
#add party column to the trust columns
trust['party'] = party
print(trust)

#save trust 
trust.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS13.csv')
print(trust)

#delete rows from 1 till 88
trust = trust.drop(trust.index[1:30223])
print(trust)

#delete row number 1
trust = trust.drop(trust.index[0])
print(trust)
trust.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS14.csv')

#delete rows from 1471 till end
trust = trust.drop(trust.index[1471:])
print(trust)
trust.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS15.csv')

#delete last two rows
trust = trust.drop(trust.index[1470:])
print(trust)
#delete last row
trust = trust.drop(trust.index[1469])
print(trust)
trust.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS16.csv')
print(trust)

#columns name and data type
print(trust.columns)
print(type(trust))
print(trust)



############ IS OK #################

#delte rows if it has 77 or 88 or 99
filtered_df = trust[~trust.isin([77, 88, 99, 66, 31, 18]).any(axis=1)]
print(filtered_df)
filtered_df.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS17.csv')


x = filtered_df.iloc[:,0:6]
y = filtered_df.iloc[:,7]
print(x)
print(y)


#convert 0 till 5 to 0 
for i in range(0,6):
    x = x.replace(i,0)
    print(x)

for i in range(5,11):
    x = x.replace(i,1)

print(x)
print(y)
y.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS18y.csv')
x.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS18x.csv')


#convert y values into 0 and 1 
# 0 represents not radical right party and 1 represent a radical raight party


print(type(y))
print(type(x))

y = pd.DataFrame(y)
print(type(y))
print(y)

y.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS18y.csv')
x.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS18x.csv')



numbers_to_replace = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15]
y.iloc[y['party'].isin(numbers_to_replace), 0] = 0

numbers_to_replace_right = [3,9,13,16,17]
y.iloc[y['party'].isin(numbers_to_replace_right), 0] = 1
print(y)


y.to_csv('/Users/x/Desktop/DicorceRandomForest/ESS20.csv')

print(x)
print(y)


x = pd.read_csv('/Users/x/Desktop/DicorceRandomForest/ESS18.csv')
y = pd.read_csv('/Users/x/Desktop/DicorceRandomForest/ESS20.csv')
print(x)
print(y)
print(x.columns)
print(y.columns)

#delete column Unnamed: 0
x = x.drop(x.columns[0], axis = 1)
print(x)
print(type(x))
y = y.drop(y.columns[0], axis = 1)
print(y)
# type of data
print(type(y))
y.to_frame()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 18)
clf = RandomForestClassifier(n_estimators=100, max_depth=4, max_features=3, bootstrap=True, random_state=18).fit(x_train, y_train.values.ravel())


prediction = clf.predict(x_test)
confusion_matrix(y_test, prediction) 
accuracy_score(y_test, prediction)

from sklearn.metrics import recall_score, f1_score, precision_score

recall = recall_score(y_test, prediction, average='macro')
f1 = f1_score(y_test, prediction, average='macro')
precision = precision_score(y_test, prediction, average='macro')
print(recall, f1,precision_score)
accuracy_score(y_test, prediction)


# Extract feature importances
model = RandomForestClassifier()  # Create a Random Forest model
model.fit(x_train, y_train)  # Train the model
#check its shape
print(y_train.shape)
    #If the shape is something like (n_samples, 1), it means y_train is a column vector.

y_train = np.ravel(y_train)

model.fit(x_train, y_train)  # Train the model


feature_importances = model.feature_importances_

# Sort features by importance in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = np.array(x_train.columns)[sorted_indices]
sorted_importances = feature_importances[sorted_indices]


# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  # Optional: Display most important feature at the top
plt.tight_layout()
plt.show()




