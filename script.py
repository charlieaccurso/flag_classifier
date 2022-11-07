import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate' ]

#Print number of countries by landmass, or continent
print(df['landmass'].value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df['landmass'].isin([3,6])]

#Print the average values of the predictors for Europe and Oceania
print(df_36.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels= df_36['landmass']

#Print the variable types for the predictors
print(df_36[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df_36[var])
# print(df_36)

#Split data into a train and test set
x_train, x_test, y_train, y_test= train_test_split(data, labels, random_state=1, test_size=0.4)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []
for depth in depths:
  dt= DecisionTreeClassifier(max_depth=depth, random_state=10)
  dt.fit(x_train, y_train)
  acc_score= dt.score(x_test, y_test)
  acc_depth.append(acc_score)

#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()
plt.clf()

#Find the largest accuracy and the depth this occurs
max_acc= np.max(acc_depth)
best_depth = depths[np.argmax(acc_depth)]
print(f"The max acc is {max_acc}")
print(f"The best depth is {best_depth}")

#Refit decision tree model with the highest accuracy and plot the decision tree
plt.figure(figsize=(14,8))
dt= DecisionTreeClassifier(max_depth=best_depth, random_state=10)
dt.fit(x_train, y_train)
tree.plot_tree(
  dt, 
  feature_names=x_train.columns, 
  class_names=['Europe','Oceania'], 
  filled=True)
plt.show()
plt.clf()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
ccp= np.logspace(-3, 0, num=20)
acc_pruned= []
for ccp_value in ccp:
  dt= DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=10,
    ccp_alpha= ccp_value)
  dt.fit(x_train, y_train)
  acc_pruned.append(dt.score(x_test, y_test))
#Plot the accuracy vs ccp_alpha
plt.plot(ccp, acc_pruned)
plt.xlabel('ccp_value')
plt.ylabel('accuracy')
plt.show()
plt.clf()

#Find the largest accuracy and the ccp value this occurs
max_acc= np.max(acc_pruned)
print(f"The max acc is {max_acc}")
best_ccp= ccp[np.argmax(acc_pruned)]
print(f"The best ccp is {best_ccp}")

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dt= DecisionTreeClassifier(
  max_depth=best_depth,
  ccp_alpha=best_ccp,
  random_state=10,
)

#Plot the final decision tree
plt.figure(figsize=(14,8))
dt.fit(x_train, y_train)
tree.plot_tree(
  dt,
  feature_names=x_train.columns,
  class_names=['Europe','Oceania'],
  filled=True)
plt.show()
plt.clf()
