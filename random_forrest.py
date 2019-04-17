#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:05:22 2019

@author: jenisha
"""
#Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import confusion_matrix

from confusion import plot_confusion_matrix

#Read CSV file
url = "https://raw.githubusercontent.com/stiwarih/hssyp/d5b655f8d161f35382d4c965791b946b7a76c0e3/project/power_spectrum.csv"
df_power = pd.read_csv(url)


# Split the dataset into training and testing sets
df_power = df_power_s1.copy()
predictors = df_power.columns[1:-2]#df_power.columns[:-13]
frac_test_size = 0.25
X_train, X_test, y_train, y_test= train_test_split(df_power[predictors],
                                                                df_power['State'], 
                                                                test_size=frac_test_size)
print("The proportion of seizure activity in training is" ,np.sum(y_train)/len(y_train))

min_depth = 10
max_depth = 100
step_tree = 10

depth_dd = list(range(min_depth,max_depth,step_tree))

def determine_depth_tree(X_train,y_train,
                           minimum_depth = 10,maximum_depth= 100,step_tree = 10,
                           num_trees = 300,num_columns = 10):
    """
    Determine optimal depth for decision tree
     
    Input:
        X_train: Data from training set
        y_train: Labels for testing set
        minimum_depth: Smallest depth to test
        maximum_depth: Largest depth to test
        step_tree: By how much the depth increases for each iteration
        num_trees:  Number of trees in the forest
        num_columns: Number of iterations of the random forrest model
    
    Output: Dataframes with accuracy for each tree depth
    

 
        results_rf[k].iloc[j] = np.sum(predictions == yy_test)
    
    """
    
    list_depth = list(range(min_depth,max_depth+step_tree,step_tree))
    
    results_rf = pd.DataFrame(index = list_depth, columns=range(num_columns))
    
    
    for i in range(num_columns):
        X_train2, X_test2, y_train2, y_test2= train_test_split(X_train,y_train, test_size=0.25)
        for j, d in enumerate(list_depth ):
            model_rf = RandomForestClassifier(n_estimators = num_trees,max_depth =d, class_weight="balanced")
            model_rf.fit(X_train2,y_train2)
            predictions = model_rf.predict(X_test2)
            results_rf[i].iloc[j]= np.sum(predictions == y_test2)

    
    return results_rf.mean(axis=1)    

#results_rf = determine_depth_tree(X_train,y_train)

       
# Results of the first model

model_rf = RandomForestClassifier(n_estimators = 500,max_depth=1000, class_weight="balanced")
model_rf.fit(X_train,y_train)
y_pred = model_rf.predict(X_test)



# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=["Non-seizure","Seizure"],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=["Non-seizure","Seizure"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
