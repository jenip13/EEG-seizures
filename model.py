#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:54:38 2019

@author: jenisha
"""
#Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from confusion import plot_confusion_matrix

#Read CSV file
#url = "https://raw.githubusercontent.com/stiwarih/hssyp/d5b655f8d161f35382d4c965791b946b7a76c0e3/project/power_spectrum.csv"
#df_power = pd.read_csv(url)



# Split the dataset into training and testing sets
df_power = df_power_s2.copy()
predictors = df_power.columns[:-2]#df_power.columns[:-13]
frac_test_size = 0.25
X_train, X_test, y_train, y_test= train_test_split(df_power[predictors],
                                                                df_power['State'], 
                                                                test_size=frac_test_size)
print("The proportion of seizure activity in training is" ,np.sum(y_train)/len(y_train))





def determine_reg_strength(X_train,y_train,
                           minimum_regularization = 0.1,maximum_regularization = 5,num_spaces = 20,
                           num_columns = 10):
    """
    Determine inverse regularization strength parameter for logistic regressiob model
    where Regularization = applying a penalty to increasing the magnitude of model parameters
     to decrease overfitting.
     
    Input:
        X_train: Data from training set
        y_train: Labels for testing set
        minimum_regularization: Smallest inverse regularization strength parameter to test
        maximum_regularization: Largest inverse regularization strength parameter to test
        num_spaces: Number of coefficients to test between minimum_regularization and maximum_regularization
        num_columns: Number of iterations of the logistic regression model
    
    Output: Dataframes with accuracy for each inverse regularization strength parameter 
    
    """
    
    list_inverse_regularization_strength = list(np.linspace(minimum_regularization,maximum_regularization,num_spaces))
    
    results_lg = pd.DataFrame(index = list_inverse_regularization_strength, columns=range(num_columns))
    
    
    for i in range(num_columns):
        X_train2, X_test2, y_train2, y_test2= train_test_split(X_train,y_train, test_size=0.25)
        for j, inverse_regularization_strength in enumerate(list_inverse_regularization_strength):
            model_lg = LogisticRegression(C = inverse_regularization_strength, class_weight="balanced")
            model_lg.fit(X_train2,y_train2)
            predictions = model_lg.predict(X_test2)
            results_lg[i].iloc[j]= np.sum(predictions == y_test2)
    
    return results_lg.mean(axis=1)       
    
#C_list = determine_reg_strength(X_train,y_train,
#                           minimum_regularization = 2,maximum_regularization = 3,num_spaces = 20,
#                           num_columns = 5)

       
# Results of the first model

model_lg = LogisticRegression(C = 2.4, class_weight="balanced")
model_lg.fit(X_train,y_train)
y_pred = model_lg.predict(X_test)




#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=["Non-seizure","Seizure"],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=["Non-seizure","Seizure"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()



true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_pred, y_test).ravel()

print("Accuracy: ", true_positive/(true_positive+false_positive))


#t = df_power_spectrum["type"][y_testing.index] 

#d2 = np.sum((predictions==1)&(y_testing==0)&(t=="day"))
#d4 = np.sum((predictions==0)&(y_testing==0)&(t=="day"))
#n2 = np.sum((predictions==1)&(y_testing==0)&((t=="night")|(t=="before")|(t=="after")))
#n4 = np.sum((predictions==0)&(y_testing==0)&((t=="night")|(t=="before")|(t=="after")))
#         
#df_pred = pd.DataFrame(columns=["Seizure","Normal - Night", "Normal - Day"], index=["Predicted Seizure","Predicted Normal"])
#df_pred["Seizure"].iloc[0] = round(a1/(a1+a3+0.0),3)
#df_pred["Seizure"].iloc[1] = round(a3/(a1+a3+0.0),3)
#df_pred["Normal - Night"].iloc[0] = round(n2/(a2+a4+0.0) ,3)   
#df_pred["Normal - Night"].iloc[1] = round(n4/(a2+a4+0.0) ,3)   
#df_pred["Normal - Day"].iloc[0] = round(d2/(a2+a4+0.0) ,3)   
#df_pred["Normal - Day"].iloc[1] = round(d4/(a2+a4+0.0) ,3)    
#df_pred 
