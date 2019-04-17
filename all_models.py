#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:34:45 2019

@author: jenisha
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBC
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix

# Split the dataset into training and testing sets
#df_power = pd.read_csv("df_power_s11.csv")
df_power = df_power_s22.copy()
predictors = df_power.columns[:-2]#df_power.columns[:-13]
t = df_power[predictors]
frac_test_size = 0.25
X_train, X_test, y_train, y_test= train_test_split(df_power[predictors],
                                                                df_power['State'], 
                                                                test_size=frac_test_size)
print("The proportion of seizure activity in training is" ,np.sum(y_train)/len(y_train))


#Gradient booster classifier
gb_classifier =  RandomForest(n_estimators=200, max_depth=5, class_weight="balanced")
gb_classifier.fit(X_train, y_train)
predictions = gb_classifier.predict(X_test)


#Importance of each feature
#fig = plt.figure()
#ax = fig.add_subplot()
#
#viz_feature_importance = FeatureImportances(gb_classifier, ax=ax)
#viz_feature_importance.fit(X_train, X_test)
#viz_feature_importance.poof()

#Classification error report
classes=["non-seizure","seizure"]
#viz_classification_report= ClassificationReport(gb_classifier , classes=classes)
#viz_classification_report.fit(X_train, y_train)  # Fit the visualizer and the model
#viz_classification_report.score(X_test, y_test)  # Evaluate the model on the test data
#c = viz_classification_report.poof()            

# Instantiate the visualizer with the classification model
#viz_ROC = ROCAUC(gb_classifier, classes=classes)
#viz_ROC.fit(X_train, y_train)  # Fit the training data to the visualizer
#viz_ROC.score(X_test, y_test)  # Evaluate the model on the test data
#g = viz_ROC.poof()             # Draw/show/poof the data



# The ConfusionMatrix visualizer taxes a model
cm = ConfusionMatrix(gb_classifier, classes=classes,label_encoder={0: 'non-seizure', 1: 'seizure'})
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
c = cm.poof()
plt.tight_layout()

#Latency
X_test_latency = df_seizures_power_22_states[df_seizures_power_22_states.columns[:-2]]
latency = gb_classifier.predict(X_test_latency)

latency_test= pd.concat([df_seizures_power_22_states['State'],pd.Series(latency)],axis=1)