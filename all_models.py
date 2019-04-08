#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:34:45 2019

@author: jenisha
"""
import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import GradientBoostingClassifier as GBC




# Split the dataset into training and testing sets
df_power = df_power_s1.copy()
predictors = df_power.columns[:-2]#df_power.columns[:-13]
frac_test_size = 0.25
X_train, X_test, y_train, y_test= train_test_split(df_power[predictors],
                                                                df_power['State'], 
                                                                test_size=frac_test_size)
print("The proportion of seizure activity in training is" ,np.sum(y_train)/len(y_train))


 
MD = [2, 5, 7, 10, 15, 20, 25]
C_param = [0.1, 0.3, 0.6, 0.8, 1, 2, 5, 10, 25, 50]
QDA_param = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
GBC_param = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900]

results_lda_0 = pd.DataFrame(index = [0], columns=range(10))
results_lda_1 = pd.DataFrame(index = [0], columns=range(10))
results_lda_oa = pd.DataFrame(index = [0], columns=range(10))
results_lda = pd.DataFrame(index = [0], columns=["class 0","class 1","overall"])
results_qda_0 = pd.DataFrame(index = QDA_param, columns=range(10))
results_qda_1 = pd.DataFrame(index = QDA_param, columns=range(10))
results_qda_oa = pd.DataFrame(index = QDA_param, columns=range(10))
results_qda = pd.DataFrame(index = QDA_param, columns=["class 0","class 1","overall"])
results_rf_0 = pd.DataFrame(index = MD, columns=range(10))
results_rf_1 = pd.DataFrame(index = MD, columns=range(10))
results_rf_oa = pd.DataFrame(index = MD, columns=range(10))
results_rf = pd.DataFrame(index = MD, columns=["class 0","class 1","overall"])
results_logit_0 = pd.DataFrame(index = C_param, columns=range(10))
results_logit_1 = pd.DataFrame(index = C_param, columns=range(10))
results_logit_oa = pd.DataFrame(index = C_param, columns=range(10))
results_logit = pd.DataFrame(index = C_param, columns=["class 0","class 1","overall"])
results_gbc_0 = pd.DataFrame(index = GBC_param, columns=range(10))
results_gbc_1 = pd.DataFrame(index = GBC_param, columns=range(10))
results_gbc_oa = pd.DataFrame(index = GBC_param, columns=range(10))
results_gbc = pd.DataFrame(index = GBC_param, columns=["class 0","class 1","overall"])

for k in range(10):
    
    XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=frac_test_size , 
                                                            random_state=k)
    
    lda = LDA()
    lda.fit(XX_train, yy_train)
    predictions = lda.predict(XX_test)
    a1 = np.sum((predictions==1)&(yy_test==1))
    a2 = np.sum((predictions==1)&(yy_test==0))
    a3 = np.sum((predictions==0)&(yy_test==1))
    a4 = np.sum((predictions==0)&(yy_test==0))
    results_lda_0[k].iloc[0] = a4/(a2+a4+0.0)
    results_lda_1[k].iloc[0] = a1/(a1+a3+0.0)
    results_lda_oa[k].iloc[0] = (a1+a4)/(a1+a2+a3+a4+0.0)
    
    for j in range(len(QDA_param)):
        qda = QDA(reg_param=QDA_param[j])
        qda.fit(XX_train, yy_train)
        predictions = qda.predict(XX_test)
        a1 = np.sum((predictions==1)&(yy_test==1))
        a2 = np.sum((predictions==1)&(yy_test==0))
        a3 = np.sum((predictions==0)&(yy_test==1))
        a4 = np.sum((predictions==0)&(yy_test==0))
        results_qda_0[k].iloc[j] = a4/(a2+a4+0.0)
        results_qda_1[k].iloc[j] = a1/(a1+a3+0.0)
        results_qda_oa[k].iloc[j] = (a1+a4)/(a1+a2+a3+a4+0.0)
    
    for j in range(len(C_param)):
        logit = LogisticRegression(C = C_param[j], class_weight="balanced")
        logit.fit(XX_train, yy_train)
        predictions = logit.predict(XX_test)
        a1 = np.sum((predictions==1)&(yy_test==1))
        a2 = np.sum((predictions==1)&(yy_test==0))
        a3 = np.sum((predictions==0)&(yy_test==1))
        a4 = np.sum((predictions==0)&(yy_test==0))
        results_logit_0[k].iloc[j] = a4/(a2+a4+0.0)
        results_logit_1[k].iloc[j] = a1/(a1+a3+0.0)
        results_logit_oa[k].iloc[j] = (a1+a4)/(a1+a2+a3+a4+0.0)
   
    for j in range(len(MD)):
        rf = RandomForest(n_estimators=300, max_depth=MD[j], class_weight="balanced")
        rf.fit(XX_train, yy_train)
        predictions = rf.predict(XX_test)
        a1 = np.sum((predictions==1)&(yy_test==1))
        a2 = np.sum((predictions==1)&(yy_test==0))
        a3 = np.sum((predictions==0)&(yy_test==1))
        a4 = np.sum((predictions==0)&(yy_test==0))
        results_rf_0[k].iloc[j] = a4/(a2+a4+0.0)
        results_rf_1[k].iloc[j] = a1/(a1+a3+0.0)
        results_rf_oa[k].iloc[j] = (a1+a4)/(a1+a2+a3+a4+0.0)
    
    for j in range(len(GBC_param)):
        gbc = GBC(n_estimators=GBC_param[j], learning_rate=0.01, max_depth=2, random_state=j)
        gbc.fit(XX_train, yy_train)
        predictions = gbc.predict(XX_test)
        a1 = np.sum((predictions==1)&(yy_test==1))
        a2 = np.sum((predictions==1)&(yy_test==0))
        a3 = np.sum((predictions==0)&(yy_test==1))
        a4 = np.sum((predictions==0)&(yy_test==0))
        results_gbc_0[k].iloc[j] = a4/(a2+a4+0.0)
        results_gbc_1[k].iloc[j] = a1/(a1+a3+0.0)
        results_gbc_oa[k].iloc[j] = (a1+a4)/(a1+a2+a3+a4+0.0)
         
results_logit["class 0"] = results_logit_0.mean(1)
results_logit["class 1"] = results_logit_1.mean(1)
results_logit["overall"] = results_logit_oa.mean(1)

results_rf["class 0"] = results_rf_0.mean(1)
results_rf["class 1"] = results_rf_1.mean(1)
results_rf["overall"] = results_rf_oa.mean(1)

results_gbc["class 0"] = results_gbc_0.mean(1)
results_gbc["class 1"] = results_gbc_1.mean(1)
results_gbc["overall"] = results_gbc_oa.mean(1)

results_lda["class 0"] = results_lda_0.mean(1)
results_lda["class 1"] = results_lda_1.mean(1)
results_lda["overall"] = results_lda_oa.mean(1)

results_qda["class 0"] = results_qda_0.mean(1)
results_qda["class 1"] = results_qda_1.mean(1)
results_qda["overall"] = results_qda_oa.mean(1)