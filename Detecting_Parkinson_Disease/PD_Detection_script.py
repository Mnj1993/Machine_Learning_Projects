#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:09:48 2024

@author: manojnagarajan
"""

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/manojnagarajan/Desktop/Machine_Learning/Parkenson Detection project/parkinsons/parkinsons.data')
df.head()

print("total number of people tested : ",df["name"].count())

features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

print(labels[labels==1].shape[0], labels[labels==0].shape[0])

scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

model=XGBClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

categories = ["Diagnosed Negative", "Diagnosed Positive"]
y_pred_per = [len(y_pred[y_pred==0])/len(y_pred), len(y_pred[y_pred==1])/len(y_pred)]  # Predicted probabilities or values

# Create bar chart
plt.bar(categories, y_pred_per, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Predicted Probability')
plt.title('Predictions by Category')
plt.ylim([0, 1])  # Assuming predictions are probabilities between 0 and 1
plt.show()