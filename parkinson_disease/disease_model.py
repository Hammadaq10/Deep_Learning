import pandas as pd
import numpy as np 
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("/Users/Hammad/Desktop/Deep_Learning/parkinson_disease/parkinsons.csv")
# print(data)

features = data.loc[:,data.columns!='status'].values[:,1:]
labels = data.loc[:,'status'].values
# print(features)
# print(labels)

scaler = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(features)
y = labels

# print(x)
# print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=10)

model1 = LogisticRegression()
model = XGBClassifier(booster='gbtree')
model.fit(x_train,y_train)
model1.fit(x_train,y_train)

pred1 = model.predict(x_test)
pred2 = model1.predict(x_test)

print(accuracy_score(y_test,pred1)*100)
print(accuracy_score(y_test,pred2)*100)
