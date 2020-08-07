# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:58:23 2020

@author: GENIUS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('features.csv')

#creating the 2015 dataset
X = dataset.iloc[:, [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,38,40,41,42,43,44]].values
y = dataset.iloc[:, 2].values
 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [20])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#creating the 2019 test set
X_test1 = dataset.iloc[:, [0,1,3,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [20])
X_test1 = onehotencoder.fit_transform(X_test1).toarray()

#training the model on the 2015 dataset
from catboost import CatBoostRegressor
regressor = CatBoostRegressor(25)
regressor.fit(X_train, y_train)

#predicting for the 2019 dataset
y_pred = regressor.predict(X_test1)
y_pred[y_pred<0] = 0

#storing the prediction values
datasetsub = pd.read_csv('coordinates.csv')
datasetsub['target_2019'] = y_pred
datasetsub.to_csv('pred_2019.csv')
