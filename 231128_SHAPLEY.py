#IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb
from multiprocessing.sharedctypes import Value
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from pathlib import Path
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier, Pool

import xgboost
import shap

# import data 
data = pd.read_csv('C:/Users/82102/Desktop/2023 Papers/1. 강바다 교수님 ML Paper/EXPORT.csv')
X = data[['family', 'age', 'EDUCATION', 'LTC', 'LTC_SERVICE', 'MARITAL_STATUS', 'SOCIAL_ENGAGEMENT', 'REGION',
         'adl', 'iadl', 'obesity', 'DRINKING', 'SMOKING', 'HANDGRIP_STRENGTH', 'COMORBIDITIES', 'FALL', 'pain', 'EXERCISE']]
y = data['MCI']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# create categorical variables 
X = pd.get_dummies(X, columns=['family', 'age', 'EDUCATION', 'LTC', 'LTC_SERVICE', 'MARITAL_STATUS', 'SOCIAL_ENGAGEMENT', 'REGION',
         'adl', 'iadl', 'obesity', 'DRINKING', 'SMOKING', 'HANDGRIP_STRENGTH', 'COMORBIDITIES', 'FALL', 'pain', 'EXERCISE'], drop_first=True)

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classes = ['NO MCI', 'MCI']

# 3) XGBoost 
# create model instance
bst = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
y_pred = bst.predict(X_test)

# compute SHAP values
explainer = shap.Explainer(bst, X)
shap_values = explainer(X)
shap.plots.bar(shap_values)
shap.plots.bar(shap_values, max_display=40)