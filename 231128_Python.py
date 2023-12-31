# IMPORT PACKAGES
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from model_utils import lr_train, rf_train

# Import data
data = pd.read_csv('./EXPORT.csv')
X = data[['family', 'age', 'EDUCATION', 'ASSETS', 'LTC', 'LTC_SERVICE', 'MARITAL_STATUS', 'SOCIAL_ENGAGEMENT', 'REGION',
         'adl', 'iadl', 'obesity', 'DRINKING', 'SMOKING', 'HANDGRIP_STRENGTH', 'COMORBIDITIES', 'FALL', 'pain', 'EXERCISE']]
y = data['DEMENTIA']

# Create categorical variables 
X = pd.get_dummies(X, columns=['family', 'age', 'EDUCATION', 'ASSETS', 'LTC', 'LTC_SERVICE', 'MARITAL_STATUS', 'SOCIAL_ENGAGEMENT', 'REGION',
         'adl', 'iadl', 'obesity', 'DRINKING', 'SMOKING', 'HANDGRIP_STRENGTH', 'COMORBIDITIES', 'FALL', 'pain', 'EXERCISE'], drop_first=True)

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classes = ['NO DEMENTIA', 'DEMENTIA']

# Define models (including Random Forest)

lr_perf_df, lr = lr_train(X_train, y_train) # hyperparamter tuning result, best lr model
rf_perf_df, rf = rf_train(X_train, y_train) # hyperparamter tuning result, best rf model

models = [
    ("Logistic Regression", lr),
    ("Light GBM", lgb.LGBMClassifier(learning_rate=0.1, max_depth=3)),
    ("XGBoost", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, objective='binary:logistic')),
    ("CatBoost", CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, loss_function='Logloss', verbose=False)),
    ("Random Forest", rf),
]

# Train and evaluate models
for model_name, model in models:
    # Hyperparameter Tuning
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    confusion = metrics.confusion_matrix(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"AUC: {auc:.4f}")  # Display AUC with 4 decimal places

    # Plot ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")

plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()