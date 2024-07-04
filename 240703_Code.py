# IMPORT PACKAGES
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

# Define the decision curve analysis function
def decision_curve_analysis(y_true, y_pred_prob, thresholds):
    """
    Perform decision curve analysis.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred_prob (array-like): Predicted probabilities.
    thresholds (array-like): Threshold values to evaluate.

    Returns:
    list: Net benefits at each threshold.
    """
    net_benefits = []
    for threshold in thresholds:
        tp = sum((y_pred_prob >= threshold) & (y_true == 1))
        fp = sum((y_pred_prob >= threshold) & (y_true == 0))
        tn = sum((y_pred_prob < threshold) & (y_true == 0))
        fn = sum((y_pred_prob < threshold) & (y_true == 1))

        nb = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
        net_benefits.append(nb)
    return net_benefits

# Import data
data = pd.read_csv('240701_Dataset.csv')

# List of continuous and categorical features
continuous_features = ['age', 'ASSETS', 'adl', 'iadl']
categorical_features = ['family', 'EDUCATION', 'LTC', 'LTC_SERVICE', 'MARITAL_STATUS', 'SOCIAL_ENGAGEMENT', 'REGION',
                        'obesity', 'DRINKING', 'SMOKING', 'HANDGRIP_STRENGTH', 'COMORBIDITIES', 'FALL', 'pain', 'EXERCISE']

# Separate features and target variable
X = data[continuous_features + categorical_features]
y = data['DEMENTIA']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Apply any necessary scaling to continuous features
scaler = MinMaxScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# Drop rows with missing values
X = X.dropna()
y = y[X.index]

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Aggregated predictions and true labels for ROC curve
all_y_true = np.array([])
all_y_pred_prob = np.array([])

# Define models (including additional models)
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Light GBM", lgb.LGBMClassifier(learning_rate=0.1, max_depth=3)),
    ("XGBoost", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, objective='binary:logistic')),
    ("CatBoost", CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, loss_function='Logloss', verbose=False)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=0)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=0)),
    ("SVM", SVC(probability=True, random_state=0)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5))
]

# Initialize figures for the combined plots
fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
fig_pr, ax_pr = plt.subplots(figsize=(10, 7))
fig_cal, ax_cal = plt.subplots(figsize=(10, 7))
fig_dec, ax_dec = plt.subplots(figsize=(10, 7))

# Initialize dictionaries to store performance metrics across folds/models
auc_scores = {model_name: [] for model_name, _ in models}
accuracies = {model_name: [] for model_name, _ in models}
auprcs = {model_name: [] for model_name, _ in models}
recalls = {model_name: [] for model_name, _ in models}
conf_matrices = {model_name: [] for model_name, _ in models}

prob_trues = {model_name: [] for model_name, _ in models}
prob_preds = {model_name: [] for model_name, _ in models}
net_benefits_dict = {model_name: [] for model_name, _ in models}
thresholds_dict = {model_name: [] for model_name, _ in models}

# Perform model training and evaluation with bootstrapping
for model_name, model in models:
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Aggregate predictions and true labels for ROC curve
        all_y_true = np.concatenate([all_y_true, y_test])
        all_y_pred_prob = np.concatenate([all_y_pred_prob, y_pred_prob])

        # Calculate performance metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        auprc = auc(recall, precision)
        confusion = metrics.confusion_matrix(y_test, y_pred)
        auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
        prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10) # calibration curve

        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = decision_curve_analysis(y_test, y_pred_prob, thresholds) # decision curve (clinical utility)

        auc_scores[model_name].append(auc_score)
        accuracies[model_name].append(accuracy)
        auprcs[model_name].append(auprc)
        recalls[model_name].append(np.mean(recall))  # Average recall values
        conf_matrices[model_name].append(confusion)

        prob_trues[model_name].extend(prob_true)
        prob_preds[model_name].extend(prob_pred)
        net_benefits_dict[model_name].append(net_benefits)
        thresholds_dict[model_name].append(thresholds)
    
    print(prob_trues)
    print(prob_preds)
    ax_cal.plot(np.array(prob_trues[model_name]), np.array(prob_preds[model_name]), marker='o', label=model_name)

    # Calculate median AUC and IQR
    median_auc = np.median(auc_scores[model_name])
    auc_iqr = np.percentile(auc_scores[model_name], 75) - np.percentile(auc_scores[model_name], 25)

    print(f"Model: {model_name}")
    print(f"Accuracy: {np.mean(accuracies[model_name])}")
    print(f"AUPRC: {np.mean(auprcs[model_name])}")
    print(f"Recall: {np.mean(recalls[model_name])}")
    print(f"Confusion Matrix:\n{np.mean(conf_matrices[model_name], axis=0)}")
    print(f"Median AUC: {median_auc:.4f} +/- {auc_iqr:.4f}")  # Display median AUC and IQR

    # Plot ROC curve using aggregated predictions
    fpr, tpr, _ = metrics.roc_curve(all_y_true, all_y_pred_prob)
    ax_roc.plot(fpr, tpr, label=f"{model_name} (Median AUC={median_auc:.4f} +/- {auc_iqr:.4f})")

    # plot AUPRC curve using aggregated predictions
    precision_vals, recall_vals, _ = precision_recall_curve(all_y_true, all_y_pred_prob)
    ax_pr.plot(recall_vals, precision_vals, label=f"{model_name} (Median AUPRC={np.mean(auprcs[model_name]):.4f})")

    # # Ensure all arrays have the same shape before averaging
    # min_len = min(len(arr) for arr in fold_prob_true)
    # fold_prob_true_trimmed = [arr[:min_len] for arr in fold_prob_true]
    # fold_prob_pred_trimmed = [arr[:min_len] for arr in fold_prob_pred]

    # # Calibration curve
    
    ax_cal.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')

    # Decision Curve Analysis
    ax_dec.plot(thresholds, np.mean(np.array(net_benefits_dict[model_name]), axis=0), label=model_name)

# Save dictionary (HW: If you want to save the result for evaluation later you can save dictionaries like this)
with open('auc_scores.pkl', 'wb') as f:
    pickle.dump(auc_scores, f)