# IMPORT PACKAGES
import os
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

# Initialize lists to store performance metrics across folds
auc_scores = []
accuracies = []
auprcs = []
recalls = []
conf_matrices = []

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


# Perform model training and evaluation with bootstrapping
for model_name, model in models:
    fold_auc_scores = []
    fold_accuracies = []
    fold_auprcs = []
    fold_recalls = []
    fold_conf_matrices = []
    
    fold_prob_true = []
    fold_prob_pred = []
    fold_net_benefits = []
    fold_thresholds = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Aggregate predictions and true labels for ROC curve
        all_y_true = np.concatenate([all_y_true, y_test])
        all_y_pred_prob = np.concatenate([all_y_pred_prob, y_pred_prob])
pr
        # Calculate performance metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        auprc = auc(recall, precision)
        confusion = metrics.confusion_matrix(y_test, y_pred)
        auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
        prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10) # calibration curve

        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = decision_curve_analysis(y_test, y_pred_prob, thresholds) # decision curve (clinical utility)

        fold_accuracies.append(accuracy)
        fold_auprcs.append(auprc)
        fold_recalls.append(np.mean(recall))  # Average recall values
        fold_conf_matrices.append(confusion)
        fold_auc_scores.append(auc_score)

        fold_prob_true.extend(prob_true)
        fold_prob_pred.extend(prob_pred)
        fold_net_benefits.append(net_benefits)
    
    print(fold_prob_true)
    print(fold_prob_pred)
    ax_cal.plot(np.array(fold_prob_true), np.array(fold_prob_pred), marker='o', label=model_name)

    # Calculate median AUC and IQR
    median_auc = np.median(fold_auc_scores)
    auc_iqr = np.percentile(fold_auc_scores, 75) - np.percentile(fold_auc_scores, 25)

    # Store overall performance metrics
    accuracies.append(np.mean(fold_accuracies))
    auprcs.append(np.mean(fold_auprcs))
    recalls.append(np.mean(fold_recalls))  # Average recall values
    conf_matrices.append(np.mean(fold_conf_matrices, axis=0))
    auc_scores.append((median_auc, auc_iqr))

    print(f"Model: {model_name}")
    print(f"Accuracy: {np.mean(fold_accuracies)}")
    print(f"AUPRC: {np.mean(fold_auprcs)}")
    print(f"Recall: {np.mean(fold_recalls)}")
    print(f"Confusion Matrix:\n{np.mean(fold_conf_matrices, axis=0)}")
    print(f"Median AUC: {median_auc:.4f} +/- {auc_iqr:.4f}")  # Display median AUC and IQR

    # Plot ROC curve using aggregated predictions
    fpr, tpr, _ = metrics.roc_curve(all_y_true, all_y_pred_prob)
    ax_roc.plot(fpr, tpr, label=f"{model_name} (Median AUC={median_auc:.4f} +/- {auc_iqr:.4f})")

    # plot AUPRC curve using aggregated predictions
    precision_vals, recall_vals, _ = precision_recall_curve(all_y_true, all_y_pred_prob)
    ax_pr.plot(recall_vals, precision_vals, label=f"{model_name} (Median AUPRC={np.mean(fold_auprcs):.4f})")

    # # Ensure all arrays have the same shape before averaging
    # min_len = min(len(arr) for arr in fold_prob_true)
    # fold_prob_true_trimmed = [arr[:min_len] for arr in fold_prob_true]
    # fold_prob_pred_trimmed = [arr[:min_len] for arr in fold_prob_pred]

    # # Calibration curve
    
    ax_cal.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')

    # Decision Curve Analysis
    ax_dec.plot(thresholds, np.mean(np.array(fold_net_benefits), axis=0), label=model_name)
