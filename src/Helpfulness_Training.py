import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.utils import resample
from collections import Counter
from logg import logger 

# function for class balancing/ mannual resampling
def bootstrap_resample(X, y):
    data = pd.concat([pd.DataFrame(X), pd.Series(y, name="target")], axis=1)
    class_counts = Counter(y)
    max_class_size = max(class_counts.values())
    resampled_data = []
    for label, count in class_counts.items():
        subset = data[data["target"] == label]
        subset_resampled = resample(subset, replace=True, n_samples=max_class_size, random_state=42) if count < max_class_size else subset
        resampled_data.append(subset_resampled)
    balanced_data = pd.concat(resampled_data, axis=0).sample(frac=1, random_state=42)
    return balanced_data.drop(columns=["target"]), balanced_data["target"]
# Evaluation for all kind of models
def evaluate_model(model, X, y, cv=5):
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    accuracy = accuracy_score(y, y_pred_cv)
    precision = precision_score(y, y_pred_cv, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred_cv, average='weighted')
    f1 = f1_score(y, y_pred_cv, average='weighted')
    return accuracy, precision, recall, f1

def random_forest(df, target_columns):
    results = []
    for target in target_columns:
        filtered_data = df[df[target].isin([0, 1, 2])]#taking only respondents of the specific target
        X, y = filtered_data.drop(columns=target_columns), filtered_data[target]
        X_resampled, y_resampled = bootstrap_resample(X, y)
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42)

        acc, prec, rec, f1 = evaluate_model(rf_model, X_resampled, y_resampled)
        results.append([target, acc, prec, rec, f1])
    return pd.DataFrame(results, columns=["Target", "CV Accuracy", "Precision", "Recall", "F1 Score"])

def neural_network(df, target_columns):
    results = []
    for target in target_columns:
        filtered_data = df[df[target].isin([0, 1, 2])]#taking only respondents of the specific target
        X, y = filtered_data.drop(columns=target_columns), filtered_data[target]
        X_resampled, y_resampled = bootstrap_resample(X, y)
        X_resampled = StandardScaler().fit_transform(X_resampled)
        nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=2000, random_state=42)
        acc, prec, rec, f1 = evaluate_model(nn_model, X_resampled, y_resampled)
        results.append([target, acc, prec, rec, f1])
    return pd.DataFrame(results, columns=["Target", "CV Accuracy", "Precision", "Recall", "F1 Score"])

def xgboost(df, target_columns):
    results = []
    for target in target_columns:
        filtered_data = df[df[target].isin([0, 1, 2])] #taking only respondents of the specific target
        X, y = filtered_data.drop(columns=target_columns), filtered_data[target]
        X_resampled, y_resampled = bootstrap_resample(X, y)
        xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
        acc, prec, rec, f1 = evaluate_model(xgb_model, X_resampled, y_resampled)
        results.append([target, acc, prec, rec, f1])
    return pd.DataFrame(results, columns=["Target", "CV Accuracy", "Precision", "Recall", "F1 Score"])
