import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Add logg.py to path
current_dir = os.path.dirname(os.path.abspath(__file__))
logg_path = os.path.abspath(os.path.join(current_dir, '..', 'utils'))
sys.path.append(logg_path)

# Import logger
from logg import logger

# Load data
try:
    file_path = os.path.abspath('./data/depression_processed_data.csv')
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded dataset from {file_path} with shape: {df.shape}")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Balancing + Evaluation
def balance_and_evaluate(model, X, y, scale=False):
    logger.info(f"Starting balance and evaluation. Scaling: {scale}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info(f"Data balanced. New shape: {X_resampled.shape}")

    if scale:
        X_resampled = StandardScaler().fit_transform(X_resampled)
        logger.info("Applied standard scaling.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    model.fit(X_train, y_train)
    logger.info(f"Model trained: {model.__class__.__name__}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f"Evaluation done. Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1

# Random Forest
def random_forest(df, target_column):
    logger.info("Running Random Forest...")
    X, y = df.drop(columns=[target_column]), df[target_column]
    model = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42)
    return pd.DataFrame([[
         "Random Forest", *balance_and_evaluate(model, X, y)
    ]], columns=[ "Model", "Accuracy", "Precision", "Recall", "F1 Score"])

# Neural Network (only on binary)
def neural_network(df, target_column):
    logger.info("Running Neural Network...")
    filtered_data = df[df[target_column].isin([0, 1])]
    X, y = filtered_data.drop(columns=[target_column]), filtered_data[target_column]
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                          max_iter=2000, random_state=42)
    return pd.DataFrame([[
        "Neural Network", *balance_and_evaluate(model, X, y, scale=True)
    ]], columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

# XGBoost
def xgboost_model(df, target_column):
    logger.info("Running XGBoost...")
    X, y = df.drop(columns=[target_column]), df[target_column]
    model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    return pd.DataFrame([[
         "XGBoost ", *balance_and_evaluate(model, X, y)
    ]], columns=[ "Model", "Accuracy", "Precision", "Recall", "F1 Score"])

# Run evaluations
logger.info("Starting model evaluations...")
target_column = 'MH7A'
results = pd.concat([
    random_forest(df, target_column),
    neural_network(df, target_column),
    xgboost_model(df, target_column)
], ignore_index=True)

# Show and save
logger.info("\nImproved Evaluation Results After Balancing:\n")
logger.info("\n" + results.to_string(index=False))

output_path = "./results/Depression_prediction_results.csv"
results.to_csv(output_path, index=False)
logger.info(f"Saved evaluation results to {output_path}")
