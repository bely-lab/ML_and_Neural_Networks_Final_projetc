import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.utils import resample

from logg import logger  # assumes you have logger set up in logg.py

def bootstrap_resample(X, y):
    data = pd.concat([pd.DataFrame(X), pd.Series(y, name="target")], axis=1)
    class_counts = Counter(y)
    max_class_size = max(class_counts.values())
    resampled_data = []

    for label, count in class_counts.items():
        subset = data[data["target"] == label]
        if count < max_class_size:
            subset_resampled = resample(subset, replace=True, n_samples=max_class_size, random_state=42)
        else:
            subset_resampled = subset
        resampled_data.append(subset_resampled)

    balanced_data = pd.concat(resampled_data, axis=0).sample(frac=1, random_state=42)
    return balanced_data.drop(columns=["target"]), balanced_data["target"]

def random_forest_demographics(df, target_columns):
    results = []

    for target in target_columns:
        try:
            if target not in df.columns:
                logger.warning(f"Target '{target}' not found in dataset. Skipping...")
                continue

            filtered_data = df.dropna(subset=[target])
            if filtered_data.empty:
                logger.warning(f"No valid data available for target '{target}'. Skipping...")
                continue

            X = filtered_data.drop(columns=target_columns)
            y = filtered_data[target]

            X_resampled, y_resampled = bootstrap_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
            )

            if X_train.empty or y_train.empty:
                logger.warning(f"Empty training set for target '{target}', skipping...")
                continue

            rf_model = RandomForestClassifier(random_state=42)
            cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
            mean_cv_accuracy = np.mean(cv_scores)

            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append([target, mean_cv_accuracy, accuracy, precision, recall, f1])

        except Exception as e:
            logger.error(f"Error processing target '{target}': {e}")

    return pd.DataFrame(results, columns=["Target", "CV Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"])

def main():
    try:
        file_path = "./data/processed_data_helpfulness.csv"
        df = pd.read_csv(file_path)
        df = df.drop(columns='Age', errors='ignore')

        demographic_targets = ["age_var1", "age_mh", "Gender", "Subjective_Income"]
        results_df = random_forest_demographics(df, demographic_targets)

        os.makedirs("./results", exist_ok=True)
        output_path = "./results/demographic_prediction_result.csv"
        results_df.to_csv(output_path, index=False)

        logger.info("Demographic prediction completed successfully.")
        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to complete demographic prediction: {e}")

if __name__ == "__main__":
    main()
